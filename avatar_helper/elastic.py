"""Shape-preserving fit: solve for the whole garment at once instead of placing vertices one by one.

The per-vertex positions produced by the fitting rules (smooth warp, skin and layer
binding) are only suggestions. From them we estimate how the fabric should stretch and
turn around each vertex, smooth that over a short distance (removing bumps, which are
small-scale), and rebuild the garment from its original edges transformed that way. The
garment's own detail (cables, folds, seams) comes from the original; only smooth changes
come from the suggestions. Collisions become "be at least here" constraints, and the
solve spreads each correction smoothly instead of moving single vertices.
"""
import numpy as np
from mathutils.kdtree import KDTree

from . import mesh_utils as mu

RIGID_WEIGHT = 100.0  # edge weight inside rigid parts: practically unbendable
LINK_WEIGHT = 10.0  # weight of links across a thin part or between interlocked pieces
CONSTRAINT_WEIGHT = 10.0  # pull of a collision constraint, relative to an edge
LINK_GAP = 0.002  # separate pieces closer than this are linked (a ring and its strap loop)


class _Graph:
    """Weighted edge graph for the solve."""

    def __init__(self, n, e0, e1, w):
        self.n, self.e0, self.e1, self.w = n, e0, e1, w
        self.deg = np.bincount(e0, weights=w, minlength=n) + np.bincount(e1, weights=w, minlength=n)

    def laplacian(self, x):
        s = np.stack([np.bincount(self.e0, weights=self.w * x[self.e1, c], minlength=self.n)
                      + np.bincount(self.e1, weights=self.w * x[self.e0, c], minlength=self.n)
                      for c in range(3)], axis=1)
        return self.deg[:, None] * x - s

    def solve(self, d, pull, pull_to, x0, iterations=600, tol=1e-7):
        """Minimise sum_e w_e |x_j - x_i - d_e|^2 + sum_i pull_i |x_i - pull_to_i|^2."""
        b = (np.stack([np.bincount(self.e1, weights=self.w * d[:, c], minlength=self.n)
                       - np.bincount(self.e0, weights=self.w * d[:, c], minlength=self.n) for c in range(3)], axis=1)
             + pull[:, None] * pull_to)
        diag = (self.deg + pull)[:, None]

        def apply(x):
            return self.laplacian(x) + pull[:, None] * x

        x = x0.copy()
        r = b - apply(x)
        z = r / diag
        p = z.copy()
        rz = (r * z).sum(0)
        b_norm = max(np.sqrt((b * b).sum(0)).max(), 1e-12)
        for _ in range(iterations):
            ap = apply(p)
            alpha = rz / np.maximum((p * ap).sum(0), 1e-30)
            x += alpha * p
            r -= alpha * ap
            if np.sqrt((r * r).sum(0)).max() < tol * b_norm:
                break
            z = r / diag
            rz_new = (r * z).sum(0)
            p = z + (rz_new / np.maximum(rz, 1e-30)) * p
            rz = rz_new
        return x


def links(verts, tris, edges, thickness):
    """Pairs tying together surfaces that share no edges: the two sides of a thin part
    (a hem's lining under its outer layer, a strap's inner and outer side), found by
    casting a short ray through the material, and separate pieces touching each other."""
    pairs = []
    if len(tris):
        bvh = mu.build_bvh(verts, tris)
        normals = mu.vertex_normals(verts, tris)
        for i, (p, n) in enumerate(zip(verts.tolist(), normals.tolist())):
            for sign in (-1.0, 1.0):
                d = (n[0] * sign, n[1] * sign, n[2] * sign)
                start = (p[0] + d[0] * 1e-5, p[1] + d[1] * 1e-5, p[2] + d[2] * 1e-5)
                loc, nrm, idx, dist = bvh.ray_cast(start, d, thickness)
                if idx is None or dist < 1e-4:
                    continue
                # The other side faces the other way; this skips the vertex's own neighbouring faces.
                if nrm[0] * n[0] + nrm[1] * n[1] + nrm[2] * n[2] > -0.3:
                    continue
                tri = tris[idx]
                j = int(tri[np.argmin(np.linalg.norm(verts[tri] - np.array(loc[:]), axis=1))])
                if j != i:
                    pairs.append((i, j))

    labels = mu.islands(len(verts), edges)
    if labels.max() > 0:
        kd = KDTree(len(verts))
        for i, p in enumerate(verts.tolist()):
            kd.insert(p, i)
        kd.balance()
        for i, p in enumerate(verts.tolist()):
            for _co, j, _d in kd.find_range(p, LINK_GAP):
                if labels[j] != labels[i] and j > i:
                    pairs.append((i, j))
                    break
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.unique(np.sort(np.array(pairs, dtype=np.int64), axis=1), axis=0)
    return pairs


def local_transforms(rest, targets, tris, e0, e1):
    """Per-vertex 3x3 map that best carries the rest edges around a vertex onto the
    suggested positions. The normal is included so the map is defined across the surface."""
    n = len(rest)
    src = np.concatenate([e0, e1])
    dst = np.concatenate([e1, e0])
    e = rest[dst] - rest[src]
    f = targets[dst] - targets[src]
    m = np.zeros((n, 3, 3))
    c = np.zeros((n, 3, 3))
    for a in range(3):
        for b in range(3):
            m[:, a, b] = np.bincount(src, weights=e[:, a] * e[:, b], minlength=n)
            c[:, a, b] = np.bincount(src, weights=f[:, a] * e[:, b], minlength=n)
    length = np.sqrt(np.maximum(np.trace(m, axis1=1, axis2=2) / np.maximum(np.bincount(src, minlength=n), 1), 1e-12))
    nr = mu.vertex_normals(rest, tris) * length[:, None]
    nt = mu.vertex_normals(targets, tris) * length[:, None]
    m += nr[:, :, None] * nr[:, None, :]
    c += nt[:, :, None] * nr[:, None, :]
    m += np.eye(3)[None] * (1e-6 * length ** 2)[:, None, None]
    return c @ np.linalg.inv(m)


def solve_fit(verts, tris, edges, targets, rigid, rigid_maps, collision_push, detail, thickness, iterations):
    """Fit the garment to `targets` while keeping its own shape.

    rigid: vertex index arrays of rigid parts; rigid_maps: matching 3x3 maps (scale x rotation).
    collision_push(x) -> per-vertex outward push needed at positions x.
    detail: distance below which the garment keeps its own shape rather than the targets'.
    """
    n = len(verts)
    link_pairs = links(verts, tris, edges, thickness)
    e0 = np.concatenate([edges[:, 0], link_pairs[:, 0]])
    e1 = np.concatenate([edges[:, 1], link_pairs[:, 1]])
    w = np.concatenate([np.ones(len(edges)), np.full(len(link_pairs), LINK_WEIGHT)])

    edge_len = float(np.median(np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1))) if len(edges) else 0.005
    # Smooth the local stretch and turn over ~`detail`: bumps are small-scale, so they go.
    a = local_transforms(verts, targets, tris, edges[:, 0], edges[:, 1]).reshape(n, 9)
    steps = int(min(400, round(2.0 * (detail / max(edge_len, 1e-5)) ** 2)))
    a = mu.Adjacency(n, edges).smooth(a, steps).reshape(n, 3, 3)
    rigid_vert = np.zeros(n, dtype=bool)
    for ids, m in zip(rigid, rigid_maps):
        a[ids] = m
        rigid_vert[ids] = True
    w = np.where(rigid_vert[e0] & rigid_vert[e1], RIGID_WEIGHT, w)

    rest_edges = verts[e1] - verts[e0]
    d = 0.5 * np.einsum("nij,nj->ni", a[e0] + a[e1], rest_edges)
    graph = _Graph(n, e0, e1, w)
    # The targets pull just enough that shapes larger than `detail` follow them.
    pull = np.full(n, (edge_len / max(detail, 1e-5)) ** 2)
    pull_to = targets.copy()
    x = graph.solve(d, pull, pull_to, targets)

    for _ in range(iterations):
        push = collision_push(x)
        hit = np.linalg.norm(push, axis=1) > 1e-7
        if not hit.any():
            break
        pull = pull.copy()
        pull[hit] = CONSTRAINT_WEIGHT
        pull_to = pull_to.copy()
        pull_to[hit] = x[hit] + push[hit]
        x = graph.solve(d, pull, pull_to, x)
    x += collision_push(x)  # whatever small amount is left
    return x
