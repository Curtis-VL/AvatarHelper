"""Numpy helpers for reading meshes and doing per-vertex math."""
import numpy as np
from mathutils.bvhtree import BVHTree


def mesh_arrays(me):
    """Return (verts (N,3) local, tris (M,3), edges (E,2)) for a mesh datablock."""
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    edges = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", edges)
    return co.reshape(-1, 3).astype(np.float64), tris.reshape(-1, 3), edges.reshape(-1, 2)


def transform_points(matrix, pts):
    m = np.array(matrix, dtype=np.float64)
    return pts @ m[:3, :3].T + m[:3, 3]


def evaluated_mesh_world(obj, depsgraph):
    """Vertices (world space), triangles and edges of an object after modifiers, shape keys and pose."""
    obj_eval = obj.evaluated_get(depsgraph)
    me = obj_eval.to_mesh()
    try:
        co, tris, edges = mesh_arrays(me)
        mw = obj_eval.matrix_world.copy()
    finally:
        obj_eval.to_mesh_clear()
    return transform_points(mw, co), tris, edges


def read_shape_key(kb, n):
    co = np.empty(n * 3, dtype=np.float32)
    kb.data.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def write_shape_key(kb, co):
    kb.data.foreach_set("co", np.ascontiguousarray(co, dtype=np.float32).ravel())


def build_bvh(verts, tris):
    return BVHTree.FromPolygons(verts.tolist(), tris.tolist(), all_triangles=True)


def nearest_on(bvh, points, max_dist=1e10):
    """Closest surface point for each point. idx is -1 where nothing was within max_dist."""
    n = len(points)
    loc = np.zeros((n, 3))
    nrm = np.zeros((n, 3))
    idx = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf)
    for i, p in enumerate(points.tolist()):
        hit_loc, hit_nrm, hit_idx, hit_dist = bvh.find_nearest(p, max_dist)
        if hit_loc is not None:
            loc[i] = hit_loc
            nrm[i] = hit_nrm
            idx[i] = hit_idx
            dist[i] = hit_dist
    return loc, nrm, idx, dist


def rays_hit(points, directions, bvh, max_dist):
    """For each point, whether a ray along its direction hits the surface within max_dist."""
    hit = np.zeros(len(points), dtype=bool)
    for i, (p, d) in enumerate(zip(points.tolist(), directions.tolist())):
        start = (p[0] + d[0] * 1e-4, p[1] + d[1] * 1e-4, p[2] + d[2] * 1e-4)
        hit[i] = bvh.ray_cast(start, d, max_dist)[2] is not None
    return hit


def normalize(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(length, 1e-12)


def face_normals(verts, tris):
    a = verts[tris[:, 0]]
    return normalize(np.cross(verts[tris[:, 1]] - a, verts[tris[:, 2]] - a))


def vertex_normals(verts, tris):
    a = verts[tris[:, 0]]
    fn = np.cross(verts[tris[:, 1]] - a, verts[tris[:, 2]] - a)  # area weighted
    n = len(verts)
    vn = np.zeros_like(verts)
    for k in range(3):
        for c in range(3):
            vn[:, c] += np.bincount(tris[:, k], weights=fn[:, c], minlength=n)
    return normalize(vn)



def boundary_loops(tris):
    """Open-edge loops of a triangle mesh, each as vertex indices in the order its triangles
    run along it. Edges that don't close into a simple loop are left out."""
    directed = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    have = set(map(tuple, directed.tolist()))
    nxt = {}
    for a, b in directed.tolist():
        if (b, a) not in have:
            nxt.setdefault(a, []).append(b)
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen or len(nxt[start]) != 1:
            continue
        loop, v = [start], nxt[start][0]
        while v != start and v not in seen and len(nxt.get(v, ())) == 1 and len(loop) <= len(nxt):
            loop.append(v)
            v = nxt[v][0]
        seen.update(loop)
        if v == start and len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
    return loops


def weld_map(verts, tris, dist):
    """Per vertex, the vertex it would be merged with: open-edge vertices of separate pieces
    lying within `dist` of each other (a leg modelled as its own part, its top edge on the
    hip's) count as one, so their seam isn't taken for a hole."""
    from mathutils.kdtree import KDTree
    rep = np.arange(len(verts))
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    rim = np.unique(u[c == 1])
    if not len(rim):
        return rep
    kd = KDTree(len(rim))
    for i in rim.tolist():
        kd.insert(verts[i].tolist(), i)
    kd.balance()
    for i in rim.tolist():
        for _co, j, _d in kd.find_range(verts[i].tolist(), dist):
            a, b = rep[i], rep[j]
            while rep[a] != a:
                a = rep[a]
            while rep[b] != b:
                b = rep[b]
            if a != b:
                rep[max(a, b)] = min(a, b)
    for i in range(len(rep)):
        r = rep[i]
        while rep[r] != r:
            r = rep[r]
        rep[i] = r
    return rep


def hole_patches(verts, tris, max_perimeter=1.5, allowed=None, weld=0.0):
    """Patches closing the holes of a mesh (a base with an opening at the crotch, fingertips
    left open under separate nails, a neck under a separate head): new vertices, and
    triangles indexing the mesh's vertices followed by the new ones. Each patch is a smooth
    membrane across its hole, wound to face the same way as the mesh around it. allowed:
    per vertex, whether holes there may be closed (e.g. only on the body itself). weld:
    separate pieces' open edges this close are taken as joined (see weld_map), and only
    what is still open is closed."""
    new_verts, new_tris = [], []
    n = len(verts)
    if weld > 0.0:
        tris = weld_map(verts, tris, weld)[tris]
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]
    for loop in boundary_loops(tris):
        if allowed is not None and not allowed[loop].all():
            continue
        rim = verts[loop]
        k = len(loop)
        perimeter = float(np.linalg.norm(rim - np.roll(rim, -1, axis=0), axis=1).sum())
        if perimeter > max_perimeter:
            continue
        step = perimeter / k
        rings = int(np.clip(round(perimeter / (2.0 * np.pi) / max(step, 1e-6)), 1, 12))
        centre = rim.mean(0)
        start = n + sum(len(v) for v in new_verts)
        pts = [rim + (centre - rim) * (r / rings) for r in range(1, rings)] + [centre[None]]
        ids = [loop] + [start + r * k + np.arange(k) for r in range(rings - 1)]
        centre_id = start + (rings - 1) * k
        tri = []
        for r in range(rings - 1):
            a, b = ids[r], ids[r + 1]
            for i in range(k):
                j = (i + 1) % k
                # The mesh runs along the rim a[i] -> a[j]; the patch, beside it, runs back.
                tri += [(a[j], a[i], b[i]), (a[j], b[i], b[j])]
        last = ids[-1]
        tri += [(last[(i + 1) % k], last[i], centre_id) for i in range(k)]
        patch = np.concatenate(pts)
        tri = np.array(tri, dtype=np.int64)
        # Relax the inside of the patch into a smooth membrane, its rim held in place.
        local = np.concatenate([rim, patch])
        lt = np.where(tri >= start, tri - start + k, -1)
        rim_pos = {int(v): i for i, v in enumerate(loop)}
        for c in range(3):
            col = tri[:, c]
            lt[col < start, c] = [rim_pos[int(v)] for v in col[col < start]]
        e = np.sort(np.concatenate([lt[:, [0, 1]], lt[:, [1, 2]], lt[:, [2, 0]]]), axis=1)
        adj = Adjacency(len(local), np.unique(e, axis=0))
        for _ in range(rings * rings * 4):
            local[k:] = adj.neighbor_mean(local)[k:]
        new_verts.append(local[k:])
        new_tris.append(tri)
    if not new_verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(new_verts), np.concatenate(new_tris)


def fill_creases(verts, tris, width, passes=8):
    """The surface with creases narrower than about `width` filled in, the way fabric
    stretched over it would bridge them (between toes, in a crotch, a navel). Points only
    ever move outwards, onto the average of the surface around them, so bumps and convex
    shapes stay as they are."""
    from mathutils.kdtree import KDTree
    if width <= 0.0 or not len(verts):
        return verts.copy()
    # The surface around each point is averaged from a coarser copy of it (points merged
    # in small cells), found once: the filling moves points by much less than `width`.
    cell = width / 6.0
    _, inv, counts = np.unique(np.floor(verts / cell).astype(np.int64), axis=0, return_inverse=True,
                               return_counts=True)
    inv = inv.ravel()
    m = len(counts)

    def coarse(points):
        return np.stack([np.bincount(inv, points[:, c], m) for c in range(3)], 1) / counts[:, None]

    kd = KDTree(m)
    for i, p in enumerate(coarse(verts).tolist()):
        kd.insert(p, i)
    kd.balance()
    rows, cols, dists = [], [], []
    for i, p in enumerate(verts.tolist()):
        hits = kd.find_range(p, width)
        rows.append(np.full(len(hits), i, dtype=np.int64))
        cols.append(np.fromiter((h[1] for h in hits), dtype=np.int64, count=len(hits)))
        dists.append(np.fromiter((h[2] for h in hits), dtype=np.float64, count=len(hits)))
    rows, cols, dists = np.concatenate(rows), np.concatenate(cols), np.concatenate(dists)
    w = np.exp(-dists ** 2 / (0.5 * width) ** 2) * counts[cols]
    total = np.maximum(np.bincount(rows, weights=w, minlength=len(verts)), 1e-12)
    out = verts.copy()
    for _ in range(passes):
        normals = vertex_normals(out, tris)
        c = coarse(out)
        mean = np.stack([np.bincount(rows, weights=w * c[cols, k], minlength=len(verts)) for k in range(3)], 1)
        lift = ((mean / total[:, None] - out) * normals).sum(1)
        out = out + np.maximum(lift, 0.0)[:, None] * normals
    return out


def falloff_weights(selected, edges, rest, falloff):
    """1 on the selection, fading to 0 over `falloff` along the surface."""
    w = selected.astype(np.float64)
    if falloff <= 0.0 or not len(edges):
        return w
    edge_len = float(np.median(np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)))
    rings = max(1, int(round(falloff / max(edge_len, 1e-5))))
    e0, e1 = edges[:, 0], edges[:, 1]
    for _ in range(rings):
        grown = w.copy()
        np.maximum.at(grown, e0, w[e1] - 1.0 / rings)
        np.maximum.at(grown, e1, w[e0] - 1.0 / rings)
        w = np.maximum(grown, 0.0)
    # Smooth ramp rather than a linear one, so the border doesn't show as a crease.
    return w * w * (3.0 - 2.0 * w)


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.stack([1.0 - v - w, v, w], axis=1)


class Adjacency:
    """Edge-based neighbour averaging, used to smooth per-vertex vector fields."""

    def __init__(self, n, edges):
        self.n = n
        self.e0 = edges[:, 0]
        self.e1 = edges[:, 1]
        self.deg = (np.bincount(self.e0, minlength=n) + np.bincount(self.e1, minlength=n)).astype(np.float64)

    def neighbor_sum(self, values):
        if values.ndim == 2:
            return np.stack([self.neighbor_sum(values[:, c]) for c in range(values.shape[1])], axis=1)
        return (np.bincount(self.e0, weights=values[self.e1], minlength=self.n)
                + np.bincount(self.e1, weights=values[self.e0], minlength=self.n))

    def neighbor_mean(self, values):
        s = self.neighbor_sum(values)
        deg = self.deg[:, None] if values.ndim == 2 else self.deg
        return np.where(deg > 0, s / np.maximum(deg, 1), values)

    def solve_membrane(self, targets, weights, stiffness, iterations=400, tol=1e-5):
        """Smooth displacement field d minimising
            sum_i w_i |d_i - t_i|^2 + stiffness * sum_edges |d_i - d_j|^2.

        Unlike plain smoothing, vertices with zero weight are interpolated from the
        surrounding constrained vertices rather than left behind. Solved with
        Jacobi-preconditioned conjugate gradients.
        """
        w = weights.astype(np.float64) + 1e-6
        diag = (w + stiffness * self.deg)[:, None]

        def apply(x):
            return w[:, None] * x + stiffness * (self.deg[:, None] * x - self.neighbor_sum(x))

        b = w[:, None] * targets
        x = np.zeros_like(b)
        r = b.copy()
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

    def smooth(self, values, iterations, weights=None, factor=0.5):
        """Weighted Laplacian smoothing. Zero-weight vertices take their value from weighted neighbours."""
        if iterations <= 0 and weights is None:
            return values.copy()
        w = np.ones(self.n) if weights is None else weights.astype(np.float64)
        num = values * w[:, None]
        den = w.copy()
        for _ in range(iterations):
            num = (1.0 - factor) * num + factor * self.neighbor_mean(num)
            den = (1.0 - factor) * den + factor * self.neighbor_mean(den)
        return np.where(den[:, None] > 1e-9, num / np.maximum(den[:, None], 1e-9), 0.0)


def islands(n, edges):
    """Connected component label per vertex."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.ravel()


def similarity_parts(src, dst):
    """Best (scale, rotation) mapping src onto dst about their centroids."""
    if len(src) < 3:
        return 1.0, np.eye(3)
    xs, xd = src - src.mean(0), dst - dst.mean(0)
    u, sv, vt = np.linalg.svd(xs.T @ xd / len(src))
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    dm = np.diag([1.0, 1.0, d])
    var = (xs ** 2).sum() / len(src)
    scale = np.trace(np.diag(sv) @ dm) / var if var > 1e-12 else 1.0
    return scale, vt.T @ dm @ u.T


def similarity_fit(src, dst):
    """Best rotation + uniform scale + translation mapping src onto dst; returns transformed src."""
    scale, rot = similarity_parts(src, dst)
    return scale * (src - src.mean(0)) @ rot.T + dst.mean(0)


def voxel_downsample(points, values, size, max_points, grid_points=None):
    """Average points and their values into a voxel grid, growing the voxels until under max_points.
    grid_points: the same points in another frame to lay the grid out in (so the result doesn't
    depend on where the points happen to be), with `size` in that frame's units."""
    grid_points = points if grid_points is None else grid_points
    while True:
        keys = np.floor(grid_points / size).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        if len(counts) <= max_points:
            break
        size *= 1.25
    inv = inv.ravel()
    m = len(counts)
    p = np.stack([np.bincount(inv, points[:, c], m) for c in range(3)], axis=1) / counts[:, None]
    v = np.stack([np.bincount(inv, values[:, c], m) for c in range(values.shape[1])], axis=1) / counts[:, None]
    return p, v
