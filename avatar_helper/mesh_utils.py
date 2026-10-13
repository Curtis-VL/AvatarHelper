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
