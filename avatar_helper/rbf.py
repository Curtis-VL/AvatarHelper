"""Radial basis function warp: a smooth 3D vector field through scattered samples."""
import numpy as np


def _pairwise_dist(a, b):
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0))


class RBFWarp:
    """Biharmonic kernel phi(r) = -r with an affine term.

    Far from the samples the field tends towards the best-fit affine transform, so it
    extrapolates sensibly to loose clothing. `smoothing` (in normalised units, where the
    sample bounding box is ~1) trades exact interpolation for a smoother field.
    """

    def __init__(self, centers, values, smoothing=0.0):
        centers = np.asarray(centers, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        self.dim = values.shape[1] if values.ndim == 2 else 3
        self.offset = centers.mean(0) if len(centers) else np.zeros(3)
        self.scale = float(max(np.ptp(centers, axis=0).max(), 1e-9)) if len(centers) else 1.0
        x = (centers - self.offset) / self.scale
        n = len(x)
        self.centers = x
        self.weights = None
        self.mean = values.mean(0) if n else np.zeros(self.dim)
        if n < 5:
            return

        a = np.zeros((n + 4, n + 4))
        a[:n, :n] = -_pairwise_dist(x, x) + smoothing * np.eye(n)
        p = np.hstack([np.ones((n, 1)), x])
        a[:n, n:] = p
        a[n:, :n] = p.T
        b = np.zeros((n + 4, self.dim))
        b[:n] = values
        try:
            sol = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(a, b, rcond=None)[0]
        self.weights = sol[:n]
        self.affine = sol[n:]

    def __call__(self, points, chunk=2048):
        pts = np.asarray(points, dtype=np.float64)
        if self.weights is None:
            return np.tile(self.mean, (len(pts), 1))
        x = (pts - self.offset) / self.scale
        out = np.empty((len(x), self.dim))
        for s in range(0, len(x), chunk):
            xs = x[s:s + chunk]
            out[s:s + chunk] = (-_pairwise_dist(xs, self.centers)) @ self.weights + self.affine[0] + xs @ self.affine[1:]
        return out


def dedupe_pairs(src, dst, eps=1e-6):
    """Drop duplicate source points (they make the RBF system singular)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if not len(src):
        return src, dst
    _, keep = np.unique(np.round(src / eps).astype(np.int64), axis=0, return_index=True)
    keep = np.sort(keep)
    return src[keep], dst[keep]
