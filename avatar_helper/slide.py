"""Sliding over the skin: points moved along a body's surface rather than through space.

Each point is bound to the nearest point of the surface under it, that point walks over the
surface by the requested offset, and the point is rebuilt at the same offset from the skin at
its new place. A band moved down over a breast follows its curve instead of going into it or
lifting off; a stocking top moved down a thigh stays wrapped around it.

Used by the Slide Along Body touch-up (on a fitted garment) and by anchors (on the
conversion's body mapping, so every layer of the outfit moves together; see anchors.py).
"""
import numpy as np

from . import mesh_utils as mu

SLIDE_REACH = 0.1  # points further than this from the skin just move by the offset
SLIDE_STEP = 0.004  # walking step over the surface
SLIDE_MAX_STEPS = 16
TURN_AWAY = 0.1  # a step this much along the normal (relative to its length) ends the walk


class SkinWalker:
    def __init__(self, verts, tris):
        self.verts = verts
        self.tris = tris
        self.normals = mu.vertex_normals(verts, tris)
        self.bvh = mu.build_bvh(verts, tris)

    def skin_point(self, p):
        """The surface point nearest to `p`."""
        loc, _nrm, idx, _dist = mu.nearest_on(self.bvh, np.asarray(p, dtype=np.float64)[None], 1.0)
        return loc[0] if idx[0] >= 0 else np.asarray(p, dtype=np.float64)

    def _normal_at(self, loc, idx):
        """Smoothly interpolated normal (like smooth shading) at surface points."""
        t = self.tris[idx]
        v = self.verts
        bary = np.clip(mu.barycentric(loc, v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]), 0.0, None)
        bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
        n = self.normals
        return mu.normalize(bary[:, :1] * n[t[:, 0]] + bary[:, 1:2] * n[t[:, 1]] + bary[:, 2:] * n[t[:, 2]])

    def walk(self, loc, idx, offsets):
        """Surface points `loc` (on triangles `idx`) walked over the surface by `offsets`.
        Each step goes along the surface, keeping its length, so the distance walked is the
        offset's length even where the surface curves away from it (under a breast).
        Returns (new points, their triangles)."""
        length = np.linalg.norm(offsets, axis=1)
        steps = int(np.clip(np.ceil(length.max(initial=0.0) / SLIDE_STEP), 1, SLIDE_MAX_STEPS))
        step = offsets / steps
        step_len = length / steps
        cur, tri = loc.copy(), idx.copy()
        for _ in range(steps):
            n = self._normal_at(cur, tri)
            along = step - (step * n).sum(1, keepdims=True) * n
            along_len = np.linalg.norm(along, axis=1)
            keep = along_len > TURN_AWAY * step_len
            along = np.where(keep[:, None], along * (step_len / np.maximum(along_len, 1e-12))[:, None], 0.0)
            new, _nrm, new_tri, _dist = mu.nearest_on(self.bvh, cur + along, 4.0 * SLIDE_STEP + step_len.max())
            ok = new_tri >= 0
            cur[ok], tri[ok] = new[ok], new_tri[ok]
        return cur, tri

    def slide(self, points, offsets, reach=SLIDE_REACH):
        """`points` moved by `offsets` along the surface, each keeping its offset from the
        skin under it (turned with the skin). Points with no skin within `reach` simply move
        by their offset."""
        out = points + offsets
        moving = np.linalg.norm(offsets, axis=1) > 1e-9
        if not moving.any():
            return out
        ids = np.nonzero(moving)[0]
        loc, _nrm, idx, _dist = mu.nearest_on(self.bvh, points[ids], reach)
        ok = idx >= 0
        ids, loc, idx = ids[ok], loc[ok], idx[ok]
        if not len(ids):
            return out
        n0 = self._normal_at(loc, idx)
        rel = points[ids] - loc
        up = (rel * n0).sum(1)
        side = rel - up[:, None] * n0
        new, new_idx = self.walk(loc, idx, offsets[ids])
        n1 = self._normal_at(new, new_idx)
        # The small sideways part of the offset (where the smooth normal differs from the
        # face's) is carried over onto the new tangent plane, same length.
        side_len = np.linalg.norm(side, axis=1)
        side = side - (side * n1).sum(1, keepdims=True) * n1
        side *= (side_len / np.maximum(np.linalg.norm(side, axis=1), 1e-12))[:, None]
        out[ids] = new + up[:, None] * n1 + side
        return out
