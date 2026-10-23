"""Fit clothing using a calibration.

Clothing is never projected onto the body. Instead:
  * a smooth space warp (RBF through the calibration's body displacements) moves every
    vertex, so loose garments keep their shape and their gap from the body;
  * close to the skin, that is blended with a surface-relative binding (position in the
    nearest triangle's frame) so tight garments follow body detail; close to a lower layer,
    the item binds to that layer instead, keeping its overlap with it;
  * parts hidden inside the source body, or tucked under a lower layer, stay that way;
  * small disconnected parts (buttons, buckles) and the "AH_Rigid" vertex group move rigidly;
  * a push-out pass only ever moves vertices outward, away from the body and lower layers;
  * shoes are not warped at all: they are placed rigidly on the target's feet and floor.
"""
import math
import os
import re

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from . import mesh_utils as mu
from . import conversions, elastic, limbs, rig
from .calibration import CALIBRATION_KEY, output_collection, shape_key_values, target_armature
from .props import is_result
from .rbf import RBFWarp
from .rig import armature_map, bone_depth, deforming_armatures

FITTED_COLLECTION = "AH_Fitted"
RIGID_GROUP = "AH_Rigid"


RIGID_TOUCH = 0.001  # rigid parts closer than this count as touching and move together


class SurfaceBinding:
    """Carries points along with a surface that moved: each point is expressed in a frame
    at its nearest surface point before, and rebuilt in the matching frame after.

    The frames use smoothly interpolated normals (like smooth shading), so offsets don't
    jump at triangle edges, which would press the surface's polygons into the clothing."""

    def __init__(self, base, fitted, tris, crease_width=0.0):
        self.base = base
        self.fitted = fitted
        self.tris = tris
        self.bvh = mu.build_bvh(base, tris)
        # Points are carried along both bodies with their narrow creases filled in, the way
        # fabric bridges them: a tight garment then keeps its shape over a crotch or toes
        # instead of being pressed into whatever detail the target has there. (Collisions
        # still use the real bodies.)
        if crease_width > 0.0:
            self.bind_base = mu.fill_creases(base, tris, crease_width)
            self.bind_fitted = mu.fill_creases(fitted, tris, crease_width)
            self.bind_bvh = mu.build_bvh(self.bind_base, tris)
        else:
            self.bind_base, self.bind_fitted, self.bind_bvh = base, fitted, self.bvh
        self.base_normals = mu.vertex_normals(self.bind_base, tris)
        self.fit_normals = mu.vertex_normals(self.bind_fitted, tris)

    @staticmethod
    def _frames(verts, vnormals, t, bary):
        n = mu.normalize(bary[:, :1] * vnormals[t[:, 0]] + bary[:, 1:2] * vnormals[t[:, 1]]
                         + bary[:, 2:] * vnormals[t[:, 2]])
        edge = verts[t[:, 1]] - verts[t[:, 0]]
        e1 = mu.normalize(edge - (edge * n).sum(1, keepdims=True) * n)
        return e1, np.cross(n, e1), n

    def surface_warp(self, pts, max_dist):
        """Returns (moved points, distance to the surface, whether a triangle was found,
        signed offset along the surface normal: > 0 on the outer side)."""
        loc, _nrm, idx, dist = mu.nearest_on(self.bind_bvh, pts, max_dist)
        ok = idx >= 0
        out = pts.copy()
        side = np.zeros(len(pts))
        if not ok.any():
            return out, dist, ok, side
        ti = idx[ok]
        t = self.tris[ti]
        b = self.bind_base
        bary = mu.barycentric(loc[ok], b[t[:, 0]], b[t[:, 1]], b[t[:, 2]])
        rel = pts[ok] - loc[ok]
        e1, e2, n = self._frames(b, self.base_normals, t, bary)
        coef = np.stack([(rel * e1).sum(1), (rel * e2).sum(1), (rel * n).sum(1)], axis=1)
        f = self.bind_fitted
        anchor = (bary[:, :1] * f[t[:, 0]] + bary[:, 1:2] * f[t[:, 1]] + bary[:, 2:] * f[t[:, 2]])
        g1, g2, gn = self._frames(f, self.fit_normals, t, bary)
        out[ok] = anchor + coef[:, :1] * g1 + coef[:, 1:2] * g2 + coef[:, 2:] * gn
        side[ok] = coef[:, 2]
        return out, dist, ok, side


class FitField(SurfaceBinding):
    """The calibration as a deformation: a smooth space warp plus binding to the body surface."""

    def __init__(self, base, fitted, tris, resolution, smoothing, max_centers=2500, grid=None, grid_unit=1.0,
                 crease_width=0.0):
        super().__init__(base, fitted, tris, crease_width)
        centers, values = mu.voxel_downsample(base, fitted - base, resolution / grid_unit, max_centers, grid)
        self.rbf = RBFWarp(centers, values, smoothing)

    @classmethod
    def from_conversion(cls, conv, src_align, tgt_align, resolution, smoothing, crease_width=0.0):
        """The conversion placed in this scene: its source side on the clothing's armature,
        its target side on the target avatar. The warp is sampled in the conversion's own
        frame, so where the clothing stands doesn't change it."""
        unit = float(np.cbrt(abs(np.linalg.det(src_align[:3, :3]))))
        return cls(mu.transform_points(src_align, conv.base), mu.transform_points(tgt_align, conv.fitted),
                   conv.tris, resolution, smoothing, grid=conv.base, grid_unit=unit, crease_width=crease_width)

    @classmethod
    def from_object(cls, cal_obj, resolution, smoothing):
        """From a calibration preview object (source body with a "Fitted" shape key)."""
        me = cal_obj.data
        keys = me.shape_keys
        if keys is None or CALIBRATION_KEY not in keys.key_blocks:
            raise ValueError(f"'{cal_obj.name}' has no '{CALIBRATION_KEY}' shape key")
        n = len(me.vertices)
        base = mu.transform_points(cal_obj.matrix_world, mu.read_shape_key(keys.reference_key, n))
        fitted = mu.transform_points(cal_obj.matrix_world, mu.read_shape_key(keys.key_blocks[CALIBRATION_KEY], n))
        _co, tris, _edges = mu.mesh_arrays(me)
        return cls(base, fitted, tris, resolution, smoothing)

    def warp(self, pts):
        return pts + self.rbf(pts)

    @property
    def bvh_fitted(self):
        """The calibrated body in the target's shape."""
        if not hasattr(self, "_bvh_fitted"):
            self._bvh_fitted = mu.build_bvh(self.fitted, self.tris)
        return self._bvh_fitted

    def nearest_vertex(self, pts):
        """Index of the nearest calibrated (target-shaped) source body vertex for each point."""
        if not hasattr(self, "_kd"):
            self._kd = KDTree(len(self.fitted))
            for i, p in enumerate(self.fitted.tolist()):
                self._kd.insert(p, i)
            self._kd.balance()
        return np.fromiter((self._kd.find(p)[1] for p in pts.tolist()), dtype=np.int64, count=len(pts))


def _smoothstep_falloff(dist, inner, blend):
    t = np.clip((dist - inner) / max(blend, 1e-6), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def rigid_islands(obj, verts, edges, max_size):
    """Vertex index arrays for parts that should move rigidly."""
    labels = mu.islands(len(verts), edges)
    in_group = np.zeros(len(verts), dtype=bool)
    vg = obj.vertex_groups.get(RIGID_GROUP)
    if vg is not None:
        gi = vg.index
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == gi and g.weight >= 0.5:
                    in_group[v.index] = True
    parts = []
    for lab in range(labels.max() + 1 if len(labels) else 0):
        ids = np.nonzero(labels == lab)[0]
        if len(ids) < 3:
            continue
        size = np.linalg.norm(np.ptp(verts[ids], axis=0))
        if size <= max_size or in_group[ids].any():
            parts.append(ids)
    return _group_touching(verts, parts, max_size)


def _group_touching(verts, parts, max_size):
    """Merge rigid parts that touch (gems set in a pendant, a bail on its ring) so they move
    as one piece; moved separately they drift into each other. A merged group still has to
    fit within max_size."""
    lo = [verts[ids].min(0) - RIGID_TOUCH for ids in parts]
    hi = [verts[ids].max(0) + RIGID_TOUCH for ids in parts]
    groups = [[k] for k in range(len(parts))]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ga, gb = groups[a], groups[b]
                if not any((lo[i] <= hi[j]).all() and (lo[j] <= hi[i]).all() for i in ga for j in gb):
                    continue
                ids = np.concatenate([parts[k] for k in ga + gb])
                if np.linalg.norm(np.ptp(verts[ids], axis=0)) > max_size:
                    continue
                groups[a] = ga + gb
                del groups[b]
                merged = True
                break
            if merged:
                break
    return [np.concatenate([parts[k] for k in g]) for g in groups]


def push_out(points, colliders):
    """Outward-only push so every point sits at least its gap outside each collider.

    colliders: (bvh, skip, depth, gaps[, surface]) tuples. `skip` marks points exempt from
    that collider (parts that sat under it on the source, like a tucked-in shirt hem), or is
    None. `depth` is how far inside a point may be and still count as colliding. `gaps`
    is the clearance to keep, per point. `surface`, a SmoothSide for the collider's mesh,
    tells inside from outside at its edges and corners too (see SmoothSide).
    """
    push = np.zeros_like(points)
    for i, p in enumerate(points.tolist()):
        best = 0.0
        pv = Vector(p)
        for bvh, skip, depth, gaps, *surface in colliders:
            if skip is not None and skip[i]:
                continue
            offset = gaps[i]
            loc, nrm, idx, d = bvh.find_nearest(p, depth + offset)
            if loc is None:
                continue
            rel = pv - loc
            side = rel.dot(nrm)
            if d > 1e-5 and abs(side) / d < 0.7:
                # Nearest to an edge or corner, not a face. On an open edge reached from
                # beyond the surface (e.g. below a shirt hem) that's no collision. Elsewhere
                # (a crease such as a crotch, seen from inside) it can be.
                if not surface or surface[0] is None or surface[0].open_edge(idx):
                    continue
                nrm = surface[0].normal(idx, loc)
                side = rel.dot(nrm)
            if side < offset:
                v = loc + nrm * offset - pv
                if v.length > best:
                    best = v.length
                    push[i] = v
    return push


class SmoothSide:
    """Inside or outside of a mesh where the nearest point is on an edge or a corner, where a
    face's own normal can't tell: by the vertex normals blended across the face there."""

    def __init__(self, verts, tris, vertex_normals):
        self.verts, self.tris, self.normals = verts, tris, vertex_normals
        e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        _u, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        self.open = (counts[inv.ravel()] == 1).reshape(3, -1).any(0)

    def open_edge(self, tri):
        return bool(self.open[tri])

    def normal(self, tri, loc):
        ids = self.tris[tri]
        bary = mu.barycentric(np.array([loc[:]]), *(self.verts[ids[k]][None] for k in range(3)))[0]
        n = np.clip(bary, 0.0, 1.0) @ self.normals[ids]
        return Vector(n).normalized()


POKE_DEPTH = 0.01  # deeper "pokes" are usually skin showing through an opening on purpose
THICKNESS_MAX = 0.006  # points this close get the same collision push (both sides of a thin part)
PUSH_FADE = 0.01  # beyond that, collision pushes fade out over this distance
THROUGH_FADE = 0.0015  # shape-preserving solver: sideways reach of a push (the solve spreads the rest)


def poke_through(points, tris, body_verts, body_normals, gaps):
    """Outward push for clothing faces that body vertices (or lower-layer vertices that
    belong under this clothing) come through.

    Catches body detail (knees, collarbones) poking out between the vertices of
    lower-resolution clothing, which the per-vertex test in push_out misses.
    Only outer-facing clothing faces count; the inner shell of thick clothing faces the
    body and is skipped. `gaps` is the clearance to keep, per clothing vertex.
    """
    push = np.zeros_like(points)
    amount = np.zeros(len(points))
    loc, nrm, idx, dist = mu.nearest_on(mu.build_bvh(points, tris), body_verts, POKE_DEPTH)
    side = ((body_verts - loc) * nrm).sum(1)
    facing = (nrm * body_normals).sum(1) > 0.5
    offset = np.where(idx >= 0, gaps[tris[np.maximum(idx, 0)]].mean(1), 0.0)
    # Facing the face (not reached around an open edge) and closer than its gap below it, or outside.
    hit = (idx >= 0) & facing & (side > -offset) & (np.abs(side) >= 0.9 * dist)
    for ti, a, n in zip(idx[hit].tolist(), (side[hit] + offset[hit]).tolist(), nrm[hit]):
        for v in tris[ti]:
            if a > amount[v]:
                amount[v] = a
                push[v] = n * a
    return push


LAYER_RANGE = 0.03  # how far apart two layers can be and still count as touching
BODY_MATCH = 0.003  # a part of the avatar mesh is body if most of it lies within this of the calibrated body


def body_parts(verts, edges, calibrated):
    """Vertices of the avatar mesh belonging to its body: the connected parts lying mostly
    on the calibrated body (`calibrated`: a BVH of it). Hair strands, accessories and
    clothing merged into the mesh lie off it."""
    labels = mu.islands(len(verts), edges)
    _loc, _nrm, idx, _dist = mu.nearest_on(calibrated, verts, BODY_MATCH)
    on = np.bincount(labels, weights=(idx >= 0).astype(float)) / np.bincount(labels)
    return on[labels] > 0.5


BODY_DEPTH = 0.03  # how deep inside the body a point can be and still be pushed out
# Clothing is thin and open: a point more than this "inside" another item has gone through
# to its other side (or was never outside it), so it isn't pushed back.
LAYER_DEPTH = 0.01


class Layer:
    """A fitted clothing item, as something for the items fitted after it to rest on."""

    def __init__(self, src_verts, fit_verts, tris):
        self.tris = tris
        self.src_verts = src_verts  # None when the source shape isn't comparable (shoes use the rest pose)
        self.fit_verts = fit_verts
        self.fit_bvh = mu.build_bvh(fit_verts, tris)
        self.binding = SurfaceBinding(src_verts, fit_verts, tris) if src_verts is not None else None


def inside_of(points, bvh, depth):
    """Points up to `depth` inside the surface (not counting points reached around an
    open edge)."""
    loc, nrm, idx, dist = mu.nearest_on(bvh, points, depth)
    side = ((points - loc) * nrm).sum(1)
    return (idx >= 0) & (side < 0.0) & (np.abs(side) >= 0.7 * np.maximum(dist, 1e-9))


def facing_out(points, normals, body_bvh):
    """Points whose surface faces away from the body. The back of a pendant or chain, or
    a garment's inner lining, faces the body: a ray from there hits whatever lies below,
    which must not count as being covered by it."""
    loc, nrm, idx, _dist = mu.nearest_on(body_bvh, points, 0.1)
    return (idx < 0) | ((normals * nrm).sum(1) > 0.3)


def layer_relations(verts, tris, layers, body_bvh):
    """How this item sat relative to each lower layer on the source.

    Returns per layer: (tucked, covered, exempt).
      tucked: this item's vertices that were under the layer (e.g. a shirt hem inside the
        trousers); they are kept under it. "Under" means a short ray outward along the
        surface hits the layer, which also works at open rims and thick waistbands.
      covered: the layer's vertices that were under this item; they must not poke through it.
      exempt: vertices never pushed out of the layer: tucked ones, and ones already inside
        it on the source (a choker under a collar), so only vertices that newly cross it
        during fitting are pushed. Parts resting on it (a necklace on a sweater) are not
        exempt: they are kept at their original gap above it.
    """
    normals = mu.vertex_normals(verts, tris)
    outer = facing_out(verts, normals, body_bvh)
    item_bvh = mu.build_bvh(verts, tris)
    out = []
    for layer in layers:
        if layer.binding is None:
            out.append((None, None, None))
            continue
        tucked = outer & mu.rays_hit(verts, normals, layer.binding.bvh, LAYER_RANGE)
        lnormals = mu.vertex_normals(layer.src_verts, layer.tris)
        covered = (facing_out(layer.src_verts, lnormals, body_bvh)
                   & mu.rays_hit(layer.src_verts, lnormals, item_bvh, LAYER_RANGE))
        exempt = tucked | inside_of(verts, layer.binding.bvh, LAYER_DEPTH)
        out.append((tucked if tucked.any() else None, covered if covered.any() else None, exempt))
    return out


def follow_layers(verts, out, body_dist, layers, relations, s):
    """Where a lower layer is closer than the body and this item rests on it (or is
    tucked under it), carry the points along with that layer instead, so the item keeps
    its exact offset from it (a shirt over or tucked into trousers stays that way, rather
    than both being fitted to the body separately and drifting into each other). Items
    merely near a layer, like a choker inside a shirt collar, keep following the body."""
    reach = s.skin_distance + s.blend_distance
    best = np.where(np.isfinite(body_dist), body_dist, np.inf)
    for layer, (tucked, _covered, _exempt) in zip(layers, relations):
        if layer.binding is None:
            continue
        near, dist, ok, side = layer.binding.surface_warp(verts, reach)
        rests = side > 0.0
        if tucked is not None:
            rests |= tucked
        closer = ok & rests & (dist < best)
        if not closer.any():
            continue
        w = _smoothstep_falloff(dist[closer], s.skin_distance, s.blend_distance)
        out[closer] = out[closer] * (1.0 - w[:, None]) + near[closer] * w[:, None]
        best[closer] = dist[closer]
    return out


STACK_RANGE = 0.03  # pieces of one garment this close count as stacked (a lining under a knit)


class StackedPieces:
    """Separate pieces of the same garment lying on each other: a knit over its inner top, a
    collar over the shirt. A point lies over another piece if a short ray inward hits it,
    facing the same way.

    A tight inner piece gets pushed out of the body where the target is fuller (or where the
    smooth fit doesn't follow the body's detail); the piece over it doesn't touch the body,
    so on its own it would stay put and the inner piece would come through it. carry()
    passes those pushes on to the pieces above, as far as needed to keep their original gap,
    while their own shape is left alone everywhere else."""

    def __init__(self, verts, tris, edges, min_gap):
        self.tris = tris
        self.rest = verts
        self.min_gap = min_gap
        self.src_gaps = {}  # lower piece -> each point's gap above it on the source
        self.labels = mu.islands(len(verts), edges)
        self.tri_labels = self.labels[tris[:, 0]] if len(tris) else np.zeros(0, dtype=np.int64)
        self.pairs = set()  # (upper piece, lower piece)
        if self.labels.max() == 0:
            return
        bvh = mu.build_bvh(verts, tris)
        normals = mu.vertex_normals(verts, tris)
        count = {}
        for i, (p, n) in enumerate(zip(verts.tolist(), normals.tolist())):
            d = (-n[0], -n[1], -n[2])
            _loc, nrm, idx, _dist = bvh.ray_cast((p[0] + d[0] * 1e-5, p[1] + d[1] * 1e-5, p[2] + d[2] * 1e-5),
                                                 d, STACK_RANGE)
            if idx is None or self.tri_labels[idx] == self.labels[i]:
                continue
            if nrm[0] * n[0] + nrm[1] * n[1] + nrm[2] * n[2] > 0.3:
                key = (int(self.labels[i]), int(self.tri_labels[idx]))
                count[key] = count.get(key, 0) + 1
        # Each pair stacks one way. A few points of the lower piece showing up as over the
        # upper one (at a folded edge) would have the two pushing each other out in turn.
        self.pairs = {(a, b) for (a, b), k in count.items() if count.get((b, a), 0) <= k}

    def any(self):
        return bool(self.pairs)

    def _lower_first(self):
        level = {p: 0 for pair in self.pairs for p in pair}
        for _ in range(len(level)):
            for a, b in self.pairs:
                level[a] = max(level[a], level[b] + 1)
        return sorted({b for _a, b in self.pairs}, key=level.get)

    def carry(self, x, push):
        """Pass the pushes on each lower piece on to the pieces lying on it (interpolated
        across its faces, at the nearest point below), lowest pieces first so a push travels
        up through several layers."""
        for lower in self._lower_first():
            lt = self.tris[self.tri_labels == lower]
            if not (np.linalg.norm(push[lt], axis=2) > 1e-7).any():
                continue
            ids = np.nonzero(np.isin(self.labels, [a for a, b in self.pairs if b == lower]))[0]
            loc, nrm, idx, _dist = mu.nearest_on(mu.build_bvh(x, lt), x[ids], STACK_RANGE)
            ok = (idx >= 0) & (((x[ids] - loc) * nrm).sum(1) > 0.0)
            if not ok.any():
                continue
            ids, loc, nrm, t = ids[ok], loc[ok], nrm[ok], lt[idx[ok]]
            bary = mu.barycentric(loc, x[t[:, 0]], x[t[:, 1]], x[t[:, 2]])
            carried = bary[:, :1] * push[t[:, 0]] + bary[:, 1:2] * push[t[:, 1]] + bary[:, 2:] * push[t[:, 2]]
            # Only as much as needed to keep the original gap: where the piece on top already
            # sits further out, a push on the piece below (often a local spike where it dips
            # into the body under a band) is taken up by the extra room rather than copied into
            # the piece on top as a lump.
            gap_now = ((x[ids] - loc) * nrm).sum(1)
            keep = np.maximum(self._src_gap(lower)[ids], self.min_gap)
            rise = (carried * nrm).sum(1)
            scale = np.clip((rise - (gap_now - keep)) / np.maximum(rise, 1e-9), 0.0, 1.0)
            carried *= scale[:, None]
            more = np.linalg.norm(carried, axis=1) > np.linalg.norm(push[ids], axis=1)
            push[ids[more]] = carried[more]
        return push

    def _src_gap(self, lower):
        if lower not in self.src_gaps:
            gaps = np.zeros(len(self.rest))
            ids = np.nonzero(np.isin(self.labels, [a for a, b in self.pairs if b == lower]))[0]
            bvh = mu.build_bvh(self.rest, self.tris[self.tri_labels == lower])
            loc, nrm, idx, _dist = mu.nearest_on(bvh, self.rest[ids], STACK_RANGE)
            gaps[ids] = np.where(idx >= 0, np.maximum(((self.rest[ids] - loc) * nrm).sum(1), 0.0), 0.0)
            self.src_gaps[lower] = gaps
        return self.src_gaps[lower]


CREASE_WIDTH = 0.02  # creases narrower than this are bridged, as fabric would (see SurfaceBinding)
LIMB_FOLLOW = True  # fingers and toes follow their bones (limbs.py)
HEM_RANGE = 0.015  # how close to an open edge a part inside the body must be to stay hidden
COVER_RANGE = 0.05  # how far out another garment hiding a part inside the body may be


def hidden_parts(field, verts, tris, covers=()):
    """Clothing tucked into the source body is hidden by design: a hem curling in under a
    waistband, a strap end disappearing into the thigh, a sock's toe squeezed inside the
    foot to fit a shoe. Such parts are carried along at the same depth rather than pushed
    out, which would crumple them.

    A thin part that merely dips into the skin (the inner side of a strap, with its outer
    side right above it) doesn't count: it is pushed out whole as usual, which keeps it
    clean. Nor does the base's body pressing through the middle of a garment with nothing
    over it (anatomy through the crotch of shorts): that is clipping on the base, and copying
    it onto an avatar without that detail sinks the garment into the body. So a part inside
    the body stays hidden only if it is near an open edge that is inside the body too (with
    none of the same garment over it), or if nothing of the same garment but another garment
    (`covers`: BVHs of the outfit's other meshes, e.g. a shoe) lies over it.

    This is decided per connected area inside the body, by majority, so one area is treated
    the same throughout: decided point by point, a gusset passing into the base's body would
    have some points kept inside and their neighbours pushed out, folding it over between.
    """
    loc, nrm, idx, _dist = mu.nearest_on(field.bvh, verts, 0.05)
    inside = (idx >= 0) & (((verts - loc) * nrm).sum(1) < 0.0)
    if not inside.any():
        return inside
    ids = np.nonzero(inside)[0]
    self_bvh = mu.build_bvh(verts, tris)
    covered_by_self = mu.rays_hit(verts[ids], nrm[ids], self_bvh, THICKNESS_MAX)
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    # Only an open edge that goes into the body itself: shorts' leg openings around the thighs
    # don't make the crotch between them a tucked-in hem.
    rim = np.unique(uniq[counts == 1])
    rim = rim[inside[rim]]
    near_rim = np.zeros(len(ids), dtype=bool)
    if len(rim):
        kd = KDTree(len(rim))
        for i in rim.tolist():
            kd.insert(verts[i].tolist(), i)
        kd.balance()
        near_rim = np.array([kd.find(verts[i].tolist())[2] < HEM_RANGE for i in ids.tolist()])
    # An inner piece under the garment's own outer layer (a crotch gusset inside the base's
    # body) is out of sight however it's fitted, and pushed out it supports the layer over it.
    under_self = mu.rays_hit(verts[ids], nrm[ids], self_bvh, COVER_RANGE)
    covered_by_other = np.zeros(len(ids), dtype=bool)
    for bvh in covers:
        covered_by_other |= mu.rays_hit(verts[ids], nrm[ids], bvh, COVER_RANGE)
    vote = np.zeros(len(verts))
    vote[ids] = ((near_rim & ~under_self) | (~covered_by_self & covered_by_other)).astype(float)
    # Connected areas inside the body (edges with both ends inside).
    both = inside[uniq[:, 0]] & inside[uniq[:, 1]]
    areas = mu.islands(len(verts), uniq[both])
    share = np.bincount(areas, weights=vote) / np.maximum(np.bincount(areas, weights=inside.astype(float)), 1e-9)
    return inside & (share[areas] > 0.5)


BRIDGE_RANGE = 0.02  # how far around a point its gap to the body is compared
BRIDGE_SPREAD = (0.002, 0.006)  # gap varying less than this: hugging the body; more: bridging


def bridging(verts, edges, gap):
    """0..1 per point: how much the garment bridges over the body there instead of hugging it.
    A garment that hugs the body keeps about the same gap to it; one stretched smooth over
    body detail (a crotch, a navel, ribs) has a gap that changes quickly, inside the detail
    one moment and well clear of it the next. Copying the base's detail into it then shows."""
    have = np.isfinite(gap)
    g = np.where(have, gap, 0.0)
    if not len(edges) or not have.any():
        return np.zeros(len(verts))
    edge_len = float(np.median(np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)))
    steps = int(min(200, round(2.0 * (BRIDGE_RANGE / max(edge_len, 1e-5)) ** 2)))
    adj = mu.Adjacency(len(verts), edges)
    sm = adj.smooth(np.stack([g, g * g], axis=1), steps, weights=have.astype(float))
    spread = np.sqrt(np.maximum(sm[:, 1] - sm[:, 0] ** 2, 0.0))
    lo, hi = BRIDGE_SPREAD
    t = np.clip((spread - lo) / (hi - lo), 0.0, 1.0)
    return np.where(have, t * t * (3.0 - 2.0 * t), 0.0)


MIN_CLEARANCE = 0.0002  # clothing never ends up closer than this to what it rests on


def clearance(verts, src_bvh, max_gap):
    """Gap each point should keep from a surface: its gap on the source, capped at Min Gap.
    Tight clothing stays as tight as designed instead of being lifted to Min Gap, which
    lifts some parts and not others and makes the fabric wobbly."""
    loc, nrm, idx, _dist = mu.nearest_on(src_bvh, verts, max_gap)
    side = ((verts - loc) * nrm).sum(1)
    return np.where(idx >= 0, np.clip(side, MIN_CLEARANCE, max_gap), max_gap)


def fit_points(obj, verts, tris, edges, field, s, solver, detail, body=None, layers=(), info=None, covers=(),
               limb_follow=None):
    """solver: 'ELASTIC' or 'DIRECT'; detail: Shape Detail for the elastic solver.
    body: (verts, normals, bvh) of the target body, or None. layers: fitted lower Layers.
    info: optional dict, receives "hidden" (the parts kept inside the body by design) and
    "poked" (a garment vertex where each point of the base comes through it, and how many
    points of the base it lies over).
    covers: BVHs of the outfit's other meshes as on the source, which can hide parts of this one.
    limb_follow: a limbs.LimbFollow for the garment's armature, or None."""
    adj = mu.Adjacency(len(verts), edges)
    relations = layer_relations(verts, tris, layers, field.bvh) if s.collide_layers else []

    hidden = hidden_parts(field, verts, tris, covers)
    if info is not None:
        info["hidden"] = hidden

    colliders = []
    pokers = []
    if body is not None:
        body_gaps = clearance(verts, field.bvh, s.collision_offset)
        colliders.append((body[2], hidden, BODY_DEPTH, body_gaps, body[3] if len(body) > 3 else None))
        # Body detail that already poked through this item on the source (e.g. toes through
        # socks meant to be hidden in shoes) may keep doing so; elsewhere it is pushed back
        # under. Each target body vertex looks this up at the nearest point of the
        # calibrated source body.
        base_normals = mu.vertex_normals(field.base, field.tris)
        loc, nrm, idx, dist = mu.nearest_on(mu.build_bvh(verts, tris), field.base, POKE_DEPTH)
        side = ((field.base - loc) * nrm).sum(1)
        poked = ((idx >= 0) & ((nrm * base_normals).sum(1) > 0.5) & (side > 0.0)
                 & (np.abs(side) >= 0.9 * np.maximum(dist, 1e-9)))
        mask = ~poked[field.nearest_vertex(body[0])] if poked.any() else np.ones(len(body[0]), dtype=bool)
        pokers.append((body[0][mask], body[1][mask], body_gaps))
        if info is not None:
            # For telling the user when the base comes through the clothing a lot: the
            # garment may be made for another shape of the base (a shrink or heel key).
            covered = (idx >= 0) & ((nrm * base_normals).sum(1) > 0.5)
            info["poked"] = (tris[idx[poked]][:, 0], int(covered.sum()))
    for layer, (_tucked, covered, exempt) in zip(layers, relations):
        gaps = (clearance(verts, layer.binding.bvh, s.collision_offset) if layer.binding is not None
                else np.full(len(verts), s.collision_offset))
        colliders.append((layer.fit_bvh, hidden if exempt is None else (hidden | exempt), LAYER_DEPTH, gaps))
        if covered is not None:
            pokers.append((layer.fit_verts[covered], mu.vertex_normals(layer.fit_verts, layer.tris)[covered], gaps))

    warped = field.warp(verts)
    near, dist, ok, side = field.surface_warp(verts, s.skin_distance + s.blend_distance)
    w = np.where(ok, _smoothstep_falloff(dist, s.skin_distance, s.blend_distance), 0.0)
    # Where the garment bridges over body detail instead of hugging it, it follows the smooth
    # warp: binding it to the skin there would copy the base's detail into it.
    w *= 1.0 - bridging(verts, edges, np.where(ok, side, np.nan))
    out = warped * (1.0 - w[:, None]) + near * w[:, None]
    if limb_follow is not None:
        follow, wf = limb_follow(obj, verts)
        out = out * (1.0 - wf[:, None]) + follow * wf[:, None]
    if relations:
        out = follow_layers(verts, out, dist, layers, relations, s)

    def collision_push(x):
        push = push_out(x, colliders)
        for pv, pn, gaps in pokers:
            poke = poke_through(x, tris, pv, pn, gaps)
            poke[hidden] = 0.0
            push = np.where((np.linalg.norm(poke, axis=1) > np.linalg.norm(push, axis=1))[:, None], poke, push)
        return push

    # Rigid parts sit where, and at the size, the material around them goes, but turn only
    # as much as the smooth warp does: following the nearby surface's detail would spin a
    # pendant resting in a cleavage, or a buckle on a curved strap.
    rigid = rigid_islands(obj, verts, edges, s.rigid_size)
    rigid_maps = []
    for ids in rigid:
        scale, _rot = mu.similarity_parts(verts[ids], out[ids])
        _scale, rot = mu.similarity_parts(verts[ids], warped[ids])
        rigid_maps.append(scale * rot)

    # Parts that were tucked under a lower layer on the source stay tucked under it.
    tucks = [(layer.fit_bvh, tucked & ~hidden) for layer, (tucked, _covered, _exempt) in zip(layers, relations)
             if tucked is not None and (tucked & ~hidden).any()]

    # Pieces of this garment lying on each other (a knit over its inner top) move out
    # together where the body pushes the lower one.
    stack = StackedPieces(verts, tris, edges, s.collision_offset)

    def carry(x, push):
        if not stack.any() or not push.any():
            return push
        push = stack.carry(x, push)
        for ids in rigid:
            push[ids] = push[ids[np.argmax(np.linalg.norm(push[ids], axis=1))]]
        return push

    if solver == 'ELASTIC':
        def elastic_push(x):
            # Collisions and tucks become constraints. Each is carried through the material
            # (not sideways): constraining only one side of a thin part would balloon or
            # squash it.
            push = collision_push(x)
            for bvh, mask in tucks:
                pull = tuck_pull(x, tris, [bvh], s.collision_offset, mask)
                push = np.where((np.linalg.norm(pull, axis=1) > np.linalg.norm(push, axis=1))[:, None], pull, push)
            if push.any():
                push = spread_pushes(x, push, THICKNESS_MAX, THROUGH_FADE)
            for ids in rigid:
                push[ids] = push[ids[np.argmax(np.linalg.norm(push[ids], axis=1))]]
            return carry(x, push)

        out = elastic.solve_fit(verts, tris, edges, out, rigid, rigid_maps, elastic_push,
                                detail, THICKNESS_MAX, s.collision_iterations)
        for ids in rigid:
            out[ids] = mu.similarity_fit(verts[ids], out[ids])
    else:
        out = _direct_fit(verts, out, adj, rigid, rigid_maps, collision_push, carry, s)
        for bvh, mask in tucks:
            out = tuck_in(out, tris, [bvh], body[2] if body is not None else None, s.collision_offset, mask=mask)
    return out


def _direct_fit(verts, out, adj, rigid, rigid_maps, collision_push, carry, s):
    """Place each vertex at its suggested position, then fix collisions vertex by vertex."""
    # Smooth the displacement, not the positions, so wrinkles and seams survive. Kept light:
    # heavier smoothing drags strap ends away from the rigid rings and buckles they hold.
    out = verts + adj.smooth(out - verts, s.displacement_smoothing)
    for ids, m in zip(rigid, rigid_maps):
        out[ids] = (verts[ids] - verts[ids].mean(0)) @ m.T + out[ids].mean(0)

    for it in range(s.collision_iterations + 1):
        push = collision_push(out)
        if not push.any():
            break
        if it < s.collision_iterations:
            # Spread through space rather than along the mesh: the two sides of a strap, or
            # a shell's inner and outer surface, are often not connected by edges. Pushing
            # only the side that collides would squash the part flat (e.g. a strap modelled
            # sinking into the source body's skin). The final pass pushes only what still
            # collides, by the small amounts left.
            push = spread_pushes(out, push, THICKNESS_MAX, PUSH_FADE)
        for ids in rigid:
            # Move the whole part by its strongest push so no corner stays inside.
            push[ids] = push[ids[np.argmax(np.linalg.norm(push[ids], axis=1))]]
        out += carry(out, push)
    return out


def spread_pushes(points, push, inner, fade):
    """Each point takes the strongest push found near it in space.

    Along the push direction (the other side of a strap, the outer shell over an inner
    one) the push applies in full up to `inner` away. Sideways, along the same surface, it
    fades smoothly over about `fade`, so a pushed patch becomes a gentle bump rather than
    a flat-topped lump."""
    mag = np.linalg.norm(push, axis=1)
    src = np.nonzero(mag > 1e-7)[0]
    # One source per 1 mm cell (the strongest) keeps the work bounded on dense meshes.
    src = src[np.argsort(-mag[src])]
    _, first = np.unique(np.floor(points[src] / 0.001).astype(np.int64), axis=0, return_index=True)
    src = src[np.sort(first)]

    kd = KDTree(len(points))
    for i, p in enumerate(points.tolist()):
        kd.insert(p, i)
    kd.balance()

    out = push.copy()
    best = mag.copy()
    for j in src.tolist():
        hits = kd.find_range(points[j].tolist(), inner + fade)
        if not hits:
            continue
        idx = np.fromiter((h[1] for h in hits), dtype=np.int64, count=len(hits))
        rel = points[idx] - points[j]
        along = rel @ (push[j] / mag[j])
        lateral = np.linalg.norm(rel - along[:, None] * (push[j] / mag[j]), axis=1)
        f = np.exp(-2.0 * (lateral / fade) ** 2) * ((along > -inner * 0.5) & (along < inner))
        v = mag[j] * f
        better = v > best[idx]
        idx, f = idx[better], f[better]
        best[idx] = v[better]
        out[idx] = push[j] * f[:, None]
    return out


def evaluated_shape(context, obj):
    """The mesh exactly as displayed (current pose and shape key values).

    Returns world-space posed vertices, triangles, edges, and the local-space vertices
    before armature deformation (the current shape key mix). Active shape keys often
    match the source body (e.g. "Foot_heel"), so they must be part of what gets fitted.
    """
    verts, tris, edges = mu.evaluated_mesh_world(obj, context.evaluated_depsgraph_get())
    if len(verts) != len(obj.data.vertices):
        raise ValueError(f"'{obj.name}': modifiers change the vertex count; apply or disable them first")
    arm_mods = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.show_viewport]
    for m in arm_mods:
        m.show_viewport = False
    context.view_layer.update()
    try:
        obj_eval = obj.evaluated_get(context.evaluated_depsgraph_get())
        me = obj_eval.to_mesh()
        try:
            unposed, _t, _e = mu.mesh_arrays(me)
        finally:
            obj_eval.to_mesh_clear()
    finally:
        for m in arm_mods:
            m.show_viewport = True
        context.view_layer.update()
    return verts, tris, edges, unposed


def write_positions(obj, local, current_mix):
    """Move the mesh so its current shape key mix lands on `local`.

    The same offset is added to every key, so each key's relative change is kept.
    """
    me = obj.data
    delta = np.ascontiguousarray(local - current_mix, dtype=np.float32).ravel()
    n = len(me.vertices)
    blocks = list(me.shape_keys.key_blocks) if me.shape_keys else []
    for kb in blocks:
        co = np.empty(n * 3, dtype=np.float32)
        kb.data.foreach_get("co", co)
        kb.data.foreach_set("co", co + delta)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    me.vertices.foreach_set("co", co + delta)
    me.update()


def fit_armature(context, arm, field, coll):
    """Copy an armature with its rest pose moved into the target proportions."""
    bones = sorted(arm.data.bones, key=bone_depth)
    mw = arm.matrix_world
    heads = np.array([(mw @ arm.pose.bones[b.name].head)[:] for b in bones])
    tails = np.array([(mw @ arm.pose.bones[b.name].tail)[:] for b in bones])
    zaxes = [(mw.to_3x3() @ arm.pose.bones[b.name].matrix.to_3x3()).col[2] for b in bones]
    new_heads, new_tails = field.warp(heads), field.warp(tails)

    new = arm.copy()
    new.data = arm.data.copy()
    new.name = arm.name + "_fitted"
    new["ah_fitted_from"] = arm.name
    if new.animation_data:
        new.animation_data_clear()
    coll.objects.link(new)

    inv = new.matrix_world.inverted()
    inv3 = inv.to_3x3()
    for o in context.selected_objects:
        o.select_set(False)
    context.view_layer.objects.active = new
    new.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        ebs = new.data.edit_bones
        for i, b in enumerate(bones):
            eb = ebs[b.name]
            head = inv @ Vector(new_heads[i])
            tail = inv @ Vector(new_tails[i])
            if (tail - head).length < 1e-5:
                tail = head + (eb.tail - eb.head).normalized() * 1e-3
            eb.head = head
            eb.tail = tail
            eb.align_roll(inv3 @ zaxes[i])
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    for pb in new.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    new.select_set(False)
    return new


HIDDEN_ATTRIBUTE = "ah_hidden"


def mark_hidden(obj, hidden):
    """Remember which parts were kept inside the body by design (a tucked-in hem), for the
    touch-up tools: they leave those alone, and push anything else back out of the body."""
    me = obj.data
    if HIDDEN_ATTRIBUTE in me.attributes:
        me.attributes.remove(me.attributes[HIDDEN_ATTRIBUTE])
    if hidden is None or len(hidden) != len(me.vertices):
        return
    # An integer attribute: Edit Mode (where the touch-up tools run) can read those, not booleans.
    attr = me.attributes.new(HIDDEN_ATTRIBUTE, 'INT', 'POINT')
    attr.data.foreach_set("value", np.asarray(hidden, dtype=np.int32))


def hidden_marks(obj):
    """The parts marked by mark_hidden, or None for meshes fitted before it existed."""
    me = obj.data
    if obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(me)
        layer = bm.verts.layers.int.get(HIDDEN_ATTRIBUTE)
        return None if layer is None else np.array([v[layer] != 0 for v in bm.verts], dtype=bool)
    attr = me.attributes.get(HIDDEN_ATTRIBUTE)
    if attr is None or attr.domain != 'POINT' or attr.data_type != 'INT':
        return None
    out = np.zeros(len(me.vertices), dtype=np.int32)
    attr.data.foreach_get("value", out)
    return out != 0


def remove_previous(names):
    names = set(names)
    for o in list(bpy.data.objects):
        if o.get("ah_fitted_from") in names:
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if data is not None and data.users == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Armature):
                    bpy.data.armatures.remove(data)


SHOE_NAMES = ("shoe", "boot", "heel", "sneaker", "sandal", "loafer", "slipper")


def is_shoe(obj):
    """"Auto" mode: mostly skinned to the feet, and nothing above the shins (high-top
    collars are usually skinned to the lower leg). A shoe-like name lowers the bar."""
    arms = deforming_armatures(obj)
    if not arms:
        return False
    arm = arms[0]
    mapping = armature_map(arm)
    feet = rig.foot_bone_names(arm, mapping, "L") | rig.foot_bone_names(arm, mapping, "R")
    shins = {mapping[k] for k in ("lower_leg.L", "lower_leg.R") if k in mapping}
    n = max(len(obj.data.vertices), 1)
    foot_share = len(rig.region_vertices(obj, feet)) / n
    below_knee_share = len(rig.region_vertices(obj, feet | shins)) / n
    if any(word in obj.name.lower() for word in SHOE_NAMES):
        return foot_share >= 0.3 and below_knee_share >= 0.9
    return foot_share >= 0.5 and below_knee_share >= 0.95


SHOE_EXTRA_SCALE = 1.15  # how much a shoe may grow beyond the foot-size estimate to cover what's inside
SHOE_STEP = 1.02
SHOE_ROOM = 0.003  # space a shoe tries to leave around the foot, for socks


def place_shoes(points, tris, edges, src_frames, tgt_frames, contents):
    """Move each shoe rigidly onto the target foot: turned to face the same way, ankle over
    ankle, and the sole on the target's floor.

    `points` is the shoe as designed (source rest pose), so heeled shoes stay heeled
    rather than being warped onto a differently shaped foot. The scale is uniform: first
    from the foot's length and width, then grown a little (up to SHOE_EXTRA_SCALE) until
    `contents` (points and normals of the target body) stay inside the shoe.
    """
    sides = [k for k in "LR" if k in src_frames and k in tgt_frames]
    if not sides:
        raise ValueError("Shoes need foot bones (and feet on both bodies) to be placed")
    labels = mu.islands(len(points), edges)
    centres = np.stack([np.bincount(labels, points[:, c]) for c in range(3)], axis=1) / np.bincount(labels)[:, None]
    ankles = np.array([src_frames[k][0] for k in sides])
    vert_side = np.argmin(np.linalg.norm(centres[:, None, :2] - ankles[None, :, :2], axis=2), axis=1)[labels]

    out = points.copy()
    for si, side in enumerate(sides):
        ids = np.nonzero(vert_side == si)[0]
        if not len(ids):
            continue
        a_s, t_s, floor_s, width_s = src_frames[side]
        a_t, t_t, floor_t, width_t = tgt_frames[side]
        yaw = math.atan2(t_t[1] - a_t[1], t_t[0] - a_t[0]) - math.atan2(t_s[1] - a_s[1], t_s[0] - a_s[0])
        c, sn = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        src = points[ids]

        def place(scale):
            moved = scale * (src - a_s) @ rot.T + a_t
            sole_offset = scale * (src[:, 2].min() - floor_s)
            moved[:, 2] += floor_t + sole_offset - moved[:, 2].min()
            return moved

        length_ratio = np.linalg.norm(t_t - a_t) / max(np.linalg.norm(t_s - a_s), 1e-6)
        width_ratio = width_t / max(width_s, 1e-6)
        base = float(np.clip(max(length_ratio, width_ratio), 0.7, 1.5))
        moved = place(base)

        # Grow until what should be inside the shoe is inside it.
        side_tris = tris[(vert_side[tris] == si).all(1)]
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        local_tris = remap[side_tris]
        content_pts, content_nrm = contents
        lo, hi = moved.min(0) - 0.02, moved.max(0) + 0.02
        near = ((content_pts >= lo) & (content_pts <= hi)).all(1)
        inside = (content_pts[near], content_nrm[near])
        scale = base
        while len(inside[0]) and len(local_tris) and scale * SHOE_STEP <= base * SHOE_EXTRA_SCALE:
            if _share_outside(moved, local_tris, *inside, room=SHOE_ROOM) <= 0.005:
                break
            scale *= SHOE_STEP
            moved = place(scale)
        out[ids] = moved
    return out


TUCK_DEPTH = 0.015  # how far a lower layer may stick out of a shoe and still be tucked in
SKIN_CLEARANCE = 0.0005  # a tucked layer never ends up closer to the skin than this


def tuck_pull(points, tris, shells, offset, mask=None):
    """Per point, the pull needed to bring it `offset` back under the shells it pokes
    out through (only where it faces the same way as the shell)."""
    normals = mu.vertex_normals(points, tris)
    pull = np.zeros_like(points)
    for bvh in shells:
        loc, nrm, idx, dist = mu.nearest_on(bvh, points, TUCK_DEPTH)
        side = ((points - loc) * nrm).sum(1)
        hit = (idx >= 0) & ((nrm * normals).sum(1) > 0.5) & (side > -offset) & (np.abs(side) >= 0.9 * dist)
        if mask is not None:
            hit &= mask
        v = np.where(hit[:, None], loc - nrm * offset - points, 0.0)
        stronger = np.linalg.norm(v, axis=1) > np.linalg.norm(pull, axis=1)
        pull[stronger] = v[stronger]
    return pull


def tuck_in(points, tris, shells, body_bvh, offset, mask=None):
    """Pull the parts of a layer that poke out through an outer shell (socks through a
    shoe, a tucked shirt hem through the trousers) back inside it, `offset` under the
    shell's surface, never into the body. `mask` limits which points may be pulled.

    Only areas facing the same way as a shell face count, so e.g. a sock leg rising out
    of the shoe opening is left alone.
    """
    pull = tuck_pull(points, tris, shells, offset, mask)
    if not pull.any():
        return points
    moved = points + spread_pushes(points, pull, THICKNESS_MAX, PUSH_FADE)
    if body_bvh is not None:
        # Where the shoe is tighter than the layer, settle the layer just on the skin.
        loc, nrm, idx, _dist = mu.nearest_on(body_bvh, moved, 0.05)
        into_body = (idx >= 0) & (((moved - loc) * nrm).sum(1) < SKIN_CLEARANCE)
        moved[into_body] = loc[into_body] + nrm[into_body] * SKIN_CLEARANCE
    return moved


def _share_outside(shell_points, shell_tris, points, normals, room=0.0):
    """Share of `points` poking out through the shell's outer faces, or closer than `room`
    under them (faces pointing the same way as the point's own surface; inner linings
    face the other way)."""
    loc, nrm, idx, dist = mu.nearest_on(mu.build_bvh(shell_points, shell_tris), points, 0.03)
    side = ((points - loc) * nrm).sum(1)
    out = (idx >= 0) & ((nrm * normals).sum(1) > 0.5) & (side > -room) & (np.abs(side) >= 0.9 * dist)
    return out.mean()


def _key_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_shape_keys(objs, values):
    """Set the clothing's shape keys named like the base body's (e.g. "Foot_heel") to the
    values the base had when the conversion was made: the conversion only fits clothing
    shaped for the base as it was then. Returns (key, old value) pairs to restore later, and
    a description of what changed."""
    wanted = {_key_name(k): (k, v) for k, v in values.items()}
    # Clothing keys often add a number to the base's name ("Foot_heel2" for "Foot_heel"):
    # matched too, when that leaves only one base key it could be.
    loose = {}
    for name, item in wanted.items():
        loose.setdefault(name.rstrip("0123456789"), []).append(item)
    saved, changes = [], []
    for o in objs:
        keys = o.data.shape_keys
        if keys is None:
            continue
        own = {_key_name(kb.name) for kb in keys.key_blocks}
        for kb in keys.key_blocks:
            if kb == keys.reference_key:
                continue
            name = _key_name(kb.name)
            match = wanted.get(name)
            if match is None:
                near = loose.get(name.rstrip("0123456789"), [])
                if len(near) == 1 and _key_name(near[0][0]) not in own:
                    match = near[0]
            if match is not None and abs(kb.value - match[1]) > 1e-4:
                saved.append((kb, kb.value))
                as_base = "" if _key_name(match[0]) == name else f" (as the base's '{match[0]}')"
                changes.append(f"{o.name} '{kb.name}' {kb.value:g} → {match[1]:g}{as_base}")
                kb.value = match[1]
    return saved, changes


def fit_clothing(context, s, progress=None, notes=None):
    """notes: optional list collecting things worth telling the user."""
    notes = [] if notes is None else notes
    if not s.conversion:
        raise ValueError("Choose a conversion (or create one) first")
    path = conversions.path_for(s.conversion)
    if not os.path.exists(path):
        raise ValueError(f"Conversion '{s.conversion}' not found in {conversions.library_dir()}")
    conv = conversions.Conversion.load(path)
    tgt_arm = target_armature(s)
    if not (s.target_body and tgt_arm):
        raise ValueError("Set the target body (with its armature)")

    objs, shoes, solvers = [], set(), {}
    for item in s.clothing:
        o = item.obj
        if o and o.type == 'MESH' and o not in objs and not is_result(o):
            objs.append(o)
            solvers[o] = (s.solver if item.solver == 'DEFAULT' else item.solver,
                          item.shape_detail if item.custom_detail else s.shape_detail)
            if item.mode == 'SHOES' or (item.mode == 'AUTO' and is_shoe(o)):
                shoes.add(o)
    if not objs:
        raise ValueError("Add clothing meshes to the list")

    arms = []
    for o in objs:
        for a in deforming_armatures(o):
            if a not in arms:
                arms.append(a)

    # Line the conversion up with each clothing armature (they may sit anywhere, or have only
    # some of the base's bones) and with the target avatar in this scene.
    tgt_align = conv.target_alignment(tgt_arm)
    src_align = {a: conv.source_alignment(a) for a in arms}
    default_align = next(iter(src_align.values()), np.eye(4))
    fields = {}

    def field_for(align):
        key = tuple(np.round(align, 5).ravel())
        if key not in fields:
            fields[key] = FitField.from_conversion(conv, align, tgt_align, s.field_resolution, s.field_smoothing,
                                                   CREASE_WIDTH)
        return fields[key]

    def align_of(o):
        obj_arms = deforming_armatures(o)
        return src_align[obj_arms[0]] if obj_arms else default_align

    tgt_now = shape_key_values(s.target_body)
    changed = sorted(k for k, v in conv.tgt_keys.items() if k in tgt_now and abs(tgt_now[k] - v) > 1e-3)
    if changed:
        notes.append(f"Your avatar's shape keys changed since the conversion was made ({', '.join(changed[:4])}); "
                     "create it again so the fit uses its current shape")

    # Pose the clothing armatures the way the source rig was posed for the conversion, and set
    # their shape keys as the base had them; both are restored afterwards (the fitted copies
    # keep the matched key values, as that's what they were fitted with).
    saved_pose = [(pb, pb.matrix_basis.copy()) for a in arms for pb in a.pose.bones]
    saved_keys, key_changes = match_shape_keys(objs, conv.src_keys)
    if key_changes:
        notes.append("Shape keys set to match the base: " + "; ".join(key_changes[:5])
                     + (f" (+{len(key_changes) - 5} more)" if len(key_changes) > 5 else ""))
    try:
        for a in arms:
            conv.apply_pose(context, a, src_align[a])

        # Shoes are taken as designed: the rest pose, before pose matching changed it.
        shoe_shapes, src_feet, tgt_feet = {}, {}, {}
        if shoes:
            rest_arms = set()
            for o in shoes:
                rest_arms.update(deforming_armatures(o))
            with rig.rest_pose(context, rest_arms):
                shoe_shapes = {o: evaluated_shape(context, o) for o in shoes}
            src_feet = {o: conv.feet(align_of(o)) for o in shoes}
            tgt_feet = rig.foot_frames(context, s.target_body, tgt_arm)

        if s.replace_previous:
            remove_previous([o.name for o in objs] + [a.name for a in arms])

        # Fingers and toes follow their bones (see limbs.py), measured in the matched pose.
        limb_maps = {}
        if LIMB_FOLLOW:
            tgt_verts = mu.evaluated_mesh_world(s.target_body, context.evaluated_depsgraph_get())[0]
            target = (tgt_verts, limbs.vertex_keys(s.target_body, tgt_arm, len(tgt_verts)))
            limb_maps = {a: limbs.LimbFollow(a, tgt_arm, src_align[a], tgt_align, conv, s.collision_offset, target)
                         for a in arms}

        def limbs_for(o):
            obj_arms = deforming_armatures(o)
            lf = limb_maps.get(obj_arms[0]) if obj_arms else None
            return lf if lf is not None and lf.any() else None

        coll = output_collection(context.scene, FITTED_COLLECTION)
        arm_map = {a: fit_armature(context, a, field_for(src_align[a]), coll) for a in arms}
        return _fit_meshes(context, s, objs, shoes, solvers, shoe_shapes, src_feet, tgt_feet,
                           field_for, align_of, arms, arm_map, coll, progress, limbs_for, notes)
    finally:
        for pb, m in saved_pose:
            pb.matrix_basis = m
        for kb, v in saved_keys:
            kb.value = v
        context.view_layer.update()


POKE_NOTE = 0.15  # share of the base under a garment coming through it worth a note
POKE_NOTE_MIN = 30  # points


def _poke_note(obj, poked_at, covered):
    """A note when much of the base comes through a garment, naming where."""
    if covered == 0 or len(poked_at) < POKE_NOTE_MIN or len(poked_at) / covered < POKE_NOTE:
        return None
    arms = deforming_armatures(obj)
    where = ""
    if arms:
        keys = limbs.vertex_keys(obj, arms[0], len(obj.data.vertices))[poked_at]
        keys = [k for k in keys if k]
        if keys:
            values, counts = np.unique(keys, return_counts=True)
            from .calibration import _part_name
            where = _part_name(str(values[np.argmax(counts)])) + " "
    return (f"The base's {where}comes through '{obj.name}' in places ({len(poked_at) / covered:.0%} of what it "
            f"covers): it may be made for another shape of the base. Check its shape keys against the base's "
            f"(e.g. a shrink or heel key), then fit again")


def _outfit_meshes(context, s, objs, arms):
    """BVH per mesh of the outfit as it is on the base: the meshes being fitted, and any
    other mesh on the same armatures (a shoe left out of Layers still hides a sock's toe)."""
    bodies = {s.source_body, s.target_body}
    meshes = [o for o in context.scene.objects
              if o.type == 'MESH' and o not in bodies and not is_result(o)
              and (o in objs or set(deforming_armatures(o)) & set(arms))]
    dg = context.evaluated_depsgraph_get()
    out = {}
    for o in meshes:
        verts, tris, _edges = mu.evaluated_mesh_world(o, dg)
        if len(tris):
            out[o] = mu.build_bvh(verts, tris)
    return out


def _fit_meshes(context, s, objs, shoes, solvers, shoe_shapes, src_feet, tgt_feet,
                field_for, align_of, arms, arm_map, coll, progress, limbs_for=lambda o: None, notes=None):
    body_verts, tt, te = mu.evaluated_mesh_world(s.target_body, context.evaluated_depsgraph_get())
    body_normals = mu.vertex_normals(body_verts, tt)
    # Avatars often have hair, accessories or clothing merged into the body mesh. Only the
    # parts the conversion matched (body, head) are collided with; clothing would otherwise
    # be pushed out of every hair strand or merged-in choker, e.g. a hood lying in long hair.
    # Decided per connected part, so a strand or choker touching the skin is still left out.
    skin = body_parts(body_verts, te, field_for(align_of(objs[0])).bvh_fitted)
    tt = tt[skin[tt].all(1)]
    body_bvh = mu.build_bvh(body_verts, tt)
    body = (body_verts[skin], body_normals[skin], body_bvh, SmoothSide(body_verts, tt, body_normals))

    # Shoes are placed first, sized to cover the target's feet. Layers listed before a shoe
    # (socks, stockings) are then tucked inside it rather than the shoe growing around them.
    placed = {}
    for o in shoes:
        verts, tris, edges, _unposed = shoe_shapes[o]
        placed[o] = place_shoes(verts, tris, edges, src_feet[o], tgt_feet, body[:2])

    covers = _outfit_meshes(context, s, objs, arms)
    layers = []
    results = []
    for i, o in enumerate(objs):
        info = {}
        if o in shoes:
            verts, tris, edges, unposed = shoe_shapes[o]
            fitted = placed[o]
        else:
            verts, tris, edges, unposed = evaluated_shape(context, o)
            fitted = fit_points(o, verts, tris, edges, field_for(align_of(o)), s, *solvers[o], body,
                                layers if s.collide_layers else (), info,
                                [bvh for c, bvh in covers.items() if c is not o], limbs_for(o))
            outer_shoes = [mu.build_bvh(placed[sh], shoe_shapes[sh][1]) for sh in shoes if objs.index(sh) > i]
            if outer_shoes:
                fitted = tuck_in(fitted, tris, outer_shoes, body_bvh, s.collision_offset)

        new = o.copy()
        new.data = o.data.copy()
        new.name = o.name + "_fitted"
        new["ah_fitted_from"] = o.name
        coll.objects.link(new)
        for m in new.modifiers:
            if m.type == 'ARMATURE' and m.object in arm_map:
                m.object = arm_map[m.object]
        if new.parent in arm_map and new.parent_type in {'OBJECT', 'ARMATURE'}:
            new.parent = arm_map[new.parent]
        write_positions(new, mu.transform_points(o.matrix_world.inverted(), fitted), unposed)
        mark_hidden(new, info.get("hidden"))
        if notes is not None and "poked" in info:
            note = _poke_note(o, *info["poked"])
            if note:
                notes.append(note)

        layers.append(Layer(None if o in shoes else verts, fitted, tris))
        results.append(new)
        if progress:
            progress((i + 1) / len(objs))

    if s.hide_originals:
        # Keep armatures that still drive a body (e.g. the source avatar's own) visible.
        in_use = {a for b in (s.source_body, s.target_body) if b for a in deforming_armatures(b)}
        for o in objs:
            o.hide_set(True)
        for a in arms:
            if a not in in_use:
                a.hide_set(True)
    return results, list(arm_map.values()), sorted(o.name for o in shoes)
