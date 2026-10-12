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

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from . import mesh_utils as mu
from . import rig
from .calibration import CALIBRATION_KEY, output_collection, source_armature, target_armature
from .rbf import RBFWarp
from .rig import armature_map, bone_depth, deforming_armatures

FITTED_COLLECTION = "AH_Fitted"
RIGID_GROUP = "AH_Rigid"


class SurfaceBinding:
    """Carries points along with a surface that moved: each point is expressed in the
    frame of its nearest triangle before, and rebuilt in that triangle's frame after."""

    def __init__(self, base, fitted, tris):
        self.base = base
        self.fitted = fitted
        self.tris = tris
        self.bvh = mu.build_bvh(base, tris)
        self.src_frames = mu.triangle_frames(base, tris)
        self.fit_frames = mu.triangle_frames(fitted, tris)

    def surface_warp(self, pts, max_dist):
        """Returns (moved points, distance to the surface, whether a triangle was found,
        signed offset along the triangle normal: > 0 on the outer side)."""
        loc, _nrm, idx, dist = mu.nearest_on(self.bvh, pts, max_dist)
        ok = idx >= 0
        out = pts.copy()
        side = np.zeros(len(pts))
        if not ok.any():
            return out, dist, ok, side
        ti = idx[ok]
        t = self.tris[ti]
        b = self.base
        bary = mu.barycentric(loc[ok], b[t[:, 0]], b[t[:, 1]], b[t[:, 2]])
        rel = pts[ok] - loc[ok]
        e1, e2, n = (f[ti] for f in self.src_frames)
        coef = np.stack([(rel * e1).sum(1), (rel * e2).sum(1), (rel * n).sum(1)], axis=1)
        f = self.fitted
        anchor = (bary[:, :1] * f[t[:, 0]] + bary[:, 1:2] * f[t[:, 1]] + bary[:, 2:] * f[t[:, 2]])
        g1, g2, gn = (fr[ti] for fr in self.fit_frames)
        out[ok] = anchor + coef[:, :1] * g1 + coef[:, 1:2] * g2 + coef[:, 2:] * gn
        side[ok] = coef[:, 2]
        return out, dist, ok, side


class FitField(SurfaceBinding):
    """The calibration as a deformation: a smooth space warp plus binding to the body surface."""

    def __init__(self, cal_obj, resolution, smoothing, max_centers=2500):
        me = cal_obj.data
        keys = me.shape_keys
        if keys is None or CALIBRATION_KEY not in keys.key_blocks:
            raise ValueError(f"'{cal_obj.name}' has no '{CALIBRATION_KEY}' shape key; run Calibrate first")
        n = len(me.vertices)
        base = mu.transform_points(cal_obj.matrix_world, mu.read_shape_key(keys.reference_key, n))
        fitted = mu.transform_points(cal_obj.matrix_world, mu.read_shape_key(keys.key_blocks[CALIBRATION_KEY], n))
        _co, tris, _edges = mu.mesh_arrays(me)
        super().__init__(base, fitted, tris)
        centers, values = mu.voxel_downsample(base, fitted - base, resolution, max_centers)
        self.rbf = RBFWarp(centers, values, smoothing)

    def warp(self, pts):
        return pts + self.rbf(pts)

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
    result = []
    for lab in range(labels.max() + 1 if len(labels) else 0):
        ids = np.nonzero(labels == lab)[0]
        if len(ids) < 3:
            continue
        size = np.linalg.norm(np.ptp(verts[ids], axis=0))
        if size <= max_size or in_group[ids].any():
            result.append(ids)
    return result


def push_out(points, colliders, offset):
    """Outward-only push so every point sits at least `offset` outside each collider.

    colliders: (bvh, skip, depth) tuples. `skip` marks points exempt from that collider
    (parts that sat under it on the source, like a tucked-in shirt hem), or is None.
    `depth` is how far inside a point may be and still count as colliding.
    """
    push = np.zeros_like(points)
    for i, p in enumerate(points.tolist()):
        best = 0.0
        pv = Vector(p)
        for bvh, skip, depth in colliders:
            if skip is not None and skip[i]:
                continue
            loc, nrm, _idx, d = bvh.find_nearest(p, depth + offset)
            if loc is None:
                continue
            rel = pv - loc
            side = rel.dot(nrm)
            # Skip hits on open edges reached from beyond the surface (e.g. below a shirt hem).
            if d > 1e-5 and abs(side) / d < 0.7:
                continue
            if side < offset:
                v = loc + nrm * offset - pv
                if v.length > best:
                    best = v.length
                    push[i] = v
    return push


POKE_DEPTH = 0.01  # deeper "pokes" are usually skin showing through an opening on purpose
THICKNESS_MAX = 0.006  # points this close get the same collision push (both sides of a thin part)
PUSH_FADE = 0.01  # beyond that, collision pushes fade out over this distance


def poke_through(points, tris, body_verts, body_normals, offset):
    """Outward push for clothing faces that body vertices (or lower-layer vertices that
    belong under this clothing) come through.

    Catches body detail (knees, collarbones) poking out between the vertices of
    lower-resolution clothing, which the per-vertex test in push_out misses.
    Only outer-facing clothing faces count; the inner shell of thick clothing faces the
    body and is skipped.
    """
    push = np.zeros_like(points)
    amount = np.zeros(len(points))
    loc, nrm, idx, dist = mu.nearest_on(mu.build_bvh(points, tris), body_verts, POKE_DEPTH)
    side = ((body_verts - loc) * nrm).sum(1)
    facing = (nrm * body_normals).sum(1) > 0.5
    # Facing the face (not reached around an open edge) and closer than `offset` below it, or outside.
    hit = (idx >= 0) & facing & (side > -offset) & (np.abs(side) >= 0.9 * dist)
    for ti, a, n in zip(idx[hit].tolist(), (side[hit] + offset).tolist(), nrm[hit]):
        for v in tris[ti]:
            if a > amount[v]:
                amount[v] = a
                push[v] = n * a
    return push


LAYER_RANGE = 0.03  # how far apart two layers can be and still count as touching
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


def colliding(points, bvh, offset, depth):
    """Points that push_out would move for this surface: closer than `offset` outside it,
    or up to `depth` inside it (not counting points reached around an open edge)."""
    loc, nrm, idx, dist = mu.nearest_on(bvh, points, depth + offset)
    side = ((points - loc) * nrm).sum(1)
    return (idx >= 0) & (side < offset) & (np.abs(side) >= 0.7 * np.maximum(dist, 1e-9))


def layer_relations(verts, tris, layers, offset):
    """How this item sat relative to each lower layer on the source.

    Returns per layer: (tucked, covered, exempt).
      tucked: this item's vertices that were under the layer (e.g. a shirt hem inside the
        trousers); they are kept under it. "Under" means a short ray outward along the
        surface hits the layer, which also works at open rims and thick waistbands.
      covered: the layer's vertices that were under this item; they must not poke through it.
      exempt: vertices never pushed out of the layer: tucked ones, and ones that already
        touched or crossed it on the source (a choker under a collar), so only vertices
        that newly cross it during fitting are pushed.
    """
    normals = mu.vertex_normals(verts, tris)
    item_bvh = mu.build_bvh(verts, tris)
    out = []
    for layer in layers:
        if layer.binding is None:
            out.append((None, None, None))
            continue
        tucked = mu.rays_hit(verts, normals, layer.binding.bvh, LAYER_RANGE)
        covered = mu.rays_hit(layer.src_verts, mu.vertex_normals(layer.src_verts, layer.tris), item_bvh, LAYER_RANGE)
        exempt = tucked | colliding(verts, layer.binding.bvh, offset, LAYER_DEPTH)
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


HEM_RANGE = 0.015  # how close to an open edge a part inside the body must be to stay hidden


def hidden_parts(field, verts, tris):
    """Clothing tucked into the source body is hidden by design: a hem curling in under a
    waistband, a strap end disappearing into the thigh, a sock's toe squeezed inside the
    foot to fit a shoe. Such parts are carried along at the same depth rather than pushed
    out, which would crumple them.

    A thin part that merely dips into the skin (the inner side of a strap, with its outer
    side right above it) doesn't count: it is pushed out whole as usual, which keeps it
    clean. So a part inside the body stays hidden only if it is near an open edge, or if
    nothing of the same garment lies over it.
    """
    loc, nrm, idx, _dist = mu.nearest_on(field.bvh, verts, 0.05)
    inside = (idx >= 0) & (((verts - loc) * nrm).sum(1) < 0.0)
    if not inside.any():
        return inside
    ids = np.nonzero(inside)[0]
    covered_by_self = mu.rays_hit(verts[ids], nrm[ids], mu.build_bvh(verts, tris), THICKNESS_MAX)
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    rim = np.unique(uniq[counts == 1])
    near_rim = np.zeros(len(ids), dtype=bool)
    if len(rim):
        kd = KDTree(len(rim))
        for i in rim.tolist():
            kd.insert(verts[i].tolist(), i)
        kd.balance()
        near_rim = np.array([kd.find(verts[i].tolist())[2] < HEM_RANGE for i in ids.tolist()])
    inside[ids] = near_rim | ~covered_by_self
    return inside


def fit_points(obj, verts, tris, edges, field, s, body=None, layers=()):
    """body: (verts, normals, bvh) of the target body, or None. layers: fitted lower Layers."""
    adj = mu.Adjacency(len(verts), edges)
    relations = layer_relations(verts, tris, layers, s.collision_offset) if s.collide_layers else []

    hidden = hidden_parts(field, verts, tris)

    colliders = [(body[2], hidden, BODY_DEPTH)] if body is not None else []
    pokers = []
    if body is not None:
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
        pokers.append((body[0][mask], body[1][mask]))
    for layer, (_tucked, covered, exempt) in zip(layers, relations):
        colliders.append((layer.fit_bvh, hidden if exempt is None else (hidden | exempt), LAYER_DEPTH))
        if covered is not None:
            pokers.append((layer.fit_verts[covered], mu.vertex_normals(layer.fit_verts, layer.tris)[covered]))

    out = field.warp(verts)
    near, dist, ok, _side = field.surface_warp(verts, s.skin_distance + s.blend_distance)
    w = np.where(ok, _smoothstep_falloff(dist, s.skin_distance, s.blend_distance), 0.0)
    out = out * (1.0 - w[:, None]) + near * w[:, None]
    if relations:
        out = follow_layers(verts, out, dist, layers, relations, s)

    # Smooth the displacement, not the positions, so wrinkles and seams survive. Kept light:
    # heavier smoothing drags strap ends away from the rigid rings and buckles they hold.
    disp = adj.smooth(out - verts, s.displacement_smoothing)
    out = verts + disp

    rigid = rigid_islands(obj, verts, edges, s.rigid_size)
    for ids in rigid:
        out[ids] = mu.similarity_fit(verts[ids], out[ids])

    for it in range(s.collision_iterations + 1):
        push = push_out(out, colliders, s.collision_offset)
        for pv, pn in pokers:
            poke = poke_through(out, tris, pv, pn, s.collision_offset)
            poke[hidden] = 0.0
            push = np.where((np.linalg.norm(poke, axis=1) > np.linalg.norm(push, axis=1))[:, None], poke, push)
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
        out += push

    # Parts that were tucked under a lower layer on the source stay tucked under it.
    for layer, (tucked, _covered, _exempt) in zip(layers, relations):
        if tucked is not None and (tucked & ~hidden).any():
            out = tuck_in(out, tris, [layer.fit_bvh], body[2] if body is not None else None,
                          s.collision_offset, mask=tucked & ~hidden)
    return out


def spread_pushes(points, push, inner, fade):
    """Each point takes the strongest push found near it in space: full strength within
    `inner`, fading linearly to nothing at `inner + fade`."""
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
        d = np.fromiter((h[2] for h in hits), dtype=np.float64, count=len(hits))
        f = np.clip(1.0 - (d - inner) / fade, 0.0, 1.0)
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


def tuck_in(points, tris, shells, body_bvh, offset, mask=None):
    """Pull the parts of a layer that poke out through an outer shell (socks through a
    shoe, a tucked shirt hem through the trousers) back inside it, `offset` under the
    shell's surface, never into the body. `mask` limits which points may be pulled.

    Only areas facing the same way as a shell face count, so e.g. a sock leg rising out
    of the shoe opening is left alone.
    """
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


def fit_clothing(context, s, progress=None):
    if not s.calibration:
        raise ValueError("Run Calibrate first")
    objs, shoes = [], set()
    for item in s.clothing:
        o = item.obj
        if o and o.type == 'MESH' and o not in objs and not o.get("ah_fitted_from"):
            objs.append(o)
            if item.mode == 'SHOES' or (item.mode == 'AUTO' and is_shoe(o)):
                shoes.add(o)
    if not objs:
        raise ValueError("Add clothing meshes to the list")

    field = FitField(s.calibration, s.field_resolution, s.field_smoothing)

    # Shoes are taken as designed: the source rest pose, before Match Pose changed it.
    shoe_shapes, src_feet, tgt_feet = {}, {}, {}
    if shoes:
        src_arm, tgt_arm = source_armature(s), target_armature(s)
        if not (src_arm and tgt_arm and s.source_body and s.target_body):
            raise ValueError("Shoes need both bodies and armatures set")
        rest_arms = {src_arm, *deforming_armatures(s.source_body)}
        for o in shoes:
            rest_arms.update(deforming_armatures(o))
        with rig.rest_pose(context, rest_arms):
            src_feet = rig.foot_frames(context, s.source_body, src_arm)
            shoe_shapes = {o: evaluated_shape(context, o) for o in shoes}
        tgt_feet = rig.foot_frames(context, s.target_body, tgt_arm)

    arms = []
    for o in objs:
        for a in deforming_armatures(o):
            if a not in arms:
                arms.append(a)
    if s.replace_previous:
        remove_previous([o.name for o in objs] + [a.name for a in arms])

    coll = output_collection(context.scene, FITTED_COLLECTION)
    arm_map = {a: fit_armature(context, a, field, coll) for a in arms}

    body = None
    body_bvh = None
    if s.target_body:
        body_verts, tt, _ = mu.evaluated_mesh_world(s.target_body, context.evaluated_depsgraph_get())
        body_bvh = mu.build_bvh(body_verts, tt)
        body = (body_verts, mu.vertex_normals(body_verts, tt), body_bvh)

    # Shoes are placed first, sized to cover the target's feet. Layers listed before a shoe
    # (socks, stockings) are then tucked inside it rather than the shoe growing around them.
    placed = {}
    for o in shoes:
        verts, tris, edges, _unposed = shoe_shapes[o]
        contents = body[:2] if body is not None else (np.zeros((0, 3)),) * 2
        placed[o] = place_shoes(verts, tris, edges, src_feet, tgt_feet, contents)

    layers = []
    results = []
    for i, o in enumerate(objs):
        if o in shoes:
            verts, tris, edges, unposed = shoe_shapes[o]
            fitted = placed[o]
        else:
            verts, tris, edges, unposed = evaluated_shape(context, o)
            fitted = fit_points(o, verts, tris, edges, field, s, body, layers if s.collide_layers else ())
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

        layers.append(Layer(None if o in shoes else verts, fitted, tris))
        results.append(new)
        if progress:
            progress((i + 1) / len(objs))

    if s.hide_originals:
        src_arm = source_armature(s)
        for o in objs:
            o.hide_set(True)
        for a in arms:
            if a is not src_arm:
                a.hide_set(True)
    return results, list(arm_map.values()), sorted(o.name for o in shoes)
