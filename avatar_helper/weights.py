"""Rig fitted clothing to the avatar's own armature, instead of the fitted outfit armature.

Clothing close to the body takes the body's own skin weights at the nearest point of the
skin, so in any pose it bends exactly like the skin under it and doesn't clip into it.
Loose parts (further from the body) keep the outfit's weights, moved over to the avatar's
matching bones, and parts driven by the outfit's own bones (skirt or ribbon bones for
physics) keep those: the bones are added to the avatar's armature, under the matching bone.
"""
import bpy
import numpy as np
from mathutils import Matrix

from . import mesh_utils as mu
from .rig import armature_map, bone_depth

MAX_INFLUENCES = 4  # bones per vertex, as Unity skins with
WEIGHT_SMOOTHING = 2  # neighbour-averaging passes, so weights don't jump between pieces


def _smoothstep_falloff(dist, inner, blend):
    t = np.clip((dist - inner) / max(blend, 1e-6), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def is_rest(arm):
    return all(pb.matrix_basis == Matrix.Identity(4) for pb in arm.pose.bones)


def vertex_weights(obj, bones):
    """Dense weights (vertices x bones) of `obj`'s vertex groups named in `bones`."""
    col = {b: i for i, b in enumerate(bones)}
    gi = {g.index: col[g.name] for g in obj.vertex_groups if g.name in col}
    w = np.zeros((len(obj.data.vertices), len(bones)))
    for v in obj.data.vertices:
        for g in v.groups:
            c = gi.get(g.group)
            if c is not None:
                w[v.index, c] = g.weight
    return w


def bone_mapping(outfit_arm, avatar_arm):
    """Outfit bone name -> avatar bone name: humanoid bones by what they are, others by name.
    Outfit bones with no match are returned separately (to be added to the avatar)."""
    out_map, av_map = armature_map(outfit_arm), armature_map(avatar_arm)
    names = {out_map[k]: av_map[k] for k in out_map if k in av_map}
    extra = []
    for b in outfit_arm.data.bones:
        if b.name in names:
            continue
        if b.name in avatar_arm.data.bones:
            names[b.name] = b.name
        else:
            extra.append(b.name)
    return names, extra


def rig_to_avatar(context, meshes, avatar_arm, body, body_mask, skin_distance, blend_distance):
    """Rebind fitted meshes (bound to fitted outfit armatures) to `avatar_arm`. `body` is the
    avatar's body mesh, `body_mask` which of its vertices are body (not hair or accessories).
    Returns the outfit bones added to the avatar's armature."""
    if not is_rest(avatar_arm):
        raise ValueError(f"Put '{avatar_arm.name}' in its rest pose first (Pose Mode > Pose > Clear Transform)")
    deform = [b.name for b in avatar_arm.data.bones if b.use_deform]
    dg = context.evaluated_depsgraph_get()
    bv, bt, _be = mu.evaluated_mesh_world(body, dg)
    if len(bv) != len(body.data.vertices):
        raise ValueError(f"'{body.name}': modifiers change the vertex count; apply or disable them first")
    body_w = vertex_weights(body, deform)
    keep = body_mask & (body_w.sum(1) > 0)
    bt = bt[keep[bt].all(1)]
    bvh = mu.build_bvh(bv, bt)

    added = {}
    for obj in meshes:
        outfit_arm = next((m.object for m in obj.modifiers if m.type == 'ARMATURE' and m.object), None)
        names, extra = bone_mapping(outfit_arm, avatar_arm) if outfit_arm else ({}, [])
        bones = list(dict.fromkeys(deform + [names[g.name] for g in obj.vertex_groups if g.name in names]
                                   + [g.name for g in obj.vertex_groups if g.name in extra]))
        col = {b: i for i, b in enumerate(bones)}

        # The outfit's own weights, moved onto the avatar's bones (and the outfit's extra bones).
        own = np.zeros((len(obj.data.vertices), len(bones)))
        src = vertex_weights(obj, [g.name for g in obj.vertex_groups])
        for gi, g in enumerate(obj.vertex_groups):
            target = names.get(g.name, g.name if g.name in extra else None)
            if target is not None:
                own[:, col[target]] += src[:, gi]
        extra_share = own[:, [col[b] for b in bones if b not in deform]].sum(1) / np.maximum(own.sum(1), 1e-9) \
            if any(b not in deform for b in bones) else np.zeros(len(own))

        # The body's weights at the nearest point of the skin.
        verts, tris, edges = mu.evaluated_mesh_world(obj, dg)
        loc, _nrm, idx, dist = mu.nearest_on(bvh, verts, skin_distance + blend_distance)
        near = idx >= 0
        skin = np.zeros_like(own)
        if near.any():
            t = bt[idx[near]]
            bary = mu.barycentric(loc[near], bv[t[:, 0]], bv[t[:, 1]], bv[t[:, 2]])
            w = sum(bary[:, i:i + 1] * body_w[t[:, i]] for i in range(3))
            skin[np.ix_(near, [col[b] for b in deform])] = w
        # Near the body: the body's weights; further out: the outfit's own. Parts on the outfit's
        # own (physics) bones keep those.
        blend = np.where(near, _smoothstep_falloff(np.where(near, dist, 1.0), skin_distance, blend_distance), 0.0)
        blend *= 1.0 - extra_share
        weights = blend[:, None] * skin + (1.0 - blend[:, None]) * own
        weights = mu.Adjacency(len(verts), edges).smooth(weights, WEIGHT_SMOOTHING) if len(edges) else weights
        weights = _limit(weights, MAX_INFLUENCES)

        _write_groups(obj, bones, weights)
        for m in obj.modifiers:
            if m.type == 'ARMATURE':
                m.object = avatar_arm
        mw = obj.matrix_world.copy()
        obj.parent = avatar_arm
        obj.parent_type = 'OBJECT'
        obj.matrix_world = mw
        for b in extra:
            if b in col and weights[:, col[b]].any():
                added.setdefault(outfit_arm, set()).add(b)

    new_bones = []
    for outfit_arm, used in added.items():
        new_bones += add_bones(context, outfit_arm, avatar_arm, used)
    return new_bones


def _limit(weights, n):
    if weights.shape[1] > n:
        cut = np.argsort(-weights, axis=1)[:, n:]
        np.put_along_axis(weights, cut, 0.0, axis=1)
    weights[weights < 1e-4] = 0.0
    total = weights.sum(1, keepdims=True)
    return np.where(total > 0, weights / np.maximum(total, 1e-9), weights)


def _write_groups(obj, bones, weights):
    obj.vertex_groups.clear()
    for c, name in enumerate(bones):
        ids = np.nonzero(weights[:, c] > 0)[0]
        if not len(ids):
            continue
        g = obj.vertex_groups.new(name=name)
        for w in np.unique(np.round(weights[ids, c], 4)):
            g.add(ids[np.round(weights[ids, c], 4) == w].tolist(), float(w), 'REPLACE')


def add_bones(context, outfit_arm, avatar_arm, used):
    """Add the outfit's own bones that are used (and the chains leading to them, e.g. a
    skirt's bones for physics) to the avatar's armature, at their fitted rest positions,
    under the avatar's matching bone."""
    names, extra = bone_mapping(outfit_arm, avatar_arm)
    extra = set(extra)
    wanted = set()
    for name in used:
        b = outfit_arm.data.bones[name]
        while b is not None and b.name in extra:
            wanted.add(b.name)
            b = b.parent
    if not wanted:
        return []
    to_avatar = Matrix(np.linalg.inv(np.array(avatar_arm.matrix_world)).tolist()) @ outfit_arm.matrix_world
    scale = to_avatar.to_scale()[0]
    bones = sorted((outfit_arm.data.bones[n] for n in wanted), key=bone_depth)
    prev_active = context.view_layer.objects.active
    prev_mode = context.object.mode if context.object else 'OBJECT'
    if prev_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    hidden = avatar_arm.hide_get()
    avatar_arm.hide_set(False)
    context.view_layer.objects.active = avatar_arm
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        ebs = avatar_arm.data.edit_bones
        for b in bones:
            eb = ebs.new(b.name)
            eb.head = (0.0, 0.0, 0.0)
            eb.tail = (0.0, 1.0, 0.0)
            m = to_avatar @ b.matrix_local
            eb.matrix = m.normalized()
            eb.length = b.length * scale
            eb.use_deform = b.use_deform
            parent = b.parent.name if b.parent else None
            parent = parent if parent in wanted else names.get(parent)
            if parent in ebs:
                eb.parent = ebs[parent]
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        avatar_arm.hide_set(hidden)
        context.view_layer.objects.active = prev_active
    return [b.name for b in bones]
