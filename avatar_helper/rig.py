"""Armature helpers: humanoid mapping, pose matching and landmark pairs."""
from contextlib import contextmanager

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import mesh_utils as mu
from .bone_map import FINGER_NAMES, humanoid_map, pose_segments

LANDMARK_COLLECTION = "AH_Landmarks"


def armature_map(arm_obj):
    return humanoid_map((b.name, b.parent.name if b.parent else None) for b in arm_obj.data.bones)


def deforming_armatures(obj):
    arms = []
    for m in obj.modifiers:
        if m.type == 'ARMATURE' and m.object and m.object not in arms:
            arms.append(m.object)
    return arms


def bone_depth(bone):
    d = 0
    while bone.parent:
        d += 1
        bone = bone.parent
    return d


def reset_pose(arm):
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)


def posed_head(arm, bone_name):
    return arm.matrix_world @ arm.pose.bones[bone_name].head


def match_pose(context, src_arm, tgt_arm, followers=(), src_body=None, tgt_body=None):
    """Rotate src_arm's bones so each limb segment points the same way as on tgt_arm.

    Followers (e.g. an outfit's own armature) receive the same world-space change per
    matching humanoid bone, so clothing bound to them moves with the body.
    When a foot or hand has no matching child bone on both rigs (e.g. no toe bones),
    the tip of that body part on the mesh is used instead, if the bodies are given.
    Returns the number of bones rotated.
    """
    src_map = armature_map(src_arm)
    tgt_map = armature_map(tgt_arm)
    followers = [a for a in followers if a is not src_arm and a is not tgt_arm]

    for arm in [src_arm] + followers:
        reset_pose(arm)
    context.view_layer.update()

    src_tips, tgt_tips = {}, {}
    if src_body and tgt_body:
        src_tips = extremity_tips(context, src_body, src_arm, src_map)
        tgt_tips = extremity_tips(context, tgt_body, tgt_arm, tgt_map)

    src_inv = src_arm.matrix_world.inverted()
    rotated = 0
    for key, children in pose_segments():
        if key not in src_map or key not in tgt_map:
            continue
        child = next((c for c in children if c in src_map and c in tgt_map), None)
        sh = posed_head(src_arm, src_map[key])
        if child is not None:
            ds = posed_head(src_arm, src_map[child]) - sh
            dt = posed_head(tgt_arm, tgt_map[child]) - posed_head(tgt_arm, tgt_map[key])
        elif key in src_tips and key in tgt_tips:
            ds = tip_position(context, src_body, src_tips[key]) - sh
            dt = tip_position(context, tgt_body, tgt_tips[key]) - posed_head(tgt_arm, tgt_map[key])
        else:
            continue
        if ds.length < 1e-6 or dt.length < 1e-6:
            continue
        q = ds.rotation_difference(dt)
        if q.angle < 1e-5:
            continue
        rot = Matrix.Translation(sh) @ q.to_matrix().to_4x4() @ Matrix.Translation(-sh)
        pb = src_arm.pose.bones[src_map[key]]
        pb.matrix = src_inv @ rot @ src_arm.matrix_world @ pb.matrix
        context.view_layer.update()
        rotated += 1

    for arm in followers:
        follow_pose(context, src_arm, src_map, arm)
    return rotated


def follow_pose(context, src_arm, src_map, arm):
    fmap = armature_map(arm)
    keys = sorted((k for k in fmap if k in src_map), key=lambda k: bone_depth(arm.data.bones[fmap[k]]))
    inv = arm.matrix_world.inverted()
    for k in keys:
        sb = src_arm.data.bones[src_map[k]]
        spb = src_arm.pose.bones[src_map[k]]
        delta = (src_arm.matrix_world @ spb.matrix) @ (src_arm.matrix_world @ sb.matrix_local).inverted()
        fb = arm.data.bones[fmap[k]]
        arm.pose.bones[fmap[k]].matrix = inv @ delta @ arm.matrix_world @ fb.matrix_local
        context.view_layer.update()


CENTRAL_KEYS = ("hips", "spine", "chest", "upper_chest", "neck", "head")
MIRROR_KEYS = ("upper_leg", "shoulder", "upper_arm")


def joint_positions(arm, mapping):
    """World positions of mapped bone heads, with centre-line joints snapped onto the
    rig's mirror plane (some rigs place e.g. the Hips head a few cm off-centre)."""
    pos = {k: posed_head(arm, n) for k, n in mapping.items()}
    pairs = [(pos[f"{k}.L"], pos[f"{k}.R"]) for k in MIRROR_KEYS if f"{k}.L" in pos and f"{k}.R" in pos]
    pairs = [(l, r) for l, r in pairs if (l - r).length > 1e-4]
    if pairs:
        normal = Vector((0.0, 0.0, 0.0))
        for l, r in pairs:
            normal += (l - r).normalized()
        normal.normalize()
        point = sum(((l + r) * 0.5 for l, r in pairs), Vector((0.0, 0.0, 0.0))) / len(pairs)
        for k in CENTRAL_KEYS:
            if k in pos:
                pos[k] = pos[k] - normal * (pos[k] - point).dot(normal)
    return pos


def joint_pairs(src_arm, tgt_arm):
    """Matching (source, target) world positions of every humanoid bone head found on both rigs."""
    src = joint_positions(src_arm, armature_map(src_arm))
    tgt = joint_positions(tgt_arm, armature_map(tgt_arm))
    return [(src[k], tgt[k]) for k in src if k in tgt]


@contextmanager
def rest_pose(context, arms):
    """Temporarily clear the pose of `arms` (e.g. to see clothing as originally designed)."""
    saved = [(pb, pb.matrix_basis.copy()) for a in arms for pb in a.pose.bones]
    for a in arms:
        reset_pose(a)
    context.view_layer.update()
    try:
        yield
    finally:
        for pb, m in saved:
            pb.matrix_basis = m
        context.view_layer.update()


def foot_bone_names(arm, mapping, side):
    key = f"foot.{side}"
    if key not in mapping:
        return set()
    bone = arm.data.bones[mapping[key]]
    return {bone.name} | {c.name for c in bone.children_recursive}


def foot_frames(context, body, arm):
    """Per side: (ankle, toe tip, floor height, foot width) in world space, from the rig and the body mesh."""
    mapping = armature_map(arm)
    verts, _tris, _edges = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())
    if len(verts) != len(body.data.vertices):
        return {}
    frames = {}
    for side in "LR":
        region = region_vertices(body, foot_bone_names(arm, mapping, side))
        if len(region) < 10:
            continue
        ankle = np.array(posed_head(arm, mapping[f"foot.{side}"])[:])
        pts = verts[region]
        tip = pts[np.argmax(np.linalg.norm(pts - ankle, axis=1))]
        forward = np.array([tip[0] - ankle[0], tip[1] - ankle[1], 0.0])
        across = np.cross([0.0, 0.0, 1.0], forward / max(np.linalg.norm(forward), 1e-9))
        width = float(np.ptp(pts @ across))
        frames[side] = (ankle, tip, float(pts[:, 2].min()), width)
    return frames


def region_vertices(body, bone_names):
    """Indices of vertices whose strongest weight belongs to one of bone_names."""
    group_names = {vg.index: vg.name for vg in body.vertex_groups}
    ids = []
    for v in body.data.vertices:
        best, best_w = None, 0.0
        for g in v.groups:
            if g.weight > best_w:
                best, best_w = group_names.get(g.group), g.weight
        if best in bone_names:
            ids.append(v.index)
    return np.array(ids, dtype=np.int64)


def _has_fingers(mapping, side):
    return any(f"{f}_1.{side}" in mapping for f in FINGER_NAMES)


def extremity_tips(context, body, arm, mapping):
    """Mesh vertex at the tip of each foot (and each hand without finger bones).

    The tip is the vertex skinned to that bone (or its children) farthest from the joint.
    Returns {humanoid key: vertex index}; the index stays valid as the rig is posed.
    """
    keys = [f"foot.{s}" for s in "LR"]
    keys += [f"hand.{s}" for s in "LR" if not _has_fingers(mapping, s)]
    keys = [k for k in keys if k in mapping]
    if not keys:
        return {}
    verts, _tris, _edges = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())
    if len(verts) != len(body.data.vertices):
        return {}
    tips = {}
    for key in keys:
        bone = arm.data.bones[mapping[key]]
        names = {bone.name} | {c.name for c in bone.children_recursive}
        region = region_vertices(body, names)
        if len(region) < 10:
            continue
        head = np.array(posed_head(arm, bone.name)[:])
        tips[key] = int(region[np.argmax(np.linalg.norm(verts[region] - head, axis=1))])
    return tips


def tip_position(context, body, index):
    verts, _tris, _edges = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())
    return Vector(verts[index])


def extremity_pairs(context, src_body, src_arm, tgt_body, tgt_arm):
    """Matching (source, target) world positions of foot tips (and hand tips without finger bones)."""
    src_map, tgt_map = armature_map(src_arm), armature_map(tgt_arm)
    src_tips = extremity_tips(context, src_body, src_arm, src_map)
    tgt_tips = extremity_tips(context, tgt_body, tgt_arm, tgt_map)
    return [(tip_position(context, src_body, si), tip_position(context, tgt_body, tgt_tips[key]))
            for key, si in src_tips.items() if key in tgt_tips]


def landmark_collection(scene, create=False):
    coll = bpy.data.collections.get(LANDMARK_COLLECTION)
    if coll is None and create:
        coll = bpy.data.collections.new(LANDMARK_COLLECTION)
        scene.collection.children.link(coll)
    return coll


def landmark_objects():
    coll = bpy.data.collections.get(LANDMARK_COLLECTION)
    if coll is None:
        return []
    return [o for o in coll.all_objects if "ah_lm_id" in o and "ah_lm_role" in o]


def landmark_pairs():
    by_id = {}
    for o in landmark_objects():
        by_id.setdefault(o["ah_lm_id"], {})[o["ah_lm_role"]] = o.matrix_world.translation.copy()
    return [(v["src"], v["tgt"]) for _k, v in sorted(by_id.items()) if "src" in v and "tgt" in v]
