"""Calibration: learn where every point of the source body ends up on the target body.

The result is stored as a mesh object (a copy of the posed source body) with a
"Fitted" shape key holding the matching target positions. Sliding that key from 0 to 1
morphs the source body onto the target, which is how the user checks the calibration,
and the key can be sculpted by hand to fix problem areas.
"""
import bpy
import numpy as np

from . import mesh_utils as mu
from .rbf import RBFWarp, dedupe_pairs
from .rig import deforming_armatures, extremity_pairs, joint_pairs, landmark_pairs

OUTPUT_COLLECTION = "AvatarHelper"
CALIBRATION_KEY = "Fitted"
ERROR_ATTRIBUTE = "AH_Error"
ERROR_DISPLAY_MAX = 0.01  # metres shown as full red in the error map


def output_collection(scene, name=OUTPUT_COLLECTION):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll not in scene.collection.children_recursive:
        scene.collection.children.link(coll)
    return coll


def source_armature(s):
    if s.source_armature:
        return s.source_armature
    arms = deforming_armatures(s.source_body) if s.source_body else []
    return arms[0] if arms else None


def target_armature(s):
    if s.target_armature:
        return s.target_armature
    arms = deforming_armatures(s.target_body) if s.target_body else []
    return arms[0] if arms else None


def calibrate(context, s, progress=None):
    src_body, tgt_body = s.source_body, s.target_body
    if not src_body or not tgt_body:
        raise ValueError("Set both the source and target body meshes")

    dg = context.evaluated_depsgraph_get()

    old = s.calibration
    if old and old.get("ah_calibration"):
        old_mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    me = bpy.data.meshes.new_from_object(src_body.evaluated_get(dg), preserve_all_data_layers=False, depsgraph=dg)
    me.transform(src_body.matrix_world)
    me.name = "AH_Calibration"
    cal = bpy.data.objects.new("AH_Calibration", me)
    output_collection(context.scene).objects.link(cal)
    cal["ah_calibration"] = 1
    cal["ah_source"] = src_body.name
    cal["ah_target"] = tgt_body.name

    base, tris, edges = mu.mesh_arrays(me)
    tgt_verts, tgt_tris, _ = mu.evaluated_mesh_world(tgt_body, dg)

    # Stage 1: coarse warp from matching joints and user landmarks.
    pairs = []
    src_arm, tgt_arm = source_armature(s), target_armature(s)
    tips = []
    if src_arm and tgt_arm:
        pairs += joint_pairs(src_arm, tgt_arm)
        tips = extremity_pairs(context, src_body, src_arm, tgt_body, tgt_arm)
        pairs += tips
    landmarks = landmark_pairs()
    pairs += landmarks
    warped = base.copy()
    if pairs:
        src_pts, dst_pts = dedupe_pairs([p[0][:] for p in pairs], [p[1][:] for p in pairs])
        warped += RBFWarp(src_pts, dst_pts - src_pts, smoothing=1e-6)(base)

    # Stage 2: non-rigid ICP. Pull each vertex towards the target surface through a
    # membrane that is stiff at first (overall shape) and loose at the end (local detail).
    # Unmatched areas (e.g. toes pointing a different way) follow their matched neighbours.
    bvh = mu.build_bvh(tgt_verts, tgt_tris)
    adj = mu.Adjacency(len(base), edges)
    iters = max(1, s.icp_iterations)
    stiff_start = max(float(s.icp_smoothing), 1.0)
    for it in range(iters):
        t = it / max(1, iters - 1)
        stiffness = stiff_start ** (1.0 - t)  # geometric: stiff_start -> 1
        normals = mu.vertex_normals(warped, tris)
        loc, nrm, idx, _dist = mu.nearest_on(bvh, warped, s.search_distance)
        valid = (idx >= 0) & ((normals * nrm).sum(1) > s.normal_threshold)
        residual = (loc - warped) * valid[:, None]
        warped += adj.solve_membrane(residual, valid.astype(np.float64), stiffness)
        if progress:
            progress((it + 1) / iters)

    _loc, _nrm, idx, dist = mu.nearest_on(bvh, warped, 1.0)
    dist = np.where(idx >= 0, dist, 1.0)

    cal.shape_key_add(name="Basis", from_mix=False)
    key = cal.shape_key_add(name=CALIBRATION_KEY, from_mix=False)
    mu.write_shape_key(key, warped)
    key.value = 1.0

    heat = np.clip(dist / ERROR_DISPLAY_MAX, 0.0, 1.0)
    colors = np.stack([heat, 1.0 - heat, np.zeros_like(heat), np.ones_like(heat)], axis=1)
    attr = me.color_attributes.new(ERROR_ATTRIBUTE, 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", colors.astype(np.float32).ravel())
    try:
        me.color_attributes.active_color = attr
    except (AttributeError, TypeError):
        pass

    s.calibration = cal
    return {
        "joints": len(pairs) - len(landmarks) - len(tips),
        "tips": len(tips),
        "landmarks": len(landmarks),
        "mean_error": float(dist.mean()),
        "p95_error": float(np.percentile(dist, 95)),
        "max_error": float(dist.max()),
    }
