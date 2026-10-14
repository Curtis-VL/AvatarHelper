"""Calibration: learn where every point of the source body ends up on the target body.

The source rig is first posed to match the target's limb directions, then the source body
is matched onto the target body. The result is saved as a conversion (see conversions.py)
for reuse without the source avatar, and kept in the scene as a hidden preview mesh: a copy
of the posed source body with a "Fitted" shape key holding the matching target positions.
Sliding that key from 0 to 1 morphs the source body onto the target, and the key can be
sculpted by hand to fix problem areas, then saved again.
"""
import json

import bpy
import numpy as np

from . import conversions
from . import mesh_utils as mu
from . import rig
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


def body_armature(body, chosen=None):
    """The armature that deforms `body`. An armature set by hand is only used if it is one of
    them, or if the body has no Armature modifier: posing any other rig (e.g. a clothing
    armature) wouldn't move the body, and the calibration would come out wrong."""
    arms = deforming_armatures(body) if body else []
    if chosen in arms or (chosen and not arms):
        return chosen
    return arms[0] if arms else None


def source_armature(s):
    return body_armature(s.source_body, s.source_armature)


def target_armature(s):
    return body_armature(s.target_body, s.target_armature)


def shape_key_values(obj):
    """{shape key name: value} of a mesh (Basis left out)."""
    keys = obj.data.shape_keys if obj and obj.type == 'MESH' else None
    if keys is None:
        return {}
    return {kb.name: float(kb.value) for kb in keys.key_blocks if kb != keys.reference_key}


def conversion_name(s):
    return s.conversion_name.strip() or f"{s.source_body.name} > {s.target_body.name}"


def create_conversion(context, s, progress=None):
    """Match the pose, calibrate, and save the result as a conversion. The source rig is
    left as it was."""
    src_body, tgt_body = s.source_body, s.target_body
    if not src_body or not tgt_body:
        raise ValueError("Set both the source and target body meshes")
    src_arm, tgt_arm = source_armature(s), target_armature(s)
    for body, arm in ((src_body, src_arm), (tgt_body, tgt_arm)):
        if arm is None and any(m.type == 'ARMATURE' and m.object is None for m in body.modifiers):
            raise ValueError(f"'{body.name}' has an Armature modifier with no armature set in it")
    if not src_arm or not tgt_arm:
        raise ValueError("Both bodies need an armature (set it, or give the body an Armature modifier)")

    posed_arms = {src_arm, *deforming_armatures(src_body)}
    with rig.rest_pose(context, posed_arms):
        src_joints = conversions.rest_joints(src_arm)
        src_feet = rig.foot_frames(context, src_body, src_arm)
        rig.match_pose(context, src_arm, tgt_arm, [a for a in posed_arms if a is not src_arm], src_body, tgt_body)
        pose = conversions.capture_pose(src_arm)
        stats = calibrate(context, s, progress)
    meta = {
        "name": conversion_name(s), "source": src_body.name, "target": tgt_body.name,
        "src_joints": {k: v.tolist() for k, v in src_joints.items()},
        "tgt_joints": {k: v.tolist() for k, v in conversions.posed_joints(tgt_arm).items()},
        "pose": {k: v.tolist() for k, v in pose.items()},
        "src_feet": {k: [np.asarray(x).tolist() for x in v] for k, v in src_feet.items()},
        # Where the armatures stood: the conversion is saved relative to them.
        "src_matrix": [list(r) for r in src_arm.matrix_world],
        "tgt_matrix": [list(r) for r in tgt_arm.matrix_world],
        "src_keys": shape_key_values(src_body), "tgt_keys": shape_key_values(tgt_body),
    }
    cal = s.calibration
    cal["ah_conversion"] = json.dumps(meta)
    stats["conversion"] = save_conversion(s, cal)
    return stats


def save_conversion(s, cal):
    """Save the calibration object (with any hand edits to its "Fitted" key) as a conversion."""
    meta = json.loads(cal["ah_conversion"])
    if s.conversion_name.strip():
        meta["name"] = s.conversion_name.strip()
    keys = cal.data.shape_keys
    n = len(cal.data.vertices)
    base = mu.transform_points(cal.matrix_world, mu.read_shape_key(keys.reference_key, n))
    fitted = mu.transform_points(cal.matrix_world, mu.read_shape_key(keys.key_blocks[CALIBRATION_KEY], n))
    _co, tris, _edges = mu.mesh_arrays(cal.data)
    conv = conversions.Conversion.from_meta(meta, base, fitted, tris)
    if "src_matrix" in meta:
        conv.to_armature_frame(np.array(meta["src_matrix"]), np.array(meta["tgt_matrix"]))
    stem = conversions.file_stem(conv.name)
    conv.save(conversions.path_for(stem))
    s.conversion = stem
    return conv.name


BODY_NEAR = 0.005  # a part of the target mesh counts as body once most of it is this close


def _body_parts(tgt_verts, labels, dist):
    """Target vertices on parts of the mesh lying mostly on the matched source body (the body
    itself, not hair or accessories merged into the mesh)."""
    near = np.bincount(labels, weights=(dist < BODY_NEAR).astype(float)) / np.bincount(labels)
    return near[labels] > 0.5


def _reverse_pull(warped, tris, normals, tgt_verts, tgt_normals, tgt_labels, s):
    """Each target body point pulls the nearest point of the warped source body towards it,
    shared among that triangle's corners. Returns per source vertex (sum of pulls, weight)."""
    num = np.zeros_like(warped)
    den = np.zeros(len(warped))
    loc, nrm, idx, dist = mu.nearest_on(mu.build_bvh(warped, tris), tgt_verts, s.search_distance)
    body = _body_parts(tgt_verts, tgt_labels, np.where(idx >= 0, dist, np.inf))
    ok = body & (idx >= 0) & ((tgt_normals * nrm).sum(1) > s.normal_threshold)
    if not ok.any():
        return num, den
    t = tris[idx[ok]]
    bary = mu.barycentric(loc[ok], warped[t[:, 0]], warped[t[:, 1]], warped[t[:, 2]])
    pull = tgt_verts[ok] - loc[ok]
    for c in range(3):
        for k in range(3):
            num[:, k] += np.bincount(t[:, c], weights=bary[:, c] * pull[:, k], minlength=len(warped))
        den += np.bincount(t[:, c], weights=bary[:, c], minlength=len(warped))
    return num, den


def _uncovered(warped, tris, tgt_verts, tgt_labels, gap=0.005):
    """Share of the target body that the matched source body doesn't reach (further than
    `gap`): a big unmatched shape shows up here, not in the source-to-target error."""
    _loc, _nrm, idx, dist = mu.nearest_on(mu.build_bvh(warped, tris), tgt_verts, 0.2)
    dist = np.where(idx >= 0, dist, np.inf)
    body = _body_parts(tgt_verts, tgt_labels, dist)
    return float((dist[body] > gap).mean()) if body.any() else 0.0


def calibrate(context, s, progress=None):
    src_body, tgt_body = s.source_body, s.target_body
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
    # Matching goes both ways: each source point is pulled to the nearest target point, and
    # each target point pulls the nearest source point to it. One way only, a much larger
    # target shape (bigger breasts) is never reached: the source points all find closer spots
    # around its base, and its front is left uncovered.
    bvh = mu.build_bvh(tgt_verts, tgt_tris)
    tgt_normals = mu.vertex_normals(tgt_verts, tgt_tris)
    tgt_edges = mu.evaluated_mesh_world(tgt_body, dg)[2]
    tgt_labels = mu.islands(len(tgt_verts), tgt_edges)
    adj = mu.Adjacency(len(base), edges)
    iters = max(1, s.icp_iterations)
    stiff_start = max(float(s.icp_smoothing), 1.0)
    for it in range(iters):
        t = it / max(1, iters - 1)
        stiffness = stiff_start ** (1.0 - t)  # geometric: stiff_start -> 1
        normals = mu.vertex_normals(warped, tris)
        loc, nrm, idx, _dist = mu.nearest_on(bvh, warped, s.search_distance)
        valid = (idx >= 0) & ((normals * nrm).sum(1) > s.normal_threshold)
        num = (loc - warped) * valid[:, None]
        den = valid.astype(np.float64)
        rnum, rden = _reverse_pull(warped, tris, normals, tgt_verts, tgt_normals, tgt_labels, s)
        num += rnum
        den += rden
        residual = num / np.maximum(den, 1e-9)[:, None]
        warped += adj.solve_membrane(residual, np.minimum(den, 1.0), stiffness)
        if progress:
            progress((it + 1) / iters)

    _loc, _nrm, idx, dist = mu.nearest_on(bvh, warped, 1.0)
    dist = np.where(idx >= 0, dist, 1.0)
    uncovered = _uncovered(warped, tris, tgt_verts, tgt_labels)

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
    # Only for checking and hand-fixing the calibration; otherwise it just overlaps the target.
    cal.hide_set(not s.show_calibration)
    cal.hide_render = True
    return {
        "joints": len(pairs) - len(landmarks) - len(tips),
        "tips": len(tips),
        "landmarks": len(landmarks),
        "mean_error": float(dist.mean()),
        "p95_error": float(np.percentile(dist, 95)),
        "max_error": float(dist.max()),
        "uncovered": uncovered,
    }
