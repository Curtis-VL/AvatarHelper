"""Touch-up tools for fitted clothing.

Restore Original Shape: the selected part of a fitted garment is rebuilt from the original
garment's shape. Only how it was stretched and turned on a larger scale is kept from the
fit, so bumps, wobble and crumpled edges go, while the rest of the garment stays put and
holds the restored part in place. Much quicker than straightening a hem with proportional
editing, and it ends up looking like the original because it is the original's shape.
"""
import os

import bpy
import numpy as np

from . import calibration, conversions, elastic, fitting
from . import mesh_utils as mu


def original_of(obj):
    """The garment a fitted copy was made from, with the same vertices."""
    name = obj.get("ah_fitted_from") or obj.get("ah_final_from")
    src = bpy.data.objects.get(name) if name else None
    if src is None or src.type != 'MESH':
        raise ValueError(f"'{obj.name}' isn't a fitted copy of an existing mesh")
    if len(src.data.vertices) != len(obj.data.vertices):
        raise ValueError(f"'{obj.name}' no longer has the same vertices as '{src.name}'")
    return src


def original_shape(obj):
    """The original garment in the shape being edited: Edit Mode edits the active shape key
    (e.g. a "crop" key that folds a long shirt up), so compare with the original's same key."""
    src = original_of(obj)
    key = obj.active_shape_key
    src_keys = src.data.shape_keys
    if key is not None and src_keys is not None and key.name in src_keys.key_blocks:
        data = src_keys.key_blocks[key.name].data
    else:
        data = src.data.vertices
    co = np.empty(len(data) * 3)
    data.foreach_get("co", co)
    return co.reshape(-1, 3)


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


def restore_shape(context, obj, coords, selected, tris, edges, scale, falloff, body=None, min_gap=0.002,
                  leave_hidden=True):
    """New local positions for the fitted mesh `obj` (currently at `coords`), with the
    `selected` vertices rebuilt from the original garment. If `body` (a mesh object) is
    given, the result is kept outside it, by each point's gap up to `min_gap`, except for the
    parts the fit kept inside the body by design (a tucked-in hem), which stay as fitted
    unless `leave_hidden` is off."""
    rest = original_shape(obj)
    if len(rest) != len(coords):
        raise ValueError(f"'{obj.name}' has {len(coords)} vertices, its original {len(rest)}: "
                         "the shape can only be restored on the mesh as fitted")
    weight = falloff_weights(selected, edges, rest, falloff)
    push = None
    if body is not None:
        push, hidden = _keep_outside(context, obj, coords, body, min_gap, leave_hidden)
        weight[hidden] = 0.0
    return elastic.restore_shape(rest, coords, tris, edges, weight, scale, push)


def _body_surface(context, body):
    """The avatar's body to keep clothing out of: only its body parts, not hair or accessories
    merged into the mesh, when the scene's conversion is there to tell them apart."""
    verts, tris, edges = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())
    s = context.scene.avatar_helper
    path = conversions.path_for(s.conversion) if s.conversion else ""
    arm = calibration.target_armature(s) if body == s.target_body else None
    if path and os.path.exists(path) and arm is not None:
        conv = conversions.Conversion.load(path)
        fitted = mu.transform_points(conv.target_alignment(arm), conv.fitted)
        keep = fitting.body_parts(verts, edges, mu.build_bvh(fitted, conv.tris))
        tris = tris[keep[tris].all(1)]
    return mu.build_bvh(verts, tris)


def _keep_outside(context, obj, coords, body, min_gap, leave_hidden=True, depth=fitting.BODY_DEPTH):
    """push(x) keeping the garment outside the body, and which points to leave alone: the
    parts the fit kept inside the body by design. Points that are inside the body for any
    other reason (squeezed in by the fit) are restored and pushed back out to Min Gap.
    For meshes fitted before this was recorded, every point inside the body is left alone."""
    mw = np.array(obj.matrix_world)
    inv = np.linalg.inv(mw[:3, :3])
    bvh = _body_surface(context, body)

    def side(points):
        loc, nrm, idx, _dist = mu.nearest_on(bvh, points, depth)
        return np.where(idx >= 0, ((points - loc) * nrm).sum(1), np.inf), loc, nrm

    before, _loc, _nrm = side(mu.transform_points(mw, coords))
    marks = fitting.hidden_marks(obj)
    hidden = marks if marks is not None else before < 0.0
    if not leave_hidden:
        hidden = np.zeros(len(coords), dtype=bool)
    clear = np.where(before >= 0.0, np.clip(before, 0.0, min_gap), min_gap)  # own gap, up to Min Gap

    def push(x):
        xw = mu.transform_points(mw, x)
        now, loc, nrm = side(xw)
        need = ~hidden & np.isfinite(now) & (now < clear)
        out = np.zeros_like(x)
        out[need] = (loc[need] + nrm[need] * clear[need, None] - xw[need]) @ inv.T
        return out

    return push, hidden
