"""Touch-up tools for fitted clothing.

Restore Original Shape: the selected part of a fitted garment is rebuilt from the original
garment's shape. Only how it was stretched and turned on a larger scale is kept from the
fit, so bumps, wobble and crumpled edges go, while the rest of the garment stays put and
holds the restored part in place. Much quicker than straightening a hem with proportional
editing, and it ends up looking like the original because it is the original's shape.
"""
import bpy
import numpy as np

from . import elastic
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


def restore_shape(context, obj, coords, selected, tris, edges, scale, falloff, body=None, min_gap=0.002):
    """New local positions for the fitted mesh `obj` (currently at `coords`), with the
    `selected` vertices rebuilt from the original garment. If `body` (a mesh object) is
    given, points that were outside it stay outside, by up to `min_gap`."""
    rest = original_shape(obj)
    weight = falloff_weights(selected, edges, rest, falloff)
    push = None
    if body is not None:
        push, inside = _keep_outside(context, obj, coords, body, min_gap)
        # Parts tucked inside the body (hidden by design) stay as fitted.
        weight[inside] = 0.0
    return elastic.restore_shape(rest, coords, tris, edges, weight, scale, push)


def _keep_outside(context, obj, coords, body, min_gap, depth=0.02):
    """push(x) keeping points that were outside the body outside it, and which points are
    inside it now."""
    mw = np.array(obj.matrix_world)
    inv = np.linalg.inv(mw[:3, :3])
    bv, bt, _ = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())
    bvh = mu.build_bvh(bv, bt)

    def side(points):
        loc, nrm, idx, _dist = mu.nearest_on(bvh, points, depth)
        return np.where(idx >= 0, ((points - loc) * nrm).sum(1), np.inf), loc, nrm

    before, _loc, _nrm = side(mu.transform_points(mw, coords))
    clear = np.clip(before, 0.0, min_gap)  # each point's own gap, up to Min Gap
    was_outside = before >= 0.0  # points already inside (tucked in on purpose) are left alone

    def push(x):
        xw = mu.transform_points(mw, x)
        now, loc, nrm = side(xw)
        need = was_outside & np.isfinite(now) & (now < clear)
        out = np.zeros_like(x)
        out[need] = (loc[need] + nrm[need] * clear[need, None] - xw[need]) @ inv.T
        return out

    return push, ~was_outside
