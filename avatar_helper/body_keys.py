"""Body shape keys: give clothing the avatar's body shape keys (bigger chest, thicker waist).

For each of the avatar's body shape keys that moves the skin near the clothing, the
clothing is fitted from the body as it is to the body with that key at 1, the same way Fit
Clothing fits from the base to the avatar (shape kept, kept outside the body and lower
layers), and the change is stored as a shape key of the same name. In Unity the clothing's
key is then driven along with the body's.

No calibration is needed: a shape key moves the body's own vertices, so where each part of
the body goes is known exactly. The solver also runs once with the body unchanged, and that
result is what each key is measured from, so a key holds only its own change and none of
the solver's (e.g. a strap pushed off skin it was already dipping into).

Avatars also have keys that hide the body under clothing (shrinking or sinking parts of it,
often listed under a "Body_Delete" or "Shrink" heading): clothing mustn't follow those. They
are spotted by name and by what they do, and offered unticked. Keys the clothing already has
from its creator are kept unless replacing them is asked for.
"""
import json
import re

import bpy
import numpy as np

from . import mesh_utils as mu
from .fitting import CREASE_WIDTH, FitField, Layer, SmoothSide, evaluated_shape, fit_points

OURS = "ah_body_keys"  # on clothing: JSON list of the keys made here (replaced when made again)
MOVE_MIN = 0.0002  # body vertices moving less than this don't count as moved by a key
REACH = 0.05  # clothing farther than this from every moved body vertex isn't affected by a key
KEY_MIN = 0.0002  # clothing moving less than this everywhere doesn't get the key

# Spotting keys that hide the body. By name: the key's own, or the heading it's listed under
# (a key like "===== Body_Delete ====="). By what it does: it sinks the skin it moves deep
# inside the rest of the body, or collapses it.
HIDE_WORDS = ("delete", "shrink", "hide", "hidden", "cull", "clip", "remove")
HEADING = re.compile(r"[=\-*#~]{3,}")
HIDE_MOVE = 0.001  # body vertices moving more than this count for what a key does
HIDE_DEPTH = 0.015  # sinking the skin it moves this far in (most of it)...
HIDE_INSIDE = 0.9  # ...to inside the rest of the body (this share of it)
HIDE_DEEP = 0.05  # or this far in: deeper than clothing could follow
HIDE_AREA = 0.3  # or leaving it less than this share of its area


def _named_hiding(name):
    name = name.lower()
    return any(w in name for w in HIDE_WORDS)


def is_hiding(b0, b1, tris):
    """The key hides the skin it moves rather than reshaping it (b0, b1: the body without and
    with it)."""
    d = b1 - b0
    moved = np.linalg.norm(d, axis=1) > HIDE_MOVE
    if moved.sum() < 3:
        return False
    t = tris[moved[tris].any(1)]
    before = _areas(b0, t).sum()
    if before > 0.0 and _areas(b1, t).sum() / before < HIDE_AREA:
        return True
    inward = -(d[moved] * mu.vertex_normals(b0, tris)[moved]).sum(1)
    depth = np.percentile(inward, 90)
    if depth > HIDE_DEEP:
        return True
    rest = tris[~moved[tris].any(1)]
    if depth < HIDE_DEPTH or not len(rest):
        return False
    loc, nrm, idx, _dist = mu.nearest_on(mu.build_bvh(b0, rest), b1[moved], 4 * HIDE_DEEP)
    inside = (idx >= 0) & (((b1[moved] - loc) * nrm).sum(1) < -HIDE_MOVE)
    return inside.mean() > HIDE_INSIDE


def _eval_body(context, body):
    return mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())[0]


def _with_value(context, body, kb, value):
    old, old_max = kb.value, kb.slider_max
    kb.slider_max = max(old_max, value)
    kb.value = value
    context.view_layer.update()
    try:
        return _eval_body(context, body)
    finally:
        kb.value = old
        kb.slider_max = old_max
        context.view_layer.update()


def _areas(verts, tris):
    return 0.5 * np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]),
                                axis=1)


def _clothing_bvhs(context, objs):
    dg = context.evaluated_depsgraph_get()
    return [mu.build_bvh(*mu.evaluated_mesh_world(o, dg)[:2]) for o in objs]


def _near(bvhs, points):
    """Indices of the clothing within REACH of any of `points`."""
    if not len(points):
        return set()
    return {i for i, bvh in enumerate(bvhs) if (mu.nearest_on(bvh, points, REACH)[2] >= 0).any()}


def candidates(context, body, objs):
    """[(key name, looks like it hides the body)] for the body's shape keys that move the skin
    near any of `objs`, in the body's order. Read from the keys' own data (quick): the body
    as it is now places them, when it has the mesh's vertices."""
    keys = body.data.shape_keys if body and body.type == 'MESH' else None
    if keys is None or not objs:
        return []
    me = body.data
    n = len(me.vertices)
    mw = body.matrix_world
    _co, tris, _edges = mu.mesh_arrays(me)
    now = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())[0]
    data = {kb.name: mu.transform_points(mw, mu.read_shape_key(kb, n)) for kb in keys.key_blocks}
    if len(now) != n:
        now = data[keys.reference_key.name]
    bvhs = _clothing_bvhs(context, objs)
    out, heading_hides = [], False
    for kb in keys.key_blocks:
        if kb == keys.reference_key:
            continue
        b0, b1 = data[kb.relative_key.name], data[kb.name]
        moved = np.linalg.norm(b1 - b0, axis=1) > MOVE_MIN
        if HEADING.search(kb.name) and moved.sum() == 0:
            heading_hides = _named_hiding(kb.name)
            continue
        if kb.mute or not _near(bvhs, now[moved]):
            continue
        out.append((kb.name, heading_hides or _named_hiding(kb.name) or is_hiding(b0, b1, tris)))
    return out


def _chain(items, shapes, base, fitted, tris, s, which, ref):
    """Fit the items in `which` (innermost first) from body `base` to `fitted`; the others
    keep `ref` and only serve as lower layers."""
    field = FitField(base, fitted, tris, s.field_resolution, s.field_smoothing, crease_width=CREASE_WIDTH)
    normals = mu.vertex_normals(fitted, tris)
    body = (fitted, normals, mu.build_bvh(fitted, tris), SmoothSide(fitted, tris, normals))
    layers, out = [], []
    for i, (o, solver, detail) in enumerate(items):
        verts, otris, oedges = shapes[i]
        if ref is None or i in which:
            x = fit_points(o, verts, otris, oedges, field, s, solver, detail, body,
                           layers if s.collide_layers else ())
        else:
            x = ref[i]
        layers.append(Layer(verts, x, otris))
        out.append(x)
    return out


PROBE = 0.01  # how far each vertex is moved to measure how the pose carries it


def to_local(context, obj):
    """A function turning a world-space change of the mesh as displayed (posed) into the
    change of its own vertices that causes it. Skinning is linear in such a change, so moving
    every vertex a little along each axis (with a temporary shape key) measures it exactly."""
    inv = np.array(obj.matrix_world.inverted().to_3x3(), dtype=np.float64)
    if not any(m.type == 'ARMATURE' and m.show_viewport for m in obj.modifiers):
        return lambda d: d @ inv.T
    me = obj.data
    if me.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    n = len(me.vertices)
    ref = mu.read_shape_key(me.shape_keys.reference_key, n)
    probe = obj.shape_key_add(name="AH_Probe", from_mix=False)
    probe.relative_key = me.shape_keys.reference_key
    cols = []
    try:
        probe.value = 1.0
        mu.write_shape_key(probe, ref)
        context.view_layer.update()
        before = mu.evaluated_mesh_world(obj, context.evaluated_depsgraph_get())[0]
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = PROBE
            mu.write_shape_key(probe, ref + step)
            me.update()
            context.view_layer.update()
            cols.append((mu.evaluated_mesh_world(obj, context.evaluated_depsgraph_get())[0] - before) / PROBE)
    finally:
        obj.shape_key_remove(probe)
        context.view_layer.update()
    # Per vertex: world change = M @ own change.
    m = np.linalg.pinv(np.stack(cols, axis=2))
    return lambda d: np.einsum("nij,nj->ni", m, d)


def _ours(obj):
    try:
        return set(json.loads(obj.get(OURS, "[]")))
    except ValueError:
        return set()


def _write_key(obj, name, delta, value, body_kb):
    """Store `delta` (local) as shape key `name`, at `value`, without changing how the mesh
    looks now: the clothing was fitted with the body's key at `value`, so its Basis takes
    that much of the key off."""
    me = obj.data
    if me.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    keys = me.shape_keys
    n = len(me.vertices)
    ref = mu.read_shape_key(keys.reference_key, n)
    kb = keys.key_blocks.get(name)
    # A key being replaced goes into the Basis as much as it shows now, so nothing moves.
    shift = -delta * value
    if kb is not None:
        shift += (mu.read_shape_key(kb, n) - ref) * kb.value
    else:
        kb = obj.shape_key_add(name=name, from_mix=False)
    if shift.any():
        for other in keys.key_blocks:
            if other != kb:
                mu.write_shape_key(other, mu.read_shape_key(other, n) + shift)
        co = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        me.vertices.foreach_set("co", (co.reshape(-1, 3) + shift).astype(np.float32).ravel())
    mu.write_shape_key(kb, mu.read_shape_key(keys.reference_key, n) + delta)
    kb.relative_key = keys.reference_key
    kb.slider_min, kb.slider_max = body_kb.slider_min, body_kb.slider_max
    kb.value = value
    me.update()


def _has_own(obj, name):
    """The clothing has this key from its creator (not one made here)."""
    own = obj.data.shape_keys
    return own is not None and name in own.key_blocks and name not in _ours(obj)


def fit_body_keys(context, body, items, names, s, replace=False, progress=None):
    """Give the clothing the body's shape keys `names`. items: [(clothing object, solver,
    shape detail)], innermost first, as they are on the body now.

    Returns {"made": {object name: [keys]}, "kept": [(object name, key)]}."""
    keys = body.data.shape_keys
    report = {"made": {}, "kept": []}
    blocks = [keys.key_blocks[n] for n in names if n in keys.key_blocks] if keys else []
    if not blocks or not items:
        return report
    b_now, btris, _e = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())
    shapes = []
    for o, _solver, _detail in items:
        verts, tris, edges, _unposed = evaluated_shape(context, o)
        shapes.append((verts, tris, edges))
    bvhs = [mu.build_bvh(v, t) for v, t, _e in shapes]
    ref = None
    local = {}
    for k, kb in enumerate(blocks):
        if progress:
            progress(k / len(blocks))
        b1 = _with_value(context, body, kb, 1.0)
        b0 = _with_value(context, body, kb, 0.0) if kb.value != 0.0 else b_now
        which = _near(bvhs, b_now[np.linalg.norm(b1 - b0, axis=1) > MOVE_MIN])
        kept = {i for i in which if not replace and _has_own(items[i][0], kb.name)}
        report["kept"] += [(items[i][0].name, kb.name) for i in sorted(kept)]
        which -= kept
        if not which:
            continue
        if ref is None:
            ref = _chain(items, shapes, b_now, b_now, btris, s, None, None)
        x0 = ref if kb.value == 0.0 else _chain(items, shapes, b_now, b0, btris, s, which, ref)
        x1 = _chain(items, shapes, b_now, b1, btris, s, which, ref)
        for i in sorted(which):
            o = items[i][0]
            delta = x1[i] - x0[i]
            if np.linalg.norm(delta, axis=1).max() < KEY_MIN:
                continue
            if o not in local:
                local[o] = to_local(context, o)
            _write_key(o, kb.name, local[o](delta), kb.value, kb)
            o[OURS] = json.dumps(sorted(_ours(o) | {kb.name}))
            report["made"].setdefault(o.name, []).append(kb.name)
    if progress:
        progress(1.0)
    return report
