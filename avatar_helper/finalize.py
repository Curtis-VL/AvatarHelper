"""Finalize: keep fitted results.

The fitted copies take over the originals' names (Unity animations and toggles refer to
objects by name) and their place in the scene's collections; the originals are renamed
"<name> (original)" and moved to a hidden collection, still there to fit again. Finalized
results are no longer touched by later fits, so you can go on to the next outfit, or the
next part of this one.

Optionally the clothing is rigged to the avatar's own armature (see weights.py) instead of
the fitted outfit armature.
"""
import os

import bpy
import numpy as np

from . import conversions, weights
from . import mesh_utils as mu
from .calibration import output_collection, target_armature
from .fitting import FITTED_COLLECTION, body_parts

ORIGINALS_COLLECTION = "AH_Originals"
ORIGINAL_SUFFIX = " (original)"


def fitted_copy(obj):
    return next((o for o in bpy.data.objects if o.get("ah_fitted_from") == obj.name), None)


def base_name(name):
    return name[:-len(ORIGINAL_SUFFIX)] if name.endswith(ORIGINAL_SUFFIX) else name


def finalize(context, s, rig_to='OUTFIT'):
    """Finalize the fitted copies of the clothing in the list. Returns (finalized names,
    bones added to the avatar's armature)."""
    pairs = [(it.obj, fitted_copy(it.obj)) for it in s.clothing if it.obj]
    pairs = [(o, f) for o, f in pairs if f is not None]
    if not pairs:
        raise ValueError("Nothing to finalize: fit the clothing in the list first")
    meshes = [f for _o, f in pairs]
    arms = []
    for f in meshes:
        for m in f.modifiers:
            if m.type == 'ARMATURE' and m.object and m.object.get("ah_fitted_from") and m.object not in arms:
                arms.append(m.object)

    added = []
    if rig_to == 'AVATAR':
        added = _rig_to_avatar(context, s, meshes)
    else:
        arms = [_finalize_armature(a, meshes, s.conversion) for a in arms]

    originals = output_collection(context.scene, ORIGINALS_COLLECTION)
    _hide_collection(context, originals)
    done = []
    for o, f in pairs:
        _take_over(o, f, originals)
        done.append(f.name)
    for a in arms:
        if a is not None and a.get("ah_fitted_from"):
            orig = bpy.data.objects.get(a["ah_fitted_from"])
            if orig is not None:
                _take_over(orig, a, originals)
                orig[_final_key(s.conversion)] = a.name
    # Fitted outfit armatures nothing uses any more (rigged to the avatar instead).
    for a in list(bpy.data.objects):
        if a.type == 'ARMATURE' and a.get("ah_fitted_from") and not _users(a):
            data = a.data
            bpy.data.objects.remove(a, do_unlink=True)
            if data.users == 0:
                bpy.data.armatures.remove(data)

    finalized = {o for o, _f in pairs}
    for i in reversed(range(len(s.clothing))):
        if s.clothing[i].obj in finalized:
            s.clothing.remove(i)
    s.clothing_index = min(s.clothing_index, len(s.clothing) - 1)
    return done, added


def _final_key(conversion):
    return "ah_final_as:" + conversion


def _users(arm):
    return [o for o in bpy.data.objects if o.parent == arm
            or any(m.type == 'ARMATURE' and m.object == arm for m in getattr(o, "modifiers", []))]


def _finalize_armature(arm, meshes, conversion):
    """The fitted armature to keep for these meshes. If this outfit's armature was finalized
    before for the same avatar (another part of the outfit), that one is reused: it's the
    same fit, and two copies would merge onto the avatar twice."""
    orig = bpy.data.objects.get(arm["ah_fitted_from"])
    earlier = bpy.data.objects.get(orig.get(_final_key(conversion), "")) if orig is not None else None
    if earlier is None or earlier.type != 'ARMATURE' or "ah_final_from" not in earlier:
        return arm
    for f in meshes:
        for m in f.modifiers:
            if m.type == 'ARMATURE' and m.object == arm:
                m.object = earlier
        if f.parent == arm:
            mw = f.matrix_world.copy()
            f.parent = earlier
            f.matrix_world = mw
    return None  # removed below once unused


def _take_over(orig, new, originals):
    """`new` takes `orig`'s name and place; `orig` is renamed and put away."""
    name = base_name(orig.name)
    data_name = base_name(orig.data.name) if orig.data else None
    orig.name = name + ORIGINAL_SUFFIX
    new.name = name
    if data_name is not None and new.data is not None and new.data.users == 1:
        if orig.data.users == 1:
            orig.data.name = data_name + ORIGINAL_SUFFIX
        new.data.name = data_name
    new["ah_final_from"] = orig.name
    if "ah_fitted_from" in new:
        del new["ah_fitted_from"]
    for c in [c for c in orig.users_collection if c.name != ORIGINALS_COLLECTION]:
        if new.name not in c.objects:
            c.objects.link(new)
        c.objects.unlink(orig)
    fitted = bpy.data.collections.get(FITTED_COLLECTION)
    if fitted is not None and new.name in fitted.objects and len(new.users_collection) > 1:
        fitted.objects.unlink(new)
    if orig.name not in originals.objects:
        originals.objects.link(orig)
    if not orig.users_collection:
        originals.objects.link(orig)
    new.hide_set(False)


def _hide_collection(context, coll):
    """Hidden, not excluded: other parts of an outfit may still be fitted from the original
    armature, which has to stay evaluated."""
    def find(layer):
        if layer.collection == coll:
            return layer
        for child in layer.children:
            found = find(child)
            if found:
                return found
        return None
    layer = find(context.view_layer.layer_collection)
    if layer is not None:
        layer.hide_viewport = True


def _rig_to_avatar(context, s, meshes):
    arm = target_armature(s)
    if not (s.target_body and arm):
        raise ValueError("Set the target body (with its armature)")
    body = s.target_body
    _v, _t, edges = mu.mesh_arrays(body.data)
    verts = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())[0]
    mask = np.ones(len(verts), dtype=bool)
    # Only the avatar's body, not hair or accessories merged into its mesh (see body_parts).
    path = conversions.path_for(s.conversion) if s.conversion else ""
    if path and os.path.exists(path) and len(verts) == len(body.data.vertices):
        conv = conversions.Conversion.load(path)
        fitted = mu.transform_points(conv.target_alignment(arm), conv.fitted)
        mask = body_parts(verts, edges, mu.build_bvh(fitted, conv.tris))
    return weights.rig_to_avatar(context, meshes, arm, body, mask, s.skin_distance, s.blend_distance)
