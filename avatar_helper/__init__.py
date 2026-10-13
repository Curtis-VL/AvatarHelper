bl_info = {
    "name": "AvatarHelper",
    "author": "AvatarHelper",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > AvatarHelper",
    "description": "Refit clothing made for one avatar base onto another",
    "category": "Rigging",
}

if "bpy" in locals():
    # F3 > Reload Scripts: reload in dependency order.
    import importlib
    for _m in (bone_map, mesh_utils, rbf, rig, conversions, calibration, updater, props, elastic, fitting, touchup, weights, finalize, layers, operators, ui):  # noqa: F821
        importlib.reload(_m)
else:
    from . import bone_map, mesh_utils, rbf, rig, conversions, calibration, updater, props, elastic, fitting, touchup, weights, finalize, layers, operators, ui

import bpy  # noqa: E402


def register():
    props.register()
    operators.register()
    ui.register()
    updater.register()


def unregister():
    updater.unregister()
    ui.unregister()
    operators.unregister()
    props.unregister()
