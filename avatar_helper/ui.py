import os

import bpy

from . import calibration, conversions, rig, updater
from .calibration import CALIBRATION_KEY

_names = {}  # conversion file stem -> display name, read once per file


def _conversion_label(stem):
    if not stem:
        return None
    path = conversions.path_for(stem)
    if not os.path.exists(path):
        return None
    if stem not in _names:
        _names[stem] = conversions.read_name(path) or stem
    return _names[stem]


def _body_rows(layout, s, body_prop, arm_prop):
    """Body field, plus the armature found from it. The armature field only shows when the
    body has no Armature modifier to find it from."""
    layout.prop(s, body_prop, text="Body")
    body = getattr(s, body_prop)
    if body and rig.deforming_armatures(body):
        arm = calibration.body_armature(body, getattr(s, arm_prop))
        layout.label(text=f"Armature: {arm.name}  (from the body)", icon='ARMATURE_DATA')
    else:
        layout.prop(s, arm_prop, text="Armature")


class AH_UL_clothing(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"{index + 1}.")
        row.prop(item, "obj", text="")
        row.prop(item, "mode", text="")
        row.prop(item, "solver", text="")
        if item.custom_detail:
            row.label(text=f"{item.shape_detail * 100:.3g} cm")


class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AvatarHelper"


class AH_PT_main(_Panel, bpy.types.Panel):
    bl_idname = "AH_PT_main"
    bl_label = "AvatarHelper"

    def draw(self, context):
        s = context.scene.avatar_helper
        layout = self.layout
        updater.draw(layout, compact=True)

        box = layout.box()
        box.label(text="Conversion (base > your avatar):")
        row = box.row(align=True)
        label = _conversion_label(s.conversion)
        row.operator("avatar_helper.choose_conversion", text=label or "Choose...",
                     icon='CHECKMARK' if label else 'ERROR')
        row.operator("avatar_helper.open_conversions", text="", icon='FILE_FOLDER')
        if not label:
            box.label(text="None yet? Use Create Conversion below", icon='INFO')

        col = layout.column()
        col.label(text="Your avatar (fit to):")
        _body_rows(col, s, "target_body", "target_armature")


class AH_PT_create(_Panel, bpy.types.Panel):
    bl_idname = "AH_PT_create"
    bl_label = "Create Conversion"
    bl_parent_id = "AH_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.avatar_helper
        layout = self.layout
        layout.label(text="Once per base + avatar pair:", icon='INFO')
        col = layout.column()
        col.label(text="Base avatar the clothing was made for:")
        _body_rows(col, s, "source_body", "source_armature")
        col.label(text="Clothing armatures are matched automatically when fitting", icon='INFO')
        col.separator()
        col.prop(s, "conversion_name")
        row = layout.row(align=True)
        row.operator("avatar_helper.create_conversion", icon='MOD_MESHDEFORM')
        row.operator("avatar_helper.check_bones", text="", icon='BONE_DATA')

        cal = s.calibration
        keys = cal.data.shape_keys if cal else None
        if keys and CALIBRATION_KEY in keys.key_blocks:
            layout.prop(s, "show_calibration")
            if s.show_calibration:
                layout.prop(keys.key_blocks[CALIBRATION_KEY], "value", text="Preview Base → Avatar", slider=True)
                layout.label(text="Colour by attribute 'AH_Error' for the error map", icon='INFO')


class AH_PT_create_advanced(_Panel, bpy.types.Panel):
    bl_label = "Advanced"
    bl_parent_id = "AH_PT_create"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.avatar_helper
        layout = self.layout

        box = layout.box()
        box.label(text=f"Landmarks: {len(rig.landmark_pairs())} pair(s)")
        row = box.row(align=True)
        row.operator("avatar_helper.add_landmark", text="Add Pair", icon='EMPTY_AXIS')
        row.operator("avatar_helper.clear_landmarks", text="", icon='TRASH')

        col = layout.column(align=True)
        col.label(text="Calibration")
        col.prop(s, "icp_iterations")
        col.prop(s, "icp_smoothing")
        col.prop(s, "search_distance")
        col.prop(s, "normal_threshold")
        layout.operator("avatar_helper.save_conversion", icon='FILE_TICK')


class AH_PT_fit(_Panel, bpy.types.Panel):
    bl_idname = "AH_PT_fit"
    bl_label = "Fit Clothing"
    bl_parent_id = "AH_PT_main"

    def draw(self, context):
        s = context.scene.avatar_helper
        layout = self.layout
        layout.label(text="Layers (innermost first):")
        row = layout.row()
        row.template_list("AH_UL_clothing", "", s, "clothing", s, "clothing_index", rows=4)
        side = row.column(align=True)
        side.operator("avatar_helper.clothing_add", text="", icon='ADD')
        side.operator("avatar_helper.clothing_remove", text="", icon='REMOVE')
        side.separator()
        side.operator("avatar_helper.clothing_move", text="", icon='TRIA_UP').direction = 'UP'
        side.operator("avatar_helper.clothing_move", text="", icon='TRIA_DOWN').direction = 'DOWN'
        layout.operator("avatar_helper.sort_layers", icon='SORTSIZE')

        if 0 <= s.clothing_index < len(s.clothing) and s.clothing[s.clothing_index].obj:
            item = s.clothing[s.clothing_index]
            box = layout.box()
            box.label(text=item.obj.name, icon='MESH_DATA')
            col = box.column(align=True)
            col.prop(item, "mode")
            col.prop(item, "solver")
            elastic = (s.solver if item.solver == 'DEFAULT' else item.solver) == 'ELASTIC'
            row = box.row(align=True)
            row.active = elastic
            row.prop(item, "custom_detail", text="Own Shape Detail")
            sub = row.row(align=True)
            sub.active = elastic and item.custom_detail
            sub.prop(item, "shape_detail", text="")

        col = layout.column(align=True)
        col.label(text="Defaults:")
        col.prop(s, "solver", text="Solver")
        col.prop(s, "shape_detail")
        layout.operator("avatar_helper.fit", icon='MOD_CLOTH')
        layout.operator("avatar_helper.fit_body_keys", icon='SHAPEKEY_DATA')

        box = layout.box()
        box.label(text="Happy with it?")
        box.prop(s, "rig_to")
        box.operator("avatar_helper.finalize", icon='CHECKMARK')


class AH_PT_touchup(_Panel, bpy.types.Panel):
    bl_label = "Touch Up"
    bl_parent_id = "AH_PT_main"

    def draw(self, context):
        layout = self.layout
        o = context.edit_object or context.object
        if not (o and ("ah_fitted_from" in o or "ah_final_from" in o)):
            layout.label(text="Select a fitted mesh, then Edit Mode", icon='INFO')
            return
        if context.mode != 'EDIT_MESH':
            layout.label(text="Enter Edit Mode and select bumpy parts", icon='INFO')
            return
        layout.label(text="Select the bumpy part (e.g. a hem):", icon='INFO')
        layout.operator("avatar_helper.restore_shape", icon='MOD_SMOOTH')


class AH_PT_help(_Panel, bpy.types.Panel):
    """Last in the sidebar, below Touch Up."""
    bl_label = "Help"
    bl_parent_id = "AH_PT_main"
    bl_options = {'HIDE_HEADER'}

    def draw(self, _context):
        self.layout.operator("avatar_helper.report_bug", icon='HELP')


class AH_PT_fit_advanced(_Panel, bpy.types.Panel):
    bl_label = "Advanced"
    bl_parent_id = "AH_PT_fit"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.avatar_helper
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Shape")
        col.prop(s, "skin_distance")
        col.prop(s, "blend_distance")
        col.prop(s, "field_resolution")
        col.prop(s, "field_smoothing")
        col.prop(s, "displacement_smoothing")
        col.prop(s, "rigid_size")
        col = layout.column(align=True)
        col.label(text="Collision")
        col.prop(s, "collision_offset")
        col.prop(s, "collision_iterations")
        col.prop(s, "collide_layers")
        col = layout.column(align=True)
        col.label(text="Output")
        col.prop(s, "replace_previous")
        col.prop(s, "hide_originals")


classes = (AH_UL_clothing, AH_PT_main, AH_PT_create, AH_PT_create_advanced, AH_PT_fit, AH_PT_fit_advanced,
           AH_PT_touchup, AH_PT_help)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
