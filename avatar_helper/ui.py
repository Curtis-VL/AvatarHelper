import bpy

from . import rig
from .calibration import CALIBRATION_KEY


class AH_UL_clothing(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"{index + 1}.")
        row.prop(item, "obj", text="")
        row.prop(item, "mode", text="")


class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AvatarHelper"


class AH_PT_main(_Panel, bpy.types.Panel):
    bl_idname = "AH_PT_main"
    bl_label = "AvatarHelper"

    def draw(self, context):
        s = context.scene.avatar_helper
        col = self.layout.column()
        col.label(text="Source (clothing made for):")
        col.prop(s, "source_body", text="Body")
        col.prop(s, "source_armature", text="Armature")
        col.separator()
        col.label(text="Target (fit to):")
        col.prop(s, "target_body", text="Body")
        col.prop(s, "target_armature", text="Armature")
        col.separator()
        col.operator("avatar_helper.check_bones", icon='BONE_DATA')


class AH_PT_pose(_Panel, bpy.types.Panel):
    bl_label = "1. Pose"
    bl_parent_id = "AH_PT_main"

    def draw(self, context):
        row = self.layout.row(align=True)
        row.operator("avatar_helper.match_pose", icon='ARMATURE_DATA')
        row.operator("avatar_helper.reset_pose", text="", icon='LOOP_BACK')
        self.layout.label(text="Keep this pose while calibrating and fitting", icon='INFO')


class AH_PT_calibrate(_Panel, bpy.types.Panel):
    bl_label = "2. Calibrate"
    bl_parent_id = "AH_PT_main"

    def draw(self, context):
        s = context.scene.avatar_helper
        layout = self.layout

        box = layout.box()
        n = len(rig.landmark_pairs())
        box.label(text=f"Landmarks: {n} pair(s) (optional)")
        row = box.row(align=True)
        row.operator("avatar_helper.add_landmark", text="Add Pair", icon='EMPTY_AXIS')
        row.operator("avatar_helper.clear_landmarks", text="", icon='TRASH')

        col = layout.column(align=True)
        col.prop(s, "icp_iterations")
        col.prop(s, "icp_smoothing")
        col.prop(s, "search_distance")
        col.prop(s, "normal_threshold")
        layout.operator("avatar_helper.calibrate", icon='MOD_MESHDEFORM')

        layout.prop(s, "calibration")
        cal = s.calibration
        keys = cal.data.shape_keys if cal else None
        if keys and CALIBRATION_KEY in keys.key_blocks:
            layout.prop(keys.key_blocks[CALIBRATION_KEY], "value", text="Preview Source → Target", slider=True)
            layout.label(text="Colour by attribute 'AH_Error' to see the error map", icon='INFO')


class AH_PT_fit(_Panel, bpy.types.Panel):
    bl_label = "3. Fit Clothing"
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
        col.prop(s, "replace_previous")
        col.prop(s, "hide_originals")
        layout.operator("avatar_helper.fit", icon='MOD_CLOTH')


classes = (AH_UL_clothing, AH_PT_main, AH_PT_pose, AH_PT_calibrate, AH_PT_fit)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
