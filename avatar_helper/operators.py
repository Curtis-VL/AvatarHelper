import bmesh
import bpy
import numpy as np

from . import calibration, conversions, finalize, fitting, layers, rig, touchup, updater
from .bone_map import missing_required
from .props import is_result

HIGH_ERROR = 0.003  # metres; a good calibration is well under 1 mm on average


def _ensure_object_mode(context):
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


class _Progress:
    def __init__(self, context):
        self.wm = context.window_manager

    def __enter__(self):
        self.wm.progress_begin(0, 100)
        return self

    def __call__(self, fraction):
        self.wm.progress_update(int(fraction * 100))

    def __exit__(self, *_):
        self.wm.progress_end()


class AH_OT_check_bones(bpy.types.Operator):
    bl_idname = "avatar_helper.check_bones"
    bl_label = "Check Bone Mapping"
    bl_description = "Report which humanoid bones were detected on both armatures"

    def execute(self, context):
        s = context.scene.avatar_helper
        found = False
        for label, arm in (("Source", calibration.source_armature(s)), ("Target", calibration.target_armature(s))):
            if not arm:
                continue
            found = True
            m = rig.armature_map(arm)
            missing = missing_required(m)
            fingers = sum(1 for k in m if "_" in k and k.split("_")[0] in ("thumb", "index", "middle", "ring", "little"))
            msg = f"{label} '{arm.name}': {len(m)} bones mapped ({fingers} finger)"
            if missing:
                self.report({'WARNING'}, msg + ", missing: " + ", ".join(missing))
            else:
                self.report({'INFO'}, msg)
        if not found:
            self.report({'ERROR'}, "Set the armatures (or bodies with armature modifiers) first")
            return {'CANCELLED'}
        return {'FINISHED'}


def _conversion_items(_self, _context):
    # Blender needs the strings kept alive while the menu is open.
    _conversion_items.cache = [(stem, name, "") for stem, name in conversions.list_library()] or \
        [("", "(no conversions saved yet)", "")]
    return _conversion_items.cache


class AH_OT_choose_conversion(bpy.types.Operator):
    bl_idname = "avatar_helper.choose_conversion"
    bl_label = "Choose Conversion"
    bl_description = "Pick a saved conversion (source base > target avatar) to fit clothing with"
    bl_property = "conversion"

    conversion: bpy.props.EnumProperty(items=_conversion_items)

    def invoke(self, context, _event):
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if self.conversion:
            context.scene.avatar_helper.conversion = self.conversion
        return {'FINISHED'}


class AH_OT_open_conversions(bpy.types.Operator):
    bl_idname = "avatar_helper.open_conversions"
    bl_label = "Open Conversions Folder"
    bl_description = "Open the folder conversions are saved in (to share, back up or delete them)"

    def execute(self, _context):
        bpy.ops.wm.path_open(filepath=conversions.library_dir())
        return {'FINISHED'}


class AH_OT_create_conversion(bpy.types.Operator):
    bl_idname = "avatar_helper.create_conversion"
    bl_label = "Create Conversion"
    bl_description = ("Match the source rig's pose to the target, learn how the source body maps onto the target "
                      "body, and save it as a conversion for fitting any outfit made for this base")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        _ensure_object_mode(context)
        try:
            with _Progress(context) as progress:
                stats = calibration.create_conversion(context, s, progress)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except OSError as e:
            self.report({'ERROR'}, f"Couldn't save the conversion: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, "Saved conversion '{conversion}' ({joints} joints, {tips} tips, {landmarks} landmarks). "
                              "Surface error: mean {m:.1f} mm, 95% {p:.1f} mm, max {x:.1f} mm".format(
                                  m=stats["mean_error"] * 1000, p=stats["p95_error"] * 1000,
                                  x=stats["max_error"] * 1000, **stats))
        if stats["mean_error"] > HIGH_ERROR:
            self.report({'WARNING'}, "The bodies didn't match up well. Check both bodies are the right meshes "
                                     "(Check Bone Mapping, and Show Calibration for the error map)")
        return {'FINISHED'}


class AH_OT_save_conversion(bpy.types.Operator):
    bl_idname = "avatar_helper.save_conversion"
    bl_label = "Save Calibration Edits"
    bl_description = ("Save the calibration preview mesh again, including hand edits to its 'Fitted' shape key, "
                      "as the conversion")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cal = context.scene.avatar_helper.calibration
        return cal is not None and "ah_conversion" in cal

    def execute(self, context):
        s = context.scene.avatar_helper
        try:
            name = calibration.save_conversion(s, s.calibration)
        except OSError as e:
            self.report({'ERROR'}, f"Couldn't save the conversion: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Saved conversion '{name}'")
        return {'FINISHED'}


class AH_OT_add_landmark(bpy.types.Operator):
    bl_idname = "avatar_helper.add_landmark"
    bl_label = "Add Landmark Pair"
    bl_description = ("Add a source/target marker pair at the 3D cursor. Place the sphere on the source body and "
                      "the cube on the same feature of the target body")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _ensure_object_mode(context)
        coll = rig.landmark_collection(context.scene, create=True)
        ids = [o["ah_lm_id"] for o in rig.landmark_objects()]
        lid = max(ids, default=0) + 1
        cursor = context.scene.cursor.location
        created = []
        for role, shape in (("src", 'SPHERE'), ("tgt", 'CUBE')):
            e = bpy.data.objects.new(f"LM_{lid:02d}_{role}", None)
            e.empty_display_type = shape
            e.empty_display_size = 0.01
            e.location = cursor
            e.show_in_front = True
            e["ah_lm_id"] = lid
            e["ah_lm_role"] = role
            coll.objects.link(e)
            created.append(e)
        for o in context.selected_objects:
            o.select_set(False)
        created[0].select_set(True)
        context.view_layer.objects.active = created[0]
        return {'FINISHED'}


class AH_OT_clear_landmarks(bpy.types.Operator):
    bl_idname = "avatar_helper.clear_landmarks"
    bl_label = "Clear Landmarks"
    bl_description = "Delete all landmark markers"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        for o in rig.landmark_objects():
            bpy.data.objects.remove(o, do_unlink=True)
        return {'FINISHED'}


class AH_OT_update_check(bpy.types.Operator):
    bl_idname = "avatar_helper.update_check"
    bl_label = "Check for Updates"
    bl_description = "Look up the latest AvatarHelper release on GitHub"

    def execute(self, _context):
        updater.check()
        return {'FINISHED'}


class AH_OT_update_install(bpy.types.Operator):
    bl_idname = "avatar_helper.update_install"
    bl_label = "Update"
    bl_description = "Download and install the latest release (takes effect after restarting Blender)"

    @classmethod
    def poll(cls, _context):
        return updater.state["status"] == "available"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, _context):
        try:
            updater.install()
        except Exception as e:
            self.report({'ERROR'}, f"Update failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Installed AvatarHelper {updater.version_text(updater.state['latest'])}: "
                              "restart Blender to finish")
        return {'FINISHED'}


class AH_OT_report_bug(bpy.types.Operator):
    bl_idname = "avatar_helper.report_bug"
    bl_label = "Report a Bug"
    bl_description = ("Open AvatarHelper's issue page on GitHub, and copy your AvatarHelper, Blender and system "
                      "versions to the clipboard to paste into the report")

    def execute(self, context):
        context.window_manager.clipboard = updater.system_info()
        bpy.ops.wm.url_open(url=updater.ISSUES_PAGE)
        self.report({'INFO'}, "Versions copied to the clipboard: paste them into your bug report")
        return {'FINISHED'}


class AH_OT_finalize(bpy.types.Operator):
    bl_idname = "avatar_helper.finalize"
    bl_label = "Finalize"
    bl_description = ("Keep the fitted clothing in the list: it takes the originals' names and place (they're "
                      "renamed '(original)' and hidden), later fits won't replace it, and the list is cleared for "
                      "the next outfit or part")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        _ensure_object_mode(context)
        try:
            done, added = finalize.finalize(context, s, s.rig_to)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        msg = "Finalized " + ", ".join(done)
        if added:
            msg += f"; added {len(added)} outfit bone(s) to the avatar's armature"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class AH_OT_restore_shape(bpy.types.Operator):
    bl_idname = "avatar_helper.restore_shape"
    bl_label = "Restore Original Shape"
    bl_description = ("Rebuild the selected part of a fitted garment from the original garment's shape, removing "
                      "bumps and wobble (e.g. along a hem) while the rest of the garment stays put")
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(
        name="Shape Size", default=0.05, min=0.005, soft_max=0.3, subtype='DISTANCE', unit='LENGTH',
        description="The selection keeps the original's shape up to this size; only larger stretching and "
                    "turning comes from the fit. Raise it if bumps remain, lower it if the part no longer "
                    "follows the body")
    falloff: bpy.props.FloatProperty(
        name="Falloff", default=0.01, min=0.0, soft_max=0.1, subtype='DISTANCE', unit='LENGTH',
        description="Blend into the unselected part over this distance, so there's no visible seam")
    keep_outside: bpy.props.BoolProperty(
        name="Keep Outside Body", default=True,
        description="Points that were outside the avatar's body (set in the panel) stay outside it")

    @classmethod
    def poll(cls, context):
        o = context.edit_object
        return o is not None and o.type == 'MESH' and ("ah_fitted_from" in o or "ah_final_from" in o)

    def execute(self, context):
        obj = context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        coords = np.array([v.co[:] for v in bm.verts])
        selected = np.array([v.select for v in bm.verts])
        if not selected.any():
            self.report({'ERROR'}, "Select the vertices to restore first")
            return {'CANCELLED'}
        tris = np.array([[l.vert.index for l in t] for t in bm.calc_loop_triangles()], dtype=np.int64).reshape(-1, 3)
        edges = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges], dtype=np.int64).reshape(-1, 2)
        s = context.scene.avatar_helper
        body = s.target_body if self.keep_outside else None
        try:
            new = touchup.restore_shape(context, obj, coords, selected, tris, edges, self.scale, self.falloff,
                                        body, s.collision_offset)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        for v, co in zip(bm.verts, new.tolist()):
            v.co = co
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}


def _unique_clothing(s):
    objs = []
    for it in s.clothing:
        if it.obj and it.obj not in objs and not is_result(it.obj):
            objs.append(it.obj)
    return objs


class AH_OT_sort_layers(bpy.types.Operator):
    bl_idname = "avatar_helper.sort_layers"
    bl_label = "Auto-Sort Layers"
    bl_description = ("Order the clothing list innermost-first from how the items overlap on the source avatar, "
                      "and remove duplicate entries")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        objs = _unique_clothing(s)
        if len(objs) < 2:
            return {'CANCELLED'}
        _ensure_object_mode(context)
        rel, gaps = layers.source_layers(context, objs, s.source_body)
        order = layers.sort_order([o.name for o in objs], rel, gaps)
        keep = ("mode", "solver", "custom_detail", "shape_detail")
        settings = {}
        for it in s.clothing:
            if it.obj and it.obj.name not in settings:
                settings[it.obj.name] = {k: getattr(it, k) for k in keep}
        s.clothing.clear()
        for name in order:
            item = s.clothing.add()
            item.obj = bpy.data.objects[name]
            for k, v in settings.get(name, {}).items():
                setattr(item, k, v)
        s.clothing_index = 0
        self.report({'INFO'}, "Layers: " + " < ".join(order))
        return {'FINISHED'}


class AH_OT_fit(bpy.types.Operator):
    bl_idname = "avatar_helper.fit"
    bl_label = "Fit Clothing"
    bl_description = "Create fitted copies of the clothing (and their armatures) in the target's proportions"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        _ensure_object_mode(context)
        warnings = []
        names = [it.obj.name for it in s.clothing if it.obj]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            warnings.append("listed twice (first entry used): " + ", ".join(dupes))
        objs = _unique_clothing(s)
        if len(objs) > 1:
            rel, _gaps = layers.source_layers(context, objs, s.source_body)
            wrong = layers.conflicts([o.name for o in objs], rel)
            if wrong:
                warnings.append("order differs from the source (" +
                                ", ".join(f"'{a}' is under '{b}'" for a, b in wrong[:3]) +
                                "); use Auto-Sort Layers")
        notes = []
        try:
            with _Progress(context) as progress:
                meshes, arms, shoes = fitting.fit_clothing(context, s, progress, notes)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        msg = f"Fitted {len(meshes)} mesh(es) and {len(arms)} armature(s) into '{fitting.FITTED_COLLECTION}'"
        if shoes:
            msg += "; placed as shoes: " + ", ".join(shoes)
        self.report({'INFO'}, msg)
        for w in warnings:
            self.report({'WARNING'}, "Layers " + w)
        for n in notes:
            self.report({'WARNING'}, n)
        return {'FINISHED'}


class AH_OT_clothing_add(bpy.types.Operator):
    bl_idname = "avatar_helper.clothing_add"
    bl_label = "Add Selected"
    bl_description = "Add the selected meshes to the clothing list (first item = innermost layer)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        present = {it.obj for it in s.clothing if it.obj}
        skip = {s.source_body, s.target_body, s.calibration}
        added = 0
        for o in context.selected_objects:
            if o.type == 'MESH' and o not in present and o not in skip and not is_result(o):
                s.clothing.add().obj = o
                added += 1
        if not added:
            s.clothing.add()
        s.clothing_index = len(s.clothing) - 1
        return {'FINISHED'}


class AH_OT_clothing_remove(bpy.types.Operator):
    bl_idname = "avatar_helper.clothing_remove"
    bl_label = "Remove"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        if 0 <= s.clothing_index < len(s.clothing):
            s.clothing.remove(s.clothing_index)
            s.clothing_index = min(s.clothing_index, len(s.clothing) - 1)
        return {'FINISHED'}


class AH_OT_clothing_move(bpy.types.Operator):
    bl_idname = "avatar_helper.clothing_move"
    bl_label = "Move"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(items=(('UP', "Up", ""), ('DOWN', "Down", "")))

    def execute(self, context):
        s = context.scene.avatar_helper
        i = s.clothing_index
        j = i - 1 if self.direction == 'UP' else i + 1
        if 0 <= i < len(s.clothing) and 0 <= j < len(s.clothing):
            s.clothing.move(i, j)
            s.clothing_index = j
        return {'FINISHED'}


classes = (
    AH_OT_check_bones,
    AH_OT_choose_conversion,
    AH_OT_open_conversions,
    AH_OT_create_conversion,
    AH_OT_save_conversion,
    AH_OT_add_landmark,
    AH_OT_clear_landmarks,
    AH_OT_sort_layers,
    AH_OT_fit,
    AH_OT_update_check,
    AH_OT_update_install,
    AH_OT_report_bug,
    AH_OT_finalize,
    AH_OT_restore_shape,
    AH_OT_clothing_add,
    AH_OT_clothing_remove,
    AH_OT_clothing_move,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
