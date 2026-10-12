import bpy

from . import calibration, fitting, layers, rig
from .bone_map import missing_required


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


def _follower_armatures(s, src_arm):
    arms = []
    objs = [it.obj for it in s.clothing if it.obj] + ([s.source_body] if s.source_body else [])
    for o in objs:
        for a in rig.deforming_armatures(o):
            if a is not src_arm and a not in arms:
                arms.append(a)
    return arms


class AH_OT_match_pose(bpy.types.Operator):
    bl_idname = "avatar_helper.match_pose"
    bl_label = "Match Pose"
    bl_description = ("Pose the source armature (and the clothing's armatures) so its limbs point the same way "
                      "as the target's, e.g. A-pose to T-pose")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        src, tgt = calibration.source_armature(s), calibration.target_armature(s)
        if not src or not tgt:
            self.report({'ERROR'}, "Source and target armatures are required")
            return {'CANCELLED'}
        _ensure_object_mode(context)
        followers = _follower_armatures(s, src)
        n = rig.match_pose(context, src, tgt, followers, s.source_body, s.target_body)
        self.report({'INFO'}, f"Rotated {n} bones; {len(followers)} clothing armature(s) followed")
        return {'FINISHED'}


class AH_OT_reset_pose(bpy.types.Operator):
    bl_idname = "avatar_helper.reset_pose"
    bl_label = "Reset Pose"
    bl_description = "Clear the pose on the source armature and the clothing's armatures"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        src = calibration.source_armature(s)
        arms = ([src] if src else []) + _follower_armatures(s, src)
        _ensure_object_mode(context)
        for a in arms:
            rig.reset_pose(a)
        context.view_layer.update()
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


class AH_OT_calibrate(bpy.types.Operator):
    bl_idname = "avatar_helper.calibrate"
    bl_label = "Calibrate"
    bl_description = "Learn how the source body maps onto the target body"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.avatar_helper
        _ensure_object_mode(context)
        try:
            with _Progress(context) as progress:
                stats = calibration.calibrate(context, s, progress)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Calibrated from {joints} joints, {tips} tips, {landmarks} landmarks. Surface error: "
                              "mean {m:.1f} mm, 95% {p:.1f} mm, max {x:.1f} mm".format(
                                  m=stats["mean_error"] * 1000, p=stats["p95_error"] * 1000,
                                  x=stats["max_error"] * 1000, **stats))
        return {'FINISHED'}


def _unique_clothing(s):
    objs = []
    for it in s.clothing:
        if it.obj and it.obj not in objs and not it.obj.get("ah_fitted_from"):
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
        modes = {it.obj.name: it.mode for it in s.clothing if it.obj}
        s.clothing.clear()
        for name in order:
            item = s.clothing.add()
            item.obj = bpy.data.objects[name]
            item.mode = modes.get(name, 'AUTO')
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
        try:
            with _Progress(context) as progress:
                meshes, arms, shoes = fitting.fit_clothing(context, s, progress)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        msg = f"Fitted {len(meshes)} mesh(es) and {len(arms)} armature(s) into '{fitting.FITTED_COLLECTION}'"
        if shoes:
            msg += "; placed as shoes: " + ", ".join(shoes)
        self.report({'INFO'}, msg)
        for w in warnings:
            self.report({'WARNING'}, "Layers " + w)
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
            if o.type == 'MESH' and o not in present and o not in skip and not o.get("ah_fitted_from"):
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
    AH_OT_match_pose,
    AH_OT_reset_pose,
    AH_OT_add_landmark,
    AH_OT_clear_landmarks,
    AH_OT_calibrate,
    AH_OT_sort_layers,
    AH_OT_fit,
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
