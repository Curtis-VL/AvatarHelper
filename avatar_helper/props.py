import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty,
                       StringProperty)

from . import conversions, updater


def _is_mesh(_self, obj):
    return obj.type == 'MESH'


def _is_armature(_self, obj):
    return obj.type == 'ARMATURE'


SOLVERS = (
    ('ELASTIC', "Shape-preserving",
     "Solve for the whole garment at once, keeping its own shape: no bumps or crumpling, "
     "slightly softer fit to body detail"),
    ('DIRECT', "Direct", "Place each vertex on its own: follows the body most closely, but can come out bumpy"),
)


def is_result(obj):
    """Made by the add-on: a fitted or finalized copy, or the calibration preview."""
    return any(k in obj for k in ("ah_fitted_from", "ah_final_from", "ah_calibration"))


def _is_clothing(_self, obj):
    s = bpy.context.scene.avatar_helper
    return obj.type == 'MESH' and not is_result(obj) and obj not in (s.source_body, s.target_body)


class AH_ClothingItem(bpy.types.PropertyGroup):
    obj: PointerProperty(name="Object", type=bpy.types.Object, poll=_is_clothing)
    mode: EnumProperty(
        name="Mode",
        items=(
            ('AUTO', "Auto", "Shoes if the item is skinned almost entirely to the feet, otherwise Fit"),
            ('FIT', "Fit", "Reshape to the target body"),
            ('SHOES', "Shoes", "Keep the original shape; place on the target's feet at floor level"),
        ),
        default='AUTO',
    )
    solver: EnumProperty(
        name="Solver",
        items=(('DEFAULT', "Default", "Use the solver chosen below the list"),) + SOLVERS,
        default='DEFAULT',
    )
    custom_detail: BoolProperty(
        name="Own Shape Detail",
        description="Use this item's own Shape Detail instead of the one below the list (e.g. higher for a knit, "
                    "lower for a skin-tight body suit)")
    shape_detail: FloatProperty(
        name="Shape Detail", default=0.05, min=0.002, soft_max=0.1, subtype='DISTANCE', unit='LENGTH',
        description="Shape-preserving solver: detail smaller than this comes from the original garment, "
                    "larger shape changes follow the target")


def _show_calibration(self, _context):
    if self.calibration and self.calibration.name in bpy.context.view_layer.objects:
        self.calibration.hide_set(not self.show_calibration)


class AH_Settings(bpy.types.PropertyGroup):
    # Conversion
    conversion: StringProperty(
        name="Conversion", description="The saved conversion (source base > target avatar) used for fitting")
    conversion_name: StringProperty(
        name="Name", description='Name for a new conversion, e.g. "Manuka > Custom". '
                                 'Defaults to "<source body> > <target body>"')
    source_armature: PointerProperty(
        name="Source Armature", type=bpy.types.Object, poll=_is_armature,
        description="Only needed if the source body has no Armature modifier: the base avatar's own armature "
                    "(not a clothing armature)")
    source_body: PointerProperty(
        name="Source Body", type=bpy.types.Object, poll=_is_mesh,
        description="Body mesh of the base avatar the clothing was made for, used only to create a conversion")
    target_armature: PointerProperty(
        name="Target Armature", type=bpy.types.Object, poll=_is_armature,
        description="Only needed if the target body has no Armature modifier: your avatar's armature")
    target_body: PointerProperty(
        name="Target Body", type=bpy.types.Object, poll=_is_mesh,
        description="Body mesh of the avatar to fit to")

    # Calibration
    calibration: PointerProperty(
        name="Calibration", type=bpy.types.Object, poll=_is_mesh,
        description="Calibration preview mesh (source body with a 'Fitted' shape key)")
    show_calibration: BoolProperty(
        name="Show Calibration", default=False, update=_show_calibration,
        description="Show the calibration preview mesh (hidden by default, as it overlaps the target body)")
    icp_iterations: IntProperty(
        name="Iterations", default=12, min=1, max=100,
        description="Surface-matching passes after the joint-based warp")
    icp_smoothing: IntProperty(
        name="Stiffness", default=25, min=0, max=500,
        description="Smoothing of early passes. Higher keeps the overall shape, lower matches detail sooner")
    search_distance: FloatProperty(
        name="Search Distance", default=0.06, min=0.001, soft_max=0.3, subtype='DISTANCE', unit='LENGTH',
        description="How far a source vertex may look for the target surface")
    normal_threshold: FloatProperty(
        name="Normal Agreement", default=0.5, min=-1.0, max=1.0,
        description="Ignore matches whose surface directions differ more than this (cosine)")

    # Clothing
    clothing: CollectionProperty(type=AH_ClothingItem)
    clothing_index: IntProperty()
    field_resolution: FloatProperty(
        name="Field Resolution", default=0.03, min=0.002, soft_max=0.1, subtype='DISTANCE', unit='LENGTH',
        description="Sample spacing of the body warp used for loose clothing")
    field_smoothing: FloatProperty(
        name="Field Smoothing", default=0.001, min=0.0, soft_max=0.05, precision=4,
        description="Higher values give a softer warp for loose clothing")
    skin_distance: FloatProperty(
        name="Skin Follow", default=0.01, min=0.0, soft_max=0.05, subtype='DISTANCE', unit='LENGTH',
        description="Clothing within this distance of the skin follows body detail exactly")
    blend_distance: FloatProperty(
        name="Blend Range", default=0.03, min=0.0, soft_max=0.1, subtype='DISTANCE', unit='LENGTH',
        description="Distance over which skin-following fades into the smooth warp")
    solver: EnumProperty(name="Solver", items=SOLVERS, default='ELASTIC',
                         description="Solver for items set to Default")
    shape_detail: FloatProperty(
        name="Shape Detail", default=0.01, min=0.002, soft_max=0.05, subtype='DISTANCE', unit='LENGTH',
        description="Shape-preserving solver, for items without their own: detail smaller than this comes from "
                    "the original garment, larger shape changes follow the target")
    displacement_smoothing: IntProperty(
        name="Smoothing", default=4, min=0, max=100,
        description="Smooths the per-vertex movement (not the shape), keeping folds intact")
    rigid_size: FloatProperty(
        name="Rigid Part Size", default=0.03, min=0.0, soft_max=0.1, subtype='DISTANCE', unit='LENGTH',
        description="Separate parts smaller than this (buttons, buckles) move without deforming. "
                    "Vertex group 'AH_Rigid' forces larger parts to be rigid too")
    collision_offset: FloatProperty(
        name="Min Gap", default=0.002, min=0.0, soft_max=0.02, subtype='DISTANCE', unit='LENGTH',
        description="Minimum distance kept outside the body and lower layers. Vertices are only pushed out, never pulled in")
    collision_iterations: IntProperty(name="Collision Passes", default=3, min=0, max=20)
    collide_layers: BoolProperty(
        name="Collide With Lower Layers", default=True,
        description="Each item is kept outside the items above it in the list")
    replace_previous: BoolProperty(
        name="Replace Previous Results", default=True,
        description="Delete earlier fitted copies of the same clothing before fitting")
    hide_originals: BoolProperty(name="Hide Originals", default=True)
    rig_to: EnumProperty(
        name="Rig To",
        items=(
            ('OUTFIT', "Outfit Armature",
             "Keep the fitted copy of the outfit's armature, to merge onto the avatar in Unity "
             "(e.g. with Modular Avatar Merge Armature)"),
            ('AVATAR', "Avatar Armature",
             "Rig the clothing to the avatar's own armature: near the body it takes the body's skin weights so "
             "it bends with the skin without clipping; the outfit's own bones (e.g. for physics) are added "
             "to the avatar's armature"),
        ),
        default='OUTFIT',
        description="What the finalized clothing is rigged to")


class AH_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    conversions_dir: StringProperty(
        name="Conversions Folder", subtype='DIR_PATH', default="",
        description="Where conversions are saved. Empty uses " + conversions.default_library_dir())
    check_updates: BoolProperty(
        name="Check for Updates on Startup", default=True,
        description="Look up the latest release on GitHub when Blender starts (in the background)")

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "conversions_dir")
        box = layout.box()
        box.label(text="Updates", icon='FILE_REFRESH')
        box.prop(self, "check_updates")
        updater.draw(box)


classes = (AH_ClothingItem, AH_Settings, AH_Preferences)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.avatar_helper = PointerProperty(type=AH_Settings)


def unregister():
    del bpy.types.Scene.avatar_helper
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
