import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty


def _is_mesh(_self, obj):
    return obj.type == 'MESH'


def _is_armature(_self, obj):
    return obj.type == 'ARMATURE'


class AH_ClothingItem(bpy.types.PropertyGroup):
    obj: PointerProperty(name="Object", type=bpy.types.Object, poll=_is_mesh)
    mode: EnumProperty(
        name="Mode",
        items=(
            ('AUTO', "Auto", "Shoes if the item is skinned almost entirely to the feet, otherwise Fit"),
            ('FIT', "Fit", "Reshape to the target body"),
            ('SHOES', "Shoes", "Keep the original shape; place on the target's feet at floor level"),
        ),
        default='AUTO',
    )


class AH_Settings(bpy.types.PropertyGroup):
    source_armature: PointerProperty(
        name="Source Armature", type=bpy.types.Object, poll=_is_armature,
        description="Armature of the avatar the clothing was made for. Auto-detected from the body if empty")
    source_body: PointerProperty(
        name="Source Body", type=bpy.types.Object, poll=_is_mesh,
        description="Body mesh of the avatar the clothing was made for")
    target_armature: PointerProperty(
        name="Target Armature", type=bpy.types.Object, poll=_is_armature,
        description="Armature of the avatar to fit to. Auto-detected from the body if empty")
    target_body: PointerProperty(
        name="Target Body", type=bpy.types.Object, poll=_is_mesh,
        description="Body mesh of the avatar to fit to")

    # Calibration
    calibration: PointerProperty(
        name="Calibration", type=bpy.types.Object, poll=_is_mesh,
        description="Calibration mesh (source body with a 'Fitted' shape key)")
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


classes = (AH_ClothingItem, AH_Settings)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.avatar_helper = PointerProperty(type=AH_Settings)


def unregister():
    del bpy.types.Scene.avatar_helper
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
