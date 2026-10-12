"""Humanoid bone detection from the naming styles commonly found on VRChat avatars.

Covers Cats/Booth style ("Upper_arm.L", "Left arm"), Unity style ("LeftUpperArm"),
VRM ("J_Bip_L_UpperArm"), Mixamo ("mixamorig:LeftArm") and MMD Japanese names.

Kept free of bpy so it can be tested outside Blender.
"""
import re

CENTRAL = ("hips", "spine", "chest", "upper_chest", "neck", "head")
SIDED = ("shoulder", "upper_arm", "lower_arm", "hand",
         "upper_leg", "lower_leg", "foot", "toes")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")

# Bones a usable humanoid rig should have. Upper chest, toes and fingers are optional.
REQUIRED = ["hips", "spine", "chest", "neck", "head"] + [
    f"{b}.{s}" for s in "LR"
    for b in ("shoulder", "upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot")
]

CORE_NAMES = {
    "hips": "hips", "hip": "hips", "pelvis": "hips",
    "spine": "spine",
    "chest": "chest", "spine1": "chest",
    "upperchest": "upper_chest", "spine2": "upper_chest",
    "neck": "neck",
    "head": "head",
    "shoulder": "shoulder", "clavicle": "shoulder",
    "upperarm": "upper_arm", "arm": "upper_arm", "uparm": "upper_arm",
    "lowerarm": "lower_arm", "forearm": "lower_arm", "elbow": "lower_arm", "lowarm": "lower_arm",
    "hand": "hand", "wrist": "hand",
    "upperleg": "upper_leg", "thigh": "upper_leg", "upleg": "upper_leg", "leg": "upper_leg",
    "lowerleg": "lower_leg", "knee": "lower_leg", "calf": "lower_leg", "shin": "lower_leg",
    "lowleg": "lower_leg",
    "foot": "foot", "ankle": "foot",
    "toes": "toes", "toe": "toes", "toebase": "toes",
}

FINGER_TOKENS = {"thumb": "thumb", "index": "index", "middle": "middle",
                 "ring": "ring", "little": "little", "pinky": "little"}

DROP_TOKENS = {"mixamorig", "j", "bip", "c", "def", "org", "bone"}

JP_CORE = {
    "下半身": "hips", "上半身": "spine", "上半身2": "chest", "首": "neck", "頭": "head",
    "肩": "shoulder", "腕": "upper_arm", "ひじ": "lower_arm", "手首": "hand",
    "足": "upper_leg", "ひざ": "lower_leg", "足首": "foot", "つま先": "toes",
}
JP_FINGERS = {"親指": "thumb", "人差指": "index", "人指": "index", "中指": "middle",
              "薬指": "ring", "小指": "little"}


def _tokens(name):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _classify_jp(name):
    side = None
    if name[:1] == "左":
        side, name = "L", name[1:]
    elif name[:1] == "右":
        side, name = "R", name[1:]
    name = name.strip()
    if name in JP_CORE:
        return ("core", JP_CORE[name], side, name)
    for prefix, finger in JP_FINGERS.items():
        if name.startswith(prefix):
            return ("finger", finger, side, name)
    return None


def classify(name):
    """Return (kind, key, side, core) for a bone name, or None.

    kind is "core" or "finger"; side is "L", "R" or None.
    """
    if re.search(r"[぀-ヿ一-鿿]", name):
        return _classify_jp(name)

    # The last side marker wins: suffixes like ".R" are more reliable than words in the
    # name (e.g. "Index Finger Left 02.R" is on the right hand).
    side = None
    rest = []
    for t in _tokens(name):
        if t in ("l", "left"):
            side = "L"
        elif t in ("r", "right"):
            side = "R"
        elif t not in DROP_TOKENS:
            rest.append(t)
    if not rest or "end" in rest or "tip" in rest:
        return None

    for t in rest:
        finger = FINGER_TOKENS.get(t)
        if finger is None and t.endswith("finger"):
            finger = FINGER_TOKENS.get(t[:-len("finger")])
        if finger:
            return ("finger", finger, side, "".join(rest))

    core = "".join(rest)
    key = CORE_NAMES.get(core)
    if key is None:
        return None
    return ("core", key, side, core)


def humanoid_map(bones):
    """Map humanoid keys to bone names.

    bones: iterable of (name, parent_name_or_None).
    Returns e.g. {"hips": "Hips", "upper_arm.L": "Upper_arm.L", "index_1.L": "IndexFinger1_L"}.
    """
    bones = list(bones)
    parents = dict(bones)

    def depth(n):
        d = 0
        p = parents.get(n)
        while p is not None and d < 1000:
            d += 1
            p = parents.get(p)
        return d

    candidates = []
    for name, _parent in bones:
        c = classify(name)
        if c:
            candidates.append((name, depth(name), c))

    # Mixamo uses UpLeg/Leg for thigh/shin, while Cats uses Leg/Knee.
    has_upleg = {c[2] for _n, _d, c in candidates if c[0] == "core" and c[3] == "upleg"}

    result = {}
    best_depth = {}
    finger_candidates = []
    for name, d, (kind, key, side, core) in candidates:
        if kind == "finger":
            finger_candidates.append((name, d, key, side))
            continue
        if core == "leg" and side in has_upleg:
            key = "lower_leg"
        if key in CENTRAL:
            if side:
                continue
            full = key
        else:
            if not side:
                continue
            full = f"{key}.{side}"
        if full not in best_depth or d < best_depth[full]:
            best_depth[full] = d
            result[full] = name

    # A finger belongs to the hand it hangs off, whatever its name says.
    hand_side = {result[f"hand.{s}"]: s for s in "LR" if f"hand.{s}" in result}

    def ancestor_hand_side(n):
        p = parents.get(n)
        steps = 0
        while p is not None and steps < 1000:
            if p in hand_side:
                return hand_side[p]
            p = parents.get(p)
            steps += 1
        return None

    fingers = {}
    for name, d, key, side in finger_candidates:
        hand = ancestor_hand_side(name)
        if hand is None and f"hand.{side}" in result:
            continue  # not on the hand: an accessory bone, e.g. a harness's "Spine Ring L"
        side = hand or side
        if side:
            fingers.setdefault((key, side), []).append((d, name))

    # Finger segments are assigned by hierarchy depth so "Thumb0"/"Thumb1" numbering differences don't matter.
    for (finger, side), lst in fingers.items():
        lst.sort()
        for i, (_d, name) in enumerate(lst[:3]):
            result[f"{finger}_{i + 1}.{side}"] = name

    return result


def missing_required(mapping):
    return [k for k in REQUIRED if k not in mapping]


def pose_segments():
    """(bone key, child key candidates) pairs in parent-first order.

    Pose matching rotates each bone so the direction from its head to the child's head
    matches the target rig.
    """
    segs = [
        ("spine", ("chest", "upper_chest", "neck")),
        ("chest", ("upper_chest", "neck")),
        ("upper_chest", ("neck",)),
        ("neck", ("head",)),
    ]
    for s in "LR":
        segs += [
            (f"shoulder.{s}", (f"upper_arm.{s}",)),
            (f"upper_arm.{s}", (f"lower_arm.{s}",)),
            (f"lower_arm.{s}", (f"hand.{s}",)),
            (f"hand.{s}", tuple(f"{f}_1.{s}" for f in ("middle", "index", "ring", "little"))),
        ]
        for f in FINGER_NAMES:
            segs += [
                (f"{f}_1.{s}", (f"{f}_2.{s}",)),
                (f"{f}_2.{s}", (f"{f}_3.{s}",)),
            ]
        segs += [
            (f"upper_leg.{s}", (f"lower_leg.{s}",)),
            (f"lower_leg.{s}", (f"foot.{s}",)),
            (f"foot.{s}", (f"toes.{s}",)),
        ]
    return segs
