"""Saved conversions: a calibration from one avatar base to another (e.g. "Manuka > Custom"),
reusable for any outfit without the source avatar in the scene.

A conversion holds everything fitting needs from the source side:
  * the calibrated body: the (pose-matched) source body and where each point of it lands
    on the target body;
  * the pose matching: per humanoid bone, how the source rig was turned to match the
    target's limb directions, so any clothing armature built on that base can be posed
    the same way, even one with only some of the bones;
  * the rest positions of both rigs' humanoid joints, used to line the conversion up with
    whatever armature the clothing uses and the target avatar in the current scene;
  * the source feet (for placing shoes);
  * the shape key values of both bodies when it was made (e.g. "Foot_heel"), so clothing
    can be fitted with matching keys.

Since version 2 everything is stored relative to the armatures (the source side to the base's
armature, the target side to the avatar's), as if calibrated at the origin: where either
avatar stood in the scene doesn't matter.
"""
import json
import os
import re

import bpy
import numpy as np
from mathutils import Matrix

from . import mesh_utils as mu
from . import rig

VERSION = 2
EXTENSION = ".ahconv"
LIBRARY_OVERRIDE = None  # tests point this at a temporary folder


def library_dir():
    """Folder where conversions are saved (set in the add-on preferences)."""
    if LIBRARY_OVERRIDE:
        path = LIBRARY_OVERRIDE
    else:
        path = ""
        addon = bpy.context.preferences.addons.get(__package__)
        if addon is not None and getattr(addon.preferences, "conversions_dir", ""):
            path = bpy.path.abspath(addon.preferences.conversions_dir)
        if not path:
            path = default_library_dir()
    os.makedirs(path, exist_ok=True)
    return path


def default_library_dir():
    home = os.path.expanduser("~")
    docs = os.path.join(home, "Documents")
    return os.path.join(docs if os.path.isdir(docs) else home, "AvatarHelper", "Conversions")


def file_stem(name):
    """A file name for a conversion name: "Manuka > Custom" -> "Manuka to Custom"."""
    stem = name.replace(">", " to ")
    stem = re.sub(r'[<:"/\\|?*]+', "_", stem)
    return re.sub(r"\s+", " ", stem).strip() or "Conversion"


def list_library():
    """[(stem, display name)] of the saved conversions, sorted by name."""
    out = []
    folder = library_dir()
    for fn in os.listdir(folder):
        if fn.endswith(EXTENSION):
            stem = fn[:-len(EXTENSION)]
            out.append((stem, read_name(os.path.join(folder, fn)) or stem))
    return sorted(out, key=lambda x: x[1].lower())


def read_name(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["meta"]))["name"]
    except (OSError, KeyError, ValueError):
        return None


def path_for(stem):
    return os.path.join(library_dir(), stem + EXTENSION)


class Conversion:
    def __init__(self, name, base, fitted, tris, src_joints, tgt_joints, pose, src_feet,
                 source="", target="", frame="world", src_keys=None, tgt_keys=None):
        self.name = name
        # "world": as in the scene at calibration (version 1 files); "armature": the source side
        # relative to the base's armature, the target side relative to the avatar's.
        self.frame = frame
        self.base = np.asarray(base, dtype=np.float64)  # pose-matched source body
        self.fitted = np.asarray(fitted, dtype=np.float64)  # the same points on the target body
        self.tris = np.asarray(tris, dtype=np.int64)
        self.src_joints = {k: np.asarray(v, dtype=np.float64) for k, v in src_joints.items()}  # rest
        self.tgt_joints = {k: np.asarray(v, dtype=np.float64) for k, v in tgt_joints.items()}
        self.pose = {k: np.asarray(v, dtype=np.float64) for k, v in pose.items()}  # 4x4 deltas
        # Per side: ankle, toe tip, a point on the floor below the ankle, foot width. Version 1
        # files store the floor as a height.
        self.src_feet = {k: tuple(np.asarray(x, dtype=np.float64) for x in v) for k, v in src_feet.items()}
        self.source = source
        self.target = target
        self.src_keys = dict(src_keys or {})  # shape key values of the source body
        self.tgt_keys = dict(tgt_keys or {})  # and of the target body

    def to_armature_frame(self, src_matrix, tgt_matrix):
        """Re-express a conversion made in world space relative to the two armatures (their
        world matrices at calibration)."""
        s_inv = np.linalg.inv(src_matrix)
        t_inv = np.linalg.inv(tgt_matrix)
        self.base = mu.transform_points(s_inv, self.base)
        self.fitted = mu.transform_points(t_inv, self.fitted)
        self.src_joints = {k: _apply(s_inv, v) for k, v in self.src_joints.items()}
        self.tgt_joints = {k: _apply(t_inv, v) for k, v in self.tgt_joints.items()}
        self.pose = {k: s_inv @ v @ src_matrix for k, v in self.pose.items()}
        scale = np.cbrt(abs(np.linalg.det(s_inv[:3, :3])))
        self.src_feet = {side: (_apply(s_inv, ankle), _apply(s_inv, tip),
                                _apply(s_inv, _floor_point(ankle, floor)), np.asarray(width * scale))
                         for side, (ankle, tip, floor, width) in self.src_feet.items()}
        self.frame = "armature"

    # ---------------------------------------------------------------- files

    def meta(self):
        return {
            "version": VERSION, "name": self.name, "source": self.source, "target": self.target,
            "frame": self.frame,
            "src_joints": {k: v.tolist() for k, v in self.src_joints.items()},
            "tgt_joints": {k: v.tolist() for k, v in self.tgt_joints.items()},
            "pose": {k: v.tolist() for k, v in self.pose.items()},
            "src_feet": {k: [np.asarray(x).tolist() for x in v] for k, v in self.src_feet.items()},
            "src_keys": self.src_keys, "tgt_keys": self.tgt_keys,
        }

    def save(self, path):
        with open(path, "wb") as f:
            np.savez_compressed(f, base=self.base.astype(np.float32), fitted=self.fitted.astype(np.float32),
                                tris=self.tris.astype(np.int32), meta=json.dumps(self.meta()))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls.from_meta(meta, data["base"], data["fitted"], data["tris"])

    @classmethod
    def from_meta(cls, meta, base, fitted, tris):
        return cls(meta["name"], base, fitted, tris, meta["src_joints"], meta["tgt_joints"], meta["pose"],
                   meta["src_feet"], meta.get("source", ""), meta.get("target", ""), meta.get("frame", "world"),
                   meta.get("src_keys"), meta.get("tgt_keys"))

    # ---------------------------------------------------------------- placement

    def source_alignment(self, arm):
        """4x4 matrix carrying the conversion's source space onto clothing armature `arm`
        (matching their humanoid joints at rest). An armature that can't be matched is taken
        to be the base's own armature, wherever it is."""
        mapping = rig.armature_map(arm)
        keys = [k for k in mapping if k in self.src_joints]
        if self.frame == "world":
            here = [np.array((arm.matrix_world @ arm.data.bones[mapping[k]].head_local)[:]) for k in keys]
            return _align([self.src_joints[k] for k in keys], here)
        here = [np.array(arm.data.bones[mapping[k]].head_local[:]) for k in keys]
        mw = np.array(arm.matrix_world)
        return mw @ _align_local([self.src_joints[k] for k in keys], here, _scale(mw))

    def target_alignment(self, arm):
        """4x4 matrix carrying the conversion's target space onto the target armature as posed now."""
        mapping = rig.armature_map(arm)
        keys = [k for k in mapping if k in self.tgt_joints]
        posed = [np.array(rig.posed_head(arm, mapping[k])[:]) for k in keys]
        if self.frame == "world":
            return _align([self.tgt_joints[k] for k in keys], posed)
        mw = np.array(arm.matrix_world)
        here = [_apply(np.linalg.inv(mw), p) for p in posed]
        return mw @ _align_local([self.tgt_joints[k] for k in keys], here, _scale(mw))

    def apply_pose(self, context, arm, align):
        """Pose a clothing armature the way the source rig was posed to match the target.
        Bones without a humanoid match follow their parents."""
        mapping = rig.armature_map(arm)
        keys = sorted((k for k in mapping if k in self.pose), key=lambda k: rig.bone_depth(arm.data.bones[mapping[k]]))
        rig.reset_pose(arm)
        context.view_layer.update()
        a = Matrix(align.tolist())
        a_inv = a.inverted()
        inv = arm.matrix_world.inverted()
        for k in keys:
            delta = a @ Matrix(self.pose[k].tolist()) @ a_inv
            bone = arm.data.bones[mapping[k]]
            arm.pose.bones[bone.name].matrix = inv @ delta @ arm.matrix_world @ bone.matrix_local
            context.view_layer.update()

    def feet(self, align):
        """Source foot frames (ankle, tip, floor, width) carried into the clothing's space."""
        scale = np.cbrt(abs(np.linalg.det(align[:3, :3])))
        out = {}
        for side, (ankle, tip, floor, width) in self.src_feet.items():
            a, t = _apply(align, ankle), _apply(align, tip)
            fl = _apply(align, _floor_point(ankle, floor))[2]
            out[side] = (a, t, float(fl), float(width) * scale)
        return out


def _floor_point(ankle, floor):
    """A point on the floor below the ankle (version 1 files store the floor as a height)."""
    floor = np.asarray(floor, dtype=np.float64)
    return floor if floor.ndim else np.array([ankle[0], ankle[1], float(floor)])


def _apply(m, p):
    return m[:3, :3] @ np.asarray(p) + m[:3, 3]


ALIGN_EXACT = 0.002  # metres; an armature whose joints all lie this close is taken as is


def _scale(m):
    return float(np.cbrt(abs(np.linalg.det(np.asarray(m)[:3, :3]))))


def _align_local(src, dst, unit=1.0):
    """Like _align, but between armature spaces (`unit`: world size of one armature unit): a
    clothing armature built on the base's has the same joints, so it's taken as is rather
    than fitted (which would add small errors, or worse with few joints). Only an armature
    that clearly differs is fitted."""
    if not len(src):
        return np.eye(4)
    src, dst = np.array(src), np.array(dst)
    resid = np.linalg.norm(src - dst, axis=1) * unit
    if np.percentile(resid, 90) <= ALIGN_EXACT:
        return np.eye(4)
    return _align(src, dst, unit)


ALIGN_OUTLIER = 0.01  # metres; joints further off than this after alignment are left out


def _align(src, dst, unit=1.0):
    """Similarity transform (4x4) taking points src onto dst. Joints that don't agree with the
    rest (a mis-named bone, one moved by the outfit's creator) are dropped and the fit redone.
    `unit`: world size of one unit of the points."""
    if not len(src):
        return np.eye(4)
    src, dst = np.array(src), np.array(dst)
    keep = np.ones(len(src), dtype=bool)
    for _ in range(3):
        m = _similarity(src[keep], dst[keep])
        resid = np.linalg.norm(src @ m[:3, :3].T + m[:3, 3] - dst, axis=1)
        new = resid <= max(ALIGN_OUTLIER / unit, 3.0 * np.median(resid))
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    return m


def _similarity(src, dst):
    m = np.eye(4)
    if len(src) < 3 or np.linalg.matrix_rank(src - src.mean(0), tol=1e-4) < 2:
        m[:3, 3] = dst.mean(0) - src.mean(0)
        return m
    scale, rot = mu.similarity_parts(src, dst)
    m[:3, :3] = scale * rot
    m[:3, 3] = dst.mean(0) - m[:3, :3] @ src.mean(0)
    return m


def capture_pose(src_arm):
    """World-space change of each humanoid bone from rest to its current pose."""
    mapping = rig.armature_map(src_arm)
    mw = src_arm.matrix_world
    out = {}
    for k, name in mapping.items():
        posed = mw @ src_arm.pose.bones[name].matrix
        rest = mw @ src_arm.data.bones[name].matrix_local
        out[k] = np.array(posed @ rest.inverted())
    return out


def rest_joints(arm):
    mapping = rig.armature_map(arm)
    return {k: np.array((arm.matrix_world @ arm.data.bones[n].head_local)[:]) for k, n in mapping.items()}


def posed_joints(arm):
    mapping = rig.armature_map(arm)
    return {k: np.array(rig.posed_head(arm, n)[:]) for k, n in mapping.items()}
