"""Fingers and toes follow their bones.

Gloves and socks are often not modelled around the base's own fingers and toes: bases come
with "shrink" shape keys that hide the hand or foot inside them, so a glove's fingers can sit
several millimetres off the base's, turned another way, with the base's fingers poking out
between them. Bound to the base's skin there, each glove finger would be pulled after the
wrong bits of finger and come out warped. These parts follow the clothing's own finger and
toe bones instead, turned and stretched onto the avatar's, and widened or narrowed by how
much thicker the avatar's fingers and toes are than the base's.
"""
import numpy as np
from mathutils import Matrix, Vector

from . import bone_map, rig
from . import mesh_utils as mu

FINE = bone_map.FINGER_NAMES + ("toes",)
MAX_BEND = 45.0  # degrees; a last bone's tail turned more than this (an odd rig) turns with its parent
RATIO_RANGE = (0.5, 2.0)
TIP_RADIUS = 0.012  # how far from a finger's axis its tip is looked for, without a thickness to go by
TIP_REACH = 0.05  # and how far beyond its last joint
BEND_BLEND = 0.5  # an extra joint bends over this share of the segment before it


def is_fine(key):
    return key.split("_")[0].split(".")[0] in FINE


def _segment_number(key):
    name = key.rsplit(".", 1)[0]
    return int(name.rsplit("_", 1)[1]) if "_" in name else 1


def _children():
    """humanoid key -> child key for each segment ending in another joint."""
    out = {}
    for key, children in bone_map.pose_segments():
        if is_fine(key):
            out[key] = children
    return out


def _parent_key(key):
    """The joint a finger or toe segment hangs off: the previous segment, or the hand / foot."""
    name, side = key.rsplit(".", 1)
    if name == "toes":
        return f"foot.{side}"
    finger, num = name.rsplit("_", 1)
    return f"hand.{side}" if num == "1" else f"{finger}_{int(num) - 1}.{side}"


def _frame_points(key, joints):
    """Three joints whose triangle gives the hand's (or foot's) turn, including its twist."""
    side = key.rsplit(".", 1)[1]
    if key.startswith("toes"):
        names = (f"lower_leg.{side}", f"foot.{side}", f"toes.{side}")
    else:
        names = (f"hand.{side}", f"index_1.{side}", f"little_1.{side}")
        if not all(n in joints for n in names):
            names = (f"lower_arm.{side}", f"hand.{side}", f"middle_1.{side}")
    return [joints[n] for n in names] if all(n in joints for n in names) else None


def _turn(src_pts, dst_pts):
    """Rotation turning one triangle of joints onto another, ignoring size."""
    _scale, rot = mu.similarity_parts(np.array(src_pts), np.array(dst_pts))
    return rot


def _rotation_between(a, b):
    return np.array(Vector(a).rotation_difference(Vector(b)).to_matrix())


def body_thickness(conv, src_align, tgt_align):
    """Per finger/toe segment: the base's and the avatar's thickness there (median distance
    of the body surface from the segment, halfway along it), in this scene's units. From
    the calibration: the base's surface points around each segment and where they ended up."""
    src = {k: mu.transform_points(src_align, v[None])[0] for k, v in conv.src_joints.items()}
    tgt = {k: mu.transform_points(tgt_align, v[None])[0] for k, v in conv.tgt_joints.items()}
    segs = []
    for key, children in bone_map.pose_segments():
        child = next((c for c in children if c in src and c in tgt), None)
        if key in src and key in tgt and child is not None:
            segs.append((key, child))
    if not segs:
        return {}
    base = mu.transform_points(src_align, conv.base)
    fitted = mu.transform_points(tgt_align, conv.fitted)
    ts, rs = _along_and_out(base, [(src[k], src[c]) for k, c in segs])
    _tt, rt = _along_and_out(fitted, [(tgt[k], tgt[c]) for k, c in segs])
    # Each base point belongs to the segment it is nearest to (measured to the segment).
    dist = np.where((ts >= 0) & (ts <= 1), rs, np.inf)
    owner = np.argmin(dist, axis=0)
    out = {}
    for i, (key, _child) in enumerate(segs):
        m = (owner == i) & (ts[i] > 0.2) & (ts[i] < 0.8) & np.isfinite(dist[i])
        if m.sum() >= 8:
            out[key] = (float(np.median(rs[i][m])), float(np.median(rt[i][m])))
    return out


def _along_and_out(points, segments):
    """Per segment (head, end) and point: how far along it (0..1) and how far out from it."""
    t_all, r_all = [], []
    for h, c in segments:
        axis = c - h
        length = max(float(np.linalg.norm(axis)), 1e-9)
        u = axis / length
        rel = points - h
        a = rel @ u
        t_all.append(a / length)
        r_all.append(np.linalg.norm(rel - np.outer(a, u), axis=1))
    return np.array(t_all), np.array(r_all)


def vertex_keys(obj, arm, n):
    """Per vertex of a body: the humanoid key of the bone it is weighted to most ("" if none).
    Bones without a humanoid match count for the nearest matched bone above them."""
    names = {v: k for k, v in rig.armature_map(arm).items()}
    key_of = {}
    for b in arm.data.bones:
        p = b
        while p is not None and p.name not in names:
            p = p.parent
        if p is not None:
            key_of[b.name] = names[p.name]
    groups = {g.index: key_of.get(g.name, "") for g in obj.vertex_groups}
    out = np.full(n, "", dtype=object)
    if len(obj.data.vertices) != n:
        return out
    for v in obj.data.vertices:
        if v.groups:
            out[v.index] = groups.get(max(v.groups, key=lambda g: g.weight).group, "")
    return out


class LimbFollow:
    """Where each point of a garment goes if its finger and toe parts follow their bones."""

    def __init__(self, garment_arm, target_arm, src_align, tgt_align, conv, min_gap=0.001, target=None):
        """target: (world vertices, humanoid key per vertex) of the avatar's body, to find where
        its fingertips and toe tips end; or None."""
        gmap, tmap = rig.armature_map(garment_arm), rig.armature_map(target_arm)
        self.garment_arm = garment_arm
        self.gmap = gmap
        heads_s = {k: np.array(rig.posed_head(garment_arm, n)[:]) for k, n in gmap.items()}
        heads_t = {k: np.array(rig.posed_head(target_arm, n)[:]) for k, n in tmap.items()}
        thickness = body_thickness(conv, src_align, tgt_align)
        self.min_gap = min_gap
        children = _children()
        # key -> (S_h, u_s, T_h, u_t, length ratio, rotation, (base, avatar) thickness, length, bend, tip)
        self.maps = {}
        keys = sorted((k for k in gmap if is_fine(k) and k in tmap), key=_segment_number)  # parents first
        frames = {}
        for k in keys:
            parent = _parent_key(k)
            if parent in self.maps:
                parent_rot, parent_len = self.maps[parent][5], self.maps[parent][4]
            else:
                if parent not in frames:
                    ps, pt = _frame_points(k, heads_s), _frame_points(k, heads_t)
                    frames[parent] = _turn(ps, pt) if ps and pt else np.eye(3)
                parent_rot, parent_len = frames[parent], None
            child = next((c for c in children.get(k, ()) if c in heads_s and c in heads_t), None)
            sh, th = heads_s[k], heads_t[k]
            if child is not None:
                vs, vt = heads_s[child] - sh, heads_t[child] - th
                len_ratio = np.linalg.norm(vt) / max(np.linalg.norm(vs), 1e-9)
            else:
                gb, tb = garment_arm.pose.bones[gmap[k]], target_arm.pose.bones[tmap[k]]
                vs = np.array((garment_arm.matrix_world @ gb.tail)[:]) - sh
                vt = np.array((target_arm.matrix_world @ tb.tail)[:]) - th
                len_ratio = parent_len if parent_len is not None else 1.0
            us = vs / max(np.linalg.norm(vs), 1e-9)
            ut = vt / max(np.linalg.norm(vt), 1e-9)
            turned = parent_rot @ us
            if child is None and np.degrees(np.arccos(np.clip(turned @ ut, -1, 1))) > MAX_BEND:
                rot = parent_rot
                ut = turned
            else:
                rot = _rotation_between(turned, ut) @ parent_rot
            if parent_len is None and child is None:
                # A single toe bone: as long as the foot grew.
                foot = _parent_key(k)
                knee = foot.replace("foot", "lower_leg")
                if all(j in heads_s and j in heads_t for j in (foot, knee)):
                    len_ratio = (np.linalg.norm(heads_t[foot] - heads_t[knee])
                                 / max(np.linalg.norm(heads_s[foot] - heads_s[knee]), 1e-9))
            body = thickness.get(k, thickness.get(parent))
            # The avatar has another joint further along that the clothing has no bone for (a
            # glove with two bones per finger, an avatar with three): bend there too.
            bend = None
            nxt = next((c for c in children.get(k, ()) if c in heads_t), None) if child is None else None
            if nxt is not None:
                after = next((c for c in children.get(nxt, ()) if c in heads_t), None)
                if after is not None:
                    vn = heads_t[after] - heads_t[nxt]
                else:
                    vn = np.array((target_arm.matrix_world @ target_arm.pose.bones[tmap[nxt]].tail)[:]) - heads_t[nxt]
                un = vn / max(np.linalg.norm(vn), 1e-9)
                if np.degrees(np.arccos(np.clip(ut @ un, -1, 1))) <= MAX_BEND:
                    at = float((heads_t[nxt] - th) @ ut)
                    if at > 1e-6:
                        bend = (at, heads_t[nxt], un, _rotation_between(ut, un) @ rot)
            # Where the avatar's fingertip (or toe tip) ends, beyond the last joint.
            tip = None
            if child is None and target is not None:
                target_verts, target_keys = target
                start, joint, end_dir = (bend[0], bend[1], bend[2]) if bend is not None else (0.0, th, ut)
                reach = 1.3 * body[1] if body is not None else TIP_RADIUS
                rel = target_verts - joint
                a = rel @ end_dir
                # Only the avatar's own finger: along the line, a thumb would find the hand.
                mine = np.isin(target_keys, [k] + ([nxt] if bend is not None else []))
                near = (mine & (a > 0) & (a < TIP_REACH)
                        & (np.linalg.norm(rel - np.outer(a, end_dir), axis=1) < reach))
                if near.any():
                    tip = (start, float(a[near].max()))
            self.maps[k] = (sh, us, th, ut, float(len_ratio), rot, body, float(np.linalg.norm(vs)), bend, tip)

    def _widen(self, radius, along, weight, len_ratio, body):
        """How much wider a finger (or toe) of the garment becomes. It keeps the gap it had over
        the base's finger, on the avatar's finger, but at least Min Gap: gloves are often made
        tighter than the base's fingers (the base hides them with a shrink shape key), and
        scaled with the fingers they would end up inside the avatar's."""
        if body is None:
            return len_ratio
        mid = (weight > 0.5) & (along > 0.2) & (along < 0.8)
        if mid.sum() < 8:
            return len_ratio
        mine = float(np.median(radius[mid]))
        base, avatar = body
        gap = max(mine - base, self.min_gap)
        return float(np.clip((avatar + gap) / max(mine, 1e-9), *RATIO_RANGE))

    def any(self):
        return bool(self.maps)

    def weights(self, obj):
        """Per vertex, its weight on each finger/toe key, from the vertex groups. Bones
        without a humanoid match count for the nearest matched bone above them."""
        arm = self.garment_arm
        key_of = {}
        names = {n: k for k, n in self.gmap.items()}
        for b in arm.data.bones:
            p = b
            while p is not None and p.name not in names:
                p = p.parent
            if p is not None:
                key_of[b.name] = names[p.name]
        groups = {g.index: key_of.get(g.name) for g in obj.vertex_groups if g.name in arm.data.bones}
        keys = list(self.maps)
        col = {k: i for i, k in enumerate(keys)}
        n = len(obj.data.vertices)
        w = np.zeros((n, len(keys)))
        total = np.zeros(n)
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group not in groups:
                    continue
                total[v.index] += g.weight
                k = groups[g.group]
                if k in col:
                    w[v.index, col[k]] += g.weight
        w /= np.maximum(total, 1e-9)[:, None]
        return keys, w

    def __call__(self, obj, verts):
        """(positions, weight): where the points go following their finger/toe bones, and how
        much of each point is finger or toe."""
        keys, w = self.weights(obj)
        out = np.zeros_like(verts)
        for i, k in enumerate(keys):
            m = w[:, i] > 0
            if not m.any():
                continue
            sh, us, th, ut, len_ratio, rot, body, length, bend, tip = self.maps[k]
            rel = verts[m] - sh
            a = rel @ us
            radial = rel - np.outer(a, us)
            ratio = self._widen(np.linalg.norm(radial, axis=1), a / max(length, 1e-9), w[m, i], len_ratio, body)
            along = a * len_ratio
            if tip is not None:
                # A garment fingertip ending short of the avatar's is lengthened to cover it.
                start, end = tip
                past = along - start
                mine = past[w[m, i] > 0.5].max(initial=0.0)
                if 0.0 < mine < end + self.min_gap:
                    along = np.where(past > 0, start + past * (end + self.min_gap) / mine, along)
            q = th + np.outer(along, ut) + ratio * radial @ rot.T
            if bend is not None:
                at, joint, un, rot_n = bend
                past = along - at
                qn = joint + np.outer(past, un) + ratio * radial @ rot_n.T
                t = np.clip(past / (BEND_BLEND * at) + 0.5, 0.0, 1.0)
                t = t * t * (3.0 - 2.0 * t)
                q = q * (1.0 - t[:, None]) + qn * t[:, None]
            out[m] += w[m, i, None] * q
        wf = w.sum(1)
        out = np.where(wf[:, None] > 1e-9, out / np.maximum(wf, 1e-9)[:, None], verts)
        return out, np.clip(wf, 0.0, 1.0)
