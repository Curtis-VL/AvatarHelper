"""Anchors: say where a part of a garment should end up on your avatar.

A fit puts each part of a garment where the base's body under it ends up on the avatar. When
the bodies are shaped differently that can differ from what the garment's design means (a band
meant to cover the nipples, a stocking top meant to end a hand's width below the crotch). An
anchor overrides it for one part: select the part on the fitted garment, Add Anchor, and drag
the marker over the skin to where the part should be. Every later fit then:
  * slides the conversion's body mapping under the part by the marker's offset over the skin
    (fading out over the anchor's Radius). The mapping is shared by the whole outfit, so
    whatever lies over or under the part (a top over a bra) moves along with it;
  * settles the part itself where the moved mapping puts the skin under it: the fit's smooth
    warp carries only most of a local change like this, so after the garment is fitted, what is
    left is slid over the skin.

Anchors are applied in the order they were made, each on top of the ones before: an anchor's
offset is measured on a fit that already had the earlier anchors.
"""
from dataclasses import dataclass

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from . import mesh_utils as mu
from .slide import SkinWalker

ANCHOR_COLLECTION = "AH_Anchors"
VERTS = "ah_anchor_verts"  # the garment's vertices the anchor holds
START = "ah_anchor_start"  # where they were on the skin when it was made (in its parent's space)
ORDER = "ah_anchor_id"
MIRROR = np.array([-1.0, 1.0, 1.0])  # conversions are stored relative to the armatures: x = 0 is the middle
ON_SKIN = 0.012  # only parts of the garment this close to the base's skin move the body mapping


def anchor_objects():
    """All anchor markers, oldest first."""
    coll = bpy.data.collections.get(ANCHOR_COLLECTION)
    if coll is None:
        return []
    return sorted((o for o in coll.all_objects if VERTS in o), key=lambda o: o.get(ORDER, 0))


def anchors_for(objs):
    """The anchors on the given garments."""
    objs = set(objs)
    return [a for a in anchor_objects() if a.ah_anchor.garment in objs]


def _parent_matrix(marker):
    return marker.parent.matrix_world if marker.parent else Matrix.Identity(4)


def create(context, garment, vert_ids, start, end, target_body=None, target_arm=None, radius=None):
    """A marker at `end` holding the garment's vertices `vert_ids`, which a fit put over the skin
    at `start` (world positions on the skin). On the target armature, so it moves with the
    avatar, and held to the target body's surface so it can only be dragged over the skin."""
    coll = bpy.data.collections.get(ANCHOR_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(ANCHOR_COLLECTION)
    if coll not in context.scene.collection.children_recursive:
        context.scene.collection.children.link(coll)
    lid = max((o.get(ORDER, 0) for o in anchor_objects()), default=0) + 1
    marker = bpy.data.objects.new(f"Anchor {lid:02d} ({garment.name})", None)
    marker.empty_display_type = 'SPHERE'
    marker.empty_display_size = 0.01
    marker.show_in_front = True
    coll.objects.link(marker)
    if target_arm is not None:
        marker.parent = target_arm
        marker.matrix_parent_inverse = Matrix.Identity(4)
    place(marker, start, end)
    marker[VERTS] = [int(i) for i in vert_ids]
    marker[ORDER] = lid
    marker.ah_anchor.garment = garment
    if radius is not None:
        marker.ah_anchor.radius = radius
    if target_body is not None:
        c = marker.constraints.new('SHRINKWRAP')
        c.target = target_body
        c.shrinkwrap_type = 'NEAREST_SURFACE'
    return marker


def place(marker, start, end):
    """Where the fit put the anchored part (`start`) and where it should be (`end`), world space."""
    inv = _parent_matrix(marker).inverted()
    marker.location = inv @ Vector(end)
    marker[START] = list(inv @ Vector(start))


def target(marker):
    """Where the marker is (world space)."""
    return np.array(marker.matrix_world.translation[:])


def offset(marker):
    """How far the marker was moved from where the fit put its part (world space)."""
    return target(marker) - np.array((_parent_matrix(marker) @ Vector(marker[START]))[:])


@dataclass
class Anchored:
    """An anchor ready for a fit (or the mirror image of one)."""
    garment: object
    verts: np.ndarray  # the garment's vertices it holds
    start: np.ndarray  # where the fit put them on the skin when the anchor was made (world space)
    offset: np.ndarray  # from there to the marker (world space)
    radius: float
    body: np.ndarray  # the conversion's body vertices under the part, which the offset moves


def _nearest(kd, points):
    return np.fromiter((kd.find(p)[1] for p in points.tolist()), dtype=np.int64, count=len(points))


def _kd(points):
    kd = KDTree(len(points))
    for i, p in enumerate(points.tolist()):
        kd.insert(p, i)
    kd.balance()
    return kd


def _footprint(kd, base, points):
    """The conversion's body vertices under the points that lie on the skin. A part bridging
    away from it (a band across a cleavage) would drag along body far under it that other parts
    hug (the breastbone); it follows the parts on the skin anyway."""
    ids = _nearest(kd, points)
    close = np.linalg.norm(points - base[ids], axis=1) < ON_SKIN
    return np.unique(ids[close] if close.any() else ids)


def prepare(context, objs, conv, align_of, tgt_align):
    """The anchors on the garments being fitted, as Anchored, plus the mirror image of those with
    Mirror on. The garments must be posed as for fitting. Also returns notes on anchors left out."""
    out, notes = [], []
    markers = anchors_for(objs)
    if not markers:
        return out, notes
    kd = _kd(conv.base)
    t3 = tgt_align[:3, :3]
    t3_inv = np.linalg.inv(t3)
    t_inv = np.linalg.inv(tgt_align)
    dg = context.evaluated_depsgraph_get()
    posed = {}
    for a in markers:
        g = a.ah_anchor.garment
        if g not in posed:
            world = mu.evaluated_mesh_world(g, dg)[0]
            posed[g] = (world, mu.transform_points(np.linalg.inv(align_of(g)), world))
        world, here = posed[g]
        ids = np.asarray(list(a[VERTS]), dtype=np.int64)
        if not len(ids) or ids.max() >= len(here):
            notes.append(f"Anchor '{a.name}' left out: '{g.name}' no longer has the vertices it holds")
            continue
        d = offset(a)
        if np.linalg.norm(d) < 1e-5:
            continue
        start = np.array((_parent_matrix(a) @ Vector(a[START]))[:])
        radius = a.ah_anchor.radius
        out.append(Anchored(g, ids, start, d, radius, _footprint(kd, conv.base, here[ids])))
        if a.ah_anchor.mirror:
            # Mirrored in the conversion's frames, where both bodies are centred on x = 0.
            m_ids = np.unique(_nearest(_kd(here), here[ids] * MIRROR))
            m_start = mu.transform_points(tgt_align, mu.transform_points(t_inv, start[None]) * MIRROR)[0]
            m_d = t3 @ ((t3_inv @ d) * MIRROR)
            out.append(Anchored(g, m_ids, m_start, m_d, radius, _footprint(kd, conv.base, here[m_ids])))
    return out, notes


def _edges(tris):
    return np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)


def apply(fitted, tris, anchored):
    """The conversion's body mapping (`fitted`: where each point of the base's body is on the
    avatar, world space) with each anchor's part of the body slid over the avatar's skin by its
    offset, fading out over its radius."""
    if not anchored:
        return fitted
    walker = SkinWalker(fitted, tris)
    edges = _edges(tris)
    out = fitted.copy()
    for a in anchored:
        held = np.zeros(len(fitted), dtype=bool)
        held[a.body] = True
        w = mu.falloff_weights(held, edges, fitted, a.radius)
        m = w > 0.0
        out[m] = walker.slide(out[m], w[m, None] * _aim(walker, out, a)[None])
    return out


AIM_PASSES = 4


def _aim(walker, points, a):
    """The offset to walk the anchor's part of the body mapping by, so that its point nearest to
    where the part was ends up on the marker. Walking the straight offset itself falls short
    where the skin curves (around a breast): the walk goes that far over the skin, not straight."""
    ref = points[a.body[np.argmin(np.linalg.norm(points[a.body] - a.start, axis=1))]]
    goal = ref + a.offset
    d = a.offset.copy()
    for _ in range(AIM_PASSES):
        miss = goal - walker.slide(ref[None], d[None])[0]
        if np.linalg.norm(miss) < 1e-4:
            break
        d = d + miss
    return d


def settle(obj, src_points, points, edges, anchored, field, walker):
    """A fitted garment (`points`, world space; `src_points` as it was on the base) with its
    anchored parts slid over the skin (`walker`, a SkinWalker over the avatar's body) the rest of
    the way to where the anchored body mapping (`field`, a FitField) puts the skin under them.
    Parts bridging away from the skin go as far as the nearest anchored part that is on it."""
    mine = [a for a in anchored if a.garment is obj]
    if not mine:
        return points
    held = np.zeros(len(points), dtype=bool)
    for a in mine:
        held[a.verts] = True
    ids = np.nonzero(held)[0]
    loc, _nrm, tri, _dist = mu.nearest_on(field.bvh, src_points[ids], ON_SKIN)
    on = tri >= 0
    if not on.any():
        return points
    t = field.tris[tri[on]]
    b = field.base
    bary = mu.barycentric(loc[on], b[t[:, 0]], b[t[:, 1]], b[t[:, 2]])
    f = field.fitted
    goal = bary[:, :1] * f[t[:, 0]] + bary[:, 1:2] * f[t[:, 1]] + bary[:, 2:] * f[t[:, 2]]
    now = np.array([walker.skin_point(p) for p in points[ids[on]]])
    rest = goal - now
    # Each moving point takes the leftover of the nearest anchored point on the skin, fading out.
    w = mu.falloff_weights(held, edges, points, max(a.radius for a in mine))
    m = np.nonzero(w > 0.0)[0]
    nearest = _nearest(_kd(points[ids[on]]), points[m])
    points = points.copy()
    points[m] = walker.slide(points[m], w[m, None] * rest[nearest])
    return points
