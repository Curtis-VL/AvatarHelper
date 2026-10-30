"""Touch-up tools for fitted clothing.

Restore Original Shape: the selected part of a fitted garment is rebuilt from the original
garment's shape. Only how it was stretched and turned on a larger scale is kept from the
fit, so bumps, wobble and crumpled edges go, while the rest of the garment stays put and
holds the restored part in place. Much quicker than straightening a hem with proportional
editing, and it ends up looking like the original because it is the original's shape.

Slide Along Body: the selected part (fading out over Falloff) moves over the avatar's skin,
keeping its distance from it, e.g. a band moved down over a breast or a stocking top down a
thigh. Unlike moving it with proportional editing, it doesn't sink into or lift off the body.
"""
import os

import bpy
import numpy as np

from . import calibration, conversions, elastic, fitting
from . import mesh_utils as mu
from .slide import SkinWalker


def original_of(obj):
    """The garment a fitted copy was made from, with the same vertices."""
    name = obj.get("ah_fitted_from") or obj.get("ah_final_from")
    src = bpy.data.objects.get(name) if name else None
    if src is None or src.type != 'MESH':
        raise ValueError(f"'{obj.name}' isn't a fitted copy of an existing mesh")
    if len(src.data.vertices) != len(obj.data.vertices):
        raise ValueError(f"'{obj.name}' no longer has the same vertices as '{src.name}'")
    return src


def original_shape(obj):
    """The original garment in the shape being edited: Edit Mode edits the active shape key
    (e.g. a "crop" key that folds a long shirt up), so compare with the original's same key."""
    src = original_of(obj)
    key = obj.active_shape_key
    src_keys = src.data.shape_keys
    if key is not None and src_keys is not None and key.name in src_keys.key_blocks:
        data = src_keys.key_blocks[key.name].data
    else:
        data = src.data.vertices
    co = np.empty(len(data) * 3)
    data.foreach_get("co", co)
    return co.reshape(-1, 3)


def restore_shape(context, obj, coords, selected, tris, edges, scale, falloff, body=None, min_gap=0.002,
                  leave_hidden=True):
    """New local positions for the fitted mesh `obj` (currently at `coords`), with the
    `selected` vertices rebuilt from the original garment. If `body` (a mesh object) is
    given, the result is kept outside it, by each point's gap up to `min_gap`, except for the
    parts the fit kept inside the body by design (a tucked-in hem), which stay as fitted
    unless `leave_hidden` is off."""
    rest = original_shape(obj)
    if len(rest) != len(coords):
        raise ValueError(f"'{obj.name}' has {len(coords)} vertices, its original {len(rest)}: "
                         "the shape can only be restored on the mesh as fitted")
    weight = mu.falloff_weights(selected, edges, rest, falloff)
    push = None
    if body is not None:
        push, hidden = outside_push(context, obj, coords, body, min_gap, leave_hidden)
        weight[hidden] = 0.0
    return elastic.restore_shape(rest, coords, tris, edges, weight, scale, push)


def body_mesh(context, body):
    """World vertices and triangles of the avatar's body to keep clothing out of: only its body
    parts, not hair or accessories merged into the mesh, when the scene's conversion is there to
    tell them apart."""
    verts, tris, edges = mu.evaluated_mesh_world(body, context.evaluated_depsgraph_get())
    s = context.scene.avatar_helper
    path = conversions.path_for(s.conversion) if s.conversion else ""
    arm = calibration.target_armature(s) if body == s.target_body else None
    if path and os.path.exists(path) and arm is not None:
        conv = conversions.Conversion.load(path)
        fitted = mu.transform_points(conv.target_alignment(arm), conv.fitted)
        keep = fitting.body_parts(verts, edges, mu.build_bvh(fitted, conv.tris))
        tris = tris[keep[tris].all(1)]
    return verts, tris


def _body_surface(context, body):
    return mu.build_bvh(*body_mesh(context, body))


def outside_push(context, obj, coords, body, min_gap, leave_hidden=True, depth=fitting.BODY_DEPTH, bvh=None):
    """push(x) keeping the garment outside the body, and which points to leave alone: the
    parts the fit kept inside the body by design. Points that are inside the body for any
    other reason (squeezed in by the fit) are restored and pushed back out to Min Gap.
    For meshes fitted before this was recorded, every point inside the body is left alone.
    `bvh`: the body's surface, if already built."""
    mw = np.array(obj.matrix_world)
    inv = np.linalg.inv(mw[:3, :3])
    if bvh is None:
        bvh = _body_surface(context, body)

    def side(points):
        loc, nrm, idx, _dist = mu.nearest_on(bvh, points, depth)
        return np.where(idx >= 0, ((points - loc) * nrm).sum(1), np.inf), loc, nrm

    before, _loc, _nrm = side(mu.transform_points(mw, coords))
    marks = fitting.hidden_marks(obj)
    hidden = marks if marks is not None else before < 0.0
    if not leave_hidden:
        hidden = np.zeros(len(coords), dtype=bool)
    clear = np.where(before >= 0.0, np.clip(before, 0.0, min_gap), min_gap)  # own gap, up to Min Gap

    def push(x, ids=None):
        """The push for the points `ids` (all if None) at local positions x; 0 elsewhere."""
        ids = np.arange(len(x)) if ids is None else ids
        xw = mu.transform_points(mw, x[ids])
        now, loc, nrm = side(xw)
        need = ~hidden[ids] & np.isfinite(now) & (now < clear[ids])
        sub = np.zeros_like(xw)
        sub[need] = (loc[need] + nrm[need] * clear[ids][need, None] - xw[need]) @ inv.T
        out = np.zeros_like(x)
        out[ids] = sub
        return out

    return push, hidden


class SlideAlongBody:
    """Slide Along Body on one fitted mesh, prepared once and then moved to any offset (it
    follows the mouse while dragging). `walker`: a SkinWalker over the avatar's body (see
    body_mesh), shared by all the meshes slid together; `push`, from outside_push, or None."""

    def __init__(self, obj, coords, selected, edges, falloff, walker, push=None):
        self.obj = obj
        self.coords = coords
        self.selected = selected
        self.mw = np.array(obj.matrix_world)
        self.world = mu.transform_points(self.mw, coords)
        self.weight = mu.falloff_weights(selected, edges, self.world, falloff)
        self.moving = self.weight > 0.0
        self.walker = walker
        self.push = push
        # The selection's most central vertex, to tell where it lies on the skin. Not its centre,
        # which for a loop around a leg is inside the leg, nearest to any side of it.
        ids = np.nonzero(selected)[0]
        self.rep = int(ids[np.argmin(np.linalg.norm(self.world[ids] - self.world[ids].mean(0), axis=1))]) \
            if len(ids) else 0

    def moved(self, offset):
        """Local positions with the selection slid by `offset` (a world-space vector)."""
        out = self.coords.copy()
        if not self.moving.any():
            return out
        offsets = self.weight[self.moving, None] * np.asarray(offset, dtype=np.float64)[None]
        world = self.walker.slide(self.world[self.moving], offsets)
        out[self.moving] = mu.transform_points(np.linalg.inv(self.mw), world)
        if self.push is not None:
            # Sliding keeps each point's own gap; this only catches what the stretching at the
            # edge of the falloff pressed into the body.
            ids = np.nonzero(self.moving)[0]
            out[ids] += self.push(out, ids)[ids]
        return out

    def surface_point(self, local=None):
        """Where the selection lies on the skin (as it is, or at the given local positions): the
        skin under its most central vertex."""
        p = self.world[self.rep] if local is None else mu.transform_points(self.mw, local[self.rep][None])[0]
        return self.walker.skin_point(p)


def slide_walker(context, body):
    """A SkinWalker over the avatar's body (its body parts only, see body_mesh)."""
    return SkinWalker(*body_mesh(context, body))
