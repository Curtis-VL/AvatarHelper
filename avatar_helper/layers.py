"""Work out clothing layer order from how the items overlap on the source avatar."""
import numpy as np

from . import mesh_utils as mu

OVERLAP_RANGE = 0.02  # only parts within this distance of another item count as overlapping
MARGIN = 0.002  # share difference needed to call one item "under" another


def _under_share(points, normals, shell):
    """Share of `points` that sit under the outer faces of `shell` (shell = (verts, tris, bvh))."""
    _v, _t, bvh = shell
    loc, nrm, idx, dist = mu.nearest_on(bvh, points, OVERLAP_RANGE)
    side = ((points - loc) * nrm).sum(1)
    under = (idx >= 0) & ((nrm * normals).sum(1) > 0.5) & (side < -0.0005) & (np.abs(side) >= 0.9 * dist)
    return under.mean() if len(points) else 0.0


def source_layers(context, objs, body):
    """Overlap relations and median gap to the source body for clothing as currently worn."""
    dg = context.evaluated_depsgraph_get()
    shapes = {o.name: mu.evaluated_mesh_world(o, dg)[:2] for o in objs}
    gaps = {}
    if body is not None:
        bv, bt, _ = mu.evaluated_mesh_world(body, dg)
        bvh = mu.build_bvh(bv, bt)
        for name, (v, _t) in shapes.items():
            loc, nrm, idx, _d = mu.nearest_on(bvh, v, 0.1)
            sd = ((v - loc) * nrm).sum(1)[idx >= 0]
            gaps[name] = float(np.median(sd)) if len(sd) else 0.1
    return relations(shapes), gaps


def relations(shapes):
    """shapes: {name: (verts, tris)} as worn on the source.

    Returns {(a, b): share of a under b}.
    """
    shells = {n: (v, t, mu.build_bvh(v, t)) for n, (v, t) in shapes.items()}
    normals = {n: mu.vertex_normals(v, t) for n, (v, t) in shapes.items()}
    rel = {}
    for a in shapes:
        for b in shapes:
            if a != b:
                rel[(a, b)] = _under_share(shapes[a][0], normals[a], shells[b])
    return rel


def is_under(rel, a, b):
    return rel[(a, b)] > rel[(b, a)] + MARGIN


def sort_order(names, rel, gaps):
    """Innermost first. Every "a is under b" found on the source is respected; where the
    source doesn't tell (items that barely touch), the current list order is kept.
    A cycle is broken at the item closest to the body."""
    below = {a: {b for b in names if b != a and is_under(rel, b, a)} for a in names}
    order, left = [], list(names)
    while left:
        ready = [a for a in left if not (below[a] & set(left))]
        nxt = ready[0] if ready else min(left, key=lambda a: gaps.get(a, 0.0))
        order.append(nxt)
        left.remove(nxt)
    return order


def conflicts(names, rel):
    """Pairs listed in the wrong order: (lower item listed later, upper item listed earlier)."""
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if is_under(rel, b, a):
                out.append((b, a))
    return out
