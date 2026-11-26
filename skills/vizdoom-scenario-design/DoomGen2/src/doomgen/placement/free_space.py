"""Free floor space: an area region shrunk by a clearance radius, minus obstacles."""

from __future__ import annotations

from typing import Iterable

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from doomgen.geometry.shapes import polygonal


def obstacle_geometry(obstacle):
    """Accept shapely geometries or objects with a `.polygon` attribute (e.g. `ObstacleSpec`)."""
    return getattr(obstacle, "polygon", obstacle)


def free_space(region, clearance: float, obstacles: Iterable = ()) -> Polygon | MultiPolygon:
    """Points at least `clearance` units from every wall of `region` and every obstacle."""
    if region is None or region.is_empty:
        return Polygon()
    free = region.buffer(-clearance, join_style="mitre") if clearance > 0 else region
    obs = [obstacle_geometry(o) for o in obstacles]
    obs = [o for o in obs if o is not None and not o.is_empty]
    if obs:
        blocked = unary_union(obs)
        if clearance > 0:
            blocked = blocked.buffer(clearance, join_style="mitre")
        free = free.difference(blocked)
    return polygonal(free, min_area=1.0)
