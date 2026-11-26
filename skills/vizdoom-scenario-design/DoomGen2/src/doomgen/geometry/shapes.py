"""Exact shape helpers for area outlines (rectangles, circles, rings, polygons)."""

from __future__ import annotations

import math
from typing import Union

import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

Shape = Union[Polygon, MultiPolygon, tuple, list]


def _polygonal_parts(geom: BaseGeometry) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        parts: list[Polygon] = []
        for g in geom.geoms:
            parts.extend(_polygonal_parts(g))
        return parts
    return []


def polygonal(geom: BaseGeometry, min_area: float = 0.0) -> Polygon | MultiPolygon:
    """Keep only the polygonal parts of `geom` larger than `min_area`."""
    parts = [p for p in _polygonal_parts(geom) if p.area > min_area]
    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def snap(geom: BaseGeometry, grid_size: float = 1.0, min_area: float = 0.0) -> Polygon | MultiPolygon:
    """Snap a polygonal geometry to the integer grid and drop degenerate parts."""
    if geom is None or geom.is_empty:
        return Polygon()
    snapped = shapely.set_precision(geom, grid_size=grid_size)
    if not snapped.is_valid:
        snapped = shapely.make_valid(snapped)
    return polygonal(snapped, min_area=min_area)


def rect(cx: float, cy: float, w: float, h: float) -> Polygon:
    """Axis-aligned rectangle centred on (cx, cy)."""
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def circle_segments(r: float, min_edge: float = 16.0, max_segments: int = 64) -> int:
    """Segment count for a circle so edges stay at least `min_edge` units long."""
    if r <= 0:
        return 8
    n = int(math.floor(2 * math.pi * r / max(min_edge, 1.0)))
    return max(8, min(max_segments, n))


def _circle_coords(cx: float, cy: float, r: float, segments: int, rotation: float = 0.0):
    return [
        (cx + r * math.cos(rotation + 2 * math.pi * i / segments),
         cy + r * math.sin(rotation + 2 * math.pi * i / segments))
        for i in range(segments)
    ]


def circle(cx: float, cy: float, r: float, segments: int | None = None, min_edge: float = 16.0) -> Polygon:
    """Regular polygon approximating a circle with edges of at least `min_edge` units."""
    n = segments if segments is not None else circle_segments(r, min_edge)
    return Polygon(_circle_coords(cx, cy, r, max(3, int(n))))


def ring(cx: float, cy: float, r_inner: float, r_outer: float, segments: int | None = None,
         min_edge: float = 16.0) -> Polygon:
    """Annulus between `r_inner` and `r_outer`."""
    n_out = segments if segments is not None else circle_segments(r_outer, min_edge)
    n_in = segments if segments is not None else circle_segments(r_inner, min_edge)
    outer = _circle_coords(cx, cy, r_outer, max(3, int(n_out)))
    inner = _circle_coords(cx, cy, r_inner, max(3, int(n_in)))
    return Polygon(outer, [inner[::-1]])


def regular_polygon(cx: float, cy: float, r: float, sides: int, rotation_deg: float = 0.0) -> Polygon:
    """Regular polygon with `sides` corners on a circle of radius `r`."""
    return Polygon(_circle_coords(cx, cy, r, max(3, int(sides)), math.radians(rotation_deg)))


def as_polygon(shape: Shape, snap_to_grid: bool = True) -> Polygon | MultiPolygon:
    """Convert a `(cx, cy, w, h)` tuple or shapely polygon into a grid-snapped polygon."""
    if isinstance(shape, (tuple, list)):
        cx, cy, w, h = shape
        poly: BaseGeometry = rect(cx, cy, w, h)
    elif isinstance(shape, BaseGeometry):
        poly = shape
    else:
        raise TypeError(f"Unsupported shape type: {type(shape)!r}")
    if snap_to_grid:
        return snap(poly)
    return polygonal(poly)
