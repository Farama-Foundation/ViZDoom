"""
Geometry module for DoomGen: exact area shapes, their arrangement and corridor routing.
"""

from doomgen.geometry.shapes import as_polygon, circle, rect, regular_polygon, ring
from doomgen.geometry.arrangement import VOID, Edge, PolygonLayout
from doomgen.geometry.routing import CorridorRoute, route_corridor

__all__ = [
    "as_polygon", "circle", "rect", "regular_polygon", "ring",
    "VOID", "Edge", "PolygonLayout",
    "CorridorRoute", "route_corridor",
]
