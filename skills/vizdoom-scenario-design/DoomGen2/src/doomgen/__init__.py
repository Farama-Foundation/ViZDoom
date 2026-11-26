"""Procedural Doom map generation: exact polygon walls plus Voronoi-based placement."""

__version__ = "0.4.0"

from doomgen.abstraction.layout import (
    LayoutGraph,
    Area,
    AreaConfig,
    Transform,
    ConnectionType
)

from doomgen.geometry.shapes import rect, circle, ring, regular_polygon, as_polygon
from doomgen.geometry.arrangement import PolygonLayout

from doomgen.doom.wad import WADWriter, DoomMapData, run_zdbsp
from doomgen.doom.translator import MapTranslator
from doomgen.doom.things import ThingType

from doomgen.builder import ProceduralMapBuilder as MapBuilder

__all__ = [
    "LayoutGraph", "Area", "AreaConfig", "Transform", "ConnectionType",
    "rect", "circle", "ring", "regular_polygon", "as_polygon",
    "PolygonLayout",
    "WADWriter", "DoomMapData", "run_zdbsp", "MapTranslator", "ThingType",
    "MapBuilder"
]
