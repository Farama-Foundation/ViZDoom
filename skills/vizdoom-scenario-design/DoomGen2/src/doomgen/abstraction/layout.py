"""
Abstract layout representation for the DoomGen map generator.
Defines Areas (abstract regions) and their relationships, independent of geometry;
the exact area polygons live in `doomgen.geometry.arrangement.PolygonLayout`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
import uuid

@dataclass
class Transform:
    """Position and orientation of an area."""
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0  # Degrees

@dataclass
class AreaConfig:
    """Doom-specific configuration for an area (sector properties)."""
    floor_height: int = 0
    ceiling_height: int = 128
    floor_texture: str = "FLOOR4_8"
    ceiling_texture: str = "CEIL3_5"
    wall_texture: str = "STARTAN3"
    middle_texture: Optional[str] = None # Explicit middle texture (usually for 2-sided lines like windows)
    lower_texture: Optional[str] = None # Explicit lower texture (overrides logic)
    upper_texture: Optional[str] = None # Explicit upper texture (overrides logic)
    light_level: int = 160
    special: int = 0
    tag: int = 0
    linedef_special: Optional[int] = None # Special for linedefs connecting to this area (e.g. for doors)
    linedef_args: List[int] = field(default_factory=list) # Args for the linedef special
    linedef_flags: Optional[int] = None # Explicit flags for the linedef (e.g. activation flags)
    properties: Dict[str, Any] = field(default_factory=dict) # Catch-all for extra properties

class Area:
    """
    An abstract region in the map.
    """
    def __init__(
        self,
        name: str,
        transform: Transform,
        config: Optional[AreaConfig] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.transform = transform
        self.config = config or AreaConfig()
        self.tags: Set[str] = set()

    def add_tag(self, tag: str):
        self.tags.add(tag)

class ConnectionType(Enum):
    OPEN = "open"       # Shared edge, passable (2-sided linedef)
    SOLID = "solid"     # Shared edge, impassable (1-sided linedef / mid-texture)
    DOOR = "door"       # Door mechanism (requires specific handling)
    WINDOW = "window"   # Window (mid-texture, passable or blocking)

@dataclass
class Connection:
    """A logical link between two areas."""
    area_a_id: str
    area_b_id: str
    type: ConnectionType = ConnectionType.OPEN
    properties: Dict[str, Any] = field(default_factory=dict)

class LayoutGraph:
    """
    The graph of areas and connections.
    """
    def __init__(self):
        self.areas: Dict[str, Area] = {}
        self.connections: List[Connection] = []

    def add_area(self, area: Area) -> str:
        self.areas[area.id] = area
        return area.id

    def get_area(self, area_id: str) -> Optional[Area]:
        return self.areas.get(area_id)

    def connect(self, id_a: str, id_b: str, type: ConnectionType = ConnectionType.OPEN, **kwargs):
        self.connections.append(Connection(id_a, id_b, type, properties=kwargs))
