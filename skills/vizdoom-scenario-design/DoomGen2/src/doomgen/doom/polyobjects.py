"""Helpers for creating Hexen/ZDoom polyobject primitives in DoomMapData."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from doomgen.doom.specials import LinedefFlags, PolyobjAction
from doomgen.doom.wad import DoomMapData


POLYOBJ_ANCHOR_THING = 9300
POLYOBJ_START_SPOT_THING = 9301
POLYOBJ_START_SPOT_CRUSH_THING = 9302
POLYOBJ_START_SPOT_HURT_THING = 9303


@dataclass
class PolyobjectConfig:
    number: int
    center: Tuple[int, int]
    width: int
    height: int
    start_spot_center: Tuple[int, int] | None = None
    floor_height: int = 0
    ceiling_height: int = 128
    texture: str = "METAL"
    light_level: int = 160
    start_spot_thing_type: int = POLYOBJ_START_SPOT_THING


class PolyobjectBuilder:
    """Builds simple rectangular polyobject geometry and setup markers."""

    def __init__(self, map_data: DoomMapData):
        self.map_data = map_data
        self._control_sector: int | None = None

    def _ensure_control_sector(self) -> int:
        if self._control_sector is None:
            self._control_sector = self.map_data.add_sector(
                floor_height=-4096,
                ceiling_height=-3968,
                floor_texture="F_SKY1",
                ceiling_texture="F_SKY1",
                light_level=0,
                special=0,
                tag=0,
            )
        return self._control_sector

    def add_control_linedef(
        self,
        action: PolyobjAction,
        args: list[int],
        x: int,
        y: int,
        length: int = 32,
    ) -> int:
        """Create an off-map control linedef carrying a polyobject action."""
        sector = self._ensure_control_sector()
        v1 = self.map_data.add_vertex(x, y)
        v2 = self.map_data.add_vertex(x + max(8, length), y)
        side = self.map_data.add_sidedef(sector=sector, middle_texture="-")
        return self.map_data.add_linedef(
            v1,
            v2,
            side,
            -1,
            int(LinedefFlags.NOT_ON_MAP),
            int(action),
            0,
            args[:5],
        )

    def add_move_trigger(
        self,
        number: int,
        speed: int,
        angle: int,
        distance: int,
        x: int,
        y: int,
    ) -> int:
        """Add a repeatable polyobject move action linedef."""
        return self.add_control_linedef(
            PolyobjAction.OR_MOVE,
            [number, speed, angle, distance, 0],
            x,
            y,
        )

    def add_rotate_trigger(
        self,
        number: int,
        speed: int,
        angle: int,
        clockwise: bool,
        x: int,
        y: int,
    ) -> int:
        """Add a repeatable polyobject rotate action linedef."""
        action = PolyobjAction.OR_ROTATE_RIGHT if clockwise else PolyobjAction.OR_ROTATE_LEFT
        return self.add_control_linedef(action, [number, speed, angle, 0, 0], x, y)

    def add_rect(self, config: PolyobjectConfig) -> int:
        """
        Add a rectangular polyobject shell.

        Returns:
            Sector index for the polyobject shell.
        """
        cx, cy = config.center
        sx, sy = config.start_spot_center if config.start_spot_center is not None else (cx, cy)
        half_w = max(8, config.width // 2)
        half_h = max(8, config.height // 2)

        sector = self.map_data.add_sector(
            floor_height=config.floor_height,
            ceiling_height=config.ceiling_height,
            floor_texture=config.texture,
            ceiling_texture=config.texture,
            light_level=config.light_level,
            special=0,
            tag=0,
        )

        v1 = self.map_data.add_vertex(cx - half_w, cy - half_h)
        v2 = self.map_data.add_vertex(cx + half_w, cy - half_h)
        v3 = self.map_data.add_vertex(cx + half_w, cy + half_h)
        v4 = self.map_data.add_vertex(cx - half_w, cy + half_h)

        sd1 = self.map_data.add_sidedef(sector=sector, middle_texture=config.texture)
        sd2 = self.map_data.add_sidedef(sector=sector, middle_texture=config.texture)
        sd3 = self.map_data.add_sidedef(sector=sector, middle_texture=config.texture)
        sd4 = self.map_data.add_sidedef(sector=sector, middle_texture=config.texture)

        self.map_data.add_linedef(v1, v2, sd1, -1, int(LinedefFlags.BLOCKING), int(PolyobjAction.START_LINE), 0, [config.number, 0, 0, 0, 0])
        self.map_data.add_linedef(v2, v3, sd2, -1, int(LinedefFlags.BLOCKING), 0, 0)
        self.map_data.add_linedef(v3, v4, sd3, -1, int(LinedefFlags.BLOCKING), 0, 0)
        self.map_data.add_linedef(v4, v1, sd4, -1, int(LinedefFlags.BLOCKING), 0, 0)

        self.map_data.add_thing(
            x=cx,
            y=cy,
            thing_type=POLYOBJ_ANCHOR_THING,
            angle=config.number,
            flags=7,
            tid=config.number,
            arg0=config.number,
            arg1=0,
            arg2=0,
            arg3=0,
            arg4=0,
        )

        self.map_data.add_thing(
            x=sx,
            y=sy,
            thing_type=config.start_spot_thing_type,
            angle=config.number,
            flags=7,
            tid=config.number,
            arg0=config.number,
            arg1=0,
            arg2=0,
            arg3=0,
            arg4=0,
        )

        return sector
