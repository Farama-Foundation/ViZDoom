"""Translate classified layout edges into Doom map data."""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
from shapely.geometry import LineString

from doomgen.doom.wad import DoomMapData
from doomgen.abstraction.layout import LayoutGraph, Area, ConnectionType
from doomgen.doom.specials import LinedefFlags

class MapTranslator:
    """Translate polygon-layout edges into sectors, sidedefs, and linedefs."""
    def __init__(self, map_data: DoomMapData, use_hexen_format: bool = False):
        self.map_data = map_data
        self.use_hexen_format = use_hexen_format
        self.area_to_sector_idx: Dict[str, int] = {}

    def translate(self, edges: Iterable[dict], graph: LayoutGraph):
        """Populate `map_data` from edge dicts and the area graph.

        Each edge has `line` (front side on the right), `front_area`,
        `back_area` ("VOID" for outer walls), `type` (void_wall/solid/portal)
        and `connection`. Areas that no edge references get no sector.
        """
        ridges = list(edges)
        referenced = set()
        for ridge in ridges:
            referenced.add(ridge["front_area"])
            referenced.add(ridge["back_area"])

        for area_id, area in graph.areas.items():
            if area_id not in referenced:
                continue
            sector_idx = self.map_data.add_sector(
                floor_height=area.config.floor_height,
                ceiling_height=area.config.ceiling_height,
                floor_texture=area.config.floor_texture,
                ceiling_texture=area.config.ceiling_texture,
                light_level=area.config.light_level,
                special=area.config.special,
                tag=area.config.tag,
                **area.config.properties
            )
            self.area_to_sector_idx[area_id] = sector_idx

        for ridge in ridges:
            line_geom: LineString = ridge["line"]
            front_area = ridge["front_area"]
            back_area = ridge["back_area"]
            rtype = ridge["type"]

            x1, y1 = int(round(line_geom.coords[0][0])), int(round(line_geom.coords[0][1]))
            x2, y2 = int(round(line_geom.coords[1][0])), int(round(line_geom.coords[1][1]))

            if x1 == x2 and y1 == y2:
                continue

            sec_front = self.area_to_sector_idx.get(front_area)
            sec_back = self.area_to_sector_idx.get(back_area)

            v1_idx = self.map_data.add_vertex(x1, y1)
            v2_idx = self.map_data.add_vertex(x2, y2)

            config_front = graph.get_area(front_area).config if front_area != "VOID" else None
            config_back = graph.get_area(back_area).config if back_area != "VOID" else None

            side_front = -1
            side_back = -1

            if rtype == "void_wall":
                if sec_front is not None:
                    side_front = self.map_data.add_sidedef(
                        sector=sec_front,
                        middle_texture=config_front.wall_texture,
                        **config_front.properties
                    )

            elif rtype == "solid":
                if sec_front is not None:
                    side_front = self.map_data.add_sidedef(
                        sector=sec_front,
                        middle_texture=config_front.wall_texture,
                        **config_front.properties
                    )
                    self.map_data.add_linedef(
                        start_vertex=v1_idx,
                        end_vertex=v2_idx,
                        front_sidedef=side_front,
                        back_sidedef=-1,
                        flags=LinedefFlags.BLOCKING
                    )

                if sec_back is not None:
                    side_back = self.map_data.add_sidedef(
                        sector=sec_back,
                        middle_texture=config_back.wall_texture,
                        **config_back.properties
                    )
                    self.map_data.add_linedef(
                        start_vertex=v2_idx,
                        end_vertex=v1_idx,
                        front_sidedef=side_back,
                        back_sidedef=-1,
                        flags=LinedefFlags.BLOCKING
                    )

                continue

            elif rtype == "portal":
                s_front = self.map_data.sectors[sec_front]
                s_back = self.map_data.sectors[sec_back]

                f_front = s_front['floor_height']
                c_front = s_front['ceiling_height']
                f_back = s_back['floor_height']
                c_back = s_back['ceiling_height']

                # Always populate portal upper/lower textures so later dynamic height
                # changes still have wall surfaces to reveal.
                if config_back:
                    lower_front = config_back.wall_texture
                    if config_back.lower_texture:
                        lower_front = config_back.lower_texture
                else:
                    lower_front = "-"

                lower_back = "-"
                upper_front = config_front.wall_texture
                if config_front.upper_texture:
                    upper_front = config_front.upper_texture

                if config_back:
                    upper_back = config_back.wall_texture
                    if config_back.upper_texture:
                        upper_back = config_back.upper_texture
                else:
                    upper_back = "-"

                lower_back = config_front.wall_texture
                if config_front.lower_texture:
                    lower_back = config_front.lower_texture

                side_front = self.map_data.add_sidedef(
                    sector=sec_front,
                    middle_texture=config_front.middle_texture if config_front.middle_texture else "-",
                    lower_texture=lower_front,
                    upper_texture=upper_front,
                    **config_front.properties
                )

                side_back = self.map_data.add_sidedef(
                    sector=sec_back,
                    middle_texture=config_back.middle_texture if config_back.middle_texture else "-",
                    lower_texture=lower_back,
                    upper_texture=upper_back,
                    **config_back.properties
                )

            flags = 0
            special = 0
            tag = 0

            connection = ridge.get("connection")

            if rtype == "portal":
                flags |= LinedefFlags.TWO_SIDED

                mid_tex = "-"
                if connection and connection.type == ConnectionType.WINDOW:
                    mid_tex = connection.properties.get("middle_texture", "MIDGRATE")
                    if connection.properties.get("blocking", True):
                        flags |= LinedefFlags.BLOCKING

                if connection and "linedef_flags" in connection.properties:
                    flags |= connection.properties["linedef_flags"]

                if connection and connection.type == ConnectionType.DOOR:
                    is_front_door = (s_front['ceiling_height'] == s_front['floor_height'])
                    is_back_door = (s_back['ceiling_height'] == s_back['floor_height'])

                    if is_back_door and not is_front_door:
                        if side_front != -1:
                            sd = self.map_data.sidedefs[side_front]
                            if sd['upper_texture'] != "-":
                                sd['upper_texture'] = config_back.wall_texture
                            if sd['lower_texture'] != "-":
                                sd['lower_texture'] = config_back.wall_texture

                    elif is_front_door and not is_back_door:
                        if side_back != -1:
                            sd = self.map_data.sidedefs[side_back]
                            if sd['upper_texture'] != "-":
                                sd['upper_texture'] = config_front.wall_texture
                            if sd['lower_texture'] != "-":
                                sd['lower_texture'] = config_front.wall_texture

                    if is_front_door and not is_back_door:
                        v1_idx, v2_idx = v2_idx, v1_idx
                        side_front, side_back = side_back, side_front
                        if self.use_hexen_format:
                            # Hexen action 1 means Polyobj_StartLine, not a manual door.
                            # When ACS owns the door logic, leave the linedef action at 0.
                            special = 0
                        else:
                            special = 1

                    elif is_back_door and not is_front_door:
                        if self.use_hexen_format:
                            special = 0
                        else:
                            special = 1

                    elif "DOOR" in config_front.wall_texture or "DOOR" in config_back.wall_texture:
                         if self.use_hexen_format:
                             special = 0
                         else:
                             special = 1

                    door_config = None
                    if is_front_door:
                        door_config = config_front
                    elif is_back_door:
                        door_config = config_back

                    if door_config and door_config.linedef_special is not None:
                        special = door_config.linedef_special

                if mid_tex != "-":
                    if side_front != -1:
                        self.map_data.sidedefs[side_front]['middle_texture'] = mid_tex
                    if side_back != -1:
                        self.map_data.sidedefs[side_back]['middle_texture'] = mid_tex

            elif rtype == "void_wall":
                flags |= LinedefFlags.BLOCKING

            args = []

            if connection:
                if "special" in connection.properties:
                    special = connection.properties["special"]
                if "tag" in connection.properties:
                    tag = connection.properties["tag"]
                if "args" in connection.properties:
                    args = connection.properties["args"]
                if "flags" in connection.properties:
                    flags |= connection.properties["flags"]

            if special > 0 and not args:
                door_config = None
                if connection and connection.type == ConnectionType.DOOR:
                    if is_front_door:
                        door_config = config_front
                    elif is_back_door:
                        door_config = config_back

                if door_config:
                    if door_config.linedef_args:
                        args = door_config.linedef_args
                    if door_config.linedef_flags is not None:
                        flags |= door_config.linedef_flags

            self.map_data.add_linedef(
                start_vertex=v1_idx,
                end_vertex=v2_idx,
                front_sidedef=side_front,
                back_sidedef=side_back,
                flags=flags,
                special=special,
                tag=tag,
                args=args
            )

    def add_player_start(self, area_id: str, x: float, y: float, angle: int = 0):
        """Add player start thing."""
        self.map_data.add_thing(
            x=int(x), y=int(y),
            thing_type=1,
            angle=angle,
            flags=7,
        )
