"""High-level API for procedural Doom map generation.

Areas keep their exact shapes (rectangles, circles, rings, any polygon) and
become straight or finely segmented round walls. Voronoi diagrams are only used
by `doomgen.placement` to decide where spawns, pickups and obstacles go.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from doomgen.abstraction.layout import Area, AreaConfig, ConnectionType, LayoutGraph, Transform
from doomgen.doom.things import ThingType
from doomgen.doom.translator import MapTranslator
from doomgen.doom.wad import WADWriter, run_zdbsp
from doomgen.geometry.arrangement import PolygonLayout
from doomgen.geometry.routing import CorridorRoute, route_corridor, slice_route
from doomgen.geometry.shapes import Shape, polygonal
from doomgen.placement.clearance import max_clearance_spots
from doomgen.placement.free_space import free_space
from doomgen.placement.nav_graph import NavGraph, build_nav_graph, pick_start_goal_pairs
from doomgen.placement.obstacles import ObstacleSpec, scatter_obstacles
from doomgen.placement.spots import SpotPool, relaxed_spots, stable_rng
from doomgen.utils.constants import PLAYER_RADIUS
from doomgen.utils.paths import get_bundled_compiler, validate_executable


class ProceduralMapBuilder:
    """Build Doom maps from exact area polygons plus Voronoi-based placement."""
    def __init__(self, bounds: tuple[float, float, float, float] | None = None, seed: int = 42):
        self.graph = LayoutGraph()
        self.seed = seed
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.wad_writer = WADWriter()
        self.map_data = self.wad_writer.create_map("MAP01")
        self.things_to_add: list[dict[str, Any]] = []
        self.name_to_id: dict[str, str] = {}
        self.layout = PolygonLayout(bounds)
        self.routes: dict[str, CorridorRoute] = {}
        self.spot_pool = SpotPool(self.layout, seed, self.rng)

    # ------------------------------------------------------------ area helpers

    def _resolve_id(self, name_or_id: str) -> str:
        return self.name_to_id.get(name_or_id, name_or_id)

    def _register(self, name: str, config: AreaConfig) -> str:
        area = Area(name=name, transform=Transform(0, 0), config=config)
        # Deterministic ids keep line orientation (and WAD bytes) stable per seed.
        area.id = f"{len(self.graph.areas):04d}:{name}"
        area_id = self.graph.add_area(area)
        self.name_to_id[name] = area_id
        return area_id

    def add_area(
        self,
        name: str,
        shape: Shape,
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_texture: str = "FLOOR4_8",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "STARTAN3",
        middle_texture: str | None = None,
        lower_texture: str | None = None,
        upper_texture: str | None = None,
        light_level: int = 160,
        special: int = 0,
        tag: int = 0,
        linedef_special: int | None = None,
        linedef_args: list[int] | None = None,
        linedef_flags: int | None = None,
        mode: str = "claim_void",
        **kwargs
    ) -> str:
        """Add an area with the exact outline `shape`.

        `mode="claim_void"` fills only space no earlier area uses;
        `mode="overwrite"` cuts the shape out of earlier areas (inner sectors,
        doors, pillars). Touching areas stay separated by solid walls until
        they are connected with `connect_adjacent`.
        """
        config = AreaConfig(
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            floor_texture=floor_texture,
            ceiling_texture=ceiling_texture,
            wall_texture=wall_texture,
            middle_texture=middle_texture,
            lower_texture=lower_texture,
            upper_texture=upper_texture,
            light_level=light_level,
            special=special,
            tag=tag,
            linedef_special=linedef_special,
            linedef_args=linedef_args or [],
            linedef_flags=linedef_flags,
            properties=kwargs
        )
        area_id = self._register(name, config)
        stored = self.layout.add(area_id, shape, mode)
        if stored.is_empty:
            print(f"Warning: Area {name} has an empty shape (outside bounds?).")
        return area_id

    def set_void(self, shape: Shape) -> None:
        """Remove the space inside `shape` from every area added so far."""
        self.layout.set_void(shape)

    def get_area_region(self, area_name_or_id: str) -> Polygon | MultiPolygon:
        """The area's effective region after layering (may be empty)."""
        return self.layout.region(self._resolve_id(area_name_or_id))

    def get_area_centroid(self, area_name_or_id: str) -> tuple[float, float] | None:
        """Centroid of the area's region."""
        region = self.get_area_region(area_name_or_id)
        if region.is_empty:
            return None
        c = region.centroid
        return (c.x, c.y)

    def _blockers(self, *exclude: str) -> Polygon | MultiPolygon:
        regions = self.layout.regions()
        others = [r for a, r in regions.items() if a not in exclude]
        return polygonal(unary_union(others)) if others else Polygon()

    # ---------------------------------------------------- corridors and stairs

    @staticmethod
    def _check_width(width: float) -> None:
        if width < 2 * PLAYER_RADIUS:
            raise ValueError(
                f"Corridor width {width} is narrower than a player ({2 * PLAYER_RADIUS} units); "
                "width is measured in map units, not Voronoi cells."
            )

    def _route(self, room_a_name: str, room_b_name: str, width: float) -> tuple[str, str, CorridorRoute | None]:
        id_a = self._resolve_id(room_a_name)
        id_b = self._resolve_id(room_b_name)
        region_a = self.layout.region(id_a)
        region_b = self.layout.region(id_b)
        if region_a.is_empty or region_b.is_empty:
            print(f"Warning: Cannot connect {room_a_name} and {room_b_name}, one has no region.")
            return id_a, id_b, None
        route = route_corridor(region_a, region_b, width, self._blockers(id_a, id_b), self.bounds)
        if route is None:
            print(f"Warning: No corridor route found between {room_a_name} and {room_b_name}.")
        return id_a, id_b, route

    def add_corridor(
        self,
        room_a_name: str,
        room_b_name: str,
        width: int = 128,
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_texture: str = "FLOOR4_8",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "STARTAN3",
        middle_texture: str | None = None,
        lower_texture: str | None = None,
        upper_texture: str | None = None,
        light_level: int = 160,
        special: int = 0,
        tag: int = 0,
        name: str | None = None,
    ) -> str:
        """Connect two areas with a straight (or single L-bend) corridor `width` units wide."""
        self._check_width(width)
        id_a, id_b, route = self._route(room_a_name, room_b_name, width)
        if route is None:
            return ""

        corridor_name = name or f"corridor_{room_a_name}_{room_b_name}"
        config = AreaConfig(
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            floor_texture=floor_texture,
            ceiling_texture=ceiling_texture,
            wall_texture=wall_texture,
            middle_texture=middle_texture,
            lower_texture=lower_texture,
            upper_texture=upper_texture,
            light_level=light_level,
            special=special,
            tag=tag
        )
        area_id = self._register(corridor_name, config)
        self.layout.add(area_id, route.polygon, "claim_void")
        self.routes[area_id] = route

        self.graph.connect(id_a, area_id, ConnectionType.OPEN)
        self.graph.connect(area_id, id_b, ConnectionType.OPEN)
        return area_id

    def add_stairs(
        self,
        room_a_name: str,
        room_b_name: str,
        num_steps: int = 4,
        width: int = 128,
        floor_texture: str = "STEP1",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "BROWN1",
        riser_texture: str = "STEP1"
    ) -> list[str]:
        """Connect two areas with a corridor sliced into `num_steps` step sectors."""
        self._check_width(width)
        id_a, id_b, route = self._route(room_a_name, room_b_name, width)
        if route is None:
            return []

        area_a = self.graph.get_area(id_a)
        area_b = self.graph.get_area(id_b)
        h_start = area_a.config.floor_height
        h_end = area_b.config.floor_height

        pieces = [p for p in slice_route(route, self.layout.region(id_a), num_steps,
                                         self.layout.region(id_b)) if not p.is_empty]
        if not pieces:
            print(f"Warning: Could not slice stairs between {room_a_name} and {room_b_name}.")
            return []
        num_steps = len(pieces)

        step_ids = []
        for k, piece in enumerate(pieces):
            t = (k + 1) / (num_steps + 1)
            h = int(round(h_start + (h_end - h_start) * t))
            config = AreaConfig(
                floor_height=h,
                ceiling_height=max(area_a.config.ceiling_height, area_b.config.ceiling_height),
                floor_texture=floor_texture,
                ceiling_texture=ceiling_texture,
                wall_texture=wall_texture,
                lower_texture=riser_texture,
                upper_texture=riser_texture
            )
            step_id = self._register(f"stairs_{room_a_name}_{room_b_name}_step_{k+1}", config)
            self.layout.add(step_id, piece, "claim_void")
            step_ids.append(step_id)

        self.graph.connect(id_a, step_ids[0], ConnectionType.OPEN)
        for k in range(len(step_ids) - 1):
            self.graph.connect(step_ids[k], step_ids[k + 1], ConnectionType.OPEN)
        self.graph.connect(step_ids[-1], id_b, ConnectionType.OPEN)
        return step_ids

    # -------------------------------------------------------------- adjacency

    @staticmethod
    def _longest_straight_run(boundary) -> LineString | None:
        if boundary.is_empty:
            return None
        lines = list(boundary.geoms) if isinstance(boundary, MultiLineString) else [boundary]
        best: LineString | None = None
        for line in lines:
            coords = list(line.coords)
            run = [coords[0]]
            for p in coords[1:]:
                if len(run) >= 2:
                    (x0, y0), (x1, y1) = run[0], run[-1]
                    cross = (x1 - x0) * (p[1] - y0) - (y1 - y0) * (p[0] - x0)
                    if abs(cross) > 1e-6 * max(1.0, math.hypot(x1 - x0, y1 - y0) ** 2):
                        seg = LineString([run[0], run[-1]])
                        if best is None or seg.length > best.length:
                            best = seg
                        run = [run[-1]]
                run.append(p)
            seg = LineString([run[0], run[-1]])
            if best is None or seg.length > best.length:
                best = seg
        return best

    def add_boundary(
        self,
        room_a_name: str,
        room_b_name: str,
        name: str = "boundary",
        floor_height: int | None = None,
        ceiling_height: int | None = None,
        floor_texture: str = "FLOOR4_8",
        ceiling_texture: str = "CEIL3_5",
        wall_texture: str = "BIGDOOR2",
        lower_texture: str | None = None,
        upper_texture: str | None = None,
        light_level: int = 160,
        special: int = 0,
        tag: int = 0,
        connection_type: ConnectionType = ConnectionType.OPEN,
        linedef_special: int | None = None,
        linedef_args: list[int] | None = None,
        linedef_flags: int | None = None,
        door_width: float | None = None,
        door_depth: float = 16,
    ) -> str:
        """Create a rectangular door/boundary sector straddling the edge shared by two areas.

        The rectangle is centred on the longest straight piece of the shared
        edge; `door_width` defaults to 128 units (or the whole piece if shorter).
        """
        id_a = self._resolve_id(room_a_name)
        id_b = self._resolve_id(room_b_name)
        run = self._longest_straight_run(self.layout.shared_boundary(id_a, id_b))
        if run is None or run.length < 1:
            print(f"Warning: No boundary found between {room_a_name} and {room_b_name}.")
            return ""

        (x1, y1), (x2, y2) = run.coords
        length = run.length
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        nx, ny = -uy, ux
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = min(door_width if door_width is not None else 128, length) / 2
        half_d = max(door_depth, 2) / 2
        door_poly = Polygon([
            (mx - ux * half_w - nx * half_d, my - uy * half_w - ny * half_d),
            (mx + ux * half_w - nx * half_d, my + uy * half_w - ny * half_d),
            (mx + ux * half_w + nx * half_d, my + uy * half_w + ny * half_d),
            (mx - ux * half_w + nx * half_d, my - uy * half_w + ny * half_d),
        ])

        area_a = self.graph.get_area(id_a)
        config = AreaConfig(
            floor_height=floor_height if floor_height is not None else area_a.config.floor_height,
            ceiling_height=ceiling_height if ceiling_height is not None else area_a.config.ceiling_height,
            floor_texture=floor_texture,
            ceiling_texture=ceiling_texture,
            wall_texture=wall_texture,
            lower_texture=lower_texture,
            upper_texture=upper_texture,
            light_level=light_level,
            special=special,
            tag=tag,
            linedef_special=linedef_special,
            linedef_args=linedef_args or [],
            linedef_flags=linedef_flags
        )
        area_id = self._register(name, config)
        self.layout.add(area_id, door_poly, "overwrite")

        if not self.are_adjacent(id_a, id_b):
            pair = {id_a, id_b}
            self.graph.connections = [
                c for c in self.graph.connections if {c.area_a_id, c.area_b_id} != pair
            ]
        self.graph.connect(id_a, area_id, connection_type)
        self.graph.connect(area_id, id_b, connection_type)
        return area_id

    def _shapes_may_touch(self, id_a: str, id_b: str) -> bool:
        shapes_a = [s.polygon for s in self.layout.shapes if s.area_id == id_a]
        shapes_b = [s.polygon for s in self.layout.shapes if s.area_id == id_b]
        return any(pa.intersects(pb) for pa in shapes_a for pb in shapes_b)

    def are_adjacent(self, room_a_name: str, room_b_name: str) -> bool:
        """True if the two areas currently share at least one boundary segment."""
        id_a = self._resolve_id(room_a_name)
        id_b = self._resolve_id(room_b_name)
        if not self._shapes_may_touch(id_a, id_b):
            return False
        return tuple(sorted((id_a, id_b))) in self.layout.touching_pairs()

    def connect_adjacent(self, room_a_name: str, room_b_name: str,
                         type: ConnectionType = ConnectionType.OPEN,
                         silent: bool = False, **kwargs) -> bool:
        """Connect two areas if they share an edge (two-sided portal unless SOLID)."""
        id_a = self._resolve_id(room_a_name)
        id_b = self._resolve_id(room_b_name)
        if self.are_adjacent(id_a, id_b):
            self.graph.connect(id_a, id_b, type, **kwargs)
            return True
        if not silent:
            print(f"Warning: {room_a_name} and {room_b_name} are not adjacent.")
        return False

    def auto_connect_touching(self, type: ConnectionType = ConnectionType.OPEN) -> int:
        """Connect every touching pair of areas that has no connection yet."""
        existing = {frozenset((c.area_a_id, c.area_b_id)) for c in self.graph.connections}
        count = 0
        for a, b in sorted(self.layout.touching_pairs()):
            if frozenset((a, b)) in existing:
                continue
            self.graph.connect(a, b, type)
            count += 1
        return count

    # ---------------------------------------------------------------- things

    def add_thing(self, thing_type: str, x: float, y: float, **kwargs):
        """Adds a thing to the map."""
        self.things_to_add.append({
            "type": thing_type,
            "x": x,
            "y": y,
            **kwargs
        })

    # ------------------------------------------------------------- placement

    def get_random_point_in_area(self, area_name_or_id: str, safe: bool = True,
                                 min_distance: float | None = None,
                                 clearance: float | None = None,
                                 spacing: float = 64) -> tuple[float, float] | None:
        """Draw a Lloyd-relaxed (centroidal Voronoi) spot from the area.

        With `safe` the spot keeps `PLAYER_RADIUS` (or `clearance`) from walls.
        Spots do not repeat until the area's pool is used up; `min_distance`
        prefers spots far from every earlier draw.
        """
        area_id = self._resolve_id(area_name_or_id)
        if clearance is None:
            clearance = PLAYER_RADIUS if safe else 0
        return self.spot_pool.draw(area_id, clearance, spacing, min_distance=min_distance)

    def get_spots(self, area_name_or_id: str, count: int | None = None, spacing: float = 64,
                  clearance: float = PLAYER_RADIUS, obstacles=()) -> list[tuple[float, float]]:
        """Centroidal-Voronoi spots: `count` of them, or one per `spacing`-sized cell."""
        area_id = self._resolve_id(area_name_or_id)
        region = self.layout.region(area_id)
        return relaxed_spots(region, count=count, spacing=spacing, clearance=clearance,
                             obstacles=obstacles,
                             rng=stable_rng(self.seed, "spots", area_id, count, spacing, clearance))

    def clearance_spots(self, area_name_or_id: str, n: int, min_clearance: float = 32,
                        min_separation: float = 64) -> list[tuple[float, float, float]]:
        """Medial-axis points `(x, y, clearance)` ranked by distance to the nearest wall."""
        region = self.get_area_region(area_name_or_id)
        return max_clearance_spots(region, n, min_clearance=min_clearance,
                                   min_separation=min_separation)

    def scatter_obstacles(
        self,
        area_name_or_id: str,
        count: int,
        *,
        footprint: tuple[float, float] = (64, 64),
        spacing: float = 64,
        kinds: tuple[str, ...] = ("box",),
        rotate: bool = False,
        clearance: float = PLAYER_RADIUS,
        keep_clear: tuple = (),
        create_areas: bool = False,
        name_prefix: str = "Obstacle",
        connection_type: ConnectionType = ConnectionType.OPEN,
        **area_kwargs,
    ) -> list[ObstacleSpec]:
        """Scatter obstacles one per clipped Voronoi cell, `spacing` units apart.

        With `create_areas`, every obstacle becomes an `overwrite` inner sector
        connected to the host area (a two-sided portal), so it can start flush
        and be raised by ACS later. Any `area_kwargs` value given as a list or
        tuple (e.g. `tag=[11, 12, 13]`) is applied per obstacle.
        """
        host_id = self._resolve_id(area_name_or_id)
        region = self.layout.region(host_id)
        specs = scatter_obstacles(
            region, count, stable_rng(self.seed, "obstacles", host_id, count),
            footprint=footprint, spacing=spacing, kinds=kinds, rotate=rotate,
            clearance=clearance, keep_clear=keep_clear,
        )
        if not create_areas:
            return specs

        for i, spec in enumerate(specs):
            kwargs = {k: (v[i % len(v)] if isinstance(v, (list, tuple)) else v)
                      for k, v in area_kwargs.items()}
            spec.area_id = self.add_area(f"{name_prefix}_{i}", spec.polygon, mode="overwrite", **kwargs)
        touching = self.layout.touching_pairs()
        for spec in specs:
            if tuple(sorted((host_id, spec.area_id))) in touching:
                self.graph.connect(host_id, spec.area_id, connection_type)
        return specs

    def _navigable_region(self, names) -> Polygon | MultiPolygon:
        if isinstance(names, str):
            names = [names]
        regions = [self.get_area_region(n) for n in names]
        return polygonal(unary_union([r for r in regions if not r.is_empty]))

    def nav_graph(self, names, clearance: float = PLAYER_RADIUS, spacing: float = 64,
                  obstacles=()) -> NavGraph:
        """Delaunay + MST navigation graph over relaxed spots in the union of `names`.

        `obstacles` (polygons or `ObstacleSpec`) are treated as solid, e.g. every
        raisable cover block when checking the worst case.
        """
        region = self._navigable_region(names)
        free = free_space(region, clearance, obstacles)
        key = names if isinstance(names, str) else "|".join(map(str, names))
        spots = relaxed_spots(region, spacing=spacing, clearance=clearance, obstacles=obstacles,
                              rng=stable_rng(self.seed, "nav", key, spacing, clearance))
        return build_nav_graph(free, spots)

    def pick_spawn_pairs(self, names, n: int, min_path_distance: float,
                         clearance: float = PLAYER_RADIUS, spacing: float = 64,
                         obstacles=()) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
        """Start/goal pairs at least `min_path_distance` apart along the navigation graph."""
        nav = self.nav_graph(names, clearance=clearance, spacing=spacing, obstacles=obstacles)
        return pick_start_goal_pairs(nav, n, min_path_distance, self.rng)

    # ----------------------------------------------------------------- build

    def build(self, output_path: str, zdbsp_path: str | None = None):
        """Generate the WAD file."""
        regions = self.layout.regions()
        for area_id, area in self.graph.areas.items():
            if area_id not in regions:
                print(f"Warning: Area {area.name} has no space left and emits no sector.")

        use_hexen = self.map_data.behavior is not None
        translator = MapTranslator(self.map_data, use_hexen_format=use_hexen)
        translator.translate(self.layout.edges_for_translation(self.graph), self.graph)

        for thing in self.things_to_add:
            t_type = 2001
            raw_type = thing["type"]

            if isinstance(raw_type, int):
                t_type = raw_type
            elif isinstance(raw_type, str):
                if hasattr(ThingType, raw_type):
                    t_type = getattr(ThingType, raw_type).value
                elif raw_type.isdigit():
                    t_type = int(raw_type)
                else:
                    print(f"Warning: Unknown thing type '{raw_type}', defaulting to Imp (2001)")

            x = thing["x"]
            y = thing["y"]
            angle = thing.get("angle", 0)
            flags = thing.get("flags", 7)

            extra_args = {k: v for k, v in thing.items() if k not in ["type", "x", "y", "angle", "flags"]}

            self.map_data.add_thing(
                x=x,
                y=y,
                thing_type=t_type,
                angle=angle,
                flags=flags,
                **extra_args
            )

        self.wad_writer.save(output_path)
        print(f"WAD written to {output_path}")

        if zdbsp_path is None:
            zdbsp_path = get_bundled_compiler("zdbsp")

        if zdbsp_path is not None and validate_executable(zdbsp_path):
            found = run_zdbsp(output_path, zdbsp_path=zdbsp_path)
            if found:
                print(f"ZDBSP: nodes built successfully for {output_path}")
            else:
                print(f"Warning: ZDBSP not found at '{zdbsp_path}'.")
        elif zdbsp_path is not None:
            print(f"Warning: ZDBSP not found at '{zdbsp_path}'. "
                  "WAD saved with empty node lumps (ZDoom will rebuild at load time).")
