"""Tests for corridor routing, stairs, doors and the polygon-based builder."""

import importlib

import pytest
from shapely.geometry import Point

from doomgen.abstraction.layout import ConnectionType
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.translator import MapTranslator
from doomgen.doom.wad import DoomMapData
from doomgen.geometry.routing import route_corridor
from doomgen.geometry.shapes import as_polygon, circle, rect


def _touch_len(a, b) -> float:
    return a.buffer(0.5).intersection(b.boundary).length


class TestRouteCorridor:
    def test_aligned_rooms_give_straight_route(self):
        a = as_polygon(rect(-600, 0, 400, 400))
        b = as_polygon(rect(600, 0, 400, 400))
        route = route_corridor(a, b, 128)
        assert route is not None and route.kind == "straight"
        free = route.polygon.difference(a.union(b))
        assert free.area == pytest.approx(800 * 128, rel=0.01)
        minx, miny, maxx, maxy = free.bounds
        assert maxy - miny == pytest.approx(128)

    def test_offset_rooms_give_l_route(self):
        a = as_polygon(rect(-800, -800, 300, 300))
        b = as_polygon(rect(800, 800, 300, 300))
        route = route_corridor(a, b, 128)
        assert route is not None and route.kind == "L"
        assert len(route.centerline.coords) == 3
        assert _touch_len(route.polygon.difference(a), a) >= 64
        assert _touch_len(route.polygon.difference(b), b) >= 64

    def test_blocker_triggers_grid_route(self):
        a = as_polygon(rect(-800, 0, 300, 300))
        b = as_polygon(rect(800, 0, 300, 300))
        blocker = as_polygon(rect(0, 0, 300, 1400))
        route = route_corridor(a, b, 128, blockers=blocker)
        assert route is not None and route.kind == "grid"
        assert route.polygon.intersection(blocker).area < 1.0

    def test_no_route_when_fully_blocked(self):
        a = as_polygon(rect(0, 0, 200, 200))
        b = as_polygon(rect(1000, 0, 200, 200))
        wall = as_polygon(rect(0, 0, 1000, 1000)).difference(as_polygon(rect(0, 0, 400, 400)))
        assert route_corridor(a, b, 128, blockers=wall, bounds=(-500, -500, 500, 500)) is None


class TestBuilderCorridors:
    def test_corridor_touches_both_rooms_and_avoids_others(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-800, 0, 400, 400))
        builder.add_area("B", (800, 0, 400, 400))
        builder.add_area("C", (0, 600, 200, 200))
        cid = builder.add_corridor("A", "B", width=128)
        assert cid
        region = builder.get_area_region(cid)
        assert region.intersection(builder.get_area_region("C")).area == 0
        assert builder.are_adjacent("A", cid)
        assert builder.are_adjacent(cid, "B")
        edges = builder.layout.edges_for_translation(builder.graph)
        portals = [e for e in edges if e["type"] == "portal"]
        assert len(portals) == 2
        assert all(e["line"].length == pytest.approx(128) for e in portals)

    def test_corridor_to_round_room(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-800, 0, 400, 400))
        builder.add_area("R", circle(600, 0, 300))
        cid = builder.add_corridor("A", "R", width=96)
        assert cid
        assert builder.are_adjacent(cid, "R")

    def test_width_in_cells_is_rejected(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-800, 0, 400, 400))
        builder.add_area("B", (800, 0, 400, 400))
        with pytest.raises(ValueError):
            builder.add_corridor("A", "B", width=2)

    def test_stairs_step_heights(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("low", (-800, 0, 400, 400), floor_height=0)
        builder.add_area("high", (800, 0, 400, 400), floor_height=96)
        steps = builder.add_stairs("low", "high", num_steps=4, width=128)
        assert len(steps) == 4
        heights = [builder.graph.get_area(s).config.floor_height for s in steps]
        assert heights == sorted(heights)
        assert 0 < heights[0] and heights[-1] < 96
        assert builder.are_adjacent("low", steps[0])
        for s1, s2 in zip(steps, steps[1:]):
            assert builder.are_adjacent(s1, s2)
        assert builder.are_adjacent(steps[-1], "high")
        # Steps are ordered from low to high along the corridor.
        xs = [builder.get_area_centroid(s)[0] for s in steps]
        assert xs == sorted(xs)


class TestDoors:
    def test_add_boundary_creates_door_rectangle(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-256, 0, 512, 512))
        builder.add_area("B", (256, 0, 512, 512))
        door = builder.add_boundary("A", "B", name="door", ceiling_height=0, floor_height=0,
                                    connection_type=ConnectionType.DOOR,
                                    door_width=128, door_depth=16)
        assert door
        region = builder.get_area_region(door)
        minx, miny, maxx, maxy = region.bounds
        assert (minx, maxx) == (-8, 8)
        assert maxy - miny == 128
        remaining = builder.layout.shared_boundary(builder.name_to_id["A"], builder.name_to_id["B"])
        assert remaining.length == pytest.approx(512 - 128)

        md = DoomMapData("MAP01")
        MapTranslator(md).translate(builder.layout.edges_for_translation(builder.graph), builder.graph)
        door_lines = [ld for ld in md.linedefs if ld.get("special") == 1]
        assert len(door_lines) >= 2

    def test_full_width_door_replaces_direct_connection(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("low", (-800, 0, 400, 400))
        builder.add_area("high", (800, 0, 400, 400), floor_height=96)
        steps = builder.add_stairs("low", "high", num_steps=4, width=128)
        low = builder.name_to_id["low"]
        door = builder.add_boundary("low", steps[0], name="door", door_width=128, door_depth=16)
        assert door
        pairs = {frozenset((c.area_a_id, c.area_b_id)) for c in builder.graph.connections}
        assert frozenset((low, steps[0])) not in pairs
        assert frozenset((low, door)) in pairs and frozenset((door, steps[0])) in pairs

    def test_partial_door_keeps_direct_connection(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-256, 0, 512, 512))
        builder.add_area("B", (256, 0, 512, 512))
        builder.connect_adjacent("A", "B")
        builder.add_boundary("A", "B", door_width=128)
        a, b = builder.name_to_id["A"], builder.name_to_id["B"]
        pairs = {frozenset((c.area_a_id, c.area_b_id)) for c in builder.graph.connections}
        assert frozenset((a, b)) in pairs

    def test_default_door_width_is_128(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-256, 0, 512, 512))
        builder.add_area("B", (256, 0, 512, 512))
        door = builder.add_boundary("A", "B")
        minx, miny, maxx, maxy = builder.get_area_region(door).bounds
        assert maxy - miny == 128

    def test_add_boundary_on_non_touching_returns_empty(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-600, 0, 200, 200))
        builder.add_area("B", (600, 0, 200, 200))
        assert builder.add_boundary("A", "B") == ""
        assert builder.connect_adjacent("A", "B", silent=True) is False


class TestBuilderBasics:
    def test_connect_adjacent_and_auto_connect(self):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-100, 0, 200, 200))
        builder.add_area("B", (100, 0, 200, 200))
        builder.add_area("C", (0, 200, 400, 200))
        assert builder.connect_adjacent("A", "B")
        assert builder.auto_connect_touching() == 2

    def test_point_in_area_is_safe(self):
        builder = ProceduralMapBuilder(seed=3)
        builder.add_area("A", (0, 0, 400, 400))
        for _ in range(20):
            x, y = builder.get_random_point_in_area("A")
            assert builder.get_area_region("A").exterior.distance(Point(x, y)) >= 16 - 1e-6

    def test_build_writes_wad(self, tmp_path):
        builder = ProceduralMapBuilder(seed=1)
        builder.add_area("A", (-800, 0, 400, 400))
        builder.add_area("B", (800, 300, 400, 400))
        builder.add_corridor("A", "B", width=128)
        builder.add_thing("PLAYER1_START", -800, 0)
        out = tmp_path / "test.wad"
        builder.build(str(out), zdbsp_path="/nonexistent/zdbsp")
        assert out.exists() and out.stat().st_size > 0


class TestVoronoiWallsRemoved:
    def test_mesh_and_seeds_modules_are_gone(self):
        for mod in ("doomgen.geometry.mesh", "doomgen.geometry.seeds", "doomgen.utils.geometry_utils"):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(mod)

    def test_builder_has_no_mesh(self):
        builder = ProceduralMapBuilder(seed=1)
        assert not hasattr(builder, "mesh")
        assert not hasattr(builder, "seeds")
        with pytest.raises(TypeError):
            ProceduralMapBuilder(num_seeds=100)
