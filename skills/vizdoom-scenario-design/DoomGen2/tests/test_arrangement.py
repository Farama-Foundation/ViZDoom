"""Tests for the exact-polygon arrangement and its translation to Doom data."""

from collections import Counter

import pytest
from shapely.geometry import MultiPolygon, Point

from doomgen.abstraction.layout import Area, AreaConfig, ConnectionType, LayoutGraph, Transform
from doomgen.doom.specials import LinedefFlags
from doomgen.doom.translator import MapTranslator
from doomgen.doom.wad import DoomMapData
from doomgen.geometry.arrangement import VOID, PolygonLayout
from doomgen.geometry.shapes import as_polygon, circle, rect, regular_polygon, ring


def _area(graph: LayoutGraph, name: str, **cfg) -> str:
    return graph.add_area(Area(name, Transform(), AreaConfig(**cfg)))


def _translate(layout: PolygonLayout, graph: LayoutGraph) -> DoomMapData:
    md = DoomMapData("MAP01")
    MapTranslator(md).translate(layout.edges_for_translation(graph), graph)
    return md


def _assert_valid(md: DoomMapData):
    for ld in md.linedefs:
        v1 = md.vertices[ld["start_vertex"]]
        v2 = md.vertices[ld["end_vertex"]]
        assert (v1["x"], v1["y"]) != (v2["x"], v2["y"]), "zero-length linedef"
    used = Counter()
    for ld in md.linedefs:
        used[ld["start_vertex"]] += 1
        used[ld["end_vertex"]] += 1
    assert all(c >= 2 for c in used.values()), "dangling vertex"
    referenced = {sd["sector"] for sd in md.sidedefs}
    assert referenced == set(range(len(md.sectors))), "unreferenced sector"


class TestShapes:
    def test_tuple_is_rect(self):
        poly = as_polygon((0, 0, 100, 50))
        assert poly.bounds == (-50, -25, 50, 25)

    def test_circle_segments_and_min_edge(self):
        assert len(circle(0, 0, 256, segments=32).exterior.coords) == 33
        auto = circle(0, 0, 64)
        coords = list(auto.exterior.coords)
        edge = Point(coords[0]).distance(Point(coords[1]))
        assert edge >= 16 * 0.95

    def test_ring_has_hole(self):
        r = ring(0, 0, 100, 200, segments=24)
        assert len(r.interiors) == 1
        assert r.contains(Point(150, 0))
        assert not r.contains(Point(0, 0))

    def test_regular_polygon(self):
        hexagon = regular_polygon(0, 0, 100, 6)
        assert len(hexagon.exterior.coords) == 7

    def test_as_polygon_snaps(self):
        poly = as_polygon(circle(0.3, 0.2, 100, segments=16))
        for x, y in poly.exterior.coords:
            assert x == int(x) and y == int(y)


class TestStraightWalls:
    def test_rect_gives_four_blocking_lines(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        room = _area(graph, "room")
        layout.add(room, rect(0, 0, 2000, 2000))
        md = _translate(layout, graph)
        assert len(md.sectors) == 1
        assert len(md.linedefs) == 4
        for ld in md.linedefs:
            assert ld["back_sidedef"] == -1
            assert ld["flags"] & LinedefFlags.BLOCKING
        _assert_valid(md)

    def test_rect_front_side_faces_room(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        room = _area(graph, "room")
        layout.add(room, rect(0, 0, 200, 200))
        for edge in layout.edges_for_translation(graph):
            (x1, y1), (x2, y2) = edge["line"].coords
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            right = Point(mx + dy * 0.01, my - dx * 0.01)
            assert edge["front_area"] == room
            assert edge["back_area"] == VOID
            assert layout.region(room).contains(right)

    def test_circle_gives_n_lines_on_outline(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        room = _area(graph, "round")
        shape = circle(0, 0, 512, segments=32)
        layout.add(room, shape)
        md = _translate(layout, graph)
        assert len(md.linedefs) == 32
        for v in md.vertices:
            assert shape.exterior.distance(Point(v["x"], v["y"])) <= 1.0
        _assert_valid(md)


class TestLayering:
    def test_overwrite_with_connection_is_portal_loop(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        outer = _area(graph, "outer")
        inner = _area(graph, "inner", floor_height=0, wall_texture="BROWN1")
        layout.add(outer, rect(0, 0, 1000, 1000))
        layout.add(inner, rect(0, 0, 128, 128), mode="overwrite")
        graph.connect(outer, inner, ConnectionType.OPEN)
        md = _translate(layout, graph)

        two_sided = [ld for ld in md.linedefs if ld["back_sidedef"] != -1]
        assert len(two_sided) == 4
        for ld in two_sided:
            assert ld["flags"] & LinedefFlags.TWO_SIDED
            for side in (ld["front_sidedef"], ld["back_sidedef"]):
                sd = md.sidedefs[side]
                assert sd["lower_texture"] != "-"
                assert sd["upper_texture"] != "-"
        _assert_valid(md)

    def test_overwrite_without_connection_is_solid_double_lines(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        outer = _area(graph, "outer")
        inner = _area(graph, "inner")
        layout.add(outer, rect(0, 0, 1000, 1000))
        layout.add(inner, rect(0, 0, 128, 128), mode="overwrite")
        edges = layout.edges_for_translation(graph)
        solid = [e for e in edges if e["type"] == "solid"]
        assert len(solid) == 4
        md = _translate(layout, graph)
        # 4 outer walls + 2 one-sided lines per inner edge.
        assert len(md.linedefs) == 12
        assert all(ld["back_sidedef"] == -1 for ld in md.linedefs)

    def test_claim_void_touching_rooms(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        a = _area(graph, "a")
        b = _area(graph, "b")
        layout.add(a, rect(-256, 0, 512, 512))
        layout.add(b, rect(256, 0, 512, 512))
        assert layout.touching_pairs() == {tuple(sorted((a, b)))}
        assert layout.shared_boundary(a, b).length == pytest.approx(512)

        shared = [e for e in layout.edges_for_translation(graph) if e["back_area"] != VOID]
        assert len(shared) == 1 and shared[0]["type"] == "solid"

        graph.connect(a, b, ConnectionType.OPEN)
        shared = [e for e in layout.edges_for_translation(graph) if e["back_area"] != VOID]
        assert len(shared) == 1 and shared[0]["type"] == "portal"
        edge = shared[0]
        (x1, y1), (x2, y2) = edge["line"].coords
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        right = Point(mx + dy * 0.01, my - dx * 0.01)
        assert layout.region(edge["front_area"]).contains(right)

    def test_claim_void_does_not_take_used_space(self):
        layout = PolygonLayout()
        layout.add("a", rect(0, 0, 200, 200))
        layout.add("b", rect(100, 0, 200, 200))
        assert layout.region("a").area == pytest.approx(200 * 200)
        assert layout.region("b").area == pytest.approx(100 * 200)

    def test_claim_void_fully_covered_is_empty(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        a = _area(graph, "a")
        b = _area(graph, "b")
        layout.add(a, rect(0, 0, 400, 400))
        layout.add(b, rect(0, 0, 100, 100))
        assert layout.region(b).is_empty
        md = _translate(layout, graph)
        assert len(md.sectors) == 1

    def test_set_void_cuts_hole(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        a = _area(graph, "a")
        layout.add(a, rect(0, 0, 400, 400))
        layout.set_void(rect(0, 0, 100, 100))
        assert layout.region(a).area == pytest.approx(400 * 400 - 100 * 100)
        md = _translate(layout, graph)
        assert len(md.linedefs) == 8
        _assert_valid(md)

    def test_overwrite_split_gives_multipolygon(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        room = _area(graph, "room")
        wall = _area(graph, "wall")
        layout.add(room, rect(0, 0, 600, 200))
        layout.add(wall, rect(0, 0, 32, 200), mode="overwrite")
        region = layout.region(room)
        assert isinstance(region, MultiPolygon)
        assert len(region.geoms) == 2
        md = _translate(layout, graph)
        assert len(md.sectors) == 2
        _assert_valid(md)

    def test_overlapping_circles_have_no_zero_length_lines(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        hub = _area(graph, "hub")
        ids = [_area(graph, f"p{i}") for i in range(4)]
        layout.add(hub, circle(0, 0, 600))
        for i, pid in enumerate(ids):
            layout.add(pid, circle(-300 + 200 * i, 37 * i, 110, segments=20), mode="overwrite")
            graph.connect(hub, pid, ConnectionType.OPEN)
        md = _translate(layout, graph)
        _assert_valid(md)

    def test_long_walls_are_split_for_zdbsp(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        sea = _area(graph, "sea")
        layout.add(sea, rect(0, 0, 40000, 512))
        edges = layout.edges_for_translation(graph)
        assert all(e["line"].length <= 8192 for e in edges)
        md = _translate(layout, graph)
        _assert_valid(md)

    def test_collinear_pieces_are_merged(self):
        graph, layout = LayoutGraph(), PolygonLayout()
        a = _area(graph, "a")
        b = _area(graph, "b")
        c = _area(graph, "c")
        layout.add(a, rect(0, 0, 1000, 200))
        layout.add(b, rect(0, 200, 200, 200))
        layout.add(c, rect(-400, 200, 200, 200))
        edges = layout.edges_for_translation(graph)
        # The bottom wall of `a` stays one line even though the top is split.
        bottom = [e for e in edges if e["front_area"] == a and e["back_area"] == VOID
                  and all(abs(y + 100) < 1e-6 for _, y in e["line"].coords)]
        assert len(bottom) == 1
        assert bottom[0]["line"].length == pytest.approx(1000)
