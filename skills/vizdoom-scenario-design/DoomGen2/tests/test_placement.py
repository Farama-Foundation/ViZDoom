"""Tests for Voronoi-based placement (spots, clearance spots, obstacles, nav graph)."""

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from doomgen.builder import ProceduralMapBuilder
from doomgen.geometry.shapes import as_polygon, circle, rect
from doomgen.placement import (
    build_nav_graph,
    clipped_voronoi_cells,
    free_space,
    is_navigable,
    lloyd_relax,
    max_clearance_spots,
    min_pairwise_distance,
    pick_start_goal_pairs,
    relaxed_spots,
    sample_points,
    scatter_obstacles,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestVoronoi:
    def test_clipped_cells_tile_region(self):
        region = as_polygon(rect(0, 0, 1000, 600))
        pts = sample_points(region, 30, _rng())
        cells = clipped_voronoi_cells(pts, region)
        assert len(cells) == 30
        assert sum(c.area for c in cells) == pytest.approx(region.area, rel=1e-6)
        for (x, y), cell in zip(pts, cells):
            assert cell.covers(Point(x, y))

    def test_lloyd_spreads_points(self):
        region = as_polygon(rect(0, 0, 1000, 1000))
        raw = sample_points(region, 50, _rng(1))
        relaxed = lloyd_relax(region, 50, _rng(1), iterations=8)
        assert min_pairwise_distance(relaxed) > 2 * min_pairwise_distance(raw)
        assert min_pairwise_distance(relaxed) > 60


class TestRelaxedSpots:
    def test_spots_inside_with_clearance(self):
        region = as_polygon(circle(0, 0, 400))
        spots = relaxed_spots(region, count=40, clearance=16, rng=_rng())
        assert len(spots) == 40
        for x, y in spots:
            p = Point(x, y)
            assert region.contains(p)
            assert region.exterior.distance(p) >= 16 - 1e-6

    def test_spacing_controls_count_and_spread(self):
        region = as_polygon(rect(0, 0, 1024, 1024))
        spots = relaxed_spots(region, spacing=128, clearance=16, rng=_rng())
        free = free_space(region, 16)
        assert len(spots) == round(free.area / 128 ** 2)
        assert min_pairwise_distance(spots) > 128 * 0.6

    def test_spots_are_deterministic(self):
        region = as_polygon(rect(0, 0, 800, 800))
        a = relaxed_spots(region, count=20, rng=_rng(7))
        b = relaxed_spots(region, count=20, rng=_rng(7))
        assert a == b

    def test_spots_avoid_obstacles(self):
        region = as_polygon(rect(0, 0, 800, 800))
        obstacle = as_polygon(rect(0, 0, 200, 200))
        spots = relaxed_spots(region, count=30, clearance=16, obstacles=[obstacle], rng=_rng())
        for x, y in spots:
            assert obstacle.distance(Point(x, y)) >= 16 - 1e-6

    def test_tiny_area_falls_back_to_representative_point(self):
        region = as_polygon(rect(0, 0, 20, 20))
        spots = relaxed_spots(region, count=5, clearance=16, rng=_rng())
        assert len(spots) == 1
        assert region.contains(Point(spots[0]))


class TestBuilderSpots:
    def test_random_point_is_deterministic_per_seed(self):
        def draw(seed):
            b = ProceduralMapBuilder(seed=seed)
            b.add_area("A", (0, 0, 1000, 1000))
            return [b.get_random_point_in_area("A") for _ in range(5)]
        assert draw(3) == draw(3)
        assert draw(3) != draw(4)

    def test_random_point_does_not_repeat_until_exhausted(self):
        b = ProceduralMapBuilder(seed=1)
        b.add_area("A", (0, 0, 256, 256))
        pool = b.spot_pool.spots(b.name_to_id["A"], 16, 64)
        draws = [b.get_random_point_in_area("A") for _ in range(len(pool))]
        assert len(set(draws)) == len(pool)
        # The pool refills once every spot was used.
        assert b.get_random_point_in_area("A") in pool

    def test_min_distance_is_respected(self):
        b = ProceduralMapBuilder(seed=2)
        b.add_area("A", (0, 0, 2000, 2000))
        first = b.get_random_point_in_area("A")
        second = b.get_random_point_in_area("A", min_distance=1000)
        assert Point(first).distance(Point(second)) >= 1000

    def test_pool_invalidates_when_geometry_changes(self):
        b = ProceduralMapBuilder(seed=2)
        b.add_area("A", (0, 0, 1000, 1000))
        b.get_random_point_in_area("A")
        b.add_area("pillar", (0, 0, 400, 400), mode="overwrite")
        pillar = b.get_area_region("pillar")
        for _ in range(30):
            assert not pillar.contains(Point(b.get_random_point_in_area("A")))

    def test_get_spots(self):
        b = ProceduralMapBuilder(seed=2)
        b.add_area("A", circle(0, 0, 500))
        spots = b.get_spots("A", count=12, clearance=32)
        assert len(spots) == 12
        region = b.get_area_region("A")
        assert all(region.exterior.distance(Point(s)) >= 32 - 1e-6 for s in spots)

    def test_empty_area_returns_none(self):
        b = ProceduralMapBuilder(seed=2)
        b.add_area("A", (0, 0, 400, 400))
        b.add_area("B", (0, 0, 100, 100))
        assert b.get_random_point_in_area("B") is None


class TestClearanceSpots:
    def test_clearance_equals_wall_distance(self):
        region = as_polygon(rect(0, 0, 800, 400))
        spots = max_clearance_spots(region, 5, min_clearance=32, min_separation=64)
        assert spots
        for x, y, c in spots:
            assert c == pytest.approx(region.boundary.distance(Point(x, y)))
            assert c >= 32
        assert spots[0][2] == pytest.approx(200, abs=2)

    def test_corridor_spots_lie_on_centreline(self):
        corridor = as_polygon(rect(0, 0, 1200, 96))
        spots = max_clearance_spots(corridor, 6, min_clearance=32, min_separation=128)
        assert len(spots) >= 4
        for x, y, c in spots:
            assert abs(y) <= 1.0
            assert c == pytest.approx(48, abs=1.0)

    def test_separation(self):
        region = as_polygon(circle(0, 0, 600))
        spots = max_clearance_spots(region, 10, min_clearance=32, min_separation=150)
        pts = [(x, y) for x, y, _ in spots]
        assert min_pairwise_distance(pts) >= 150

    def test_tiny_area_still_gets_a_spot(self):
        region = as_polygon(rect(0, 0, 40, 40))
        spots = max_clearance_spots(region, 3, min_clearance=32)
        assert len(spots) == 1
        assert region.contains(Point(spots[0][0], spots[0][1]))

    def test_builder_wrapper(self):
        b = ProceduralMapBuilder(seed=1)
        b.add_area("A", (0, 0, 1000, 1000))
        b.add_area("P", (0, 0, 200, 200), mode="overwrite")
        spots = b.clearance_spots("A", 4, min_clearance=64)
        assert len(spots) == 4
        pillar = b.get_area_region("P")
        assert all(pillar.distance(Point(x, y)) >= 64 - 1e-6 for x, y, _ in spots)


class TestNavGraph:
    def test_edges_inside_free_space_and_mst_spans(self):
        import networkx as nx
        from shapely.geometry import LineString

        from doomgen.placement import build_nav_graph, is_navigable

        # U-shaped room: edges must not cut across the missing middle.
        region = as_polygon(rect(0, 0, 1200, 1200)).difference(as_polygon(rect(0, 200, 400, 1000)))
        free = free_space(region, 16)
        spots = relaxed_spots(region, spacing=96, clearance=16, rng=_rng())
        nav = build_nav_graph(free, spots)
        assert nav.graph.number_of_edges() > 0
        for u, v in nav.graph.edges():
            assert free.buffer(0.5).covers(LineString([nav.points[u], nav.points[v]]))
        assert is_navigable(nav)
        assert nx.is_tree(nav.mst)
        assert nav.mst.number_of_nodes() == len(spots)

    def test_disconnected_free_space_is_not_navigable(self):
        from doomgen.placement import build_nav_graph, is_navigable

        region = as_polygon(rect(-400, 0, 300, 300)).union(as_polygon(rect(400, 0, 300, 300)))
        free = free_space(region, 16)
        nav = build_nav_graph(free, relaxed_spots(region, spacing=64, rng=_rng()))
        assert not is_navigable(nav)
        assert is_navigable(nav, required=[(-400, 0), (-350, 50)])
        assert not is_navigable(nav, required=[(-400, 0), (400, 0)])

    def test_pick_pairs_respects_path_distance(self):
        from doomgen.placement import build_nav_graph, pick_start_goal_pairs

        region = as_polygon(rect(0, 0, 1200, 1200)).difference(as_polygon(rect(0, 200, 400, 1000)))
        free = free_space(region, 16)
        nav = build_nav_graph(free, relaxed_spots(region, spacing=96, rng=_rng()))
        pairs = pick_start_goal_pairs(nav, 5, 1200, _rng(3))
        assert len(pairs) == 5
        for start, goal, dist in pairs:
            assert dist >= 1200
            # Path distance exceeds the straight-line distance around the U.
            assert dist >= Point(start).distance(Point(goal)) - 1e-6

    def test_builder_spawn_pairs(self):
        b = ProceduralMapBuilder(seed=4)
        b.add_area("A", (0, 0, 2000, 2000))
        pairs = b.pick_spawn_pairs("A", 3, min_path_distance=1200)
        assert len(pairs) == 3
        region = b.get_area_region("A")
        for s, g, d in pairs:
            assert d >= 1200
            assert region.contains(Point(s)) and region.contains(Point(g))


class TestObstacles:
    def test_spacing_containment_and_navigability(self):
        from doomgen.placement import build_nav_graph, is_navigable, scatter_obstacles

        region = as_polygon(rect(0, 0, 2000, 2000))
        specs = scatter_obstacles(region, 20, _rng(), footprint=(96, 96), spacing=64,
                                  kinds=("box", "circle"), rotate=True)
        assert len(specs) == 20
        polys = [s.polygon for s in specs]
        for p in polys:
            assert region.contains(p)
            assert region.exterior.distance(p) >= 64 - 1.5
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                assert polys[i].distance(polys[j]) >= 64 - 1.5
        free = free_space(region, 16, polys)
        assert free.geom_type == "Polygon"
        nav = build_nav_graph(free, relaxed_spots(region, spacing=96, obstacles=polys, rng=_rng()))
        assert is_navigable(nav)

    def test_too_many_obstacles_warns(self, capsys):
        from doomgen.placement import scatter_obstacles

        region = as_polygon(rect(0, 0, 512, 512))
        specs = scatter_obstacles(region, 60, _rng(), footprint=(64, 64), spacing=64)
        assert len(specs) < 60
        assert "obstacles fit" in capsys.readouterr().out

    def test_keep_clear_points(self):
        from doomgen.placement import scatter_obstacles

        region = as_polygon(rect(0, 0, 1500, 1500))
        keep = [(-600, -600), (600, 600)]
        specs = scatter_obstacles(region, 25, _rng(5), footprint=(96, 96), spacing=64,
                                  keep_clear=keep)
        for s in specs:
            for k in keep:
                assert s.polygon.distance(Point(k)) >= 16

    def test_builder_creates_flush_connected_obstacles(self):
        from doomgen.doom.translator import MapTranslator
        from doomgen.doom.wad import DoomMapData

        b = ProceduralMapBuilder(seed=8)
        b.add_area("arena", (0, 0, 1600, 1600))
        specs = b.scatter_obstacles("arena", 6, footprint=(96, 96), spacing=96,
                                    create_areas=True, tag=[11, 12, 13, 14, 15, 16],
                                    wall_texture="BROWN1")
        assert len(specs) == 6
        tags = sorted(b.graph.get_area(s.area_id).config.tag for s in specs)
        assert tags == [11, 12, 13, 14, 15, 16]
        edges = b.layout.edges_for_translation(b.graph)
        assert not any(e["type"] == "solid" for e in edges)
        md = DoomMapData("MAP01")
        MapTranslator(md).translate(edges, b.graph)
        two_sided = [ld for ld in md.linedefs if ld["back_sidedef"] != -1]
        assert two_sided
        for ld in two_sided:
            for side in (ld["front_sidedef"], ld["back_sidedef"]):
                assert md.sidedefs[side]["lower_texture"] != "-"

    def test_builder_obstacles_are_deterministic(self):
        def run(seed):
            b = ProceduralMapBuilder(seed=seed)
            b.add_area("arena", (0, 0, 1600, 1600))
            return [s.polygon.wkt for s in b.scatter_obstacles("arena", 5, footprint=(64, 64))]
        assert run(1) == run(1)
