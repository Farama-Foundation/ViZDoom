"""Navigation graph over placement spots: Delaunay edges inside free space plus an MST."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import shapely
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import Polygon
from shapely.prepared import prep


@dataclass
class NavGraph:
    points: list[tuple[float, float]]
    graph: nx.Graph
    mst: nx.Graph
    free: object = field(default_factory=Polygon)

    def nearest_node(self, x: float, y: float) -> Optional[int]:
        """Closest node that can be reached from (x, y) in a straight line through free space."""
        if not self.points:
            return None
        pts = np.asarray(self.points)
        order = np.argsort(((pts - (x, y)) ** 2).sum(axis=1))
        tolerant = prep(self.free.buffer(0.5)) if not self.free.is_empty else None
        for i in order[:16]:
            if tolerant is None or tolerant.covers(shapely.LineString([(x, y), tuple(pts[i])])):
                return int(i)
        return int(order[0])

    def path_distance(self, a: int, b: int) -> float:
        try:
            return float(nx.dijkstra_path_length(self.graph, a, b, weight="weight"))
        except nx.NetworkXNoPath:
            return float("inf")


def _candidate_edges(pts: np.ndarray) -> set[tuple[int, int]]:
    n = len(pts)
    if n < 2:
        return set()
    if n == 2:
        return {(0, 1)}
    try:
        tri = Delaunay(pts)
        edges = set()
        for a, b, c in tri.simplices:
            for u, v in ((a, b), (b, c), (a, c)):
                edges.add((int(min(u, v)), int(max(u, v))))
        return edges
    except QhullError:
        # Collinear points: connect consecutive points along the line.
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        return {(int(min(u, v)), int(max(u, v))) for u, v in zip(order, order[1:])}


def build_nav_graph(free, points: Sequence[Sequence[float]]) -> NavGraph:
    """Delaunay graph of `points`, keeping only edges whose segment stays in `free`."""
    pts_list = [(float(p[0]), float(p[1])) for p in points]
    pts = np.asarray(pts_list, dtype=float).reshape(-1, 2)
    graph = nx.Graph()
    for i, (x, y) in enumerate(pts_list):
        graph.add_node(i, pos=(x, y))

    edges = sorted(_candidate_edges(pts))
    if edges and free is not None and not free.is_empty:
        lines = shapely.linestrings([[pts[u], pts[v]] for u, v in edges])
        inside = shapely.covers(free.buffer(0.5), lines)
        for (u, v), ok in zip(edges, inside):
            if ok:
                graph.add_edge(u, v, weight=float(np.hypot(*(pts[u] - pts[v]))))
    mst = nx.minimum_spanning_tree(graph, weight="weight")
    return NavGraph(pts_list, graph, mst, free if free is not None else Polygon())


def is_navigable(nav: NavGraph, required: Iterable[Sequence[float]] = ()) -> bool:
    """True if the graph is connected, or if all `required` points share one component."""
    if nav.graph.number_of_nodes() == 0:
        return False
    required = list(required)
    if not required:
        return nx.is_connected(nav.graph)
    nodes = [nav.nearest_node(float(p[0]), float(p[1])) for p in required]
    comp = nx.node_connected_component(nav.graph, nodes[0])
    return all(n in comp for n in nodes)


def pick_start_goal_pairs(nav: NavGraph, n: int, min_path_distance: float,
                          rng: np.random.Generator) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Up to `n` (start, goal, path_distance) pairs at least `min_path_distance` apart along the graph.

    Each start node is used at most once; distance is the Dijkstra path length.
    """
    nodes = list(nav.graph.nodes())
    if len(nodes) < 2 or n <= 0:
        return []
    order = list(rng.permutation(nodes))
    pairs = []
    for start in order:
        lengths = nx.single_source_dijkstra_path_length(nav.graph, int(start), weight="weight")
        far = sorted(k for k, d in lengths.items() if d >= min_path_distance)
        if not far:
            continue
        goal = int(far[int(rng.integers(len(far)))])
        pairs.append((nav.points[int(start)], nav.points[goal], float(lengths[goal])))
        if len(pairs) >= n:
            break
    return pairs
