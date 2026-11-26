"""Voronoi-based placement: spots, obstacles, navigation graphs and clearance points.

Walls come from exact polygons (`doomgen.geometry`); this package only decides
where things go inside them, following the sparse generator of
"Beyond Specialization: Robust RL Navigation via Procedural Map Generators".
"""

from doomgen.placement.free_space import free_space
from doomgen.placement.voronoi import clipped_voronoi_cells, lloyd_relax, sample_points
from doomgen.placement.spots import SpotPool, min_pairwise_distance, relaxed_spots
from doomgen.placement.clearance import max_clearance_spots
from doomgen.placement.nav_graph import NavGraph, build_nav_graph, is_navigable, pick_start_goal_pairs
from doomgen.placement.obstacles import (
    ObstaclePlacementError,
    ObstacleSpec,
    make_footprint,
    scatter_obstacles,
)

__all__ = [
    "free_space",
    "clipped_voronoi_cells", "lloyd_relax", "sample_points",
    "SpotPool", "relaxed_spots", "min_pairwise_distance",
    "max_clearance_spots",
    "NavGraph", "build_nav_graph", "is_navigable", "pick_start_goal_pairs",
    "ObstacleSpec", "ObstaclePlacementError", "make_footprint", "scatter_obstacles",
]
