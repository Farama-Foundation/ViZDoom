"""Sparse obstacle scattering: one obstacle per clipped Voronoi cell, with guaranteed gaps.

Every cell is clipped to the region shrunk by `spacing / 2` and then inset by
another `spacing / 2`, so any two obstacles (and any obstacle and a wall) are at
least `spacing` apart. Layouts are resampled until the free floor for a player
of radius `clearance` stays connected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon

from doomgen.geometry.shapes import as_polygon, circle, polygonal, rect, regular_polygon
from doomgen.placement.clearance import max_clearance_spots
from doomgen.placement.free_space import free_space
from doomgen.placement.nav_graph import build_nav_graph, is_navigable
from doomgen.placement.spots import relaxed_spots
from doomgen.placement.voronoi import clipped_voronoi_cells, lloyd_relax


@dataclass
class ObstacleSpec:
    kind: str
    polygon: Polygon
    center: tuple[float, float]
    cell: Polygon
    area_id: Optional[str] = None


class ObstaclePlacementError(RuntimeError):
    """Raised when no navigable obstacle layout was found."""


def make_footprint(kind: str, center: tuple[float, float], size: tuple[float, float],
                   angle: float = 0.0) -> Polygon:
    """Box, circle or regular-polygon footprint centred on `center`."""
    cx, cy = center
    w, h = size
    if kind == "circle":
        return circle(cx, cy, min(w, h) / 2)
    if kind.startswith("poly"):
        sides = int(kind[4:] or 6)
        return regular_polygon(cx, cy, min(w, h) / 2, sides, rotation_deg=angle)
    poly = rect(cx, cy, w, h)
    if angle:
        poly = affinity.rotate(poly, angle, origin=(cx, cy))
    return poly


def _n_parts(geom) -> int:
    if geom.is_empty:
        return 0
    return len(getattr(geom, "geoms", [geom]))


def _fit(kind: str, allowed, seed_pt, size, angle, min_scale: float) -> Optional[Polygon]:
    centers = [seed_pt]
    best = max_clearance_spots(allowed, 1, min_clearance=0, sample_step=8)
    if best:
        centers.append((best[0][0], best[0][1]))
    scale = 1.0
    while scale >= min_scale - 1e-9:
        sw, sh = size[0] * scale, size[1] * scale
        for c in centers:
            if not allowed.contains(Point(c)):
                continue
            fp = as_polygon(make_footprint(kind, c, (sw, sh), angle))
            if not fp.is_empty and allowed.buffer(0.5).contains(fp):
                return fp
        scale *= 0.85
    return None


def scatter_obstacles(region, count: int, rng: np.random.Generator, *,
                      footprint: Sequence[float] = (64, 64), spacing: float = 64,
                      kinds: Sequence[str] = ("box",), rotate: bool = False,
                      clearance: float = 16, min_scale: float = 0.5,
                      max_attempts: int = 10, keep_clear: Sequence[Sequence[float]] = ()
                      ) -> list[ObstacleSpec]:
    """Place up to `count` obstacles in `region` (see module docstring).

    `keep_clear` lists points (e.g. spawns) that must stay reachable and
    uncovered. Returns fewer obstacles (with a warning) when they do not fit.
    """
    if region is None or region.is_empty or count <= 0:
        return []
    if spacing < 2 * clearance:
        print(f"Warning: obstacle spacing {spacing} is below the player diameter {2 * clearance}.")

    seed_region = polygonal(region.buffer(-spacing / 2, join_style="mitre"))
    if seed_region.is_empty:
        print("Warning: region too small for any obstacle.")
        return []
    base_parts = _n_parts(free_space(region, clearance))

    for _attempt in range(max_attempts):
        seeds = lloyd_relax(seed_region, count, rng)
        cells = clipped_voronoi_cells(seeds, seed_region)
        specs: list[ObstacleSpec] = []
        for (sx, sy), cell in zip(seeds, cells):
            allowed = polygonal(cell.buffer(-spacing / 2, join_style="mitre"))
            for kx, ky in keep_clear:
                allowed = polygonal(allowed.difference(Point(kx, ky).buffer(clearance + spacing / 2)))
            if allowed.is_empty:
                continue
            kind = str(kinds[int(rng.integers(len(kinds)))])
            angle = float(rng.uniform(0, 180)) if rotate else 0.0
            fp = _fit(kind, allowed, (float(sx), float(sy)), footprint, angle, min_scale)
            if fp is None:
                continue
            c = fp.centroid
            specs.append(ObstacleSpec(kind, fp, (c.x, c.y), cell))

        free = free_space(region, clearance, [s.polygon for s in specs])
        if _n_parts(free) > base_parts:
            continue
        if keep_clear or specs:
            probe = relaxed_spots(region, spacing=max(spacing, 64), clearance=clearance,
                                  obstacles=[s.polygon for s in specs], rng=rng, iterations=2)
            nav = build_nav_graph(free, probe)
            if not is_navigable(nav, required=keep_clear):
                continue
        if len(specs) < count:
            print(f"Warning: only {len(specs)} of {count} obstacles fit with spacing {spacing}.")
        return specs

    raise ObstaclePlacementError(
        f"No navigable layout for {count} obstacles after {max_attempts} attempts; "
        "reduce count, footprint or spacing."
    )
