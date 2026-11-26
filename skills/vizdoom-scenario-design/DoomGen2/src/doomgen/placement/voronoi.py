"""Clipped Voronoi cells and Lloyd relaxation inside arbitrary polygons."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon


def sample_points(region, n: int, rng: np.random.Generator, max_rounds: int = 50) -> np.ndarray:
    """Uniform random points inside `region` (rejection sampling on its bounding box)."""
    if n <= 0 or region.is_empty:
        return np.empty((0, 2))
    minx, miny, maxx, maxy = region.bounds
    out = []
    have = 0
    for _ in range(max_rounds):
        batch = rng.uniform([minx, miny], [maxx, maxy], size=(max(2 * (n - have), 16), 2))
        inside = batch[shapely.contains_xy(region, batch[:, 0], batch[:, 1])]
        out.append(inside)
        have += len(inside)
        if have >= n:
            break
    pts = np.vstack(out) if out else np.empty((0, 2))
    if len(pts) < n:
        rp = region.representative_point()
        pad = np.tile([rp.x, rp.y], (n - len(pts), 1))
        pts = np.vstack([pts, pad])
    return pts[:n]


def clipped_voronoi_cells(points, clip) -> list[Polygon]:
    """Voronoi cell of each point (same order as `points`) clipped to `clip`."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [clip]
    diagram = shapely.voronoi_polygons(MultiPoint(pts), extend_to=clip, ordered=True)
    cells = np.array(list(diagram.geoms), dtype=object)
    if len(cells) != len(pts):
        # Duplicate input points collapse into one cell; match cells by containment.
        matched = []
        for x, y in pts:
            hit = next((c for c in cells if c.covers(shapely.Point(x, y))), Polygon())
            matched.append(hit)
        cells = np.array(matched, dtype=object)
    return list(shapely.intersection(cells, clip))


def lloyd_relax(region, n: int, rng: np.random.Generator, iterations: int = 5,
                initial: np.ndarray | None = None) -> np.ndarray:
    """Centroidal-Voronoi points: random samples moved to their clipped-cell centroids."""
    if n <= 0 or region.is_empty:
        return np.empty((0, 2))
    if n == 1:
        c = region.centroid
        if not region.contains(c):
            c = region.representative_point()
        return np.array([[c.x, c.y]])
    pts = sample_points(region, n, rng) if initial is None else np.asarray(initial, dtype=float)
    for _ in range(iterations):
        # Tiny jitter keeps coincident points from collapsing into one cell.
        jitter = rng.normal(scale=1e-3, size=pts.shape)
        cells = clipped_voronoi_cells(pts + jitter, region)
        new_pts = pts.copy()
        for i, cell in enumerate(cells):
            if cell is None or cell.is_empty:
                continue
            c = cell.centroid
            if not region.contains(c):
                c = cell.representative_point()
            new_pts[i] = (c.x, c.y)
        pts = new_pts
    return pts
