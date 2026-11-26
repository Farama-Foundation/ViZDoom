"""Max-clearance spots on the (Voronoi-approximated) medial axis of a region."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import MultiPoint, MultiPolygon, Polygon


def _boundary_samples(region, step: float) -> np.ndarray:
    polys = [region] if isinstance(region, Polygon) else list(region.geoms)
    pts = []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            dense = shapely.segmentize(ring, max_segment_length=step)
            pts.extend(list(dense.coords)[:-1])
    return np.unique(np.round(np.asarray(pts, dtype=float), 6), axis=0)


def medial_axis_candidates(region, sample_step: float = 16) -> np.ndarray:
    """Voronoi-edge vertices of boundary samples that fall inside `region`."""
    samples = _boundary_samples(region, sample_step)
    if len(samples) < 3:
        return np.empty((0, 2))
    edges = shapely.voronoi_polygons(MultiPoint(samples), only_edges=True)
    coords = shapely.get_coordinates(edges)
    if len(coords) == 0:
        return np.empty((0, 2))
    # Edge midpoints as well, so long straight medial edges are sampled too.
    parts = shapely.get_parts(edges)
    mids = shapely.get_coordinates(shapely.line_interpolate_point(parts, 0.5, normalized=True))
    cand = np.unique(np.round(np.vstack([coords, mids]), 3), axis=0)
    inside = shapely.contains_xy(region, cand[:, 0], cand[:, 1])
    return cand[inside]


def max_clearance_spots(region, n: int, *, min_clearance: float = 32, min_separation: float = 64,
                        sample_step: float = 16) -> list[tuple[float, float, float]]:
    """Up to `n` points ranked by distance to the nearest wall, at least `min_separation` apart.

    Returns `(x, y, clearance)`. If no point reaches `min_clearance`, the single
    clearest point is returned so small areas still get a spot.
    """
    if region is None or region.is_empty or n <= 0:
        return []
    cand = medial_axis_candidates(region, sample_step)
    if len(cand) == 0:
        rp = region.representative_point()
        return [(rp.x, rp.y, float(region.boundary.distance(rp)))]
    clear = shapely.distance(region.boundary, shapely.points(cand))
    order = np.argsort(-clear, kind="stable")

    picked: list[tuple[float, float, float]] = []
    for i in order:
        c = float(clear[i])
        if c < min_clearance:
            break
        x, y = float(cand[i, 0]), float(cand[i, 1])
        if all((x - px) ** 2 + (y - py) ** 2 >= min_separation ** 2 for px, py, _ in picked):
            picked.append((x, y, c))
            if len(picked) >= n:
                break
    if not picked:
        i = int(order[0])
        picked.append((float(cand[i, 0]), float(cand[i, 1]), float(clear[i])))
    return picked
