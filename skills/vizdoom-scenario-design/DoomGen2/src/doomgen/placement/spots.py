"""Evenly spread, wall-safe spots from Lloyd-relaxed (centroidal) Voronoi seeds."""

from __future__ import annotations

import math
import zlib
from typing import Iterable, Optional

import numpy as np

from doomgen.placement.free_space import free_space
from doomgen.placement.voronoi import lloyd_relax

MAX_SPOTS = 1500


def spot_count(free, spacing: float) -> int:
    """Number of spots so each owns roughly a `spacing` x `spacing` cell."""
    if free.is_empty:
        return 0
    return int(max(1, min(MAX_SPOTS, round(free.area / max(spacing, 1.0) ** 2))))


def relaxed_spots(region, *, count: Optional[int] = None, spacing: float = 64,
                  clearance: float = 16, rng: np.random.Generator,
                  obstacles: Iterable = (), iterations: int = 5) -> list[tuple[float, float]]:
    """Centroidal-Voronoi spots inside `region`, at least `clearance` from walls/obstacles.

    If no point has the requested clearance (area smaller than the player),
    the region's representative point is returned instead.
    """
    if region is None or region.is_empty:
        return []
    free = free_space(region, clearance, obstacles)
    if free.is_empty:
        rp = region.representative_point()
        return [(rp.x, rp.y)]
    n = count if count is not None else spot_count(free, spacing)
    pts = lloyd_relax(free, int(n), rng, iterations=iterations)
    return [(float(x), float(y)) for x, y in pts]


def stable_rng(seed: int, *parts) -> np.random.Generator:
    """RNG derived from the builder seed and a key, independent of call order."""
    words = [int(seed) & 0xFFFFFFFF]
    for p in parts:
        words.append(zlib.crc32(str(p).encode("utf-8")))
    return np.random.default_rng(words)


class SpotPool:
    """Cached relaxed spots per area; hands out unused spots first.

    The cache is keyed on `layout.version`, so adding geometry invalidates it.
    """

    def __init__(self, layout, seed: int, rng: np.random.Generator):
        self.layout = layout
        self.seed = seed
        self.rng = rng
        self._cache: dict[tuple, tuple[int, list[tuple[float, float]]]] = {}
        self._used: dict[tuple, set[int]] = {}
        self.taken: list[tuple[float, float]] = []

    def spots(self, area_id: str, clearance: float, spacing: float) -> list[tuple[float, float]]:
        key = (area_id, float(clearance), float(spacing))
        cached = self._cache.get(key)
        if cached is not None and cached[0] == self.layout.version:
            return cached[1]
        region = self.layout.region(area_id)
        spots = relaxed_spots(region, spacing=spacing, clearance=clearance,
                              rng=stable_rng(self.seed, area_id, clearance, spacing))
        self._cache[key] = (self.layout.version, spots)
        self._used[key] = set()
        return spots

    def draw(self, area_id: str, clearance: float, spacing: float,
             min_distance: Optional[float] = None) -> Optional[tuple[float, float]]:
        """Pick a random unused spot; with `min_distance`, keep away from earlier draws."""
        spots = self.spots(area_id, clearance, spacing)
        if not spots:
            return None
        key = (area_id, float(clearance), float(spacing))
        used = self._used.setdefault(key, set())
        unused = [i for i in range(len(spots)) if i not in used]
        if not unused:
            used.clear()
            unused = list(range(len(spots)))

        if min_distance is not None and self.taken:
            taken = np.asarray(self.taken)
            pts = np.asarray([spots[i] for i in unused])
            d = np.sqrt(((pts[:, None, :] - taken[None, :, :]) ** 2).sum(-1)).min(axis=1)
            ok = [unused[k] for k in np.nonzero(d >= min_distance)[0]]
            if ok:
                unused = ok
            else:
                unused = [unused[int(np.argmax(d))]]

        idx = int(unused[int(self.rng.integers(len(unused)))])
        used.add(idx)
        spot = spots[idx]
        self.taken.append(spot)
        return spot


def min_pairwise_distance(points) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return math.inf
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())
