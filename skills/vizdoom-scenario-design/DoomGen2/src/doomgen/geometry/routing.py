"""Corridor routing between exact area regions: straight, then L-bend, then grid."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Literal, Optional

from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import substring, unary_union

from doomgen.geometry.shapes import polygonal

RouteKind = Literal["straight", "L", "grid"]

# How far the stored corridor polygon reaches into the rooms; the arrangement
# trims it back with `claim_void`, which hides integer-snapping gaps.
_OVERLAP = 4.0


@dataclass
class CorridorRoute:
    polygon: Polygon | MultiPolygon
    centerline: LineString
    width: float
    kind: RouteKind


def _parts(geom) -> list[Polygon]:
    geom = polygonal(geom)
    if geom.is_empty:
        return []
    return [geom] if isinstance(geom, Polygon) else list(geom.geoms)


def _contact(poly, region) -> float:
    """Length of `region`'s boundary that `poly` sits against."""
    return poly.buffer(0.5).intersection(region.boundary).length


def _band(path: LineString, width: float) -> Polygon:
    return path.buffer(width / 2.0, cap_style="flat", join_style="mitre", mitre_limit=4.0)


def _evaluate(path: LineString, width: float, region_a, region_b, blockers) -> Optional[Polygon | MultiPolygon]:
    """Corridor polygon for `path` if it links A and B without touching blockers."""
    if path.length <= 0:
        return None
    band = _band(path, width)
    free = band.difference(unary_union([region_a, region_b]))
    min_contact = min(width * 0.5, 32.0)
    for piece in _parts(free):
        if piece.area < 1.0:
            continue
        if _contact(piece, region_a) < min_contact or _contact(piece, region_b) < min_contact:
            continue
        if blockers is not None and not blockers.is_empty and piece.intersection(blockers).area > 1.0:
            continue
        return polygonal(band.intersection(piece.buffer(_OVERLAP, join_style="mitre")))
    return None


def _anchor(region) -> tuple[float, float]:
    pt = region.centroid
    if not region.contains(pt):
        pt = region.representative_point()
    return (round(pt.x), round(pt.y))


def _straight_candidates(region_a, region_b, width: float) -> list[LineString]:
    ax0, ay0, ax1, ay1 = region_a.bounds
    bx0, by0, bx1, by1 = region_b.bounds
    (acx, acy), (bcx, bcy) = _anchor(region_a), _anchor(region_b)
    paths = []

    lo, hi = max(ax0, bx0) + width / 2, min(ax1, bx1) - width / 2
    if hi >= lo:
        xs = [round((lo + hi) / 2), round(min(max(acx, lo), hi)), round(min(max(bcx, lo), hi))]
        for x in dict.fromkeys(xs):
            paths.append(LineString([(x, acy), (x, bcy)]))

    lo, hi = max(ay0, by0) + width / 2, min(ay1, by1) - width / 2
    if hi >= lo:
        ys = [round((lo + hi) / 2), round(min(max(acy, lo), hi)), round(min(max(bcy, lo), hi))]
        for y in dict.fromkeys(ys):
            paths.append(LineString([(acx, y), (bcx, y)]))

    paths.sort(key=lambda p: p.length)
    return paths


def _l_candidates(region_a, region_b) -> list[LineString]:
    (ax, ay), (bx, by) = _anchor(region_a), _anchor(region_b)
    if ax == bx or ay == by:
        return []
    return [
        LineString([(ax, ay), (bx, ay), (bx, by)]),
        LineString([(ax, ay), (ax, by), (bx, by)]),
    ]


def _grid_path(region_a, region_b, width: float, blockers, bounds=None,
               max_cells: int = 160) -> Optional[LineString]:
    """Orthogonal A* over a coarse grid; cells in other areas are blocked."""
    allx0, ally0, allx1, ally1 = unary_union([region_a, region_b]).bounds
    if blockers is not None and not blockers.is_empty:
        bx0, by0, bx1, by1 = blockers.bounds
        allx0, ally0 = min(allx0, bx0), min(ally0, by0)
        allx1, ally1 = max(allx1, bx1), max(ally1, by1)
    margin = 2 * width
    allx0, ally0, allx1, ally1 = allx0 - margin, ally0 - margin, allx1 + margin, ally1 + margin
    if bounds is not None:
        allx0, ally0 = max(allx0, bounds[0]), max(ally0, bounds[1])
        allx1, ally1 = min(allx1, bounds[2]), min(ally1, bounds[3])

    cell = float(width)
    nx = int(math.ceil((allx1 - allx0) / cell))
    ny = int(math.ceil((ally1 - ally0) / cell))
    if nx <= 0 or ny <= 0 or nx * ny > max_cells * max_cells:
        return None

    # Align the grid so one cell centre sits on A's anchor point.
    ax, ay = _anchor(region_a)
    ox = ax - (math.ceil((ax - allx0) / cell) + 0.5) * cell
    oy = ay - (math.ceil((ay - ally0) / cell) + 0.5) * cell

    def center(i, j):
        return (ox + (i + 0.5) * cell, oy + (j + 0.5) * cell)

    def cell_box(i, j):
        x, y = center(i, j)
        return box(x - cell / 2, y - cell / 2, x + cell / 2, y + cell / 2)

    nx += 1
    ny += 1
    blocked = set()
    starts, goals = set(), set()
    for i in range(nx):
        for j in range(ny):
            c = cell_box(i, j)
            if blockers is not None and not blockers.is_empty and c.intersection(blockers).area > 1.0:
                blocked.add((i, j))
                continue
            if c.intersection(region_a).area >= 0.5 * c.area:
                starts.add((i, j))
            elif c.intersection(region_b).area >= 0.5 * c.area:
                goals.add((i, j))
    if not starts or not goals:
        return None

    gx = sum(g[0] for g in goals) / len(goals)
    gy = sum(g[1] for g in goals) / len(goals)

    def h(n):
        return abs(n[0] - gx) + abs(n[1] - gy)

    open_heap = []
    best = {}
    came = {}
    no_dir = (0, 0)
    for s in sorted(starts):
        best[(s, no_dir)] = 0.0
        heapq.heappush(open_heap, (h(s), 0.0, s, no_dir))
    end_state = None
    while open_heap:
        _, g, node, direction = heapq.heappop(open_heap)
        if node in goals:
            end_state = (node, direction)
            break
        if g > best.get((node, direction), math.inf):
            continue
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (node[0] + d[0], node[1] + d[1])
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or nb in blocked:
                continue
            # Walking inside A is free; turns cost extra so routes stay straight.
            step = 0.0 if (node in starts and nb in starts) else 1.0
            if direction != no_dir and d != direction:
                step += 0.5
            ng = g + step
            if ng < best.get((nb, d), math.inf):
                best[(nb, d)] = ng
                came[(nb, d)] = (node, direction)
                heapq.heappush(open_heap, (ng + h(nb), ng, nb, d))
    if end_state is None:
        return None

    cells = []
    state = end_state
    while state is not None:
        cells.append(state[0])
        state = came.get(state)
    cells.reverse()
    # Trim the walk inside A to the last start cell.
    last_start = max(i for i, c in enumerate(cells) if c in starts)
    cells = cells[last_start:]
    pts = [center(*c) for c in cells]
    if len(pts) < 2:
        return None
    line = LineString(pts).simplify(0)
    # Extend both ends half a cell so the band reaches into the rooms.
    coords = list(line.coords)
    def _extend(p, q):
        dx, dy = p[0] - q[0], p[1] - q[1]
        n = math.hypot(dx, dy) or 1.0
        return (p[0] + dx / n * cell / 2, p[1] + dy / n * cell / 2)
    coords[0] = _extend(coords[0], coords[1])
    coords[-1] = _extend(coords[-1], coords[-2])
    return LineString(coords)


def route_corridor(region_a, region_b, width: float, blockers=None,
                   bounds: tuple[float, float, float, float] | None = None) -> Optional[CorridorRoute]:
    """Find a corridor between two regions: straight, then one L bend, then a grid route."""
    if region_a.is_empty or region_b.is_empty:
        return None
    for path in _straight_candidates(region_a, region_b, width):
        poly = _evaluate(path, width, region_a, region_b, blockers)
        if poly is not None and not poly.is_empty:
            return CorridorRoute(poly, path, width, "straight")
    for path in _l_candidates(region_a, region_b):
        poly = _evaluate(path, width, region_a, region_b, blockers)
        if poly is not None and not poly.is_empty:
            return CorridorRoute(poly, path, width, "L")
    path = _grid_path(region_a, region_b, width, blockers, bounds)
    if path is not None:
        poly = _evaluate(path, width, region_a, region_b, blockers)
        if poly is not None and not poly.is_empty:
            return CorridorRoute(poly, path, width, "grid")
    return None


def slice_route(route: CorridorRoute, region_a, num_steps: int,
                region_b=None) -> list[Polygon | MultiPolygon]:
    """Cut a corridor polygon into `num_steps` consecutive pieces along its centreline.

    Only the part of the route outside the two rooms is divided, so every step
    has the same length. Pieces overlap slightly; add them with `claim_void` in
    order so earlier steps win.
    """
    line = route.centerline
    rooms = region_a if region_b is None else unary_union([region_a, region_b])
    free = route.polygon.difference(rooms)
    if free.is_empty:
        free = route.polygon
    ts = [line.project(Point(p)) for p in _ring_points(free)]
    t0, t1 = (min(ts), max(ts)) if ts else (0.0, line.length)
    # Orient from room A to room B.
    if line.project(region_a.centroid) > line.length / 2:
        line = LineString(list(line.coords)[::-1])
        t0, t1 = line.length - t1, line.length - t0
    num_steps = max(1, int(num_steps))
    step_len = (t1 - t0) / num_steps
    pieces = []
    for k in range(num_steps):
        a = t0 + k * step_len if k > 0 else 0.0
        b = t0 + (k + 1) * step_len + (1.0 if k < num_steps - 1 else 0.0)
        if k == num_steps - 1:
            b = line.length
        seg = substring(line, max(0.0, a), min(line.length, b))
        if seg.length <= 0:
            continue
        slab = seg.buffer(route.width, cap_style="flat", join_style="mitre", mitre_limit=4.0)
        pieces.append(polygonal(route.polygon.intersection(slab)))
    return pieces


def _ring_points(geom) -> list[tuple[float, float]]:
    pts = []
    for p in _parts(geom):
        pts.extend(p.exterior.coords)
    return pts
