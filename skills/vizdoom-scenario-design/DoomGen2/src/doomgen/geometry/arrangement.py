"""Exact polygon arrangement: resolves layered area shapes into Doom-ready edges.

Every area keeps its exact outline. All outlines are noded together and
polygonized into faces; each face is owned by the area that wins at that point
when the shapes are replayed in insertion order (`claim_void` fills only
unowned space, `overwrite` takes everything, `VOID` clears). Face boundaries
between different owners become the edges handed to `MapTranslator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, polygonize, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

from doomgen.abstraction.layout import Connection, ConnectionType, LayoutGraph
from doomgen.geometry.shapes import as_polygon, polygonal
from doomgen.utils.constants import MIN_SECTOR_AREA

VOID = "VOID"

# ZDBSP's blockmap asserts on linedefs longer than a signed 16-bit span; keep a wide margin.
MAX_LINE_LENGTH = 8192.0

AreaMode = Literal["claim_void", "overwrite"]


class Edge(TypedDict):
    """One wall segment; `front_area` lies on the right-hand side of `line`."""
    line: LineString
    front_area: str
    back_area: str
    type: Literal["void_wall", "solid", "portal"]
    connection: Optional[Connection]


@dataclass
class AreaShape:
    area_id: str
    polygon: Polygon | MultiPolygon
    mode: str
    order: int


def _key(pt) -> tuple[float, float]:
    return (round(pt[0], 6), round(pt[1], 6))


def _ring_lines(poly: Polygon) -> list[LineString]:
    lines = [LineString(poly.exterior.coords)]
    lines.extend(LineString(r.coords) for r in poly.interiors)
    return lines


def _polys(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in geom.geoms if isinstance(g, Polygon)]


class PolygonLayout:
    """Ordered collection of exact area shapes plus derived topology."""

    def __init__(self, bounds: tuple[float, float, float, float] | None = None):
        self.bounds = bounds
        self.shapes: list[AreaShape] = []
        self.version = 0
        self._cache_version = -1
        self._faces: list[Polygon] = []
        self._owners: list[Optional[str]] = []
        self._regions: dict[str, Polygon | MultiPolygon] = {}
        self._segments: list[tuple[tuple[float, float], tuple[float, float], Optional[str], Optional[str]]] = []

    # ------------------------------------------------------------------ input

    def add(self, area_id: str, polygon, mode: AreaMode = "claim_void") -> Polygon | MultiPolygon:
        """Record a shape for `area_id`; returns the snapped polygon that was stored."""
        if mode not in ("claim_void", "overwrite"):
            raise ValueError(f"Unknown area mode: {mode!r}")
        poly = as_polygon(polygon)
        if self.bounds is not None and not poly.is_empty:
            poly = polygonal(poly.intersection(shapely.box(*self.bounds)))
        if poly.is_empty:
            return poly
        self.shapes.append(AreaShape(area_id, poly, mode, len(self.shapes)))
        self.version += 1
        return poly

    def set_void(self, polygon) -> None:
        """Remove the space inside `polygon` from every area added so far."""
        self.add(VOID, polygon, "overwrite")

    # --------------------------------------------------------------- resolving

    def _resolve(self) -> None:
        if self._cache_version == self.version:
            return
        self._cache_version = self.version
        self._faces, self._owners, self._regions, self._segments = [], [], {}, []
        if not self.shapes:
            return

        lines = []
        for s in self.shapes:
            for p in _polys(s.polygon):
                lines.extend(_ring_lines(p))
        noded = unary_union(lines)
        faces = [orient(f, sign=1.0) for f in polygonize(noded) if f.area > 1e-9]

        tree = STRtree([s.polygon for s in self.shapes])
        prepared = [prep(s.polygon) for s in self.shapes]
        owners: list[Optional[str]] = []
        for f in faces:
            pt = f.representative_point()
            owner: Optional[str] = None
            for idx in sorted(int(i) for i in tree.query(pt)):
                if not prepared[idx].contains(pt):
                    continue
                s = self.shapes[idx]
                if s.area_id == VOID:
                    owner = None
                elif s.mode == "overwrite" or owner is None:
                    owner = s.area_id
            owners.append(owner)

        # Directed half-edges: each face lies on the left of its CCW-oriented rings.
        half: dict[tuple, int] = {}
        face_edges: list[list[tuple]] = []
        for fi, f in enumerate(faces):
            edges_f = []
            for r in [f.exterior, *f.interiors]:
                coords = list(r.coords)
                for a, b in zip(coords, coords[1:]):
                    ka, kb = _key(a), _key(b)
                    if ka == kb:
                        continue
                    half[(ka, kb)] = fi
                    edges_f.append((ka, kb))
            face_edges.append(edges_f)

        # Absorb slivers (from crossing diagonal/curved outlines) into the
        # neighbour with the longest shared boundary.
        for fi, f in enumerate(faces):
            if owners[fi] is None or f.area >= MIN_SECTOR_AREA:
                continue
            shared: dict[Optional[str], float] = {}
            for ka, kb in face_edges[fi]:
                nb = half.get((kb, ka))
                if nb is None:
                    continue
                o = owners[nb]
                if o == owners[fi]:
                    continue
                length = ((ka[0] - kb[0]) ** 2 + (ka[1] - kb[1]) ** 2) ** 0.5
                shared[o] = shared.get(o, 0.0) + length
            if shared:
                owners[fi] = max(shared.items(), key=lambda kv: kv[1])[0]

        self._faces, self._owners = faces, owners

        by_area: dict[str, list[Polygon]] = {}
        for f, o in zip(faces, owners):
            if o is not None:
                by_area.setdefault(o, []).append(f)
        self._regions = {a: polygonal(unary_union(fs)) for a, fs in by_area.items()}

        # Classified segments, oriented so the `front` owner is on the right.
        seen = set()
        segments = []
        for fi, edges_f in enumerate(face_edges):
            for ka, kb in edges_f:
                if (ka, kb) in seen:
                    continue
                seen.add((ka, kb))
                seen.add((kb, ka))
                left = owners[fi]
                rf = half.get((kb, ka))
                right = owners[rf] if rf is not None else None
                if left == right:
                    continue
                if right is None or (left is not None and left < right):
                    # Void always goes to the back; area pairs get a canonical
                    # orientation so collinear pieces can be merged.
                    segments.append((kb, ka, left, right))
                else:
                    segments.append((ka, kb, right, left))
        self._segments = segments

    # ----------------------------------------------------------------- queries

    def regions(self) -> dict[str, Polygon | MultiPolygon]:
        """Effective region of every non-empty area after layering."""
        self._resolve()
        return dict(self._regions)

    def region(self, area_id: str) -> Polygon | MultiPolygon:
        self._resolve()
        return self._regions.get(area_id, Polygon())

    def occupied(self) -> Polygon | MultiPolygon:
        """Union of all non-void regions."""
        self._resolve()
        return polygonal(unary_union(list(self._regions.values()))) if self._regions else Polygon()

    def area_at(self, x: float, y: float) -> Optional[str]:
        """Owner area at a point, or None for void."""
        self._resolve()
        pt = Point(x, y)
        for f, o in zip(self._faces, self._owners):
            if f.covers(pt):
                return o
        return None

    def shared_boundary(self, a: str, b: str) -> MultiLineString | LineString:
        """Merged boundary lines shared by areas `a` and `b`."""
        self._resolve()
        pair = {a, b}
        segs = [LineString([p, q]) for p, q, f, bk in self._segments if {f, bk} == pair]
        if not segs:
            return MultiLineString()
        return linemerge(segs)

    def touching_pairs(self) -> set[tuple[str, str]]:
        """All pairs of non-void areas that share at least one boundary segment."""
        self._resolve()
        pairs = set()
        for _, _, f, b in self._segments:
            if f is not None and b is not None:
                pairs.add(tuple(sorted((f, b))))
        return pairs

    # ------------------------------------------------------------- translation

    def edges_for_translation(self, graph: LayoutGraph) -> list[Edge]:
        """Classify, orient and merge boundary segments for `MapTranslator`."""
        self._resolve()

        conn_lookup: dict[frozenset, Connection] = {}
        for conn in graph.connections:
            conn_lookup.setdefault(frozenset((conn.area_a_id, conn.area_b_id)), conn)

        degree: dict[tuple, int] = {}
        for p, q, _, _ in self._segments:
            degree[p] = degree.get(p, 0) + 1
            degree[q] = degree.get(q, 0) + 1

        groups: dict[tuple, list[tuple]] = {}
        for p, q, f, b in self._segments:
            groups.setdefault((f, b), []).append((p, q))

        edges: list[Edge] = []
        for (front, back), segs in groups.items():
            front_id = front if front is not None else VOID
            back_id = back if back is not None else VOID
            connection = None
            if front_id == VOID or back_id == VOID:
                etype = "void_wall"
            else:
                etype = "solid"
                connection = conn_lookup.get(frozenset((front_id, back_id)))
                if connection is not None and connection.type in (
                    ConnectionType.OPEN, ConnectionType.DOOR, ConnectionType.WINDOW
                ):
                    etype = "portal"
            for p, q in _merge_collinear(segs, degree):
                for a, b in _split_long(p, q):
                    edges.append({
                        "line": LineString([a, b]),
                        "front_area": front_id,
                        "back_area": back_id,
                        "type": etype,
                        "connection": connection,
                    })
        return edges


def _split_long(p, q, max_len: float = MAX_LINE_LENGTH) -> list[tuple]:
    """Split a segment into equal pieces no longer than `max_len` (vertices on the integer grid)."""
    length = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
    n = int(length // max_len) + 1 if length > max_len else 1
    if n == 1:
        return [(p, q)]
    pts = [p]
    for k in range(1, n):
        t = k / n
        pts.append((round(p[0] + (q[0] - p[0]) * t), round(p[1] + (q[1] - p[1]) * t)))
    pts.append(q)
    return list(zip(pts, pts[1:]))


def _collinear(a, b, c, tol: float = 1e-6) -> bool:
    cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    la = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    lb = ((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2) ** 0.5
    return abs(cross) <= tol * max(la * lb, 1.0)


def _merge_collinear(segs: list[tuple], degree: dict[tuple, int]) -> list[tuple]:
    """Join consecutive collinear segments through vertices used by exactly two segments."""
    out_of = {p: (p, q) for p, q in segs}
    into = {q: (p, q) for p, q in segs}
    used = set()
    merged = []
    for seg in segs:
        if seg in used:
            continue
        start, end = seg
        # Walk backwards to the chain start.
        while start in into and degree.get(start, 0) == 2:
            prev = into[start]
            if prev in used or prev == seg or not _collinear(prev[0], start, end):
                break
            used.add(prev)
            start = prev[0]
        used.add(seg)
        while end in out_of and degree.get(end, 0) == 2:
            nxt = out_of[end]
            if nxt in used or not _collinear(start, end, nxt[1]):
                break
            used.add(nxt)
            end = nxt[1]
        merged.append((start, end))
    return merged
