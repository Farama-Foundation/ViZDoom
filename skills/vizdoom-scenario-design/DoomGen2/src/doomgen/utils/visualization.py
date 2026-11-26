"""
Visualization tools for DoomGen maps.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    from doomgen.geometry.arrangement import PolygonLayout


def _area_color(area_id: str):
    h = int(hashlib.sha256(area_id.encode("utf-8")).hexdigest(), 16)
    r = ((h & 0xFF) / 255.0) * 0.5 + 0.5  # Pastel
    g = (((h >> 8) & 0xFF) / 255.0) * 0.5 + 0.5
    b = (((h >> 16) & 0xFF) / 255.0) * 0.5 + 0.5
    return (r, g, b)


def _polygon_patch(poly: Polygon, **kwargs) -> PathPatch:
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        coords = list(ring.coords)
        verts.extend(coords)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(verts, codes), **kwargs)


def _polys(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def visualize_layout(
    layout: "PolygonLayout",
    output_path: str,
    spots: Optional[Iterable[Sequence[float]]] = None,
    nav=None,
    obstacles: Optional[Iterable] = None,
    show_labels: bool = True,
) -> None:
    """Render the resolved area regions plus optional placement overlays.

    `spots` are (x, y[, clearance]) points, `nav` a `NavGraph` and
    `obstacles` a list of `ObstacleSpec` (or shapely polygons).
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    regions = layout.regions()

    for area_id, region in regions.items():
        color = _area_color(area_id)
        for poly in _polys(region):
            ax.add_patch(_polygon_patch(poly, facecolor=color, edgecolor="black",
                                        alpha=0.8, linewidth=1.0))
        if show_labels:
            c = region.representative_point()
            label = area_id.split(":", 1)[-1]
            ax.text(c.x, c.y, label, fontsize=6, ha="center", va="center", color="black")

    if obstacles:
        for obs in obstacles:
            geom = getattr(obs, "polygon", obs)
            for poly in _polys(geom):
                ax.add_patch(_polygon_patch(poly, facecolor="dimgray", edgecolor="black",
                                            alpha=0.9, linewidth=0.8))

    if nav is not None:
        pts = nav.points
        for u, v in nav.graph.edges():
            ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color="steelblue",
                    linewidth=0.4, alpha=0.6)
        for u, v in nav.mst.edges():
            ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color="navy", linewidth=1.2)

    if spots:
        spots = list(spots)
        ax.scatter([s[0] for s in spots], [s[1] for s in spots], s=12, c="red", zorder=5)

    if regions:
        minx = min(r.bounds[0] for r in regions.values())
        miny = min(r.bounds[1] for r in regions.values())
        maxx = max(r.bounds[2] for r in regions.values())
        maxy = max(r.bounds[3] for r in regions.values())
        margin = 64
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)
    else:
        ax.autoscale()

    ax.set_aspect("equal")
    plt.title(f"Polygon layout ({len(regions)} areas)")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
