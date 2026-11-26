"""Geometry and WAD validation utilities."""

from __future__ import annotations

from typing import Optional, Any, Union
from pathlib import Path

from shapely.geometry import Polygon, Point, MultiPolygon, GeometryCollection
from shapely.validation import make_valid as shapely_make_valid

from doomgen.utils.constants import (
    MIN_SECTOR_AREA,
    MIN_CORRIDOR_WIDTH,
    MAP_MIN_COORD,
    MAP_MAX_COORD,
)


def snap_to_grid(value: float, grid_size: int = 8) -> int:
    """Snap a coordinate to the Doom grid."""
    return int(round(value / grid_size) * grid_size)


def snap_polygon_to_grid(polygon: Polygon, grid_size: int = 8) -> Polygon:
    """Snap every polygon vertex to the Doom grid."""
    coords = list(polygon.exterior.coords)
    snapped = [(snap_to_grid(x, grid_size), snap_to_grid(y, grid_size)) for x, y in coords]
    return Polygon(snapped)


def to_integer_coords(polygon: Polygon) -> Polygon:
    """Round polygon coordinates to integers."""
    coords = list(polygon.exterior.coords)
    int_coords = [(int(round(x)), int(round(y))) for x, y in coords]
    return Polygon(int_coords)


def make_valid(geom: Union[Polygon, MultiPolygon, GeometryCollection]) -> Union[Polygon, MultiPolygon]:
    """Wrap Shapely's `make_valid` for consistency."""
    return shapely_make_valid(geom)


def ensure_polygon(geom: Union[Polygon, MultiPolygon, GeometryCollection]) -> Polygon:
    """Extract a single polygon, using the largest one if needed."""
    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 0:
            raise ValueError("Empty MultiPolygon")
        return max(geom.geoms, key=lambda p: p.area)

    if isinstance(geom, GeometryCollection):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
        if not polygons:
            raise ValueError("GeometryCollection contains no polygons")
        return max(polygons, key=lambda p: p.area)

    raise ValueError(f"Cannot convert {type(geom).__name__} to Polygon")


def validate_polygon(
    polygon: Polygon,
    min_area: float = MIN_SECTOR_AREA,
    check_bounds: bool = True
) -> tuple[bool, list[str]]:
    """Validate geometry, size, and map bounds for a Doom polygon."""
    issues = []

    if not polygon.is_valid:
        issues.append(f"Invalid geometry: {polygon.geom_type}")

    if polygon.is_empty:
        issues.append("Polygon is empty")
        return False, issues

    if polygon.area < min_area:
        issues.append(f"Area too small: {polygon.area:.2f} < {min_area}")

    if not polygon.is_simple:
        issues.append("Polygon has self-intersections")

    if check_bounds:
        bounds = polygon.bounds
        min_x, min_y, max_x, max_y = bounds

        if min_x < MAP_MIN_COORD or max_x > MAP_MAX_COORD:
            issues.append(f"X coordinates out of bounds: [{min_x}, {max_x}]")
        if min_y < MAP_MIN_COORD or max_y > MAP_MAX_COORD:
            issues.append(f"Y coordinates out of bounds: [{min_y}, {max_y}]")

    num_vertices = len(polygon.exterior.coords) - 1
    if num_vertices < 3:
        issues.append(f"Too few vertices: {num_vertices}")

    return len(issues) == 0, issues


def validate_corridor_width(
    polygon: Polygon,
    min_width: float = MIN_CORRIDOR_WIDTH
) -> tuple[bool, float]:
    """Estimate whether a corridor is at least `min_width` wide."""
    eroded = polygon.buffer(-min_width / 2)

    if eroded.is_empty:
        low, high = 0.0, min_width
        while high - low > 1.0:
            mid = (low + high) / 2
            test = polygon.buffer(-mid / 2)
            if test.is_empty:
                high = mid
            else:
                low = mid

        return False, low

    return True, min_width


def validate_thing_placement(
    x: float,
    y: float,
    polygon: Polygon,
    radius: float = 16.0
) -> tuple[bool, str]:
    """Validate a thing placement against sector bounds and wall clearance."""
    point = Point(x, y)

    if not polygon.contains(point):
        return False, "Position is outside sector"

    buffered_point = point.buffer(radius)
    if not polygon.contains(buffered_point):
        return False, "Too close to walls"

    return True, "Valid"


def validate_wad(filepath: str | Path) -> tuple[bool, list[str]]:
    """Perform a basic structural sanity check on a WAD file."""
    filepath = Path(filepath)
    issues = []

    if not filepath.exists():
        return False, ["File does not exist"]

    with open(filepath, 'rb') as f:
        header = f.read(12)

        if len(header) < 12:
            return False, ["File too small for WAD header"]

        signature = header[:4].decode('ascii', errors='ignore')
        if signature not in ('IWAD', 'PWAD'):
            issues.append(f"Invalid WAD signature: {signature}")

        import struct
        num_lumps = struct.unpack('<I', header[4:8])[0]
        dir_offset = struct.unpack('<I', header[8:12])[0]

        file_size = filepath.stat().st_size

        if dir_offset > file_size:
            issues.append(f"Directory offset beyond file size: {dir_offset} > {file_size}")

        expected_dir_size = num_lumps * 16
        if dir_offset + expected_dir_size > file_size:
            issues.append("Directory extends beyond file size")

        f.seek(dir_offset)
        has_map = False

        for _ in range(num_lumps):
            entry = f.read(16)
            if len(entry) < 16:
                break

            name = entry[8:16].rstrip(b'\x00').decode('ascii', errors='ignore')

            if name.startswith('MAP') or (len(name) == 4 and name[0] == 'E' and name[2] == 'M'):
                has_map = True
                break

        if not has_map:
            issues.append("No map markers found")

    return len(issues) == 0, issues


def check_sector_closure(vertices: list[tuple[int, int]]) -> tuple[bool, str]:
    """Check whether a vertex loop closes back on itself."""
    if len(vertices) < 3:
        return False, "Too few vertices for a sector"

    first = vertices[0]
    last = vertices[-1]

    if first != last:
        dx = abs(first[0] - last[0])
        dy = abs(first[1] - last[1])
        if dx > 1 or dy > 1:
            return False, f"Sector not closed: gap from {last} to {first}"

    return True, "Sector is closed"


def estimate_bsp_complexity(polygon: Polygon) -> dict:
    """Estimate rough BSP complexity metrics for a polygon."""
    vertices = len(polygon.exterior.coords) - 1

    holes = len(polygon.interiors)

    convex_hull = polygon.convex_hull
    convexity_ratio = polygon.area / convex_hull.area if convex_hull.area > 0 else 1.0

    return {
        'vertex_count': vertices,
        'hole_count': holes,
        'convexity_ratio': convexity_ratio,
        'estimated_segs': vertices * 2,
        'estimated_ssectors': max(1, vertices // 3),
        'complexity_score': vertices * (1 + holes) / convexity_ratio,
    }
