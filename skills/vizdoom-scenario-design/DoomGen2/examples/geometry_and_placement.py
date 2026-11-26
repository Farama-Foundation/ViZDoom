"""Geometry and Voronoi placement demo.

Shows exact shapes (rect, circle), corridors, stairs with a door, damaging
floors, obstacle scattering, navigation graphs, path-distance spawn pairs,
pickup spots and max-clearance spots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from doomgen.abstraction.layout import ConnectionType
from doomgen.builder import ProceduralMapBuilder
from doomgen.geometry.shapes import circle


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "geometry_and_placement.wad"


def build_geometry_and_placement(
    output_path: str | Path,
    seed: int = 42,
    plot_path: str | Path | None = None,
) -> None:
    builder = ProceduralMapBuilder(seed=seed)

    builder.add_area("Main Hall", (0, 0, 1600, 1200), ceiling_height=192, floor_texture="FLOOR4_8")
    builder.add_area(
        "Cave",
        circle(1150, 0, 400),
        floor_height=-16,
        ceiling_height=160,
        floor_texture="NUKAGE1",
        special=16,
    )
    builder.connect_adjacent("Main Hall", "Cave", ConnectionType.OPEN)

    builder.add_area(
        "Upper Deck",
        (-1120, 0, 320, 320),
        floor_height=64,
        ceiling_height=224,
        floor_texture="RROCK19",
    )
    step_ids = builder.add_stairs(
        "Main Hall",
        "Upper Deck",
        num_steps=4,
        width=128,
        floor_texture="STEP1",
        riser_texture="STEP1",
        wall_texture="BROWN1",
    )
    builder.add_boundary(
        "Main Hall",
        step_ids[0],
        name="Stair Door",
        connection_type=ConnectionType.DOOR,
        floor_height=0,
        ceiling_height=0,
        wall_texture="BIGDOOR2",
        door_width=128,
        door_depth=16,
    )

    builder.add_area("Side Room", (0, -960, 320, 256), ceiling_height=160, floor_texture="CEIL5_1")
    builder.add_corridor("Main Hall", "Side Room", width=128, ceiling_height=144, floor_texture="CEIL3_5")

    obstacles = builder.scatter_obstacles(
        "Main Hall",
        10,
        footprint=(96, 96),
        spacing=96,
        kinds=("box", "circle"),
        rotate=True,
        create_areas=True,
        floor_height=64,
        wall_texture="CRATE1",
        floor_texture="CRATOP2",
    )

    nav = builder.nav_graph(["Main Hall", "Cave"], spacing=128, obstacles=obstacles)
    (sx, sy), (gx, gy), _dist = builder.pick_spawn_pairs(
        ["Main Hall", "Cave"], 1, min_path_distance=1400, obstacles=obstacles
    )[0]
    builder.add_thing("PLAYER1_START", sx, sy)
    builder.add_thing("PLAYER2_START", gx, gy)

    pickups = builder.get_spots("Main Hall", count=6, clearance=32, obstacles=obstacles)
    for x, y in pickups:
        builder.add_thing("STIMPACK", x, y)
    cx, cy, _ = builder.clearance_spots("Cave", 1)[0]
    builder.add_thing("MEDIKIT", cx, cy)
    builder.add_thing("SHOTGUN", 0, -960)
    builder.add_thing("IMP", -1120, 0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    builder.build(str(output))

    if plot_path is not None:
        from doomgen.utils.visualization import visualize_layout

        plot = Path(plot_path)
        plot.parent.mkdir(parents=True, exist_ok=True)
        visualize_layout(builder.layout, str(plot), spots=pickups + [(cx, cy)], nav=nav)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=Path, default=None, help="optional layout PNG path")
    args = parser.parse_args()
    build_geometry_and_placement(args.output, seed=args.seed, plot_path=args.plot)


if __name__ == "__main__":
    main()
