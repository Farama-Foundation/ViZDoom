"""Demo: exact walls plus Voronoi-based placement (spots, obstacles, nav graph, clearance)."""

import os

from doomgen.abstraction.layout import ConnectionType
from doomgen.builder import ProceduralMapBuilder
from doomgen.geometry.shapes import circle
from doomgen.utils.visualization import visualize_layout


def create_placement_map():
    print("Generating placement demo map...")
    builder = ProceduralMapBuilder(seed=42)

    # Straight-walled hall and a round cave sharing an edge.
    builder.add_area("Main Hall", (0, 0, 1600, 1200), ceiling_height=160)
    builder.add_area("Cave", circle(1150, 0, 400), floor_height=-16, floor_texture="NUKAGE1")
    builder.connect_adjacent("Main Hall", "Cave", ConnectionType.OPEN)

    # Cover crates: one per clipped Voronoi cell, always a player-sized gap apart.
    obstacles = builder.scatter_obstacles(
        "Main Hall", 10, footprint=(96, 96), spacing=96, kinds=("box", "circle"), rotate=True,
        create_areas=True, floor_height=64, wall_texture="CRATE1", floor_texture="CRATOP2",
    )

    # Navigation graph over the free floor, treating every crate as solid.
    nav = builder.nav_graph(["Main Hall", "Cave"], spacing=128, obstacles=obstacles)
    pairs = builder.pick_spawn_pairs(["Main Hall", "Cave"], 1, min_path_distance=1500,
                                     obstacles=obstacles)
    (sx, sy), (gx, gy), dist = pairs[0]
    print(f"Spawn pair path distance: {dist:.0f}")
    builder.add_thing("PLAYER1_START", sx, sy)
    builder.add_thing("PLAYER2_START", gx, gy)

    # Pickups on centroidal-Voronoi spots and the cave's most open point.
    pickups = builder.get_spots("Main Hall", count=8, clearance=32, obstacles=obstacles)
    for x, y in pickups:
        builder.add_thing("STIMPACK", x, y)
    cx, cy, _ = builder.clearance_spots("Cave", 1)[0]
    builder.add_thing("MEDIKIT", cx, cy)

    os.makedirs("examples/output", exist_ok=True)
    builder.build("examples/output/placement_test.wad")
    visualize_layout(builder.layout, "examples/output/placement_test.png",
                     spots=pickups + [(cx, cy)], nav=nav)
    print("Visualization saved to examples/output/placement_test.png")


if __name__ == "__main__":
    create_placement_map()
