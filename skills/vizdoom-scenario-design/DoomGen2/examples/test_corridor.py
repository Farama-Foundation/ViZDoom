from doomgen.builder import ProceduralMapBuilder
import os

def create_corridor_map():
    print("Generating Corridor Map...")

    builder = ProceduralMapBuilder(bounds=(-512, -512, 1024, 512), seed=42)

    # Room A
    builder.add_area(
        name="Room A",
        shape=(0, 0, 256, 256),
        floor_texture="FLOOR4_8"
    )

    # Room B
    builder.add_area(
        name="Room B",
        shape=(512, 0, 256, 256),
        floor_texture="FLOOR4_8"
    )

    # Corridor (width in map units)
    builder.add_corridor("Room A", "Room B", width=128, ceiling_height=112, floor_texture="CEIL3_5")

    # Player Start
    builder.add_thing("PLAYER1_START", 0, 0, angle=0)

    # Build
    os.makedirs("examples/output", exist_ok=True)
    builder.build("examples/output/corridor_test.wad")

if __name__ == "__main__":
    create_corridor_map()
