from doomgen.builder import ProceduralMapBuilder
from doomgen.abstraction.layout import ConnectionType
import os

def create_stairs_map():
    print("Generating Stairs Map...")

    # Initialize Builder
    builder = ProceduralMapBuilder(bounds=(-512, -512, 512, 512), seed=99)

    # Lower Room (Safe)
    # Right edge at -200 + 128 = -72
    builder.add_area(
        name="Lower Room",
        shape=(-200, 0, 256, 256),
        floor_height=0,
        ceiling_height=128,
        floor_texture="FLOOR4_8"
    )

    # Upper Room (Damaging)
    # Left edge at 200 - 128 = 72
    # Gap between -72 and 72 is 144 units.
    builder.add_area(
        name="Upper Room",
        shape=(200, 0, 256, 256),
        floor_height=64, # Higher
        ceiling_height=192,
        floor_texture="NUKAGE1", # Damaging look
        special=16 # Damage 20%
    )

    # Create Stairs using the new helper method
    # This will automatically find the path between the rooms and create steps
    step_ids = builder.add_stairs(
        "Lower Room",
        "Upper Room",
        num_steps=4,
        width=128, # Map units
        floor_texture="STEP1",
        riser_texture="STEP1",
        wall_texture="BROWN1"
    )

    # Add Door between Lower Room and Stairs
    # We use add_boundary to insert a door sector between the Lower Room and the first step
    builder.add_boundary(
        "Lower Room",
        step_ids[0], # First step ID
        name="Stair Door",
        connection_type=ConnectionType.DOOR,
        floor_height=0,
        ceiling_height=0, # Closed
        floor_texture="GATE1",
        ceiling_texture="GATE1",
        wall_texture="BIGDOOR2",
        door_width=128, # Full stair width, so the door is the only way through
        door_depth=16
    )

    # Add Player Start in Lower Room
    builder.add_thing("PLAYER1_START", -200, 0, angle=0)

    # Build
    os.makedirs("examples/output", exist_ok=True)
    builder.build("examples/output/stairs_test.wad")

    # Visualize
    from doomgen.utils.visualization import visualize_layout
    visualize_layout(builder.layout, "examples/output/stairs_test.png")
    print("Visualization saved to examples/output/stairs_test.png")

if __name__ == "__main__":
    create_stairs_map()
