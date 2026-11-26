import os
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType

def main():
    print("Generating Central Hub Map with Builder...")

    # 1. Initialize the Builder
    builder = ProceduralMapBuilder(bounds=(-1024, -1024, 1024, 1024), seed=42)

    # 2. Add Rooms
    # Central Hub
    builder.add_area(
        name="central_hub",
        shape=(0, 0, 256, 256),
        floor_height=0,
        ceiling_height=192,
        floor_texture="FLOOR4_8",
        ceiling_texture="CEIL3_5"
    )

    # Slime Pit
    builder.add_area(
        name="slime_pit",
        shape=(-600, 400, 180, 180),
        floor_height=-16,
        floor_texture="NUKAGE1",
        special=16 # Damage 20%
    )

    # Hell Keep
    builder.add_area(
        name="hell_keep",
        shape=(600, 400, 200, 200),
        floor_height=32,
        floor_texture="RROCK19",
        ceiling_texture="FLOOR6_1"
    )

    # Tech Lab
    builder.add_area(
        name="tech_lab",
        shape=(0, -600, 160, 160),
        floor_height=0,
        ceiling_texture="TLITE6_4"
    )

    # 3. Add Corridors (width in map units)
    builder.add_corridor("central_hub", "slime_pit", width=128)
    builder.add_corridor("central_hub", "hell_keep", width=128)
    builder.add_corridor("central_hub", "tech_lab", width=128)

    # 4. Add Things
    builder.add_thing("PLAYER1_START", 0, 0, angle=90)
    builder.add_thing("SHOTGUN_GUY", -600, 400)
    builder.add_thing("IMP", 600, 400)
    builder.add_thing("DEMON", 0, -600)

    # 5. Save
    output_path = os.path.join("examples", "output", "central_hub.wad")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    builder.build(output_path)

if __name__ == "__main__":
    main()
