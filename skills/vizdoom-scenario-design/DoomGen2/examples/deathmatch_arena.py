"""Procedural deathmatch arena with raised cover, custom DECORATE target and BatchGenerator.

Demonstrates:
- flush cover sectors raised per episode in OPEN (Floor_MoveToValue);
- Voronoi pick_spawn_pairs for separated deathmatch starts;
- custom DECORATE monster triggering ACS_ExecuteAlways on Death;
- literal-class-name SpawnSpot with ThingCount guard;
- fixed-point global int 0:reward (+= 1.0) and USER50/51 exports reset in OPEN;
- seedable batch sweeps with BatchGenerator (--batch-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from doomgen.batch.generator import BatchGenerator
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.geometry.shapes import circle, rect
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "deathmatch_arena.wad"
COVER_TAG_BASE = 200
TARGET_SPOT_TID = 3000
TARGET_TID = 3100


class DeathmatchArenaScenario(Scenario):
    def get_default_config(self) -> dict[str, Any]:
        return {
            "seed": 42,
            "arena_size": 1280,
            "arena_shape": "rect",
            "cover_count": 8,
            "cover_density_pct": 60,
            "cover_height": 64,
            "acc_path": None,
        }

    def validate_config(self) -> None:
        cfg = self.config
        if int(cfg["arena_size"]) < 640:
            raise ValueError("arena_size must be >= 640")
        if cfg["arena_shape"] not in ("rect", "circle"):
            raise ValueError("arena_shape must be 'rect' or 'circle'")
        if not 0 <= int(cfg["cover_density_pct"]) <= 100:
            raise ValueError("cover_density_pct must be in [0, 100]")

    def build_acs(self, cover_count: int) -> ACSBuilder:
        cfg = self.config
        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_define("COVER_TAG_BASE", COVER_TAG_BASE)
        acs.add_define("COVER_COUNT", cover_count)
        acs.add_define("COVER_DENSITY", int(cfg["cover_density_pct"]))
        acs.add_define("COVER_HEIGHT", int(cfg["cover_height"]))
        acs.add_define("TARGET_SPOT_TID", TARGET_SPOT_TID)
        acs.add_define("TARGET_TID", TARGET_TID)
        acs.add_global_var("reward", 0, "int")
        acs.add_global_var("targets_killed", 50, "int")
        acs.add_global_var("cover_raised", 51, "int")

        acs.add_script(
            ScriptType.OPEN,
            """
            reward = 0;
            targets_killed = 0;
            cover_raised = 0;
            int i;
            for (i = 0; i < COVER_COUNT; i++) {
                if (Random(1, 100) <= COVER_DENSITY) {
                    Floor_MoveToValue(COVER_TAG_BASE + i, 512, COVER_HEIGHT);
                    cover_raised++;
                }
            }
            while (TRUE) {
                if (ThingCount(T_NONE, TARGET_TID) == 0) {
                    SpawnSpot("ArenaTarget", TARGET_SPOT_TID, TARGET_TID, 0);
                }
                Delay(70);
            }
            """,
            number=1,
        )
        acs.add_script(
            ScriptType.VOID,
            """
            targets_killed++;
            reward += 1.0;
            """,
            number=10,
        )
        return acs

    def generate(self, output_path: str) -> None:
        cfg = self.config
        size = int(cfg["arena_size"])
        half = size // 2
        shape = rect(0, 0, size, size) if cfg["arena_shape"] == "rect" else circle(0, 0, half, segments=32)

        builder = ProceduralMapBuilder(
            bounds=(-half - 128, -half - 128, half + 128, half + 128),
            seed=int(cfg["seed"]),
        )
        builder.add_area(
            "Arena",
            shape,
            floor_height=0,
            ceiling_height=224,
            floor_texture="FLOOR0_3",
            ceiling_texture="F_SKY1",
            wall_texture="STONE2",
            light_level=208,
        )

        # Flush candidate cover blocks; ACS lifts a random subset each episode.
        cover_tags = [COVER_TAG_BASE + i for i in range(int(cfg["cover_count"]))]
        obstacles = builder.scatter_obstacles(
            "Arena",
            int(cfg["cover_count"]),
            footprint=(96, 96),
            spacing=112,
            kinds=("box",),
            rotate=True,
            create_areas=True,
            floor_height=0,
            ceiling_height=224,
            floor_texture="FLOOR0_3",
            ceiling_texture="F_SKY1",
            wall_texture="BROWN96",
            tag=cover_tags,
        )

        pairs = builder.pick_spawn_pairs(
            "Arena", 2, min_path_distance=half, spacing=96, obstacles=obstacles
        )
        for p1, p2, _dist in pairs:
            builder.add_thing(ThingType.DEATHMATCH_START, *p1)
            builder.add_thing(ThingType.DEATHMATCH_START, *p2)
        builder.add_thing(ThingType.PLAYER1_START, *pairs[0][0])

        tx, ty, _ = builder.clearance_spots("Arena", 1)[0]
        builder.add_thing("MAP_SPOT", tx, ty, tid=TARGET_SPOT_TID)
        for sx, sy in builder.get_spots("Arena", count=4, clearance=32, obstacles=obstacles):
            builder.add_thing("SHELLS", sx, sy)

        builder.wad_writer.add_lump(
            "DECORATE",
            b"""
            ACTOR ArenaTarget : BaronOfHell 15030 {
                Health 150
                DropItem "None"
                States {
                Death:
                    BOSS H 0 ACS_ExecuteAlways(10, 0)
                    Goto Super::Death
                }
            }
            """,
        )

        acs = self.build_acs(cover_count=len(obstacles))
        builder.map_data.behavior = acs.compile(acc_path=cfg["acc_path"])
        builder.map_data.scripts = acs.to_code()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        builder.build(str(out))


def generate_batch(batch_dir: Path, acc_path: str | None = None) -> list[Path]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    generator = BatchGenerator(
        DeathmatchArenaScenario,
        {
            "seed": [1, 2],
            "arena_shape": ["rect", "circle"],
            "acc_path": [acc_path],
        },
    )
    written: list[Path] = []
    for scenario in generator.generate_scenarios(base_name="arena"):
        out = batch_dir / f"{scenario.name}.wad"
        scenario.generate(str(out))
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-dir", type=Path, default=None, help="emit a seed x shape batch instead")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", choices=("rect", "circle"), default="rect")
    parser.add_argument("--acc", default=None)
    args = parser.parse_args()
    if args.batch_dir is not None:
        generate_batch(args.batch_dir, acc_path=args.acc)
    else:
        DeathmatchArenaScenario(
            {"seed": args.seed, "arena_shape": args.shape, "acc_path": args.acc}
        ).generate(str(args.output))


if __name__ == "__main__":
    main()
