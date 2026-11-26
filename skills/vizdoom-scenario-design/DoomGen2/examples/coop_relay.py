"""Two-player cooperative relay: private HUD clue, grated window and explicit doors.

Player 0 (lane A) sees a private HUD clue logged to their notifications buffer
(SetHudSize + BIGFONT + HUDMSG_LOG). Player 1 (lane B) faces three per-episode
shuffled coloured torches and presses USE next to the matching torch to open
both lanes' progression doors.

Exports (reset in OPEN):
    USER40 relay_secret   1=RED, 2=GREEN, 3=BLUE (diagnostic only)
    USER41 relay_solved   0 until the matching torch is used, then 1
    USER42 relay_wrong    cumulative wrong USE presses
    USER43 relay_clue     1 once the private clue has been shown
    USER44 relay_finished 1 once both players reach the exit rooms
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "coop_relay.wad"


class CoopRelayScenario(Scenario):
    def get_default_config(self) -> dict[str, Any]:
        return {
            "seed": 42,
            "room_width": 512,
            "room_height": 320,
            "exit_width": 256,
            "use_range": 64,
            "acc_path": None,
        }

    def validate_config(self) -> None:
        if int(self.config["room_width"]) < 384 or int(self.config["room_height"]) < 256:
            raise ValueError("room_width must be >= 384 and room_height >= 256")
        if int(self.config["use_range"]) < 32:
            raise ValueError("use_range must be >= 32")

    def build_acs(self, switch_x: int, exit_x: int, use_range: int) -> ACSBuilder:
        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_define("USE_RANGE", use_range)
        acs.add_define("SWITCH_X", switch_x)
        acs.add_define("EXIT_X", exit_x)
        acs.add_global_var("reward", 0, "int")
        acs.add_global_var("relay_secret", 40, "int")
        acs.add_global_var("relay_solved", 41, "int")
        acs.add_global_var("relay_wrong", 42, "int")
        acs.add_global_var("relay_clue", 43, "int")
        acs.add_global_var("relay_finished", 44, "int")
        acs.add_global_code("int slot_color[3];")

        # OPEN resets every exported global and randomises secret + torch order per episode.
        acs.add_script(
            ScriptType.OPEN,
            """
            reward = 0;
            relay_secret = Random(1, 3);
            relay_solved = 0;
            relay_wrong = 0;
            relay_clue = 0;
            relay_finished = 0;

            slot_color[0] = 1;
            slot_color[1] = 2;
            slot_color[2] = 3;
            int i, j, tmp;
            for (i = 2; i > 0; i--) {
                j = Random(0, i);
                tmp = slot_color[i];
                slot_color[i] = slot_color[j];
                slot_color[j] = tmp;
            }

            // Literal actor class names avoid ZDoom 2.8.1 runtime-string SpawnSpot issues.
            if (relay_secret == 1) SpawnSpot("RedTorch", 2000, 3000, 0);
            else if (relay_secret == 2) SpawnSpot("GreenTorch", 2000, 3000, 0);
            else SpawnSpot("BlueTorch", 2000, 3000, 0);

            for (i = 0; i < 3; i++) {
                if (slot_color[i] == 1) SpawnSpot("RedTorch", 2001 + i, 3001 + i, 0);
                else if (slot_color[i] == 2) SpawnSpot("GreenTorch", 2001 + i, 3001 + i, 0);
                else SpawnSpot("BlueTorch", 2001 + i, 3001 + i, 0);
            }
            """,
            number=1,
        )

        setup = """
            int pn = PlayerNumber();
            ClearInventory();
            Thing_ChangeTID(1000 + pn, 0);
            Thing_ChangeTID(0, 1000 + pn);
        """
        acs.add_script(ScriptType.ENTER, setup, number=2)
        acs.add_script(ScriptType.RESPAWN, setup, number=3)

        # Private clue loop runs as Player 0 activator so HUDMSG_LOG only reaches Player 0.
        acs.add_script(
            ScriptType.ENTER,
            """
            if (PlayerNumber() != 0) terminate;
            Delay(5);
            while (!relay_solved) {
                SetHudSize(136, 102, 1);
                SetFont("BIGFONT");
                if (relay_secret == 1)
                    HudMessage(s:"CODE RED"; HUDMSG_PLAIN | HUDMSG_LOG, 1, CR_RED, 68.0, 51.0, 1.3);
                else if (relay_secret == 2)
                    HudMessage(s:"CODE GREEN"; HUDMSG_PLAIN | HUDMSG_LOG, 1, CR_GREEN, 68.0, 51.0, 1.3);
                else
                    HudMessage(s:"CODE BLUE"; HUDMSG_PLAIN | HUDMSG_LOG, 1, CR_BLUE, 68.0, 51.0, 1.3);
                relay_clue = 1;
                Delay(35);
            }
            """,
            number=4,
        )

        # Operator switch monitor + joint exit check.
        acs.add_script(
            ScriptType.OPEN,
            """
            int cooldown = 0;
            while (!relay_finished) {
                if (!relay_solved) {
                    if (cooldown > 0) {
                        cooldown--;
                    } else if (PlayerInGame(1) && (GetPlayerInput(1, INPUT_BUTTONS) & BT_USE)) {
                        int px = GetActorX(1001) >> 16;
                        int py = GetActorY(1001) >> 16;
                        int best = -1;
                        int i;
                        for (i = 0; i < 3; i++) {
                            int dx = px - (GetActorX(2001 + i) >> 16);
                            int dy = py - (GetActorY(2001 + i) >> 16);
                            if (dx < 0) dx = -dx;
                            if (dy < 0) dy = -dy;
                            if (dx <= USE_RANGE && dy <= USE_RANGE) {
                                best = i;
                                break;
                            }
                        }
                        if (best >= 0) {
                            if (slot_color[best] == relay_secret) {
                                relay_solved = 1;
                                reward += 1.0;
                                Door_Open(101, 64);
                                Door_Open(102, 64);
                            } else {
                                relay_wrong++;
                                cooldown = 35;
                            }
                        }
                    }
                } else if (PlayerInGame(0) && PlayerInGame(1)) {
                    if ((GetActorX(1000) >> 16) >= EXIT_X && (GetActorX(1001) >> 16) >= EXIT_X) {
                        relay_finished = 1;
                    }
                }
                Delay(1);
            }
            """,
            number=5,
        )
        return acs

    def generate(self, output_path: str) -> None:
        cfg = self.config
        rw, rh, ew = int(cfg["room_width"]), int(cfg["room_height"]), int(cfg["exit_width"])
        gap = 64
        lane_y = rh // 2 + gap // 2
        door_depth = 16
        door_x = rw // 2 + door_depth // 2
        exit_cx = rw // 2 + door_depth + ew // 2

        builder = ProceduralMapBuilder(bounds=(-1024, -1024, 1536, 1024), seed=int(cfg["seed"]))

        lane_a = builder.add_area("LaneA", (0, lane_y, rw, rh), ceiling_height=160, floor_texture="FLOOR4_8")
        lane_b = builder.add_area("LaneB", (0, -lane_y, rw, rh), ceiling_height=160, floor_texture="FLOOR4_8")

        # Grated viewing booth connecting the two lanes (blocking, see-through).
        window = builder.add_area(
            "Window",
            (-rw // 4, 0, 128, gap),
            floor_height=32,
            ceiling_height=112,
            floor_texture="CEIL5_1",
            wall_texture="MIDGRATE",
        )
        builder.connect_adjacent(lane_a, window, ConnectionType.OPEN)
        builder.connect_adjacent(lane_b, window, ConnectionType.OPEN)

        # Explicit door sectors (more reliable than Voronoi boundary doors).
        door_a = builder.add_area(
            "DoorA", (door_x, lane_y, door_depth, 128), floor_height=0, ceiling_height=0,
            wall_texture="BIGDOOR2", tag=101,
        )
        door_b = builder.add_area(
            "DoorB", (door_x, -lane_y, door_depth, 128), floor_height=0, ceiling_height=0,
            wall_texture="BIGDOOR2", tag=102,
        )
        exit_a = builder.add_area("ExitA", (exit_cx, lane_y, ew, rh), ceiling_height=160, floor_texture="FLAT14")
        exit_b = builder.add_area("ExitB", (exit_cx, -lane_y, ew, rh), ceiling_height=160, floor_texture="FLAT14")

        for a, b in ((lane_a, door_a), (door_a, exit_a), (lane_b, door_b), (door_b, exit_b)):
            builder.connect_adjacent(a, b, ConnectionType.OPEN)

        builder.add_thing(ThingType.PLAYER1_START, -rw // 3, lane_y, angle=0)
        builder.add_thing(ThingType.PLAYER2_START, -rw // 4, -lane_y, angle=0)

        # Clue torch spot (Lane A) and 3 switch spots (Lane B).
        switch_x = rw // 2 - 72
        builder.add_thing("MAP_SPOT", rw // 4, lane_y, tid=2000)
        for idx, dy in enumerate((-96, 0, 96)):
            builder.add_thing("MAP_SPOT", switch_x, -lane_y + dy, tid=2001 + idx)

        acs = self.build_acs(switch_x=switch_x, exit_x=rw // 2 + door_depth, use_range=int(cfg["use_range"]))
        builder.map_data.behavior = acs.compile(acc_path=cfg["acc_path"])
        builder.map_data.scripts = acs.to_code()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        builder.build(str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc", default=None)
    args = parser.parse_args()
    CoopRelayScenario({"seed": args.seed, "acc_path": args.acc}).generate(str(args.output))


if __name__ == "__main__":
    main()
