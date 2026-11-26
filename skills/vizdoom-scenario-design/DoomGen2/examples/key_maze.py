"""Seedable grid maze with reachability-checked key, locked door and ACS reward.

Demonstrates deterministic procedural layouts, Door_LockedRaise (special 13),
fixed-point global int 0:reward (+= 1.0) and USER1 exit tracking reset in OPEN.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Any

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "key_maze.wad"


class KeyMazeScenario(Scenario):
    def get_default_config(self) -> dict[str, Any]:
        return {
            "seed": 42,
            "grid_w": 5,
            "grid_h": 5,
            "cell_size": 256,
            "enemy_prob": 0.35,
            "pickup_prob": 0.25,
            "acc_path": None,
        }

    def validate_config(self) -> None:
        if int(self.config["grid_w"]) < 2 or int(self.config["grid_h"]) < 2:
            raise ValueError("grid_w and grid_h must be >= 2")
        if int(self.config["cell_size"]) < 128:
            raise ValueError("cell_size must be >= 128")

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = int(cfg["seed"])
        rng = random.Random(seed)
        grid_w, grid_h = int(cfg["grid_w"]), int(cfg["grid_h"])
        cell = int(cfg["cell_size"])
        offset_x = -(grid_w - 1) * cell / 2
        offset_y = -(grid_h - 1) * cell / 2
        half_span = max(grid_w, grid_h) * cell
        builder = ProceduralMapBuilder(bounds=(-half_span, -half_span, half_span, half_span), seed=seed)

        start_pos = (0, 0)
        exit_pos = (grid_w - 1, grid_h - 1)
        coords: dict[tuple[int, int], tuple[float, float]] = {}

        for gx in range(grid_w):
            for gy in range(grid_h):
                px, py = offset_x + gx * cell, offset_y + gy * cell
                coords[(gx, gy)] = (px, py)
                floor_tex = "FLAT14" if (gx, gy) == start_pos else (
                    "CEIL5_1" if (gx, gy) == exit_pos else "FLOOR4_8"
                )
                builder.add_area(
                    f"Room_{gx}_{gy}",
                    shape=(px, py, cell, cell),
                    floor_height=0,
                    ceiling_height=128,
                    floor_texture=floor_tex,
                    ceiling_texture="CEIL3_5",
                    wall_texture="STARTAN3",
                )

        builder.add_thing("PLAYER1_START", *coords[start_pos], angle=90)

        visited = {start_pos}
        stack = [start_pos]
        connections: list[tuple[tuple[int, int], tuple[int, int]]] = []
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {start_pos: []}

        while stack:
            cx, cy = stack[-1]
            nbrs = [
                (cx + dx, cy + dy)
                for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))
                if 0 <= cx + dx < grid_w and 0 <= cy + dy < grid_h and (cx + dx, cy + dy) not in visited
            ]
            if nbrs:
                nxt = rng.choice(nbrs)
                connections.append(((cx, cy), nxt))
                adj.setdefault((cx, cy), []).append(nxt)
                adj.setdefault(nxt, []).append((cx, cy))
                visited.add(nxt)
                stack.append(nxt)
            else:
                stack.pop()

        # Place Red Keycard in a room reachable from start without entering the exit room.
        reachable: list[tuple[int, int]] = []
        seen = {start_pos}
        queue = deque([start_pos])
        while queue:
            curr = queue.popleft()
            if curr != start_pos:
                reachable.append(curr)
            for nxt in adj.get(curr, []):
                if nxt not in seen and nxt != exit_pos:
                    seen.add(nxt)
                    queue.append(nxt)
        key_pos = rng.choice(reachable)
        builder.add_thing("RED_KEYCARD", *coords[key_pos])

        for p1, p2 in connections:
            room_a, room_b = f"Room_{p1[0]}_{p1[1]}", f"Room_{p2[0]}_{p2[1]}"
            if exit_pos in (p1, p2):
                builder.add_boundary(
                    room_a,
                    room_b,
                    name=f"RedDoor_{room_a}_{room_b}",
                    connection_type=ConnectionType.DOOR,
                    floor_height=0,
                    ceiling_height=0,
                    wall_texture="DOORRED",
                    linedef_special=13,  # Door_LockedRaise
                    linedef_args=[0, 16, 150, 1, 0],  # lock 1 = Red Card
                    linedef_flags=0x600,  # repeatable + player use
                    door_width=128,
                    door_depth=16,
                )
            else:
                builder.connect_adjacent(room_a, room_b, ConnectionType.OPEN)

        for pos, (px, py) in coords.items():
            if pos in (start_pos, exit_pos):
                continue
            if rng.random() < float(cfg["enemy_prob"]):
                builder.add_thing(rng.choice(("IMP", "SHOTGUN_GUY", "ZOMBIEMAN")), px, py)
            if rng.random() < float(cfg["pickup_prob"]):
                builder.add_thing(rng.choice(("MEDIKIT", "SHELLS", "CLIP")), px + 32, py + 32)

        ex, ey = coords[exit_pos]
        builder.add_thing("MAP_SPOT", ex, ey, tid=100)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("reward", 0, "int")
        acs.add_global_var("maze_cleared", 1, "int")
        acs.add_script(
            ScriptType.OPEN,
            """
            reward = 0;
            maze_cleared = 0;
            while (!maze_cleared) {
                int dx = (GetActorX(1000) - GetActorX(100)) >> 16;
                int dy = (GetActorY(1000) - GetActorY(100)) >> 16;
                if (GetActorProperty(1000, APROP_Health) > 0 && dx * dx + dy * dy <= 96 * 96) {
                    maze_cleared = 1;
                    reward += 1.0;
                    Exit_Normal(0);
                }
                Delay(4);
            }
            """,
            number=1,
        )
        acs.add_script(ScriptType.ENTER, "Thing_ChangeTID(0, 1000);", number=2)
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
    KeyMazeScenario({"seed": args.seed, "acc_path": args.acc}).generate(str(args.output))


if __name__ == "__main__":
    main()
