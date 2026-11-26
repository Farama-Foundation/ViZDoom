import os
import random
from typing import Any, Dict, List, Tuple

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class StealthLabyrinthScenario(Scenario):
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "start_room_size": 320,
            "hub_size": 448,
            "num_branches": 1,
            "rooms_per_branch": 2,
            "branch_room_width": 416,
            "branch_room_length": 896,
            "start_center_y": -560,
            "branch_offset": 980,
            "turret_wall_margin": 144,
            "first_room_turret_wall_margin": 320,
            "corridor_width": 4,
            "min_enemies_per_room": 1,
            "max_enemies_per_room": 1,
            "turret_health": 140,
            "first_room_turret_health": 90,
            "turret_damage": 2,
            "turret_projectile_speed": 14, # hard is 18
            "turret_burst_count": 2,
            "turret_refire_tics": 14,
        }

    def validate_config(self) -> None:
        if int(self.config["start_room_size"]) < 192:
            raise ValueError("start_room_size must be >= 192")
        if int(self.config["hub_size"]) < 256:
            raise ValueError("hub_size must be >= 256")
        if int(self.config["num_branches"]) < 1 or int(self.config["num_branches"]) > 3:
            raise ValueError("num_branches must be in [1, 3]")
        if int(self.config["rooms_per_branch"]) < 1:
            raise ValueError("rooms_per_branch must be >= 1")
        if int(self.config["branch_room_width"]) < 256:
            raise ValueError("branch_room_width must be >= 256")
        if int(self.config["branch_room_length"]) < 512:
            raise ValueError("branch_room_length must be >= 512")
        if int(self.config["start_center_y"]) > -192:
            raise ValueError("start_center_y must stay below the hub")
        if int(self.config["branch_offset"]) < 640:
            raise ValueError("branch_offset must be >= 640")
        if int(self.config["turret_wall_margin"]) < 64:
            raise ValueError("turret_wall_margin must be >= 64")
        if int(self.config["first_room_turret_wall_margin"]) < int(self.config["turret_wall_margin"]):
            raise ValueError("first_room_turret_wall_margin must be >= turret_wall_margin")
        if int(self.config["corridor_width"]) < 3:
            raise ValueError("corridor_width must be >= 3")
        if int(self.config["min_enemies_per_room"]) < 1:
            raise ValueError("min_enemies_per_room must be >= 1")
        if int(self.config["max_enemies_per_room"]) < int(self.config["min_enemies_per_room"]):
            raise ValueError("max_enemies_per_room must be >= min_enemies_per_room")
        if int(self.config["max_enemies_per_room"]) > 2:
            raise ValueError("max_enemies_per_room must be <= 2")
        if int(self.config["turret_health"]) < 40:
            raise ValueError("turret_health must be >= 40")
        if int(self.config["first_room_turret_health"]) < 20:
            raise ValueError("first_room_turret_health must be >= 20")
        if int(self.config["first_room_turret_health"]) > int(self.config["turret_health"]):
            raise ValueError("first_room_turret_health must be <= turret_health")
        if int(self.config["turret_damage"]) < 1:
            raise ValueError("turret_damage must be >= 1")
        if int(self.config["turret_projectile_speed"]) < 8:
            raise ValueError("turret_projectile_speed must be >= 8")
        if int(self.config["turret_burst_count"]) < 1:
            raise ValueError("turret_burst_count must be >= 1")
        if int(self.config["turret_refire_tics"]) < 4:
            raise ValueError("turret_refire_tics must be >= 4")

    @staticmethod
    def _square_check(px: str, py: str, cx: int, cy: int, radius: int) -> str:
        return StealthLabyrinthScenario._rect_check(px, py, cx, cy, radius, radius)

    @staticmethod
    def _rect_check(px: str, py: str, cx: int, cy: int, half_w: int, half_h: int) -> str:
        scale = 1 << 16
        return (
            f"{px} >= {(cx - half_w) * scale} && {px} <= {(cx + half_w) * scale} && "
            f"{py} >= {(cy - half_h) * scale} && {py} <= {(cy + half_h) * scale}"
        )

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)
        random.seed(seed)
        print(f"seed: {seed}")

        start_room_size = int(cfg["start_room_size"])
        hub_size = int(cfg["hub_size"])
        num_branches = int(cfg["num_branches"])
        rooms_per_branch = int(cfg["rooms_per_branch"])
        branch_room_width = int(cfg["branch_room_width"])
        branch_room_length = int(cfg["branch_room_length"])
        start_center_y = int(cfg["start_center_y"])
        branch_offset = int(cfg["branch_offset"])
        turret_wall_margin = int(cfg["turret_wall_margin"])
        first_room_turret_wall_margin = int(cfg["first_room_turret_wall_margin"])
        corridor_width = int(cfg["corridor_width"])
        min_enemies_per_room = int(cfg["min_enemies_per_room"])
        max_enemies_per_room = int(cfg["max_enemies_per_room"])
        turret_health = int(cfg["turret_health"])
        first_room_turret_health = int(cfg["first_room_turret_health"])
        turret_damage = int(cfg["turret_damage"])
        turret_projectile_speed = int(cfg["turret_projectile_speed"])
        turret_burst_count = int(cfg["turret_burst_count"])
        turret_refire_tics = int(cfg["turret_refire_tics"])

        start_center = (0, start_center_y)
        hub_center = (0, 0)
        branch_templates = [
            {"label": "North", "axis": "y", "sign": 1},
            {"label": "West", "axis": "x", "sign": -1},
            {"label": "East", "axis": "x", "sign": 1},
        ]
        active_branches = branch_templates[:num_branches]
        room_step = branch_room_length + 128
        furthest_branch_extent = branch_offset + (rooms_per_branch - 1) * room_step + branch_room_length
        bounds_half = max(
            abs(start_center_y) + start_room_size,
            furthest_branch_extent,
            branch_offset + branch_room_width,
        ) + 512

        builder = ProceduralMapBuilder(
            bounds=(-bounds_half, -bounds_half, bounds_half, bounds_half),
            seed=seed,
        )

        missile_burst_lines = []
        for _ in range(turret_burst_count):
            missile_burst_lines.append('    TROO F 0 A_CustomMissile("SLTurretBolt", 32, 0, 0, 0, 0)')
            missile_burst_lines.append("    TROO F 2 Bright")
        decorate_str = f"""
ACTOR SLTurretBolt 16001 {{
  Radius 6
  Height 8
  Speed {turret_projectile_speed}
  Damage {turret_damage}
  Projectile
  +NOGRAVITY
  +BLOODLESSIMPACT
  RenderStyle Add
  Alpha 0.85
  States {{
  Spawn:
    BAL1 AB 4 Bright
    Loop
  Death:
    BAL1 CDE 4 Bright
    Stop
  }}
}}

ACTOR SLTurretGuard : DoomImp 16000 {{
  Health {turret_health}
  Speed 0
  Radius 24
  Height 56
  PainChance 0
  DropItem "None"
  +DONTTHRUST
  +LOOKALLAROUND
  +MISSILEMORE
  +MISSILEEVENMORE
  States {{
  Spawn:
    TROO AB 5 A_Look
    Loop
  See:
    TROO A 2 A_Chase
    Loop
  Missile:
    TROO E 0 A_FaceTarget
{os.linesep.join(missile_burst_lines)}
    TROO G {turret_refire_tics}
    Goto See
  Pain:
    TROO H 2
    Goto See
  Death:
    TROO I 8
    TROO J 8 A_Scream
    TROO K 6
    TROO L 6 A_NoBlocking
    TROO M 4
    TROO N 1 A_FadeOut(0.10)
    Wait
  }}
}}
"""
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode("utf-8"))

        room_specs: List[Tuple[str, Tuple[int, int], int, int, int, str, str, str]] = [
            ("Start", start_center, 100, start_room_size, start_room_size, "FLOOR4_8", "CEIL3_5", "STONE2"),
            ("Hub", hub_center, 101, hub_size, hub_size, "FLOOR4_8", "CEIL3_5", "STONE2"),
        ]
        branch_specs: List[Dict[str, Any]] = []
        next_room_tag = 102
        for branch_idx, branch in enumerate(active_branches):
            room_names: List[str] = []
            room_tags: List[int] = []
            room_centers: List[Tuple[int, int]] = []
            room_dims: List[Tuple[int, int]] = []
            room_enemy_counts: List[int] = []

            for room_idx in range(rooms_per_branch):
                offset = branch_offset + room_idx * room_step
                if branch["axis"] == "y":
                    center = (0, branch["sign"] * offset)
                    dims = (branch_room_width, branch_room_length)
                else:
                    center = (branch["sign"] * offset, 0)
                    dims = (branch_room_length, branch_room_width)

                room_name = f'{branch["label"]} Room {room_idx + 1}'
                room_tag = next_room_tag
                next_room_tag += 1
                room_specs.append(
                    (room_name, center, room_tag, dims[0], dims[1], "FLOOR5_2", "CEIL3_5", "METAL1")
                )
                room_names.append(room_name)
                room_tags.append(room_tag)
                room_centers.append(center)
                room_dims.append(dims)
                if rooms_per_branch == 1:
                    room_enemy_counts.append(1)
                else:
                    room_enemy_counts.append(random.randint(min_enemies_per_room, max_enemies_per_room))

            branch_specs.append(
                {
                    "label": branch["label"],
                    "axis": branch["axis"],
                    "sign": branch["sign"],
                    "room_names": room_names,
                    "room_tags": room_tags,
                    "room_centers": room_centers,
                    "room_dims": room_dims,
                    "room_enemy_counts": room_enemy_counts,
                    "corr_tags": [],
                }
            )

        area_centers: Dict[str, Tuple[int, int]] = {}
        for name, (cx, cy), tag, width, height, floor_tex, ceil_tex, wall_tex in room_specs:
            area_centers[name] = (cx, cy)
            builder.add_area(
                name=name,
                shape=(cx, cy, width, height),
                floor_height=0,
                ceiling_height=192,
                floor_texture=floor_tex,
                ceiling_texture=ceil_tex,
                wall_texture=wall_tex,
                light_level=0,
                tag=tag,
            )

        corridor_specs = [
            ("Start", "Hub", 200),
        ]
        next_corr_tag = 201
        for branch in branch_specs:
            corridor_specs.append(("Hub", branch["room_names"][0], next_corr_tag))
            branch["corr_tags"].append(next_corr_tag)
            next_corr_tag += 1

            for room_idx in range(1, rooms_per_branch):
                corridor_specs.append(
                    (branch["room_names"][room_idx - 1], branch["room_names"][room_idx], next_corr_tag)
                )
                branch["corr_tags"].append(next_corr_tag)
                next_corr_tag += 1

        # corridor_width is expressed in the legacy ~64-unit cells.
        corridor_width_units = corridor_width * 64
        corridor_centers: Dict[int, Tuple[int, int]] = {}
        for room_a, room_b, tag in corridor_specs:
            corridor_id = builder.add_corridor(
                room_a,
                room_b,
                width=corridor_width_units,
                floor_texture="FLOOR4_8",
                ceiling_height=192,
                wall_texture="METAL1",
                light_level=0,
                tag=tag,
            )
            if not corridor_id:
                raise RuntimeError(f"Could not route corridor between {room_a} and {room_b}")
            ax, ay = area_centers[room_a]
            bx, by = area_centers[room_b]
            corridor_centers[tag] = ((ax + bx) // 2, (ay + by) // 2)

        builder.add_thing(ThingType.PLAYER1_START, -40, start_center[1] - 32, angle=90)
        builder.add_thing(ThingType.PLAYER2_START, 40, start_center[1] - 32, angle=90)
        builder.add_thing(ThingType.PLAYER3_START, -40, start_center[1] + 32, angle=90)
        builder.add_thing(ThingType.PLAYER4_START, 40, start_center[1] + 32, angle=90)

        # enemy_specs: List[Dict[str, Any]] = []
        total_enemy_count = 0
        next_enemy_tid = 3000
        next_spot_tid = 5000
        for branch_idx, branch in enumerate(branch_specs):
            room_checks = []
            for (cx, cy), (width, height) in zip(branch["room_centers"], branch["room_dims"]):
                room_checks.append(
                    self._rect_check("p1x", "p1y", cx, cy, width // 2 + 64, height // 2 + 64)
                )

            corr_checks = []
            for corr_tag in branch["corr_tags"]:
                corr_center = corridor_centers[corr_tag]
                corr_checks.append(self._square_check("p1x", "p1y", corr_center[0], corr_center[1], 240))

            room_enemy_specs: List[List[Dict[str, Any]]] = []
            for room_idx, ((cx, cy), (width, height), enemy_count) in enumerate(
                zip(branch["room_centers"], branch["room_dims"], branch["room_enemy_counts"])
            ):
                room_enemies: List[Dict[str, Any]] = []
                wall_margin = turret_wall_margin
                if room_idx == 0:
                    wall_margin = first_room_turret_wall_margin
                if branch["axis"] == "y":
                    enemy_positions = [
                        (cx, cy + branch["sign"] * (height // 2 - wall_margin)),
                        (cx - width // 4, cy + branch["sign"] * (height // 2 - wall_margin - 96)),
                    ]
                else:
                    enemy_positions = [
                        (cx + branch["sign"] * (width // 2 - wall_margin), cy),
                        (cx + branch["sign"] * (width // 2 - wall_margin - 96), cy - height // 4),
                    ]

                for enemy_slot in range(enemy_count):
                    spot_tid = next_spot_tid
                    tid = next_enemy_tid
                    next_spot_tid += 1
                    next_enemy_tid += 1
                    total_enemy_count += 1
                    pos = enemy_positions[min(enemy_slot, len(enemy_positions) - 1)]
                    builder.add_thing("MAP_SPOT", pos[0], pos[1], tid=spot_tid)
                    room_enemies.append(
                        {
                            "spot_tid": spot_tid,
                            "tid": tid,
                            "aim": (cx << 16, cy << 16),
                            "room_idx": room_idx,
                            "spawn_health": first_room_turret_health if room_idx == 0 else turret_health,
                        }
                    )

                room_enemy_specs.append(room_enemies)

            branch["room_checks"] = room_checks
            branch["corr_checks"] = corr_checks
            branch["room_enemy_specs"] = room_enemy_specs

        builder.add_thing(ThingType.TECH_LAMP, start_center[0] - 96, start_center[1] - 32)
        builder.add_thing(ThingType.TECH_LAMP, start_center[0] + 96, start_center[1] - 32)
        builder.add_thing(ThingType.TALL_RED_PILLAR, hub_center[0], hub_center[1])
        for branch in branch_specs:
            for room_idx, ((cx, cy), (width, height)) in enumerate(zip(branch["room_centers"], branch["room_dims"])):
                if branch["axis"] == "y":
                    lamp_a = (cx - min(120, width // 4), cy - min(220, height // 4))
                    lamp_b = (cx + min(120, width // 4), cy + min(60, height // 6))
                else:
                    lamp_a = (cx + min(220, width // 4), cy - min(120, height // 4))
                    lamp_b = (cx - min(60, width // 6), cy + min(120, height // 4))
                builder.add_thing(ThingType.FLOOR_LAMP, lamp_a[0], lamp_a[1])
                builder.add_thing(ThingType.FLOOR_LAMP, lamp_b[0], lamp_b[1])

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("sl_started", 1, "int")
        acs.add_global_var("sl_p1_alive", 41, "int")
        acs.add_global_var("sl_p2_alive", 42, "int")
        acs.add_global_var("sl_team_dist", 43, "int")
        acs.add_global_var("sl_enemies_left", 44, "int")
        acs.add_global_var("sl_target_lit", 45, "int")
        acs.add_global_var("sl_enemies_dead", 46, "int")
        acs.add_global_var("sl_total_enemies", 47, "int")
        acs.add_global_var("sl_rooms_seen", 48, "int")
        acs.add_global_var("sl_total_rooms", 49, "int")
        acs.add_global_var("sl_team_hp", 50, "int")
        acs.add_global_var("sl_targets_lit", 51, "int")

        enemy_state_init_lines = []
        branch_light_lines = []
        branch_logic_lines = []
        room_seen_init_lines = []
        target_lit_init_lines = []
        for branch_idx, branch in enumerate(branch_specs):
            lit_var = f"branch_attack_lit_{branch_idx}"
            active_room_var = f"branch_active_room_{branch_idx}"
            room_light_lines = []
            room_seen_case_lines = []
            room_pending_lines = []
            room_spawn_lines = []
            for room_idx, (room_tag, room_check) in enumerate(zip(branch["room_tags"], branch["room_checks"])):
                seen_var = f"branch_room_seen_{branch_idx}_{room_idx}"
                lit_seen_var = f"lit_seen_{branch_idx}_{room_idx}"
                room_seen_init_lines.append(f"    int {seen_var} = 0;\n")
                target_lit_init_lines.append(f"    int {lit_seen_var} = 0;\n")
                room_seen_case_lines.append(
                    f"""
            if (!{seen_var} && {active_room_var} == {room_idx} && sl_p1_alive && sl_p2_alive && ({room_check})) {{
                {seen_var} = 1;
                rooms_seen_count++;
            }}"""
                )
                room_var = f"branch_room_lit_{branch_idx}_{room_idx}"
                room_light_lines.append(
                    f"""
            int {room_var} = 0;
            if (sl_p1_alive && ({room_check})) {{
                {room_var} = 1;
            }}
            if ({room_var}) {{
                Light_ChangeToValue({room_tag}, 224);
            }} else {{
                Light_ChangeToValue({room_tag}, 0);
            }}"""
                )
            corridor_light_lines = []
            for corr_idx, (corr_tag, corr_check) in enumerate(zip(branch["corr_tags"], branch["corr_checks"])):
                corr_var = f"branch_corr_lit_{branch_idx}_{corr_idx}"
                corridor_light_lines.append(
                    f"""
            int {corr_var} = 0;
            if (sl_p1_alive && ({corr_check})) {{
                {corr_var} = 1;
            }}
            if ({corr_var}) {{
                Light_ChangeToValue({corr_tag}, 176);
            }} else {{
                    Light_ChangeToValue({corr_tag}, 0);
            }}"""
                )
            for room_idx, room_enemies in enumerate(branch["room_enemy_specs"]):
                room_pending_lines.append(f"\n            int branch_room_pending_{branch_idx}_{room_idx} = 0;")
                for enemy in room_enemies:
                    temp_tid = 7000 + enemy["tid"]
                    spawned_var = f"enemy_spawned_{enemy['tid']}"
                    dead_var = f"enemy_dead_{enemy['tid']}"
                    enemy_state_init_lines.append(f"    int {spawned_var} = 0;\n")
                    enemy_state_init_lines.append(f"    int {dead_var} = 0;\n")
                    room_pending_lines.append(
                        f"""
            if ({spawned_var} && !{dead_var}) {{
                if (ThingCount(T_NONE, {enemy["tid"]}) <= 0) {{
                    {dead_var} = 1;
                }} else if (GetActorProperty({enemy["tid"]}, APROP_Health) <= 0) {{
                    {dead_var} = 1;
                }}
            }}
            if (!{dead_var}) {{
                branch_room_pending_{branch_idx}_{room_idx} = 1;
            }}"""
                    )
                    room_spawn_lines.append(
                        f"""
            if ({active_room_var} == {room_idx} && !{spawned_var} && !{dead_var}) {{
                int spawn_ok_{enemy["tid"]} = SpawnSpotFacing("SLTurretGuard", {enemy["spot_tid"]}, {temp_tid});
                if (spawn_ok_{enemy["tid"]} && ThingCount(T_NONE, {temp_tid}) > 0) {{
                    Thing_ChangeTID({temp_tid}, {enemy["tid"]});
                    SetActorProperty({enemy["tid"]}, APROP_Health, {enemy["spawn_health"]});
                    SetActorProperty({enemy["tid"]}, APROP_Invulnerable, 1);
                    {spawned_var} = 1;
                }}
            }}"""
                    )
            branch_light_lines.append(
                f"""
            {"".join(room_light_lines)}
            {"".join(corridor_light_lines)}
            {"".join(room_pending_lines)}
            int {active_room_var} = -1;
"""
            )
            for room_idx, _room_enemies in enumerate(branch["room_enemy_specs"]):
                branch_light_lines.append(
                    f"""
            if ({active_room_var} < 0 && branch_room_pending_{branch_idx}_{room_idx}) {{
                {active_room_var} = {room_idx};
            }}"""
                )
            lit_case_lines = []
            for room_idx, (room_check, corr_check) in enumerate(zip(branch["room_checks"], branch["corr_checks"])):
                lit_seen_var = f"lit_seen_{branch_idx}_{room_idx}"
                lit_case_lines.append(
                    f"""
            if ({active_room_var} == {room_idx} && sl_p1_alive && (({room_check}) || ({corr_check}))) {{
                {lit_var} = 1;
                if (!{lit_seen_var} && sl_p2_alive) {{
                    {lit_seen_var} = 1;
                    targets_lit_count++;
                }}
            }}"""
                )
            branch_light_lines.append(
                f"""
            int {lit_var} = 0;
            {"".join(room_spawn_lines)}
            {"".join(room_seen_case_lines)}
            {"".join(lit_case_lines)}"""
            )
            room_logic_lines = []
            for room_idx, room_enemies in enumerate(branch["room_enemy_specs"]):
                for enemy in room_enemies:
                    spawned_var = f"enemy_spawned_{enemy['tid']}"
                    dead_var = f"enemy_dead_{enemy['tid']}"
                    room_logic_lines.append(
                        f"""
            if (!{dead_var}) {{
                enemies_left++;
            }}
            if ({spawned_var} && !{dead_var}) {{
                if (ThingCount(T_NONE, {enemy["tid"]}) <= 0) {{
                    {dead_var} = 1;
                }} else if (GetActorProperty({enemy["tid"]}, APROP_Health) <= 0) {{
                    {dead_var} = 1;
                }}
            }}
            if ({spawned_var} && !{dead_var}) {{
                if ({active_room_var} == {room_idx} && {lit_var}) {{
                    SetActorProperty({enemy["tid"]}, APROP_Invulnerable, 0);
                    sl_target_lit = 1;

                    target_tid = 0;
                    if (sl_p1_alive && sl_p2_alive) {{
                        int d1_{enemy["tid"]} = SLDist({enemy["aim"][0]}, {enemy["aim"][1]}, p1x, p1y);
                        int d2_{enemy["tid"]} = SLDist({enemy["aim"][0]}, {enemy["aim"][1]}, p2x, p2y);
                        if (d2_{enemy["tid"]} < d1_{enemy["tid"]}) {{
                            target_tid = 1001;
                        }} else {{
                            target_tid = 1000;
                        }}
                    }} else if (sl_p1_alive) {{
                        target_tid = 1000;
                    }} else if (sl_p2_alive) {{
                        target_tid = 1001;
                    }}

                    if (target_tid > 0) {{
                        Thing_Hate({enemy["tid"]}, target_tid, 6);
                    }}
                }} else {{
                    SetActorProperty({enemy["tid"]}, APROP_Invulnerable, 1);
                }}
            }}"""
                    )
            branch_logic_lines.append("".join(room_logic_lines))

        start_room_check = self._square_check(
            "p1x", "p1y", start_center[0], start_center[1], start_room_size // 2 + 72
        )
        hub_room_check = self._square_check(
            "p1x", "p1y", hub_center[0], hub_center[1], hub_size // 2 + 72
        )
        start_corr_check = self._square_check(
            "p1x", "p1y", corridor_centers[200][0], corridor_centers[200][1], 220
        )

        script_setup = """
        int pn = PlayerNumber();
        int tid = 1000 + pn;
        Thing_ChangeTID(0, tid);

        SetActorProperty(0, APROP_Health, 100);
        SetActorProperty(0, APROP_SpawnHealth, 100);
        SetActorProperty(0, APROP_Speed, 1.0);
        ClearInventory();

        if (pn == 0) {
            sl_started = 0;
            sl_p1_alive = 0;
            sl_p2_alive = 0;
            sl_team_dist = 0;
            sl_enemies_left = __ENEMY_COUNT__;
            sl_target_lit = 0;
            sl_enemies_dead = 0;
            sl_total_enemies = __ENEMY_COUNT__;
            sl_rooms_seen = 0;
            sl_total_rooms = __ROOM_COUNT__;
            sl_team_hp = 200;
            sl_targets_lit = 0;

            TakeInventory("Fist", 999);
            TakeInventory("Chainsaw", 999);
            TakeInventory("Pistol", 999);
            TakeInventory("Shotgun", 999);
            TakeInventory("SuperShotgun", 999);
            TakeInventory("Chaingun", 999);
            TakeInventory("RocketLauncher", 999);
            TakeInventory("PlasmaRifle", 999);
            TakeInventory("BFG9000", 999);
            TakeInventory("Clip", 9999);
            TakeInventory("Shell", 9999);
            TakeInventory("RocketAmmo", 9999);
            TakeInventory("Cell", 9999);
        } else if (pn == 1) {
            SetAmmoCapacity("Clip", 10000);
            GiveInventory("Chaingun", 1);
            GiveInventory("Clip", 240);
            SetWeapon("Chaingun");
        }

        ACS_ExecuteAlways(10, 0);
        """
        script_setup = script_setup.replace("__ENEMY_COUNT__", str(total_enemy_count))
        script_setup = script_setup.replace("__ROOM_COUNT__", str(num_branches * rooms_per_branch))
        acs.add_script(ScriptType.ENTER, script_setup, number=1)
        acs.add_script(ScriptType.RESPAWN, script_setup, number=3)

        script_open = f"""
#define TOTAL_ENEMIES __ENEMY_COUNT__
#define TOTAL_ROOMS __ROOM_COUNT__

function int SLAlive(int tid)
{{
    if (tid <= 0) return 0;
    if (ThingCount(T_NONE, tid) <= 0) return 0;
    return GetActorProperty(tid, APROP_Health) > 0;
}}

function int SLDist(int ax, int ay, int bx, int by)
{{
    int dx = ax - bx;
    int dy = ay - by;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    if (dy > dx) dx = dy;
    return dx >> 16;
}}

script 2 OPEN
{{
    int p1x, p1y, p2x, p2y;
    int p1hp, p2hp;
    int enemies_left;
    int target_tid;
    int rooms_seen_count = 0;
    int targets_lit_count = 0;
    int startup_ready_loops = 0;
{"".join(room_seen_init_lines)}
{"".join(target_lit_init_lines)}
{"".join(enemy_state_init_lines)}

    sl_started = 0;
    sl_total_enemies = TOTAL_ENEMIES;
    sl_total_rooms = TOTAL_ROOMS;
    sl_p1_alive = 0;
    sl_p2_alive = 0;
    sl_team_dist = 0;
    sl_enemies_left = TOTAL_ENEMIES;
    sl_target_lit = 0;
    sl_enemies_dead = 0;
    sl_rooms_seen = 0;
    sl_team_hp = 200;
    sl_targets_lit = 0;

    while (TRUE)
    {{
        sl_p1_alive = SLAlive(1000);
        sl_p2_alive = SLAlive(1001);

        if (sl_p1_alive)
        {{
            p1x = GetActorX(1000);
            p1y = GetActorY(1000);
            TakeActorInventory(1000, "Fist", 999);
            TakeActorInventory(1000, "Chainsaw", 999);
            TakeActorInventory(1000, "Pistol", 999);
            TakeActorInventory(1000, "Shotgun", 999);
            TakeActorInventory(1000, "SuperShotgun", 999);
            TakeActorInventory(1000, "Chaingun", 999);
            TakeActorInventory(1000, "RocketLauncher", 999);
            TakeActorInventory(1000, "PlasmaRifle", 999);
            TakeActorInventory(1000, "BFG9000", 999);
            TakeActorInventory(1000, "Clip", 9999);
            TakeActorInventory(1000, "Shell", 9999);
            TakeActorInventory(1000, "RocketAmmo", 9999);
            TakeActorInventory(1000, "Cell", 9999);
        }}
        else
        {{
            p1x = 0;
            p1y = 0;
        }}

        if (sl_p2_alive)
        {{
            p2x = GetActorX(1001);
            p2y = GetActorY(1001);
        }}
        else
        {{
            p2x = 0;
            p2y = 0;
        }}

        if (!sl_started)
        {{
            if (sl_p1_alive && sl_p2_alive)
            {{
                startup_ready_loops++;
                if (startup_ready_loops >= 4)
                {{
                    sl_started = 1;
                }}
            }}
            else
            {{
                startup_ready_loops = 0;
            }}
        }}

        if (sl_p1_alive && sl_p2_alive)
        {{
            sl_team_dist = SLDist(p1x, p1y, p2x, p2y);
        }}
        else
        {{
            sl_team_dist = 0;
        }}

        p1hp = 0;
        p2hp = 0;
        if (sl_p1_alive)
        {{
            p1hp = GetActorProperty(1000, APROP_Health);
            if (p1hp < 0) p1hp = 0;
        }}
        if (sl_p2_alive)
        {{
            p2hp = GetActorProperty(1001, APROP_Health);
            if (p2hp < 0) p2hp = 0;
        }}
        sl_team_hp = p1hp + p2hp;

        if (sl_p1_alive && (({start_room_check}) || ({start_corr_check})))
        {{
            Light_ChangeToValue(100, 224);
        }}
        else
        {{
            Light_ChangeToValue(100, 0);
        }}

        if (sl_p1_alive && ({hub_room_check}))
        {{
            Light_ChangeToValue(101, 160);
        }}
        else
        {{
            Light_ChangeToValue(101, 0);
        }}

        if (sl_p1_alive && ({start_corr_check}))
        {{
            Light_ChangeToValue(200, 160);
        }}
        else
        {{
            Light_ChangeToValue(200, 0);
        }}

        sl_target_lit = 0;
        {"".join(branch_light_lines)}
        sl_rooms_seen = rooms_seen_count;
        sl_targets_lit = targets_lit_count;

        enemies_left = 0;
        {"".join(branch_logic_lines)}
        sl_enemies_left = enemies_left;
        sl_enemies_dead = TOTAL_ENEMIES - enemies_left;

        if (sl_started && (!sl_p1_alive || !sl_p2_alive))
        {{
            Delay(1);
            Exit_Normal(0);
        }}
        if (sl_started && sl_enemies_dead >= TOTAL_ENEMIES)
        {{
            Delay(1);
            Exit_Normal(0);
        }}

        Delay(2);
    }}
}}
"""
        script_open = script_open.replace("__ENEMY_COUNT__", str(total_enemy_count))
        script_open = script_open.replace("__ROOM_COUNT__", str(num_branches * rooms_per_branch))
        acs.add_global_code(script_open)

        script_hud = """
        int self_tid = ActivatorTID();
        while (TRUE) {
            if (self_tid > 0 && ThingCount(T_NONE, self_tid) <= 0) {
                break;
            }

            SetHudSize(320, 200, 1);
            HudMessage(
                d:sl_enemies_left;
                HUDMSG_PLAIN, 30, CR_WHITE, 160.0, 12.0, 0.0
            );
            if (sl_target_lit) {
                HudMessage(
                    s:"LIT";
                    HUDMSG_PLAIN, 31, CR_GOLD, 160.0, 24.0, 0.0
                );
            } else {
                HudMessage(
                    s:"DARK";
                    HUDMSG_PLAIN, 31, CR_WHITE, 160.0, 24.0, 0.0
                );
            }
            Delay(1);
        }
        """
        acs.add_script(ScriptType.VOID, script_hud, number=10)

        builder.map_data.scripts = acs.to_code()
        try:
            builder.map_data.behavior = acs.compile()
        except Exception as e:
            print(f"ACS compile warning: {e}")
            if hasattr(e, "stderr") and e.stderr:
                print(f"ACC stderr: {e.stderr}")

        builder.build(output_path)


if __name__ == "__main__":
    out_dir = os.path.join("examples", "benchmark", "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stealth_labyrinth.wad")
    print(f"Generating {path}...")
    StealthLabyrinthScenario().generate(path)
    print("Done.")
