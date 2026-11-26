import os
import random
import math
from typing import Any, Dict, List, Tuple

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.geometry.shapes import rect
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class LavaPitScenario(Scenario):
    @staticmethod
    def _max_plate_size_for_platform(spec: Dict[str, int], fallback: int) -> int:
        return max(
            fallback,
            min(
                spec["width"] // 2 - 20,
                spec["depth"] - 24,
            ),
        )

    @staticmethod
    def _plate_layout_for_platform(
        spec: Dict[str, int],
        plate_size: int,
        bootstrap_only: bool,
        max_candidate_count: int = 4,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        half_width = spec["width"] // 2
        half_depth = spec["depth"] // 2

        center_pad_x = 8
        edge_pad_x = 12
        edge_pad_y = 12
        min_vertical_gap = 4

        min_x = (plate_size // 2) + center_pad_x
        max_x = half_width - (plate_size // 2) - edge_pad_x
        if max_x < min_x:
            fallback_x = max(plate_size // 2, half_width - (plate_size // 2))
            x_centers = [fallback_x]
        elif max_x - min_x >= plate_size:
            x_centers = [min_x, max_x]
        else:
            x_centers = [(min_x + max_x) // 2]

        max_y = half_depth - (plate_size // 2) - edge_pad_y
        min_row_offset = (plate_size // 2) + min_vertical_gap
        if max_y >= min_row_offset:
            row_offset = min(max_y, max(min_row_offset, int(spec["depth"] * 0.18)))
            y_centers = [-row_offset, row_offset]
        else:
            y_centers = [0]

        bootstrap_positions = [(x_centers[-1], 0)]

        if bootstrap_only:
            return bootstrap_positions, bootstrap_positions

        randomized_positions: List[Tuple[int, int]] = []
        if len(x_centers) == 2 and len(y_centers) == 2:
            randomized_positions = [
                (x_centers[0], y_centers[0]),
                (x_centers[1], y_centers[0]),
                (x_centers[0], y_centers[1]),
                (x_centers[1], y_centers[1]),
            ]
        elif len(x_centers) == 2:
            randomized_positions = [(x_centers[0], 0), (x_centers[1], 0)]
        elif len(y_centers) == 2:
            randomized_positions = [(x_centers[0], y_centers[0]), (x_centers[0], y_centers[1])]
        else:
            randomized_positions = bootstrap_positions

        return bootstrap_positions, randomized_positions[:max_candidate_count]

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "random_seed_on_generate": True,
            "num_platforms": 12,
            "max_players": 2,
            "bootstrap_bridges": 3,
            "first_handoff_platform_width": 560,
            "first_handoff_platform_depth": 360,
            "first_handoff_gap": 160,
            "first_handoff_plate_scale": 1.9,
            "bridge0_support_latch_tics": 105,
            "platform_width_min": 360,
            "platform_width_max": 460,
            "platform_depth_min": 240,
            "platform_depth_max": 320,
            "gap_min": 260,
            "gap_max": 360,
            "lane_jitter": 48,
            "lane_span": 360,
            "curve_amplitude": 220,
            "curve_waves": 1,
            "platform_floor_height": 0,
            "lava_height": -80,
            "lava_damage": 100,
            "poll_tics": 1,
            "runtime_jitter_tics": 0,
        }

    def validate_config(self) -> None:
        platforms = int(self.config["num_platforms"])
        if platforms < 3 or platforms > 200:
            raise ValueError("num_platforms must be in range [3, 200]")

        max_players = int(self.config["max_players"])
        if max_players < 1 or max_players > 2:
            raise ValueError("max_players must be in range [1, 2]")

        if int(self.config["platform_width_min"]) < 96:
            raise ValueError("platform_width_min must be >= 96")
        if int(self.config["platform_width_max"]) < int(self.config["platform_width_min"]):
            raise ValueError("platform_width_max must be >= platform_width_min")

        if int(self.config["platform_depth_min"]) < 96:
            raise ValueError("platform_depth_min must be >= 96")
        if int(self.config["platform_depth_max"]) < int(self.config["platform_depth_min"]):
            raise ValueError("platform_depth_max must be >= platform_depth_min")

        if int(self.config["gap_min"]) < 112:
            raise ValueError("gap_min must be >= 112")
        if int(self.config["gap_max"]) < int(self.config["gap_min"]):
            raise ValueError("gap_max must be >= gap_min")

        if int(self.config["curve_amplitude"]) < 0:
            raise ValueError("curve_amplitude must be >= 0")
        if int(self.config["curve_waves"]) < 1:
            raise ValueError("curve_waves must be >= 1")
        if int(self.config["poll_tics"]) < 1:
            raise ValueError("poll_tics must be >= 1")
        if int(self.config["runtime_jitter_tics"]) < 0:
            raise ValueError("runtime_jitter_tics must be >= 0")
        if int(self.config["bootstrap_bridges"]) < 0:
            raise ValueError("bootstrap_bridges must be >= 0")
        if int(self.config["first_handoff_platform_width"]) < 128:
            raise ValueError("first_handoff_platform_width must be >= 128")
        if int(self.config["first_handoff_platform_depth"]) < 120:
            raise ValueError("first_handoff_platform_depth must be >= 120")
        if int(self.config["first_handoff_gap"]) < 112:
            raise ValueError("first_handoff_gap must be >= 112")
        if float(self.config["first_handoff_plate_scale"]) < 1.0:
            raise ValueError("first_handoff_plate_scale must be >= 1.0")
        if int(self.config["bridge0_support_latch_tics"]) < 0:
            raise ValueError("bridge0_support_latch_tics must be >= 0")

    def _choose_seed(self) -> int:
        cfg = self.config
        if cfg["seed"] is not None:
            return int(cfg["seed"])
        if bool(cfg.get("random_seed_on_generate", True)):
            return random.randint(0, 999999)
        return 42

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = self._choose_seed()
        rng = random.Random(seed)

        num_platforms = int(cfg["num_platforms"])
        max_players = int(cfg["max_players"])
        bootstrap_bridges = min(max(0, int(cfg.get("bootstrap_bridges", 0))), max(0, num_platforms - 1))
        platform_floor_height = int(cfg["platform_floor_height"])
        lava_height = int(cfg["lava_height"])
        lava_damage = int(cfg["lava_damage"])
        poll_tics = int(cfg["poll_tics"])
        runtime_jitter_tics = int(cfg["runtime_jitter_tics"])

        platform_width_min = int(cfg["platform_width_min"])
        platform_width_max = int(cfg["platform_width_max"])
        platform_depth_min = int(cfg["platform_depth_min"])
        platform_depth_max = int(cfg["platform_depth_max"])
        gap_min = int(cfg["gap_min"])
        gap_max = int(cfg["gap_max"])
        first_handoff_platform_width = int(cfg["first_handoff_platform_width"])
        first_handoff_platform_depth = int(cfg["first_handoff_platform_depth"])
        first_handoff_gap = int(cfg["first_handoff_gap"])
        first_handoff_plate_scale = float(cfg["first_handoff_plate_scale"])
        bridge0_support_latch_tics = int(cfg["bridge0_support_latch_tics"])
        lane_jitter = int(cfg["lane_jitter"])
        lane_span = int(cfg["lane_span"])
        curve_amplitude = int(cfg["curve_amplitude"])
        curve_waves = int(cfg["curve_waves"])

        # Keep generated geometry inside classic Doom-safe coordinate budget
        # when num_platforms is large (e.g. 120+).
        safe_chain_span = 44000
        est_span = max(1, num_platforms) * (platform_width_max + gap_max)
        scale = min(1.0, safe_chain_span / est_span)

        # Preserve larger platform footprints while shrinking gaps more aggressively.
        size_scale = min(1.0, max(0.78, scale ** 0.55))
        gap_scale = min(1.0, max(0.28, scale ** 1.60))
        lane_scale = min(1.0, max(0.45, scale ** 1.10))

        eff_platform_width_min = max(128, int(round(platform_width_min * size_scale)))
        eff_platform_width_max = max(
            eff_platform_width_min, int(round(platform_width_max * size_scale))
        )
        eff_platform_depth_min = max(120, int(round(platform_depth_min * size_scale)))
        eff_platform_depth_max = max(
            eff_platform_depth_min, int(round(platform_depth_max * size_scale))
        )
        eff_gap_min = max(112, int(round(gap_min * gap_scale)))
        eff_gap_max = max(eff_gap_min, int(round(gap_max * gap_scale)))
        eff_first_handoff_platform_width = max(
            eff_platform_width_min, int(round(first_handoff_platform_width * size_scale))
        )
        eff_first_handoff_platform_depth = max(
            eff_platform_depth_min, int(round(first_handoff_platform_depth * size_scale))
        )
        eff_first_handoff_gap = max(112, int(round(first_handoff_gap * gap_scale)))
        eff_lane_jitter = max(0, int(round(lane_jitter * lane_scale)))
        eff_lane_span = max(120, int(round(lane_span * lane_scale)))
        eff_curve_amplitude = max(0, int(round(curve_amplitude * lane_scale)))

        platform_specs: List[Dict[str, int]] = []
        gaps = [rng.randint(eff_gap_min, eff_gap_max) for _ in range(num_platforms - 1)]
        if gaps:
            gaps[0] = min(gaps[0], eff_first_handoff_gap)
        if bootstrap_bridges > 0:
            easy_gap_max = min(eff_gap_max, eff_gap_min + 48)
            for i in range(min(bootstrap_bridges, len(gaps))):
                gaps[i] = rng.randint(eff_gap_min, easy_gap_max)
            gaps[0] = min(gaps[0], eff_first_handoff_gap)

        for i in range(num_platforms):
            width = rng.randint(eff_platform_width_min, eff_platform_width_max)
            depth = rng.randint(eff_platform_depth_min, eff_platform_depth_max)
            if i < 2:
                width = max(width, eff_first_handoff_platform_width)
                depth = max(depth, eff_first_handoff_platform_depth)

            if i == 0:
                x = 0
                y = 0
            else:
                prev = platform_specs[i - 1]
                x = int(prev["x"] + (prev["width"] // 2) + gaps[i - 1] + (width // 2))
                if i <= bootstrap_bridges:
                    curve_offset = 0
                    early_jitter = min(eff_lane_jitter, 24)
                    random_walk_y = prev["y"] + rng.randint(-early_jitter, early_jitter)
                else:
                    t = i / max(1, num_platforms - 1)
                    curve_offset = int(
                        math.sin(t * 2.0 * math.pi * curve_waves) * eff_curve_amplitude
                    )
                    random_walk_y = prev["y"] + rng.randint(-eff_lane_jitter, eff_lane_jitter)
                y = int((2 * random_walk_y + curve_offset) / 3)
                y = max(-eff_lane_span, min(eff_lane_span, y))

            platform_specs.append({"x": x, "y": y, "width": width, "depth": depth})

        min_x_raw = min(spec["x"] - spec["width"] // 2 for spec in platform_specs)
        max_x_raw = max(spec["x"] + spec["width"] // 2 for spec in platform_specs)
        min_y_raw = min(spec["y"] - spec["depth"] // 2 for spec in platform_specs)
        max_y_raw = max(spec["y"] + spec["depth"] // 2 for spec in platform_specs)

        center_shift_x = -((min_x_raw + max_x_raw) // 2)
        center_shift_y = -((min_y_raw + max_y_raw) // 2)
        for spec in platform_specs:
            spec["x"] += center_shift_x
            spec["y"] += center_shift_y

        min_x = min(spec["x"] - spec["width"] // 2 for spec in platform_specs)
        max_x = max(spec["x"] + spec["width"] // 2 for spec in platform_specs)
        min_y = min(spec["y"] - spec["depth"] // 2 for spec in platform_specs)
        max_y = max(spec["y"] + spec["depth"] // 2 for spec in platform_specs)

        margin = max(280, int(round(560 * scale)))
        bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
        lava_center_x = (min_x + max_x) // 2
        lava_center_y = (min_y + max_y) // 2
        lava_width = (max_x - min_x) + (2 * margin)
        lava_depth = (max_y - min_y) + (2 * margin)

        builder = ProceduralMapBuilder(bounds=bounds, seed=seed)

        lava_tag = 9000
        bridge_tag_base = 9100
        green_plate_tag_base = 9300
        red_plate_tag_base = 9500
        outer_ceiling_height = 512
        outer_ceiling_texture = "CEIL5_2"
        platform_floor_texture = "FLOOR5_1"
        goal_floor_texture = "CEIL5_2"
        bridge_floor_texture = "FLAT5_4"
        platform_wall_texture = "STARTAN3"

        # Linedefs longer than 32767 units overflow ZDBSP, so very long seas are
        # split into same-tagged strips joined by flush portals.
        lava_strip_count = max(1, math.ceil(lava_width / 16000))
        lava_strip_edges = [
            bounds[0] + (lava_width * k) // lava_strip_count for k in range(lava_strip_count + 1)
        ]
        for strip in range(lava_strip_count):
            strip_left = lava_strip_edges[strip]
            strip_right = lava_strip_edges[strip + 1]
            builder.add_area(
                name="LavaSea" if strip == 0 else f"LavaSea_{strip:02d}",
                shape=rect(
                    (strip_left + strip_right) / 2,
                    lava_center_y,
                    strip_right - strip_left,
                    lava_depth,
                ),
                floor_height=lava_height,
                ceiling_height=outer_ceiling_height,
                floor_texture="LAVA1",
                ceiling_texture=outer_ceiling_texture,
                wall_texture="ASHWALL2",
                lower_texture="ASHWALL2",
                upper_texture="ASHWALL2",
                light_level=200,
                tag=lava_tag,
                mode="overwrite",
            )
        platform_names: List[str] = []

        for i, spec in enumerate(platform_specs):
            name = f"Platform_{i:02d}"
            is_goal_platform = i == num_platforms - 1
            builder.add_area(
                name=name,
                shape=rect(spec["x"], spec["y"], spec["width"], spec["depth"]),
                floor_height=platform_floor_height,
                ceiling_height=outer_ceiling_height,
                floor_texture=goal_floor_texture if is_goal_platform else platform_floor_texture,
                ceiling_texture=outer_ceiling_texture,
                wall_texture=platform_wall_texture,
                lower_texture=platform_wall_texture,
                upper_texture=platform_wall_texture,
                light_level=252 if is_goal_platform else 232,
                tag=7000 + i,
                mode="overwrite",
            )
            platform_names.append(name)

        bridge_tags: List[int] = []
        green_plate_positions_for_bridge: List[List[Tuple[int, int]]] = []
        red_plate_positions_for_bridge: List[List[Tuple[int, int]]] = []
        green_plate_tags_for_bridge: List[List[int]] = []
        red_plate_tags_for_bridge: List[List[int]] = []
        green_plate_sizes_for_bridge: List[int] = []
        red_plate_sizes_for_bridge: List[int] = []
        max_plate_candidate_count = 4

        for i in range(num_platforms - 1):
            left = platform_specs[i]
            right = platform_specs[i + 1]

            left_edge = left["x"] + left["width"] // 2
            right_edge = right["x"] - right["width"] // 2
            bx = (left_edge + right_edge) // 2
            by = (left["y"] + right["y"]) // 2

            x_span = right_edge - left_edge
            bridge_w = int(max(96, x_span + 72))
            bridge_h = int(max(108, abs(right["y"] - left["y"]) + 88))

            bridge_tag = bridge_tag_base + i
            builder.add_area(
                name=f"Bridge_{i:02d}",
                shape=rect(bx, by, bridge_w, bridge_h),
                floor_height=lava_height,
                ceiling_height=outer_ceiling_height,
                floor_texture="LAVA1",
                ceiling_texture=outer_ceiling_texture,
                wall_texture=platform_wall_texture,
                lower_texture=platform_wall_texture,
                upper_texture=platform_wall_texture,
                light_level=176,
                tag=bridge_tag,
                mode="overwrite",
            )
            bridge_tags.append(bridge_tag)

        base_plate_size = 72 if num_platforms >= 60 else 80
        min_half_width = min(spec["width"] // 2 for spec in platform_specs)
        min_depth = min(spec["depth"] for spec in platform_specs)
        plate_size = max(
            24,
            min(
                base_plate_size,
                min_half_width - 8,
                min_depth - 24,
            ),
        )
        green_plate_positions_for_bridge = []
        red_plate_positions_for_bridge = []
        green_plate_tags_for_bridge = []
        red_plate_tags_for_bridge = []
        first_handoff_green_plate_size = plate_size
        first_handoff_red_plate_size = plate_size
        if num_platforms > 1:
            first_handoff_green_plate_size = min(
                int(round(plate_size * first_handoff_plate_scale)),
                self._max_plate_size_for_platform(platform_specs[0], plate_size),
            )
            first_handoff_red_plate_size = min(
                int(round(plate_size * first_handoff_plate_scale)),
                self._max_plate_size_for_platform(platform_specs[1], plate_size),
            )
        for i in range(num_platforms - 1):
            bootstrap_only = i < bootstrap_bridges
            green_plate_size = first_handoff_green_plate_size if i == 0 else plate_size
            red_plate_size = first_handoff_red_plate_size if i == 0 else plate_size
            green_plate_sizes_for_bridge.append(green_plate_size)
            red_plate_sizes_for_bridge.append(red_plate_size)
            _, green_positions = self._plate_layout_for_platform(
                platform_specs[i],
                plate_size=green_plate_size,
                bootstrap_only=bootstrap_only,
                max_candidate_count=max_plate_candidate_count,
            )
            _, red_positions = self._plate_layout_for_platform(
                platform_specs[i + 1],
                plate_size=red_plate_size,
                bootstrap_only=bootstrap_only,
                max_candidate_count=max_plate_candidate_count,
            )
            green_plate_positions_for_bridge.append(green_positions)
            red_plate_positions_for_bridge.append(red_positions)
            green_plate_tags_for_bridge.append(
                [
                    green_plate_tag_base + (i * max_plate_candidate_count) + variant
                    for variant in range(len(green_positions))
                ]
            )
            red_plate_tags_for_bridge.append(
                [
                    red_plate_tag_base + (i * max_plate_candidate_count) + variant
                    for variant in range(len(red_positions))
                ]
            )

        for i, spec in enumerate(platform_specs):
            x = spec["x"]
            y = spec["y"]

            if i < num_platforms - 1:
                green_plate_size = green_plate_sizes_for_bridge[i]
                for variant, ((offset_x, offset_y), tag) in enumerate(
                    zip(green_plate_positions_for_bridge[i], green_plate_tags_for_bridge[i])
                ):
                    px = x + offset_x
                    py = y + offset_y
                    builder.add_area(
                        name=f"GreenPlate_{i:02d}_v{variant}",
                        shape=rect(px, py, green_plate_size, green_plate_size),
                        floor_height=platform_floor_height,
                        ceiling_height=outer_ceiling_height,
                        floor_texture=platform_floor_texture,
                        ceiling_texture=outer_ceiling_texture,
                        wall_texture=platform_wall_texture,
                        light_level=232,
                        tag=tag,
                        mode="overwrite",
                    )

            if i > 0:
                red_plate_size = red_plate_sizes_for_bridge[i - 1]
                for variant, ((offset_x, offset_y), tag) in enumerate(
                    zip(red_plate_positions_for_bridge[i - 1], red_plate_tags_for_bridge[i - 1])
                ):
                    px = x - offset_x
                    py = y + offset_y
                    builder.add_area(
                        name=f"RedPlate_{i:02d}_v{variant}",
                        shape=rect(px, py, red_plate_size, red_plate_size),
                        floor_height=platform_floor_height,
                        ceiling_height=outer_ceiling_height,
                        floor_texture=platform_floor_texture,
                        ceiling_texture=outer_ceiling_texture,
                        wall_texture=platform_wall_texture,
                        light_level=232,
                        tag=tag,
                        mode="overwrite",
                    )

        builder.auto_connect_touching()

        spawn_offsets = [(-28, -18), (28, 18), (-28, 18), (28, -18)]
        sx = platform_specs[0]["x"]
        sy = platform_specs[0]["y"]
        for i in range(4):
            dx, dy = spawn_offsets[i]
            builder.add_thing("MAP_SPOT", sx + dx, sy + dy, tid=100 + i)

        if num_platforms > 1:
            next_dx = platform_specs[1]["x"] - platform_specs[0]["x"]
            next_dy = platform_specs[1]["y"] - platform_specs[0]["y"]
            spawn_angle = int(round(math.degrees(math.atan2(next_dy, next_dx)))) % 360
        else:
            spawn_angle = 0

        builder.add_thing("PLAYER1_START", sx - 28, sy - 18, angle=spawn_angle)
        builder.add_thing("PLAYER2_START", sx + 28, sy + 18, angle=spawn_angle)
        builder.add_thing("PLAYER3_START", sx - 28, sy + 18, angle=spawn_angle)
        builder.add_thing("PLAYER4_START", sx + 28, sy - 18, angle=spawn_angle)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")

        acs.add_script(
            ScriptType.ENTER,
            f"""
            int pn = PlayerNumber();
            if (pn < 0 || pn >= {max_players}) terminate;
            Thing_ChangeTID(0, 1000 + pn);
            SetActorProperty(0, APROP_Health, 100);
            // Ensure players have no weapons (fists only)
            ClearInventory();
            GiveInventory("Fist", 1);
            SetWeapon("Fist");
            """,
            number=998,
        )

        acs.add_script(
            ScriptType.RESPAWN,
            f"""
            int pn = PlayerNumber();
            int p_tid;
            int spot;
            if (pn < 0 || pn >= {max_players}) terminate;
            p_tid = 1000 + pn;
            Thing_ChangeTID(0, p_tid);
            SetActorProperty(0, APROP_Health, 100);
            spot = 100 + pn;
            SetActorPosition(0, GetActorX(spot), GetActorY(spot), GetActorZ(spot), 0);
            // Remove all weapons on respawn so players keep only fists
            ClearInventory();
            GiveInventory("Fist", 1);
            SetWeapon("Fist");
            """,
            number=997,
        )

        bridge_blocks: List[str] = []
        for i, bridge_tag in enumerate(bridge_tags):
            count_checks = "\n".join(
                [
                    f"hold_{i} += ThingCountSector(0, {1000 + p}, green_active_tag[{i}]);\n"
                    f"hold_{i} += ThingCountSector(0, {1000 + p}, red_active_tag[{i}]);"
                    for p in range(max_players)
                ]
            )

            if i == 0 and bridge0_support_latch_tics > 0:
                support_logic = f"""
                if (hold_{i} > 0) {{
                    bridge_support_timer[{i}] = {bridge0_support_latch_tics};
                }} else if (bridge_support_timer[{i}] > 0) {{
                    bridge_support_timer[{i}]--;
                    hold_{i} = 1;
                }}
                """
            else:
                support_logic = ""

            block = f"""
            int hold_{i} = 0;
            {count_checks}
            {support_logic}

            if (hold_{i} > 0) {{
                if (active_bridge < {i + 1}) active_bridge = {i + 1};
                if (bridge_state[{i}] != 1) {{
                    bridge_state[{i}] = 1;
                    Floor_MoveToValue({bridge_tag}, 96, {platform_floor_height}, 0);
                    ChangeFloor({bridge_tag}, "{bridge_floor_texture}");
                    Sector_SetDamage({bridge_tag}, 0, 0);
                    Light_ChangeToValue({bridge_tag}, 248);
                }}
            }} else {{
                if (bridge_state[{i}] != 0) {{
                    bridge_state[{i}] = 0;
                    Floor_MoveToValue({bridge_tag}, 160, {lava_height}, 0);
                    ChangeFloor({bridge_tag}, "LAVA1");
                    Sector_SetDamage({bridge_tag}, {lava_damage}, 14);
                    Light_ChangeToValue({bridge_tag}, 176);
                }}
            }}
            """
            bridge_blocks.append(block)

        green_color_lines = "\n".join(
            [
                "\n".join(
                    [
                        f"Floor_MoveToValue({tag}, 96, {platform_floor_height}, 0);"
                        f"\nChangeFloor({tag}, \"{platform_floor_texture}\");"
                        f"\nLight_ChangeToValue({tag}, 232);"
                        for tag in tags
                    ]
                )
                for tags in green_plate_tags_for_bridge
            ]
        )
        red_color_lines = "\n".join(
            [
                "\n".join(
                    [
                        f"Floor_MoveToValue({tag}, 96, {platform_floor_height}, 0);"
                        f'\nChangeFloor({tag}, "{platform_floor_texture}");'
                        f"\nLight_ChangeToValue({tag}, 232);"
                        for tag in tags
                    ]
                )
                for tags in red_plate_tags_for_bridge
            ]
        )
        goal_color_line = f"Sector_SetColor({7000 + num_platforms - 1}, 88, 156, 220);"
        plate_randomization_lines: List[str] = []
        for i in range(len(bridge_tags)):
            green_tags = green_plate_tags_for_bridge[i]
            red_tags = red_plate_tags_for_bridge[i]
            green_branch = " ".join(
                [
                    f'if (green_roll_{i} == {variant}) green_active_tag[{i}] = {tag};'
                    for variant, tag in enumerate(green_tags)
                ]
            )
            red_branch = " ".join(
                [
                    f'if (red_roll_{i} == {variant}) red_active_tag[{i}] = {tag};'
                    for variant, tag in enumerate(red_tags)
                ]
            )
            if len(green_tags) == 1:
                plate_assignment_lines = (
                    f"green_active_tag[{i}] = {green_tags[0]};\n"
                    f"red_active_tag[{i}] = {red_tags[0]};"
                )
            else:
                roll_max = len(green_tags) - 1
                plate_assignment_lines = (
                    f"int green_roll_{i} = Random(0, {roll_max});\n"
                    f"int red_roll_{i} = Random(0, {roll_max});\n"
                    f"{green_branch}\n"
                    f"{red_branch}"
                )
            green_active_visuals = " ".join(
                [
                    (
                        f'if (green_active_tag[{i}] == {tag}) {{ Floor_MoveToValue({tag}, 64, {platform_floor_height + 2}, 0); '
                        f'ChangeFloor({tag}, "FLAT23"); Sector_SetColor({tag}, 0, 255, 0); Light_ChangeToValue({tag}, 248); }}'
                    )
                    for tag in green_tags
                ]
            )
            red_active_visuals = " ".join(
                [
                    (
                        f'if (red_active_tag[{i}] == {tag}) {{ Floor_MoveToValue({tag}, 64, {platform_floor_height + 2}, 0); '
                        f'ChangeFloor({tag}, "FLAT23"); Sector_SetColor({tag}, 255, 0, 0); Light_ChangeToValue({tag}, 248); }}'
                    )
                    for tag in red_tags
                ]
            )
            plate_randomization_lines.append(
                f"""
                {plate_assignment_lines}
                {green_active_visuals}
                {red_active_visuals}
                """
            )
        best_platform_update_lines: List[str] = []
        best_var_names = [
            "lavapit_p1_best_platform",
            "lavapit_p2_best_platform",
        ]
        current_var_names = [
            "lavapit_p1_current_platform",
            "lavapit_p2_current_platform",
        ]
        for p in range(max_players):
            tid = 1000 + p
            best_var = best_var_names[p]
            current_var = current_var_names[p]
            current_platform_lines = [
                f"int p{p}_platform = 0;",
                f"if (PlayerInGame({p}) && ThingCount(T_NONE, {tid}) > 0) {{",
            ]
            current_platform_lines.extend(
                [
                    (
                        f"    if (ThingCountSector(0, {tid}, {7000 + i}) > 0 && "
                        f"p{p}_platform < {i + 1}) p{p}_platform = {i + 1};"
                    )
                    for i in range(num_platforms)
                ]
            )
            current_platform_lines.extend(
                [
                    f"    {current_var} = p{p}_platform;",
                    f"    if (p{p}_platform > {best_var}) {best_var} = p{p}_platform;",
                    "    active_players += 1;",
                    "    if (have_joint == 0) {",
                    f"        joint_platform = {best_var};",
                    "        have_joint = 1;",
                    f"    }} else if ({best_var} < joint_platform) {{",
                    f"        joint_platform = {best_var};",
                    "    }",
                    f"    if ({best_var} > lead_platform) lead_platform = {best_var};",
                    "}",
                ]
            )
            current_platform_lines.append(f"else {{ {current_var} = 0; }}")
            best_platform_update_lines.append("\n".join(current_platform_lines))
        frontier_probe_lines = "\n".join(
            [
                (
                    f"frontier_entry_held += ThingCountSector(0, {1000 + p}, green_active_tag[frontier_idx]);\n"
                    f"frontier_exit_held += ThingCountSector(0, {1000 + p}, red_active_tag[frontier_idx]);\n"
                    f"frontier_bridge_occupied += ThingCountSector(0, {1000 + p}, {bridge_tag_base} + frontier_idx);"
                )
                for p in range(max_players)
            ]
        )

        acs.add_global_var("lavapit_joint_best_platform", 11, "int")
        acs.add_global_var("lavapit_platform_reached_global", 12, "int")
        acs.add_global_var("lavapit_active_bridge_global", 13, "int")
        acs.add_global_var("lavapit_success_flag", 14, "int")
        acs.add_global_var("lavapit_p1_best_platform", 15, "int")
        acs.add_global_var("lavapit_p2_best_platform", 16, "int")
        acs.add_global_var("lavapit_goal_platform", 17, "int")
        acs.add_global_var("lavapit_p1_current_platform", 18, "int")
        acs.add_global_var("lavapit_p2_current_platform", 19, "int")
        acs.add_global_var("lavapit_frontier_hold", 20, "int")
        acs.add_global_var("lavapit_frontier_bridge_occ", 21, "int")
        acs.add_global_var("lavapit_frontier_exit_hold", 22, "int")
        acs.add_global_var("lavapit_frontier_bridge_on", 23, "int")

        open_script = f"""
        int bridge_state[{len(bridge_tags)}];
        int bridge_support_timer[{len(bridge_tags)}];
        int green_active_tag[{len(bridge_tags)}];
        int red_active_tag[{len(bridge_tags)}];
        int i;

        for (i = 0; i < {len(bridge_tags)}; i++) {{
            bridge_state[i] = -1;
            bridge_support_timer[i] = 0;
            green_active_tag[i] = 0;
            red_active_tag[i] = 0;
        }}

        Sector_SetDamage({lava_tag}, {lava_damage}, 14);
        {green_color_lines}
        {red_color_lines}
        {goal_color_line}
        {' '.join(plate_randomization_lines)}
        lavapit_joint_best_platform = 0;
        lavapit_platform_reached_global = 0;
        lavapit_active_bridge_global = 0;
        lavapit_success_flag = 0;
        lavapit_p1_best_platform = 0;
        lavapit_p2_best_platform = 0;
        lavapit_goal_platform = {num_platforms};
        lavapit_p1_current_platform = 0;
        lavapit_p2_current_platform = 0;
        lavapit_frontier_hold = 0;
        lavapit_frontier_bridge_occ = 0;
        lavapit_frontier_exit_hold = 0;
        lavapit_frontier_bridge_on = 0;

        while (TRUE)
        {{
            int active_players = 0;
            int joint_platform = 0;
            int lead_platform = 0;
            int have_joint = 0;
            int active_bridge = 0;
            int frontier_entry_held = 0;
            int frontier_exit_held = 0;
            int frontier_bridge_on = 0;
            int frontier_bridge_occupied = 0;
            {' '.join(bridge_blocks)}
            {'\n'.join(best_platform_update_lines)}
            if (have_joint == 0) joint_platform = 0;
            if (joint_platform > 0 && joint_platform < {num_platforms}) {{
                int frontier_idx = joint_platform - 1;
                {frontier_probe_lines}
                if (frontier_entry_held > 0) frontier_entry_held = 1;
                if (frontier_exit_held > 0) frontier_exit_held = 1;
                if (bridge_state[frontier_idx] == 1) frontier_bridge_on = 1;
                if (frontier_bridge_occupied > 0) frontier_bridge_occupied = 1;
            }}
            lavapit_joint_best_platform = joint_platform;
            lavapit_platform_reached_global = lead_platform;
            lavapit_active_bridge_global = active_bridge;
            lavapit_frontier_hold = frontier_entry_held;
            lavapit_frontier_bridge_occ = frontier_bridge_occupied;
            lavapit_frontier_exit_hold = frontier_exit_held;
            lavapit_frontier_bridge_on = frontier_bridge_on;
            if (lavapit_success_flag == 0 && active_players > 0 && joint_platform >= {num_platforms}) {{
                lavapit_success_flag = 1;
                Delay(1);
                Exit_Normal(0);
            }}
            Delay({poll_tics} + Random(0, {runtime_jitter_tics}));
        }}
        """
        acs.add_script(ScriptType.OPEN, open_script, number=1)

        acs.add_script(
            ScriptType.DEATH,
            """
            Delay(1);
            Exit_Normal(0);
            """,
            number=996,
        )

        builder.map_data.scripts = acs.to_code()
        try:
            compiled = acs.compile()
            builder.map_data.behavior = (
                compiled if compiled else b"ACSE\x08\x00\x00\x00\x00\x00\x00\x00"
            )
        except Exception as exc:
            print(f"ACS compile warning: {exc}")
            try:
                debug_path = os.path.join(os.path.dirname(output_path), "lavapit_debug.acs")
                acs.save_source(debug_path)
                print(f"Wrote ACS debug source to {debug_path}")
            except Exception:
                pass
            builder.map_data.behavior = b"ACSE\x08\x00\x00\x00\x00\x00\x00\x00"

        builder.build(output_path)
        print(f"Lavapit WAD written to {output_path} (seed={seed}, platforms={num_platforms})")


if __name__ == "__main__":
    output = "examples/benchmark/output/lavapit.wad"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    scenario = LavaPitScenario()
    scenario.generate(output)
