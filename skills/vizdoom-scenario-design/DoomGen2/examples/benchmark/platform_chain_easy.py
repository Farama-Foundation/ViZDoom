import os
import random
from typing import Any, Dict, List, Tuple

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.polyobjects import (
    POLYOBJ_START_SPOT_THING,
    PolyobjectBuilder,
    PolyobjectConfig,
)
from doomgen.geometry.shapes import rect
from doomgen.logic.acs_builder import ACSBuilder, ScriptType

PLATFORM_SIZE = 144


def route_grid_origin(level_count: int, spacing: int) -> Tuple[int, int]:
    x0 = -((level_count - 1) * spacing) // 2
    return x0, 0


def route_positions(level_count: int, spacing: int) -> List[Tuple[int, int]]:
    x0, y0 = route_grid_origin(level_count, spacing)

    positions: List[Tuple[int, int]] = []
    for i in range(level_count):
        positions.append((x0 + i * spacing, y0))

    return positions


def route_index_from_world(level_count: int, spacing: int, x_world: int, y_world: int) -> int:
    x0, y0 = route_grid_origin(level_count, spacing)
    _ = y_world - y0
    return max(0, min(level_count - 1, (x_world - x0 + spacing // 2) // spacing))


def platform_center_for_index(level_count: int, spacing: int, idx: int) -> Tuple[int, int]:
    positions = route_positions(level_count, spacing)
    clamped = max(0, min(level_count - 1, idx))
    return positions[clamped]


def safe_route_checkpoint(
    *,
    level_count: int,
    spacing: int,
    base_height: int,
    height_step: int,
    platform_w: int,
    platform_h: int,
    x_world: int,
    y_world: int,
    z_world: int,
    min_safe_z_margin: int = 32,
) -> int | None:
    idx = route_index_from_world(level_count, spacing, x_world, y_world)
    center_x, center_y = platform_center_for_index(level_count, spacing, idx)
    base_z = base_height + idx * height_step

    if abs(x_world - center_x) > platform_w // 2:
        return None
    if abs(y_world - center_y) > platform_h // 2:
        return None
    if z_world < base_z - min_safe_z_margin:
        return None

    return idx


def advance_route_checkpoint(current_idx: int, safe_idx: int | None) -> int:
    if safe_idx is None:
        return current_idx
    if safe_idx == current_idx + 1:
        return safe_idx
    return current_idx


class PlatformChainScenario(Scenario):

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "random_seed_on_generate": True,
            "level_count": 20,
            "base_height": 0,
            "height_step": 12,
            "safe_start_levels_min": 2,
            "safe_start_levels_max": 2,
            "max_players": 4,
            "lava_height": -96,
            "lava_damage": 50,
            "polyobject_count": 8,
            "map_radius": 1800,
            "jump_gap": 36,
            "chain_max_len": 250,
        }

    def validate_config(self) -> None:
        players = int(self.config["max_players"])
        if players < 1 or players > 4:
            raise ValueError("max_players must be in range [1, 4]")

        if int(self.config["height_step"]) < 8:
            raise ValueError("height_step must be >= 8")

        if int(self.config.get("chain_max_len", 250)) < 64:
            raise ValueError("chain_max_len must be >= 64")

        if int(self.config["jump_gap"]) < 16:
            raise ValueError("jump_gap must be >= 16")

        level_count = int(self.config["level_count"])
        safe_start_levels_min = int(self.config.get("safe_start_levels_min", 2))
        safe_start_levels_max = int(self.config.get("safe_start_levels_max", 2))
        if level_count != 20:
            raise ValueError("platform_chain_easy must use exactly 20 levels")
        if safe_start_levels_min != 2 or safe_start_levels_max != 2:
            raise ValueError("platform_chain_easy must keep the first 2 levels as easy starters")

    def _choose_seed(self) -> int:
        cfg = self.config
        if cfg["seed"] is not None:
            return int(cfg["seed"])
        if bool(cfg.get("random_seed_on_generate", True)):
            return random.randint(0, 999999)
        return 42

    def _route_positions(self, level_count: int, spacing: int) -> List[Tuple[int, int]]:
        return route_positions(level_count, spacing)

    @staticmethod
    def _route_grid_origin(level_count: int, spacing: int) -> Tuple[int, int]:
        return route_grid_origin(level_count, spacing)

    @staticmethod
    def _route_index_from_world(level_count: int, spacing: int, x_world: int, y_world: int) -> int:
        return route_index_from_world(level_count, spacing, x_world, y_world)

    @staticmethod
    def _platform_center_for_index(level_count: int, spacing: int, idx: int) -> Tuple[int, int]:
        return platform_center_for_index(level_count, spacing, idx)

    @staticmethod
    def _safe_route_checkpoint(
        *,
        level_count: int,
        spacing: int,
        base_height: int,
        height_step: int,
        platform_w: int,
        platform_h: int,
        x_world: int,
        y_world: int,
        z_world: int,
        min_safe_z_margin: int = 32,
    ) -> int | None:
        return safe_route_checkpoint(
            level_count=level_count,
            spacing=spacing,
            base_height=base_height,
            height_step=height_step,
            platform_w=platform_w,
            platform_h=platform_h,
            x_world=x_world,
            y_world=y_world,
            z_world=z_world,
            min_safe_z_margin=min_safe_z_margin,
        )

    @staticmethod
    def _mover_start_center(
        ax: int,
        ay: int,
        bx: int,
        by: int,
        mover_panel_h: int,
        mover_start_gap: int,
    ) -> Tuple[int, int]:
        mx = (ax + bx) // 2
        my = (ay + by) // 2
        offset = (mover_panel_h // 2) + mover_start_gap

        if abs(bx - ax) >= abs(by - ay):
            return mx, my + offset
        return mx + offset, my

    def _decorate_lump(self) -> str | None:
        return None

    def _enter_visual_script(self, max_players: int, chain_max_len: int) -> str:
        return ""

    def _chain_visual_lines(
        self,
        follower: int,
        leader: int,
        max_players: int,
        chain_max_len: int,
    ) -> str:
        return ""

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = self._choose_seed()

        level_count = int(cfg["level_count"])
        max_players = int(cfg["max_players"])
        base_height = int(cfg["base_height"])
        height_step = int(cfg["height_step"])
        jump_gap = int(cfg["jump_gap"])
        platform_w = PLATFORM_SIZE
        platform_h = PLATFORM_SIZE
        spacing = platform_w + jump_gap
        bridge_len = max(188, jump_gap + 64)
        route_half_extent = ((level_count - 1) * spacing + platform_w) // 2
        map_radius = max(int(cfg["map_radius"]), route_half_extent + 100)
        chain_max_len = int(cfg.get("chain_max_len", 250))
        safe_start_levels_min = int(cfg.get("safe_start_levels_min", 2))
        safe_start_levels_max = int(cfg.get("safe_start_levels_max", 2))
        polyobject_count = max(1, int(cfg.get("polyobject_count", 1)))

        builder = ProceduralMapBuilder(
            bounds=(-map_radius, -map_radius, map_radius, map_radius),
            seed=seed,
        )

        decorate_lump = self._decorate_lump()
        if decorate_lump:
            builder.wad_writer.add_lump("DECORATE", decorate_lump.encode("utf-8"))

        platform_floor = "FLOOR4_8"
        lift_floor = "FLAT14"
        vertical_crusher_floor = "BLOOD1"

        platform_wall = "METAL2"
        platform_light = 208
        mover_panel_w = 56
        mover_panel_h = 20
        mover_start_gap = 10
        mover_top_z_offset = 80

        platform_tags: List[int] = []
        platform_base_z: List[int] = []
        platform_names: List[str] = []
        platform_centers: List[Tuple[int, int]] = []
        bridge_tags: List[int] = []
        obstacle_candidates: List[Tuple[int, int, int]] = []

        positions = self._route_positions(level_count, spacing)

        lava_height = int(cfg["lava_height"])
        lava_tag = 900
        builder.add_area(
            name="LavaBasin",
            shape=rect(0, 0, map_radius * 2 - 100, map_radius * 2 - 100),
            floor_height=lava_height,
            ceiling_height=2048,
            floor_texture="LAVA1",
            ceiling_texture="RROCK16",
            wall_texture="METAL2",
            lower_texture="METAL2",
            upper_texture="METAL2",
            light_level=196,
            tag=lava_tag,
            mode="claim_void",
        )

        for idx in range(level_count):
            x, y = positions[idx]
            z = base_height + idx * height_step
            tag = 7000 + idx
            name = f"L{idx + 1:03d}"

            builder.add_area(
                name=name,
                shape=rect(x, y, platform_w, platform_h),
                floor_height=z,
                ceiling_height=2048,
                floor_texture=platform_floor,
                ceiling_texture="RROCK16",
                wall_texture=platform_wall,
                lower_texture=platform_wall,
                upper_texture=platform_wall,
                light_level=platform_light,
                tag=tag,
                mode="overwrite",
            )

            platform_tags.append(tag)
            platform_base_z.append(z)
            platform_names.append(name)
            platform_centers.append((x, y))

            if idx >= 12 and idx % 7 == 0:
                obstacle_candidates.append((idx, x + 18, y - 14))

        for idx in range(6, level_count - 1, 4):
            ax, ay = platform_centers[idx]
            bx, by = platform_centers[idx + 1]
            horizontal = abs(bx - ax) >= abs(by - ay)

            mx = (ax + bx) // 2
            my = (ay + by) // 2
            bw, bh = (bridge_len, 36) if horizontal else (36, bridge_len)

            bridge_tag = 8200 + len(bridge_tags)
            bridge_z = min(platform_base_z[idx], platform_base_z[idx + 1]) + 4

            builder.add_area(
                name=f"Bridge_{idx + 1:03d}",
                shape=rect(mx, my, bw, bh),
                floor_height=bridge_z,
                ceiling_height=2048,
                floor_texture="STEP1",
                ceiling_texture="RROCK16",
                wall_texture="METAL2",
                lower_texture="METAL2",
                upper_texture="METAL2",
                light_level=220,
                tag=bridge_tag,
                mode="overwrite",
            )
            bridge_tags.append(bridge_tag)

        spawn_points: List[Tuple[int, int]] = []
        for i in range(max_players):
            centroid = builder.get_area_centroid(platform_names[i])
            if centroid is None:
                sx, sy = platform_centers[i]
            else:
                sx, sy = int(centroid[0]), int(centroid[1])
            spawn_points.append((sx, sy))

        while len(spawn_points) < 4:
            spawn_points.append(spawn_points[-1])

        for i in range(4):
            sx, sy = spawn_points[i]
            builder.add_thing("MAP_SPOT", sx, sy, tid=100 + i)

        builder.add_thing("PLAYER1_START", spawn_points[0][0], spawn_points[0][1], angle=0)
        builder.add_thing("PLAYER2_START", spawn_points[1][0], spawn_points[1][1], angle=0)
        builder.add_thing("PLAYER3_START", spawn_points[2][0], spawn_points[2][1], angle=0)
        builder.add_thing("PLAYER4_START", spawn_points[3][0], spawn_points[3][1], angle=0)

        builder.auto_connect_touching()

        po_builder = PolyobjectBuilder(builder.map_data)
        poly_nums: List[int] = []
        usable_pairs = max(1, level_count - 1)
        actual_poly_count = min(polyobject_count, usable_pairs)

        for po_idx in range(actual_poly_count):
            sample = int(((po_idx + 1) * usable_pairs) / (actual_poly_count + 1))
            pair_idx = min(level_count - 2, max(0, sample))

            ax, ay = platform_centers[pair_idx]
            bx, by = platform_centers[pair_idx + 1]
            za = platform_base_z[pair_idx]
            zb = platform_base_z[pair_idx + 1]

            start_x, start_y = self._mover_start_center(
                ax,
                ay,
                bx,
                by,
                mover_panel_h,
                mover_start_gap,
            )
            start_z = min(za, zb)

            poly_num = po_idx + 1
            dummy_x = map_radius + 512 + (po_idx * 192)
            dummy_y = map_radius + 512

            po_builder.add_rect(
                PolyobjectConfig(
                    number=poly_num,
                    center=(dummy_x, dummy_y),
                    start_spot_center=(start_x, start_y),
                    width=mover_panel_w,
                    height=mover_panel_h,
                    floor_height=start_z + 8,
                    ceiling_height=start_z + mover_top_z_offset,
                    texture="METAL",
                    light_level=224,
                    start_spot_thing_type=POLYOBJ_START_SPOT_THING,
                )
            )
            poly_nums.append(poly_num)

        for idx, ox, oy in obstacle_candidates:
            builder.add_thing(30 if idx % 2 == 0 else 31, ox, oy)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("pc_max_chain_dist_global", 51, "int")
        acs.add_global_var("pc_chain_break_events_global", 52, "int")
        acs.add_global_var("pc_dragged_links_global", 53, "int")
        acs.add_global_var("pc_level_reached_global", 54, "int")
        acs.add_global_var("pc_route_progress_global", 55, "int")
        for player_idx in range(max_players):
            acs.add_map_var(f"pc_player_checkpoint_{player_idx}", "int", player_idx)
        for idx in range(level_count):
            acs.add_map_var(f"pc_level_variant_{idx}", "int", 0)
            acs.add_map_var(f"pc_level_phase_{idx}", "int", 0)

        route_cols = level_count
        route_rows = 1
        route_x0, route_y0 = self._route_grid_origin(level_count, spacing)
        route_last_idx = level_count - 1

        acs.add_global_code(
            f"""
function int PcClampInt(int value, int lo, int hi) {{
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}}

function int PcRouteIndexFromWorld(int x_world, int y_world) {{
    int col = PcClampInt(((x_world - ({route_x0})) + {spacing // 2}) / {spacing}, 0, {route_cols - 1});
    int row = PcClampInt(((y_world - ({route_y0})) + {spacing // 2}) / {spacing}, 0, {route_rows - 1});
    int route_col = col;
    if ((row % 2) == 1) {{
        route_col = {route_cols - 1} - col;
    }}

    return PcClampInt((row * {route_cols}) + route_col, 0, {route_last_idx});
}}

function int PcCenterXForIndex(int idx) {{
    int clamped = PcClampInt(idx, 0, {route_last_idx});
    int row = clamped / {route_cols};
    int route_col = clamped - (row * {route_cols});
    int world_col = route_col;
    if ((row % 2) == 1) {{
        world_col = {route_cols - 1} - route_col;
    }}

    return ({route_x0}) + world_col * {spacing};
}}

function int PcCenterYForIndex(int idx) {{
    int clamped = PcClampInt(idx, 0, {route_last_idx});
    int row = clamped / {route_cols};
    return ({route_y0}) + row * {spacing};
}}

function int PcAbsInt(int value) {{
    if (value < 0) return -value;
    return value;
}}

function int PcBaseZForIndex(int idx) {{
    return {base_height} + PcClampInt(idx, 0, {route_last_idx}) * {height_step};
}}

function int PcSafeRouteIndex(int x_fix, int y_fix, int z_fix) {{
    int idx = PcRouteIndexFromWorld(x_fix >> 16, y_fix >> 16);
    int x_world = x_fix >> 16;
    int y_world = y_fix >> 16;
    int z_world = z_fix >> 16;

    if (PcAbsInt(x_world - PcCenterXForIndex(idx)) > {platform_w // 2}) return -1;
    if (PcAbsInt(y_world - PcCenterYForIndex(idx)) > {platform_h // 2}) return -1;
    if (z_world < (PcBaseZForIndex(idx) - 32)) return -1;

    return idx;
}}

function int PcAdvanceCheckpoint(int current_idx, int safe_idx) {{
    if (safe_idx == current_idx + 1) return safe_idx;
    return current_idx;
}}
"""
        )

        acs.add_script(
            ScriptType.ENTER,
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
            switch (pn) {{
                {chr(10).join([f'case {player_idx}: pc_player_checkpoint_{player_idx} = {player_idx}; break;' for player_idx in range(max_players)])}
            }}

            {self._enter_visual_script(max_players, chain_max_len)}
            """,
            number=12,
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
            switch (pn) {{
                {chr(10).join([f'case {player_idx}: pc_player_checkpoint_{player_idx} = {player_idx}; break;' for player_idx in range(max_players)])}
            }}
            """,
            number=11,
        )

        acs.add_script(
            ScriptType.VOID,
            f"""
            while (TRUE)
            {{
                {chr(10).join([f'Polyobj_Move({n}, 16, {192 if i % 2 == 0 else 64}, {128 + (i % 3) * 32});' for i, n in enumerate(poly_nums)])}
                Delay(5 * 35);
                {chr(10).join([f'Polyobj_Move({n}, 16, {64 if i % 2 == 0 else 192}, {128 + (i % 3) * 32});' for i, n in enumerate(poly_nums)])}
                Delay(5 * 35);
            }}
            """,
            number=10,
        )

        acs.add_script(
            ScriptType.DEATH,
            """
            Delay(1);
            Exit_Normal(0);
            """,
            number=13,
        )

        dynamic_lines: List[str] = []

        for idx, tag in enumerate(platform_tags):
            z = platform_base_z[idx]
            low = z
            lift_high = z + 152
            crusher_open_ceiling = 2048

            dynamic_lines.append(
                f"""
                if (pc_level_variant_{idx} == 1) {{
                    ChangeFloor({tag}, "{lift_floor}");
                    Sector_SetDamage({tag}, 0, 0);
                    if (((tick + pc_level_phase_{idx}) % 170) < 85) {{
                        Floor_CrushStop({tag});
                        Floor_MoveToValue({tag}, 8, {lift_high}, 0);
                    }} else {{
                        Floor_CrushStop({tag});
                        Floor_MoveToValue({tag}, 8, {low}, 0);
                    }}
                    Ceiling_CrushStop({tag});
                    Ceiling_MoveToValue({tag}, 64, {crusher_open_ceiling}, 0);
                }} else if (pc_level_variant_{idx} == 2) {{
                    ChangeFloor({tag}, "{vertical_crusher_floor}");
                    Floor_CrushStop({tag});
                    Floor_MoveToValue({tag}, 14, {low}, 0);

                    Ceiling_CrushAndRaise({tag}, 64, 200, 0);
                }} else {{
                    ChangeFloor({tag}, "{platform_floor}");
                    Sector_SetDamage({tag}, 0, 0);
                    Floor_CrushStop({tag});
                    Floor_MoveToValue({tag}, 14, {low}, 0);
                    Ceiling_CrushStop({tag});
                    Ceiling_MoveToValue({tag}, 64, {crusher_open_ceiling}, 0);
                }}
                """
            )

        bridge_texture_lines: List[str] = []
        for t in bridge_tags:
            bridge_texture_lines.append(f'ChangeFloor({t}, "STEP1");')

        chain_lines: List[str] = []
        for follower in range(1, max_players):
            leader = follower - 1
            chain_lines.append(
                f"""
                if (PlayerInGame({leader}) && PlayerInGame({follower})) {{
                    int a_tid_{follower} = {1000 + leader};
                    int b_tid_{follower} = {1000 + follower};
                    if (ThingCount(0, a_tid_{follower}) > 0 && ThingCount(0, b_tid_{follower}) > 0) {{
                        int ax_{follower} = GetActorX(a_tid_{follower});
                        int ay_{follower} = GetActorY(a_tid_{follower});
                        int az_{follower} = GetActorZ(a_tid_{follower});
                        int bx_{follower} = GetActorX(b_tid_{follower});
                        int by_{follower} = GetActorY(b_tid_{follower});
                        int bz_{follower} = GetActorZ(b_tid_{follower});

                        int dx_{follower} = ax_{follower} - bx_{follower};
                        int dy_{follower} = ay_{follower} - by_{follower};
                        int adx_{follower} = dx_{follower};
                        int ady_{follower} = dy_{follower};
                        if (adx_{follower} < 0) adx_{follower} = -adx_{follower};
                        if (ady_{follower} < 0) ady_{follower} = -ady_{follower};

                        int dist_{follower};
                        if (adx_{follower} > ady_{follower}) {{
                            dist_{follower} = adx_{follower} + (ady_{follower} >> 1);
                        }} else {{
                            dist_{follower} = ady_{follower} + (adx_{follower} >> 1);
                        }}

                        if (dist_{follower} > max_chain_dist) max_chain_dist = dist_{follower};

                        int chain_limit_{follower} = {chain_max_len} << 16;
                        int render_ax_{follower} = ax_{follower};
                        int render_ay_{follower} = ay_{follower};
                        int render_bx_{follower} = bx_{follower};
                        int render_by_{follower} = by_{follower};
                        int tether_state_{follower} = 0;

                        if (dist_{follower} > (chain_limit_{follower} * 9) / 10) {{
                            tether_state_{follower} = 2;
                        }} else if (dist_{follower} > (chain_limit_{follower} * 3) / 4) {{
                            tether_state_{follower} = 1;
                        }}

                        if (dist_{follower} > chain_limit_{follower} && dist_{follower} > 0) {{
                            dragged_links_count++;
                            chain_break_events++;
                            tether_state_{follower} = 2;
                            int keep_half_{follower} = ({max(64, chain_max_len - 24)} << 15);
                            int ratio_{follower} = FixedDiv(keep_half_{follower}, dist_{follower});
                            int mx_{follower} = (ax_{follower} + bx_{follower}) >> 1;
                            int my_{follower} = (ay_{follower} + by_{follower}) >> 1;
                            int nax_{follower} = mx_{follower} + FixedMul(dx_{follower}, ratio_{follower});
                            int nay_{follower} = my_{follower} + FixedMul(dy_{follower}, ratio_{follower});
                            int nbx_{follower} = mx_{follower} - FixedMul(dx_{follower}, ratio_{follower});
                            int nby_{follower} = my_{follower} - FixedMul(dy_{follower}, ratio_{follower});

                            SetActorPosition(a_tid_{follower}, nax_{follower}, nay_{follower}, az_{follower}, 0);
                            SetActorPosition(b_tid_{follower}, nbx_{follower}, nby_{follower}, bz_{follower}, 0);

                            render_ax_{follower} = nax_{follower};
                            render_ay_{follower} = nay_{follower};
                            render_bx_{follower} = nbx_{follower};
                            render_by_{follower} = nby_{follower};
                        }}

                        {self._chain_visual_lines(follower, leader, max_players, chain_max_len)}
                    }}
                }}
                """
            )

        dynamic_block = "\n".join(dynamic_lines)
        bridge_texture_block = "\n".join(bridge_texture_lines)
        chain_block = "\n".join(chain_lines)
        level_init_lines: List[str] = []
        for idx in range(level_count):
            level_init_lines.append(f"pc_level_phase_{idx} = Random(0, 169);")

            level_init_lines.append(
                f"""
                if ({idx} < safe_start_levels) {{
                    pc_level_variant_{idx} = 0;
                }} else {{
                    int level_roll_{idx} = Random(0, 99);
                    if (level_roll_{idx} < 50) {{
                        pc_level_variant_{idx} = 0;
                    }} else if (level_roll_{idx} < 85) {{
                        pc_level_variant_{idx} = 1;
                    }} else {{
                        pc_level_variant_{idx} = 2;
                    }}
                }}
                """
            )
        level_init_block = "\n".join(level_init_lines)

        acs_open = f"""
        int tick = 0;
        int safe_start_levels = Random({safe_start_levels_min}, {safe_start_levels_max});
        int chain_break_events = 0;
        int max_chain_dist = 0;
        int dragged_links_count = 0;
        Sector_SetDamage({lava_tag}, {int(cfg['lava_damage'])}, 14);
        ACS_ExecuteAlways(10, 0);

        pc_max_chain_dist_global = 0;
        pc_chain_break_events_global = 0;
        pc_dragged_links_global = 0;
        pc_level_reached_global = 0;
        {level_init_block}

        while (TRUE)
        {{
            if (tick % 5 == 0)
            {{
                max_chain_dist = 0;
                dragged_links_count = 0;

                int min_lvl = {route_last_idx};
                int joint_progress_fixed = {route_last_idx} << 16;
                int any_active = 0;
                int p_check;
                for (p_check = 0; p_check < {max_players}; p_check++) {{
                    if (PlayerInGame(p_check)) {{
                        int ct = 1000 + p_check;
                        if (ThingCount(0, ct) > 0) {{
                            int safe_idx = PcSafeRouteIndex(GetActorX(ct), GetActorY(ct), GetActorZ(ct));
                            switch (p_check) {{
                                {chr(10).join([f'case {player_idx}: pc_player_checkpoint_{player_idx} = PcAdvanceCheckpoint(pc_player_checkpoint_{player_idx}, safe_idx); break;' for player_idx in range(max_players)])}
                            }}
                            int clvl;
                            switch (p_check) {{
                                {chr(10).join([f'case {player_idx}: clvl = pc_player_checkpoint_{player_idx}; break;' for player_idx in range(max_players)])}
                                default: clvl = 0; break;
                            }}

                            if (clvl < min_lvl) min_lvl = clvl;
                            if ((clvl << 16) < joint_progress_fixed) joint_progress_fixed = clvl << 16;
                            any_active = 1;
                        }}
                    }}
                }}
                if (any_active) {{
                    pc_level_reached_global = min_lvl;
                    pc_route_progress_global = joint_progress_fixed;
                    if (min_lvl >= {route_last_idx}) {{
                        Delay(1);
                        Exit_Normal(0);
                    }}
                }} else {{
                    pc_level_reached_global = 0;
                    pc_route_progress_global = 0;
                }}

                {bridge_texture_block}
                {dynamic_block}
                {chain_block}

                pc_max_chain_dist_global = max_chain_dist;
                pc_chain_break_events_global = chain_break_events;
                pc_dragged_links_global = dragged_links_count;
            }}

            Delay(1);
            tick++;
        }}
        """
        acs.add_script(ScriptType.OPEN, acs_open, number=1)

        try:
            print(f"Compiling ACS for {self.name}...")
            compiled = acs.compile()
            builder.map_data.behavior = compiled if compiled else b"ACSE\x08\x00\x00\x00\x00\x00\x00\x00"
        except Exception as e:
            print(f"ACS compile warning: {e}")
            try:
                debug_path = os.path.join(os.path.dirname(output_path), "platform_chain_debug.acs")
                acs.save_source(debug_path)
                print(f"Wrote ACS debug source to {debug_path}")
            except Exception:
                pass
            builder.map_data.behavior = b"ACSE\x08\x00\x00\x00\x00\x00\x00\x00"

        builder.map_data.scripts = acs.to_code()

        builder.build(output_path)
        print(f"Platform Chain WAD written to {output_path} (seed={seed})")

    def _enter_visual_script(self, max_players: int, chain_max_len: int) -> str:
        bar_cases = []
        for i in range(21):
            filled = "=" * i
            empty = "." * (20 - i)
            bar_cases.append(f'case {i}: tether_bar = "o{filled}{empty}o"; break;')
        bar_switch = "\n                    ".join(bar_cases)

        return f"""
        int overlay_tick = 0;
        int chain_limit = {chain_max_len} << 16;

        while (TRUE)
        {{
            if (overlay_tick % 5 == 0)
            {{
                int farthest = 0;
                int teammate_found = 0;
                int my_x = GetActorX(p_tid);
                int my_y = GetActorY(p_tid);

                int other;
                for (other = 0; other < {max_players}; other++)
                {{
                    if (other == pn) continue;
                    if (!PlayerInGame(other)) continue;

                    int other_tid = 1000 + other;
                    if (ThingCount(0, other_tid) <= 0) continue;

                    int dx = my_x - GetActorX(other_tid);
                    int dy = my_y - GetActorY(other_tid);
                    int adx = dx;
                    int ady = dy;
                    if (adx < 0) adx = -adx;
                    if (ady < 0) ady = -ady;

                    int d;
                    if (adx > ady) {{
                        d = adx + (ady >> 1);
                    }} else {{
                        d = ady + (adx >> 1);
                    }}

                    if (!teammate_found || d > farthest) farthest = d;
                    teammate_found = 1;
                }}

                SetHudSize(320, 240, 1);
                SetFont("SMALLFONT");

                if (!teammate_found)
                {{
                    HudMessage(s:""; HUDMSG_PLAIN, 211, CR_WHITE, 0.5, 0.86, 0.2);
                    HudMessage(s:""; HUDMSG_PLAIN, 212, CR_WHITE, 0.5, 0.90, 0.2);
                }}
                else
                {{
                    int tension_idx = 0;
                    if (chain_limit > 0) {{
                        tension_idx = (farthest * 20) / chain_limit;
                    }}
                    if (tension_idx < 0) tension_idx = 0;
                    if (tension_idx > 20) tension_idx = 20;

                    str tether_bar = "";
                    switch (tension_idx) {{
                        {bar_switch}
                    }}

                    if (pc_dragged_links_global > 0 || farthest > chain_limit)
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_RED, 0.5, 0.90, 0.2);
                    }}
                    else if (farthest > (chain_limit * 9) / 10)
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_RED, 0.5, 0.90, 0.2);
                    }}
                    else if (farthest > (chain_limit * 3) / 4)
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_GOLD, 0.5, 0.90, 0.2);
                    }}
                    else
                    {{
                        HudMessage(s:tether_bar; HUDMSG_PLAIN, 212, CR_GREEN, 0.5, 0.90, 0.2);
                    }}
                }}
            }}

            Delay(1);
            overlay_tick++;
        }}
        """

    def _decorate_lump(self) -> str | None:
        return """
ACTOR ChainNodeCalm {
  Radius 1
  Height 1
  Scale 0.13
  RenderStyle Add
  Alpha 0.55
  +NOBLOCKMAP +NOGRAVITY +NOINTERACTION +DONTSPLASH
  States {
    Spawn:
      APLS AB 3 BRIGHT // Arachnotron plasma (Green/Blue)
      Loop
  }
}

ACTOR ChainNodeWarn {
  Radius 1
  Height 1
  Scale 0.13
  RenderStyle Add
  Alpha 0.6
  +NOBLOCKMAP +NOGRAVITY +NOINTERACTION +DONTSPLASH
  States {
    Spawn:
      BAL1 AB 3 BRIGHT // Mancubus fireball (Orange)
      Loop
  }
}

ACTOR ChainNodeHot {
  Radius 1
  Height 1
  Scale 0.13
  RenderStyle Add
  Alpha 0.65
  Translation "160:166=176:182,225:227=176:178"
  +NOBLOCKMAP +NOGRAVITY +NOINTERACTION +DONTSPLASH
  States {
    Spawn:
      BAL1 AB 3 BRIGHT // Round orb (translated to red)
      Loop
  }
}
"""

    def _chain_visual_lines(
        self,
        follower: int,
        leader: int,
        max_players: int,
        chain_max_len: int,
    ) -> str:
        tether_nodes = 30 # How dense it is
        tether_tid_base = 6000 + (follower * 64)
        return f"""
                        int node_idx_{follower};
                        for (node_idx_{follower} = 0; node_idx_{follower} < {tether_nodes}; node_idx_{follower}++) {{
                            Thing_Remove({tether_tid_base} + node_idx_{follower});
                        }}

                        if (dist_{follower} > 0) {{
                            int tether_az_{follower} = az_{follower} + (20 << 16);
                            int tether_bz_{follower} = bz_{follower} + (20 << 16);
                            int tether_dx_{follower} = render_bx_{follower} - render_ax_{follower};
                            int tether_dy_{follower} = render_by_{follower} - render_ay_{follower};
                            int tether_dz_{follower} = tether_bz_{follower} - tether_az_{follower};

                            for (node_idx_{follower} = 0; node_idx_{follower} < {tether_nodes}; node_idx_{follower}++) {{
                                int frac_{follower} = FixedDiv((node_idx_{follower} + 1) << 16, {tether_nodes + 1} << 16);
                                int sx_{follower} = render_ax_{follower} + FixedMul(tether_dx_{follower}, frac_{follower});
                                int sy_{follower} = render_ay_{follower} + FixedMul(tether_dy_{follower}, frac_{follower});
                                int sz_{follower} = tether_az_{follower} + FixedMul(tether_dz_{follower}, frac_{follower});
                                int spawn_tid_{follower} = {tether_tid_base} + node_idx_{follower};

                                if (tether_state_{follower} >= 2) {{
                                    Spawn("ChainNodeHot", sx_{follower}, sy_{follower}, sz_{follower}, spawn_tid_{follower});
                                }} else if (tether_state_{follower} == 1) {{
                                    Spawn("ChainNodeWarn", sx_{follower}, sy_{follower}, sz_{follower}, spawn_tid_{follower});
                                }} else {{
                                    Spawn("ChainNodeCalm", sx_{follower}, sy_{follower}, sz_{follower}, spawn_tid_{follower});
                                }}
                            }}
                        }}
        """

if __name__ == "__main__":
    output = "examples/benchmark/output/platform_chain_easy.wad"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    scenario = PlatformChainScenario()
    scenario.generate(output)
