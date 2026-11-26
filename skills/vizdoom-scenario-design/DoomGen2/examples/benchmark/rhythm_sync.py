import os
import random
import struct
import zlib
from typing import Any, Dict

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def solid_rgba_png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    row = bytes(rgba) * width
    raw = b"".join(b"\x00" + row for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(raw, level=9)),
            _png_chunk(b"IEND", b""),
        ]
    )


def border_rgba_png(
    width: int,
    height: int,
    rgba: tuple[int, int, int, int],
    *,
    thickness: int,
) -> bytes:
    transparent = (0, 0, 0, 0)
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            pixel = rgba if (
                x < thickness
                or x >= width - thickness
                or y < thickness
                or y >= height - thickness
            ) else transparent
            row.extend(pixel)
        rows.append(b"\x00" + bytes(row))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=9)),
            _png_chunk(b"IEND", b""),
        ]
    )


class RhythmSyncScenario(Scenario):
    def get_default_config(self) -> Dict[str, Any]:
        return {
        "seed": 42,
        "random_seed_on_generate": True,
        "num_sections": 20,
        "intro_regular_sections": 8,
        "jitter_stage_count": 6,
        "delta_t_tics": 16,  # Starting tolerance window (tics)
        "delta_t_minimum": 8,  # Minimum tolerance after decay
        "delta_t_decay": 1,  # Decrease per stage completed
        "beat_tics": 72,
        "jitter_tics": 24,
        "stage_step": 512,
        "room_width": 512,
        "room_height": 256,
        "top_y": 128,
        "bottom_y": -128,
        "switch_width": 32,
        "switch_height": 45,
        "switch_wall_margin": 32, # How far away from wall, reduce to jitter more
        "switch_intro_stages": 8, # How many stage to start actual randomizing
        "switch_intro_jitter_frac": 0.125, # How much random in intro stages
        "switch_pos_seed_offset": 0,
        "switch_use_range": 64,
    }

    def validate_config(self) -> None:
        num_sections = int(self.config["num_sections"])
        intro_regular = int(self.config["intro_regular_sections"])
        jitter_stage_count = int(self.config["jitter_stage_count"])
        delta_t_tics = int(self.config["delta_t_tics"])
        delta_t_minimum = int(self.config["delta_t_minimum"])
        delta_t_decay = int(self.config["delta_t_decay"])
        beat_tics = int(self.config["beat_tics"])
        jitter_tics = int(self.config["jitter_tics"])
        switch_wall_margin = int(self.config["switch_wall_margin"])
        switch_intro_stages = int(self.config["switch_intro_stages"])
        switch_intro_jitter_frac = float(self.config["switch_intro_jitter_frac"])

        if num_sections < 1:
            raise ValueError("num_sections must be >= 1")
        if intro_regular < 1:
            raise ValueError("intro_regular_sections must be >= 1")
        if intro_regular > num_sections:
            raise ValueError("intro_regular_sections must be <= num_sections")
        if jitter_stage_count < 0:
            raise ValueError("jitter_stage_count must be >= 0")
        if intro_regular + jitter_stage_count > num_sections:
            raise ValueError("intro_regular_sections + jitter_stage_count must be <= num_sections")
        if delta_t_tics < 1:
            raise ValueError("delta_t_tics must be >= 1")
        if delta_t_minimum < 1:
            raise ValueError("delta_t_minimum must be >= 1")
        if delta_t_minimum > delta_t_tics:
            raise ValueError("delta_t_minimum must be <= delta_t_tics")
        if delta_t_decay < 0:
            raise ValueError("delta_t_decay must be >= 0")
        if beat_tics < 1:
            raise ValueError("beat_tics must be >= 1")
        if jitter_tics < 0:
            raise ValueError("jitter_tics must be >= 0")
        if switch_wall_margin < 0:
            raise ValueError("switch_wall_margin must be >= 0")
        if switch_intro_stages < 0:
            raise ValueError("switch_intro_stages must be >= 0")
        if switch_intro_stages > num_sections:
            raise ValueError("switch_intro_stages must be <= num_sections")
        if not (0.0 <= switch_intro_jitter_frac <= 1.0):
            raise ValueError("switch_intro_jitter_frac must be in [0.0, 1.0]")

    def _choose_seed(self) -> int:
        if self.config["seed"] is not None:
            return int(self.config["seed"])
        if bool(self.config.get("random_seed_on_generate", True)):
            return random.randint(0, 999999)
        return 42

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = self._choose_seed()

        num_sections = int(cfg["num_sections"])
        intro_regular = int(cfg["intro_regular_sections"])
        jitter_stage_count = int(cfg["jitter_stage_count"])
        delta_t_tics = int(cfg["delta_t_tics"])
        delta_t_minimum = int(cfg["delta_t_minimum"])
        delta_t_decay = int(cfg["delta_t_decay"])
        beat_tics = int(cfg["beat_tics"])
        jitter_tics = int(cfg["jitter_tics"])

        stage_step = int(cfg["stage_step"])
        room_w = int(cfg["room_width"])
        room_h = int(cfg["room_height"])
        top_y = int(cfg["top_y"])
        bottom_y = int(cfg["bottom_y"])

        switch_w = int(cfg["switch_width"])
        switch_h = int(cfg["switch_height"])
        switch_wall_margin = int(cfg["switch_wall_margin"])
        switch_intro_stages = int(cfg["switch_intro_stages"])
        switch_intro_jitter_frac = float(cfg["switch_intro_jitter_frac"])
        switch_pos_seed_offset = int(cfg["switch_pos_seed_offset"])
        switch_use_range = int(cfg["switch_use_range"])
        switch_marker_actor_a = "BlueTorch"
        switch_marker_actor_b = "GreenTorch"

        # Hardcoded theme
        floor_tex = "FLAT19"
        ceil_tex = "PLANET1"
        wall_tex = "SILVER1"
        door_wall_tex = "DOORTRAK"
        door_flat_tex = "GATE1"
        window_tex = "MIDGRATE"
        door_width = min(128, room_h)
        door_depth = 16

        start_x = 0
        stage_x = [stage_step * i for i in range(1, num_sections + 1)]
        exit_x = stage_step * (num_sections + 1)

        print(
            f"Generating rhythm sync benchmark "
            f"(sections={num_sections}, intro_regular={intro_regular}, seed={seed})"
        )

        builder = ProceduralMapBuilder(
            bounds=(-800, -900, exit_x + 900, 900),
            seed=seed,
        )

        overlay_lumps = {
            "RS_CLEAR": solid_rgba_png(320, 240, (0, 0, 0, 0)),
            "RS_BEAT": border_rgba_png(320, 240, (255, 0, 0, 255), thickness=18),
            "RS_GOOD": solid_rgba_png(320, 240, (0, 255, 0, 255)),
            "RS_FAIL": solid_rgba_png(320, 240, (255, 0, 0, 255)),
        }
        for lump_name, lump_data in overlay_lumps.items():
            builder.wad_writer.add_lump(lump_name, lump_data)

        builder.add_area(
            "Start_A",
            shape=(start_x, top_y, room_w, room_h),
            floor_height=0,
            ceiling_height=128,
            floor_texture=floor_tex,
            ceiling_texture=ceil_tex,
            wall_texture=wall_tex,
        )
        builder.add_area(
            "Start_B",
            shape=(start_x, bottom_y, room_w, room_h),
            floor_height=0,
            ceiling_height=128,
            floor_texture=floor_tex,
            ceiling_texture=ceil_tex,
            wall_texture=wall_tex,
        )

        for idx, x_pos in enumerate(stage_x, start=1):
            stage_a = f"Stage{idx}_A"
            stage_b = f"Stage{idx}_B"

            builder.add_area(
                stage_a,
                shape=(x_pos, top_y, room_w, room_h),
                floor_height=0,
                ceiling_height=128,
                floor_texture=floor_tex,
                ceiling_texture=ceil_tex,
                wall_texture=wall_tex,
            )
            builder.add_area(
                stage_b,
                shape=(x_pos, bottom_y, room_w, room_h),
                floor_height=0,
                ceiling_height=128,
                floor_texture=floor_tex,
                ceiling_texture=ceil_tex,
                wall_texture=wall_tex,
            )

            builder.connect_adjacent(
                stage_a,
                stage_b,
                ConnectionType.WINDOW,
                middle_texture=window_tex,
                blocking=True,
            )

        builder.add_area(
            "Exit_A",
            shape=(exit_x, top_y, room_w, room_h),
            floor_height=0,
            ceiling_height=128,
            floor_texture=floor_tex,
            ceiling_texture=ceil_tex,
            wall_texture=wall_tex,
        )
        builder.add_area(
            "Exit_B",
            shape=(exit_x, bottom_y, room_w, room_h),
            floor_height=0,
            ceiling_height=128,
            floor_texture=floor_tex,
            ceiling_texture=ceil_tex,
            wall_texture=wall_tex,
        )

        builder.connect_adjacent("Start_A", "Stage1_A", ConnectionType.OPEN)
        builder.connect_adjacent("Start_B", "Stage1_B", ConnectionType.OPEN)

        transition_tags_a: Dict[int, int] = {}
        transition_tags_b: Dict[int, int] = {}

        for idx in range(1, num_sections):
            from_a = f"Stage{idx}_A"
            from_b = f"Stage{idx}_B"
            to_a = f"Stage{idx + 1}_A"
            to_b = f"Stage{idx + 1}_B"

            door_a = builder.add_boundary(
                from_a,
                to_a,
                name=f"Door_A_{idx}_{idx + 1}",
                connection_type=ConnectionType.DOOR,
                floor_height=0,
                ceiling_height=0,
                wall_texture=door_wall_tex,
                floor_texture=door_flat_tex,
                ceiling_texture=door_flat_tex,
                door_width=door_width,
                door_depth=door_depth,
            )
            door_b = builder.add_boundary(
                from_b,
                to_b,
                name=f"Door_B_{idx}_{idx + 1}",
                connection_type=ConnectionType.DOOR,
                floor_height=0,
                ceiling_height=0,
                wall_texture=door_wall_tex,
                floor_texture=door_flat_tex,
                ceiling_texture=door_flat_tex,
                door_width=door_width,
                door_depth=door_depth,
            )

            tag_a = 1100 + idx
            tag_b = 2100 + idx
            builder.graph.get_area(door_a).config.tag = tag_a
            builder.graph.get_area(door_b).config.tag = tag_b
            transition_tags_a[idx] = tag_a
            transition_tags_b[idx] = tag_b

        door_a_exit = builder.add_boundary(
            f"Stage{num_sections}_A",
            "Exit_A",
            name="Door_A_Exit",
            connection_type=ConnectionType.DOOR,
            floor_height=0,
            ceiling_height=0,
            wall_texture=door_wall_tex,
            floor_texture=door_flat_tex,
            ceiling_texture=door_flat_tex,
            door_width=door_width,
            door_depth=door_depth,
        )
        door_b_exit = builder.add_boundary(
            f"Stage{num_sections}_B",
            "Exit_B",
            name="Door_B_Exit",
            connection_type=ConnectionType.DOOR,
            floor_height=0,
            ceiling_height=0,
            wall_texture=door_wall_tex,
            floor_texture=door_flat_tex,
            ceiling_texture=door_flat_tex,
            door_width=door_width,
            door_depth=door_depth,
        )

        exit_tag_a = 1100 + num_sections
        exit_tag_b = 2100 + num_sections
        builder.graph.get_area(door_a_exit).config.tag = exit_tag_a
        builder.graph.get_area(door_b_exit).config.tag = exit_tag_b

        max_dx = (room_w // 2) - (switch_w // 2) - switch_wall_margin
        max_dy = (room_h // 2) - (switch_h // 2) - switch_wall_margin
        if max_dx < 0 or max_dy < 0:
            raise ValueError(
                "switch dimensions + switch_wall_margin exceed stage room bounds"
            )

        intro_jitter_dx = int(max_dx * switch_intro_jitter_frac)
        intro_jitter_dy = int(max_dy * switch_intro_jitter_frac)

        builder.add_thing("PLAYER1_START", start_x - 80, top_y, angle=0)
        builder.add_thing("PLAYER2_START", start_x - 80, bottom_y, angle=0)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")

        # Internal map vars (not exposed to ViZDoom)
        acs.add_map_var("delta_t", "int", delta_t_tics)
        acs.add_map_var("delta_t_min", "int", delta_t_minimum)
        acs.add_map_var("delta_t_dec", "int", delta_t_decay)
        acs.add_map_var("tick_serial", "int", 0)
        acs.add_map_var("press_tic_a", "int", -999999)
        acs.add_map_var("press_tic_b", "int", -999999)

        # Global vars exposed to ViZDoom as USER24-USER38
        acs.add_global_var("current_stage", 24, "int")
        acs.add_global_var("completed_stages", 25, "int")
        acs.add_global_var("current_stage_type", 26, "int")
        acs.add_global_var("pending_a", 27, "int")
        acs.add_global_var("pending_b", 28, "int")
        acs.add_global_var("failed", 29, "int")
        acs.add_global_var("finished", 30, "int")
        acs.add_global_var("current_cue_owner", 31, "int")
        acs.add_global_var("num_sections_total", 32, "int")
        acs.add_global_var("switch_x_a", 33, "int")
        acs.add_global_var("switch_y_a", 34, "int")
        acs.add_global_var("switch_x_b", 35, "int")
        acs.add_global_var("switch_y_b", 36, "int")
        acs.add_global_var("cue_visible_a", 37, "int")
        acs.add_global_var("cue_visible_b", 38, "int")

        for idx in range(1, num_sections + 1):
            acs.add_map_var(f"stage_type_{idx}", "int", 0)
            acs.add_map_var(f"cue_owner_{idx}", "int", -1)

        acs.add_map_var("switch_base_dx_a", "int", 0)
        acs.add_map_var("switch_base_dy_a", "int", 0)
        acs.add_map_var("switch_base_dx_b", "int", 0)
        acs.add_map_var("switch_base_dy_b", "int", 0)
        acs.add_map_var("switch_episode_seed", "int", 0)
        acs.add_map_var("single_observer_owner_offset", "int", 0)

        get_stage_type_lines = ["function int GetStageType(int stage) {"]
        for idx in range(1, num_sections + 1):
            prefix = "if" if idx == 1 else "else if"
            get_stage_type_lines.append(f"    {prefix} (stage == {idx}) return stage_type_{idx};")
        get_stage_type_lines.append("    return 0;")
        get_stage_type_lines.append("}")

        get_cue_owner_lines = ["function int GetCueOwner(int stage) {"]
        for idx in range(1, num_sections + 1):
            prefix = "if" if idx == 1 else "else if"
            get_cue_owner_lines.append(f"    {prefix} (stage == {idx}) return cue_owner_{idx};")
        get_cue_owner_lines.append("    return -1;")
        get_cue_owner_lines.append("}")

        assign_stage_type_lines = ["function void AssignStageTypes(void) {", "    single_observer_owner_offset = Random(0, 1);"]
        for idx in range(1, num_sections + 1):
            def stage_type_for_index(stage, regular_sections, jitter_sections):
                if stage <= regular_sections:
                    return 0
                if stage <= regular_sections + jitter_sections:
                    return 1
                return 2
            stage_type = stage_type_for_index(idx, intro_regular, jitter_stage_count)
            if stage_type == 0:
                assign_stage_type_lines.append(f"    stage_type_{idx} = 0;")
                assign_stage_type_lines.append(f"    cue_owner_{idx} = -1;")
            elif stage_type == 1:
                assign_stage_type_lines.append(f"    stage_type_{idx} = 1;")
                assign_stage_type_lines.append(f"    cue_owner_{idx} = -1;")
            else:
                single_idx = idx - intro_regular - jitter_stage_count - 1
                assign_stage_type_lines.append(f"    stage_type_{idx} = 2;")
                assign_stage_type_lines.append(
                    f"    cue_owner_{idx} = (single_observer_owner_offset + {single_idx}) % 2;"
                )
        assign_stage_type_lines.append("}")

        switch_calc_lines = [
            "function int PositiveMod(int value, int mod) {",
            "    int rem = value % mod;",
            "    if (rem < 0) rem = rem + mod;",
            "    return rem;",
            "}",
            "function int EpisodeRandFromStage(int stage, int side, int axis, int lo, int hi) {",
            "    int span = (hi - lo) + 1;",
            "    int hash = switch_episode_seed + (stage * 1103) + (side * 1999) + (axis * 3571);",
            "    hash = hash * 1103515 + 12345;",
            "    if (hash < 0) hash = -hash;",
            "    return lo + PositiveMod(hash, span);",
            "}",
            "function int GetSwitchDXA(int stage) {",
            f"    if (stage <= {switch_intro_stages}) {{",
            f"        return ClampInt(switch_base_dx_a + EpisodeRandFromStage(stage, 0, 0, -{intro_jitter_dx}, {intro_jitter_dx}), -{max_dx}, {max_dx});",
            "    }",
            f"    return EpisodeRandFromStage(stage, 0, 0, -{max_dx}, {max_dx});",
            "}",
            "function int GetSwitchDYA(int stage) {",
            f"    if (stage <= {switch_intro_stages}) {{",
            f"        return ClampInt(switch_base_dy_a + EpisodeRandFromStage(stage, 0, 1, -{intro_jitter_dy}, {intro_jitter_dy}), -{max_dy}, {max_dy});",
            "    }",
            f"    return EpisodeRandFromStage(stage, 0, 1, -{max_dy}, {max_dy});",
            "}",
            "function int GetSwitchDXB(int stage) {",
            f"    if (stage <= {switch_intro_stages}) {{",
            f"        return ClampInt(switch_base_dx_b + EpisodeRandFromStage(stage, 1, 0, -{intro_jitter_dx}, {intro_jitter_dx}), -{max_dx}, {max_dx});",
            "    }",
            f"    return EpisodeRandFromStage(stage, 1, 0, -{max_dx}, {max_dx});",
            "}",
            "function int GetSwitchDYB(int stage) {",
            f"    if (stage <= {switch_intro_stages}) {{",
            f"        return ClampInt(switch_base_dy_b + EpisodeRandFromStage(stage, 1, 1, -{intro_jitter_dy}, {intro_jitter_dy}), -{max_dy}, {max_dy});",
            "    }",
            f"    return EpisodeRandFromStage(stage, 1, 1, -{max_dy}, {max_dy});",
            "}",
        ]

        assign_switch_positions_lines = ["function void AssignSwitchPositions(void) {"]
        if switch_pos_seed_offset > 0:
            assign_switch_positions_lines.append("    int burn = 0;")
            assign_switch_positions_lines.append(
                f"    while (burn < {switch_pos_seed_offset}) {{"
            )
            assign_switch_positions_lines.append("        Random(0, 255);")
            assign_switch_positions_lines.append("        burn = burn + 1;")
            assign_switch_positions_lines.append("    }")
        assign_switch_positions_lines.append(f"    switch_base_dx_a = Random(-{max_dx}, {max_dx});")
        assign_switch_positions_lines.append(f"    switch_base_dy_a = Random(-{max_dy}, {max_dy});")
        assign_switch_positions_lines.append(f"    switch_base_dx_b = Random(-{max_dx}, {max_dx});")
        assign_switch_positions_lines.append(f"    switch_base_dy_b = Random(-{max_dy}, {max_dy});")
        assign_switch_positions_lines.append("    switch_episode_seed = Random(0, 32767);")
        assign_switch_positions_lines.append("    int stage = 1;")
        assign_switch_positions_lines.append(f"    while (stage <= {num_sections}) {{")
        assign_switch_positions_lines.append(
            "        int stage_switch_x_a = GetSwitchXA(stage);"
        )
        assign_switch_positions_lines.append(
            "        int stage_switch_y_a = GetSwitchYA(stage);"
        )
        assign_switch_positions_lines.append(
            "        int stage_switch_x_b = GetSwitchXB(stage);"
        )
        assign_switch_positions_lines.append(
            "        int stage_switch_y_b = GetSwitchYB(stage);"
        )
        assign_switch_positions_lines.append(
            f'        Spawn("{switch_marker_actor_a}", stage_switch_x_a << 16, stage_switch_y_a << 16, 0, 5000 + stage);'
        )
        assign_switch_positions_lines.append(
            f'        Spawn("{switch_marker_actor_b}", stage_switch_x_b << 16, stage_switch_y_b << 16, 0, 6000 + stage);'
        )
        assign_switch_positions_lines.append("        stage = stage + 1;")
        assign_switch_positions_lines.append("    }")
        assign_switch_positions_lines.append("}")

        switch_x_a_lines = [
            "function int GetSwitchXA(int stage) {",
            f"    return ({stage_step} * stage) + GetSwitchDXA(stage);",
            "}",
        ]

        switch_y_a_lines = [
            "function int GetSwitchYA(int stage) {",
            f"    return {top_y} + GetSwitchDYA(stage);",
            "}",
        ]

        switch_x_b_lines = [
            "function int GetSwitchXB(int stage) {",
            f"    return ({stage_step} * stage) + GetSwitchDXB(stage);",
            "}",
        ]

        switch_y_b_lines = [
            "function int GetSwitchYB(int stage) {",
            f"    return {bottom_y} + GetSwitchDYB(stage);",
            "}",
        ]

        complete_stage_lines = [
            "function void CompleteStage(int stage) {",
            "    if (failed || finished) return;",
            "    completed_stages = completed_stages + 1;",
        ]

        if num_sections == 1:
            complete_stage_lines.append("    if (stage == 1) {")
            complete_stage_lines.append(f"        Door_Open({exit_tag_a}, 16, 0);")
            complete_stage_lines.append(f"        Door_Open({exit_tag_b}, 16, 0);")
            complete_stage_lines.append("        finished = 1;")
            complete_stage_lines.append('        ShowOverlayForAllPlayers("RS_GOOD");')
            complete_stage_lines.append("        ACS_ExecuteAlways(907, 0);")
            complete_stage_lines.append("    }")
        else:
            for idx in range(1, num_sections):
                keyword = "if" if idx == 1 else "else if"
                complete_stage_lines.append(f"    {keyword} (stage == {idx}) {{")
                complete_stage_lines.append(f"        Door_Open({transition_tags_a[idx]}, 16, 0);")
                complete_stage_lines.append(f"        Door_Open({transition_tags_b[idx]}, 16, 0);")
                complete_stage_lines.append(f"        current_stage = {idx + 1};")
                complete_stage_lines.append("        SyncPublicState();")
                complete_stage_lines.append("    }")

            complete_stage_lines.append(f"    else if (stage == {num_sections}) {{")
            complete_stage_lines.append(f"        Door_Open({exit_tag_a}, 16, 0);")
            complete_stage_lines.append(f"        Door_Open({exit_tag_b}, 16, 0);")
            complete_stage_lines.append("        finished = 1;")
            complete_stage_lines.append('        ShowOverlayForAllPlayers("RS_GOOD");')
            complete_stage_lines.append("        ACS_ExecuteAlways(907, 0);")
            complete_stage_lines.append("    }")

        complete_stage_lines.append("    ResetPending();")
        # Apply delta_t decay after each stage completion
        complete_stage_lines.append("    // Decay timing tolerance to increase difficulty")
        complete_stage_lines.append("    if (delta_t > delta_t_min) {")
        complete_stage_lines.append("        delta_t = delta_t - delta_t_dec;")
        complete_stage_lines.append("        if (delta_t < delta_t_min) delta_t = delta_t_min;")
        complete_stage_lines.append("    }")
        complete_stage_lines.append("}")

        acs.add_global_code(
            "\n".join(
                [
                    """
function void ResetPending(void) {
    pending_a = 0;
    pending_b = 0;
    press_tic_a = -999999;
    press_tic_b = -999999;
}

function void SyncPublicState(void) {
    current_stage_type = GetStageType(current_stage);
    current_cue_owner = GetCueOwner(current_stage);
    switch_x_a = GetSwitchXA(current_stage);
    switch_y_a = GetSwitchYA(current_stage);
    switch_x_b = GetSwitchXB(current_stage);
    switch_y_b = GetSwitchYB(current_stage);
}

function void TriggerFail(void) {
    if (failed || finished) {
        return;
    }

    failed = 1;
    ShowOverlayForAllPlayers("RS_FAIL");
    ACS_ExecuteAlways(907, 0);
}

function void ShowOverlayForActivator(str lump) {
    SetHudSize(320, 240, 1);
    SetFont(lump);
    HudMessage(s:"A"; HUDMSG_PLAIN, 700, CR_UNTRANSLATED, 160.0, 120.0, 0.0);
}

function void ShowOverlayForAllPlayers(str lump) {
    int player_idx = 0;
    while (player_idx < 2) {
        SetActivatorToPlayer(player_idx);
        ShowOverlayForActivator(lump);
        player_idx = player_idx + 1;
    }
}

function int ClampInt(int value, int lo, int hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}
""",
                    "\n".join(get_stage_type_lines),
                    "\n".join(get_cue_owner_lines),
                    "\n".join(assign_stage_type_lines),
                    "\n".join(switch_calc_lines),
                    "\n".join(assign_switch_positions_lines),
                    "\n".join(switch_x_a_lines),
                    "\n".join(switch_y_a_lines),
                    "\n".join(switch_x_b_lines),
                    "\n".join(switch_y_b_lines),
                    f"""
function int ShouldShowCueForPlayer(int player_num) {{
    int stage_type = GetStageType(current_stage);
    if (stage_type == 2) {{
        return player_num == GetCueOwner(current_stage);
    }}
    return 1;
}}

function int CurrentWaitTics(void) {{
    int stage_type = GetStageType(current_stage);
    if (stage_type == 1) {{
        return {beat_tics} + Random(-{jitter_tics}, {jitter_tics});
    }}
    return {beat_tics};
}}

function int AbsInt(int value) {{
    if (value < 0) {{
        return -value;
    }}
    return value;
}}

function void TryUseSwitchForPlayer(int player_num) {{
    if (failed || finished) {{
        return;
    }}

    int side = -1;
    if (player_num == 0) {{
        side = 0;
    }} else if (player_num == 1) {{
        side = 1;
    }} else {{
        return;
    }}

    int sx;
    int sy;

    if (side == 0) {{
        sx = GetSwitchXA(current_stage);
        sy = GetSwitchYA(current_stage);
    }} else {{
        sx = GetSwitchXB(current_stage);
        sy = GetSwitchYB(current_stage);
    }}

    int px = GetActorX(0) >> 16;
    int py = GetActorY(0) >> 16;

    if (AbsInt(px - sx) <= {switch_use_range} && AbsInt(py - sy) <= {switch_use_range}) {{
        HandleSwitchPress(current_stage, side);
    }}
}}

function void HandleSwitchPress(int stage, int side) {{
    if (failed || finished) {{
        return;
    }}

    if (stage != current_stage) {{
        return;
    }}

    int now = Timer();

    if (side == 0) {{
        if (!pending_a) {{
            Thing_Remove(5000 + stage);
            Spawn("RedTorch", GetSwitchXA(stage) << 16, GetSwitchYA(stage) << 16, 0, 5000 + stage);
        }}
        if (pending_b) {{
            if ((now - press_tic_b) <= delta_t) {{
                CompleteStage(stage);
            }} else {{
                TriggerFail();
            }}
        }} else {{
            pending_a = 1;
            press_tic_a = now;
        }}
    }} else {{
        if (!pending_b) {{
            Thing_Remove(6000 + stage);
            Spawn("RedTorch", GetSwitchXB(stage) << 16, GetSwitchYB(stage) << 16, 0, 6000 + stage);
        }}
        if (pending_a) {{
            if ((now - press_tic_a) <= delta_t) {{
                CompleteStage(stage);
            }} else {{
                TriggerFail();
            }}
        }} else {{
            pending_b = 1;
            press_tic_b = now;
        }}
    }}
}}
""",
                    "\n".join(complete_stage_lines),
                ]
            )
        )

        acs.add_script(
            ScriptType.OPEN,
            f"""
ResetPending();
AssignStageTypes();
AssignSwitchPositions();
current_stage = 1;
completed_stages = 0;
failed = 0;
finished = 0;
cue_visible_a = 0;
cue_visible_b = 0;
num_sections_total = {num_sections};
SyncPublicState();
ACS_ExecuteAlways(901, 0);
ACS_ExecuteAlways(902, 0);
""",
            number=900,
        )

        acs.add_script(
            ScriptType.VOID,
            """
while (!failed && !finished) {
    Delay(CurrentWaitTics());

    if (failed || finished) {
        terminate;
    }

    tick_serial = tick_serial + 1;
}
""",
            number=901,
        )

        acs.add_script(
            ScriptType.VOID,
            """
while (!failed && !finished) {
    int now = Timer();

    if (pending_a && !pending_b && (now - press_tic_a) > delta_t) {
        TriggerFail();
    }

    if (pending_b && !pending_a && (now - press_tic_b) > delta_t) {
        TriggerFail();
    }

    Delay(1);
}
""",
            number=902,
        )

        acs.add_script(
            ScriptType.ENTER,
            """
Thing_ChangeTID(0, 1000 + PlayerNumber());
SetActorProperty(0, APROP_Health, 100);
ClearInventory();
""",
            number=903,
        )

        acs.add_script(
            ScriptType.ENTER,
            """
int pn = PlayerNumber();
int seen = tick_serial;
int prev_buttons = 0;
int cue_remaining = 0;
int cue_visible = 0;

ACS_ExecuteAlways(905, 0, 0);

while (!failed && !finished) {
    int buttons = GetPlayerInput(-1, INPUT_BUTTONS);

    if ((buttons & BT_USE) && !(prev_buttons & BT_USE)) {
        TryUseSwitchForPlayer(pn);
    }
    prev_buttons = buttons;

    if (failed || finished) {
        break;
    }

    if (tick_serial != seen) {
        seen = tick_serial;
        if (ShouldShowCueForPlayer(pn)) {
            cue_remaining = 8;
        }
    }

    if (cue_remaining > 0) {
        if (!cue_visible) {
            ACS_ExecuteAlways(905, 0, 1);
            cue_visible = 1;
        }
        cue_remaining = cue_remaining - 1;
    } else if (cue_visible) {
        ACS_ExecuteAlways(905, 0, 0);
        cue_visible = 0;
    }

    // Mirror local cue_visible to global for RL telemetry
    if (pn == 0) {
        cue_visible_a = cue_visible;
    } else if (pn == 1) {
        cue_visible_b = cue_visible;
    }

    Delay(1);
}
""",
            number=904,
        )

        acs.add_script(
            ScriptType.VOID,
            """
if (show_beat) {
    ShowOverlayForActivator("RS_BEAT");
} else {
    ShowOverlayForActivator("RS_CLEAR");
}
""",
            number=905,
            args=["show_beat"],
        )

        acs.add_script(
            ScriptType.VOID,
            """
Delay(1);
Exit_Normal(0);
""",
            number=907,
        )

        builder.map_data.scripts = acs.to_code()
        try:
            builder.map_data.behavior = acs.compile()
        except Exception as exc:
            print(f"ACS compile warning: {exc}")
            if hasattr(exc, "stderr") and exc.stderr:
                print(f"ACC stderr: {exc.stderr}")
            builder.map_data.behavior = b"ACSE\x08\x00\x00\x00\x00\x00\x00\x00"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        builder.build(output_path)
        print(f"WAD written to {output_path}")


if __name__ == "__main__":
    output = "examples/benchmark/output/rhythm_sync.wad"
    scenario = RhythmSyncScenario(config={"seed": random.randint(0, 999999)})
    scenario.generate(output)
