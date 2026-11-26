import os
import random
from typing import Any, Dict

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class RhythmSyncScenario(Scenario):
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": None,
            "random_seed_on_generate": True,
            "num_sections": 20,
            "intro_regular_sections": 1,
            "delta_t_tics": 36,
            "beat_tics": 140,
            "jitter_tics": 35,
            "stage_step": 512,
            "room_width": 512,
            "room_height": 256,
            "top_y": 128,
            "bottom_y": -128,
            "switch_width": 64,
            "switch_height": 90,
            "switch_x_offset": 144,
            "switch_use_range": 96,
        }

    def validate_config(self) -> None:
        num_sections = int(self.config["num_sections"])
        intro_regular = int(self.config["intro_regular_sections"])
        delta_t_tics = int(self.config["delta_t_tics"])
        beat_tics = int(self.config["beat_tics"])
        jitter_tics = int(self.config["jitter_tics"])

        if num_sections < 1:
            raise ValueError("num_sections must be >= 1")
        if intro_regular < 1:
            raise ValueError("intro_regular_sections must be >= 1")
        if intro_regular > num_sections:
            raise ValueError("intro_regular_sections must be <= num_sections")
        if delta_t_tics < 1:
            raise ValueError("delta_t_tics must be >= 1")
        if beat_tics < 1:
            raise ValueError("beat_tics must be >= 1")
        if jitter_tics < 0:
            raise ValueError("jitter_tics must be >= 0")

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
        delta_t_tics = int(cfg["delta_t_tics"])
        beat_tics = int(cfg["beat_tics"])
        jitter_tics = int(cfg["jitter_tics"])

        stage_step = int(cfg["stage_step"])
        room_w = int(cfg["room_width"])
        room_h = int(cfg["room_height"])
        top_y = int(cfg["top_y"])
        bottom_y = int(cfg["bottom_y"])

        switch_w = int(cfg["switch_width"])
        switch_h = int(cfg["switch_height"])
        switch_x_offset = int(cfg["switch_x_offset"])
        switch_use_range = int(cfg["switch_use_range"])

        # Hardcoded theme
        floor_tex = "FLAT19"
        ceil_tex = "PLANET1"
        wall_tex = "SILVER1"
        door_wall_tex = "DOORTRAK"
        door_flat_tex = "GATE1"
        window_tex = "MIDGRATE"
        switch_tex = "COMPBLUE"  # Used for side of console
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

        switch_positions_a: Dict[int, tuple[int, int]] = {}
        switch_positions_b: Dict[int, tuple[int, int]] = {}
        for idx, x_pos in enumerate(stage_x, start=1):
            sx = x_pos + switch_x_offset
            switch_positions_a[idx] = (sx, top_y)
            switch_positions_b[idx] = (sx, bottom_y)

            switch_specs = [
                (f"Stage{idx}_A", f"Switch{idx}_A", sx, top_y),
                (f"Stage{idx}_B", f"Switch{idx}_B", sx, bottom_y),
            ]

            for room_name, switch_name, sx, sy in switch_specs:
                builder.add_area(
                    switch_name,
                    shape=(sx, sy, switch_w, switch_h),
                    mode="overwrite",
                    floor_height=40,
                    ceiling_height=128,
                    floor_texture="TLITE6_4",
                    ceiling_texture=ceil_tex,
                    wall_texture="COMPBLUE",
                )
                builder.connect_adjacent(
                    room_name,
                    switch_name,
                    ConnectionType.OPEN,
                )

        builder.add_thing("PLAYER1_START", start_x - 80, top_y, angle=0)
        builder.add_thing("PLAYER2_START", start_x - 80, bottom_y, angle=0)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")

        acs.add_map_var("delta_t", "int", delta_t_tics)
        acs.add_map_var("env_signal", "int", 0)
        acs.add_map_var("tick_serial", "int", 0)
        acs.add_map_var("current_stage", "int", 1)
        acs.add_map_var("pending_a", "int", 0)
        acs.add_map_var("pending_b", "int", 0)
        acs.add_map_var("press_tic_a", "int", -999999)
        acs.add_map_var("press_tic_b", "int", -999999)
        acs.add_map_var("failed", "int", 0)
        acs.add_map_var("finished", "int", 0)

        for idx in range(1, num_sections + 1):
            acs.add_map_var(f"stage_type_{idx}", "int", 0)

        get_stage_type_lines = ["function int GetStageType(int stage) {"]
        for idx in range(1, num_sections + 1):
            prefix = "if" if idx == 1 else "else if"
            get_stage_type_lines.append(f"    {prefix} (stage == {idx}) return stage_type_{idx};")
        get_stage_type_lines.append("    return 0;")
        get_stage_type_lines.append("}")

        assign_stage_type_lines = ["function void AssignStageTypes(void) {"]
        for idx in range(1, num_sections + 1):
            if idx <= intro_regular:
                assign_stage_type_lines.append(f"    stage_type_{idx} = 0;")
            else:
                assign_stage_type_lines.append(f"    stage_type_{idx} = Random(0, 2);")
        assign_stage_type_lines.append("}")

        switch_x_a_lines = ["function int GetSwitchXA(int stage) {"]
        for idx in range(1, num_sections + 1):
            prefix = "if" if idx == 1 else "else if"
            switch_x_a_lines.append(f"    {prefix} (stage == {idx}) return {switch_positions_a[idx][0]};")
        switch_x_a_lines.append("    return 0;")
        switch_x_a_lines.append("}")

        switch_y_a_lines = ["function int GetSwitchYA(int stage) {"]
        for idx in range(1, num_sections + 1):
            prefix = "if" if idx == 1 else "else if"
            switch_y_a_lines.append(f"    {prefix} (stage == {idx}) return {switch_positions_a[idx][1]};")
        switch_y_a_lines.append("    return 0;")
        switch_y_a_lines.append("}")

        switch_x_b_lines = ["function int GetSwitchXB(int stage) {"]
        for idx in range(1, num_sections + 1):
            prefix = "if" if idx == 1 else "else if"
            switch_x_b_lines.append(f"    {prefix} (stage == {idx}) return {switch_positions_b[idx][0]};")
        switch_x_b_lines.append("    return 0;")
        switch_x_b_lines.append("}")

        switch_y_b_lines = ["function int GetSwitchYB(int stage) {"]
        for idx in range(1, num_sections + 1):
            prefix = "if" if idx == 1 else "else if"
            switch_y_b_lines.append(f"    {prefix} (stage == {idx}) return {switch_positions_b[idx][1]};")
        switch_y_b_lines.append("    return 0;")
        switch_y_b_lines.append("}")

        complete_stage_lines = [
            "function void CompleteStage(int stage) {",
            "    if (failed || finished) return;",
        ]

        if num_sections == 1:
            complete_stage_lines.append("    if (stage == 1) {")
            complete_stage_lines.append(f"        Door_Open({exit_tag_a}, 16, 0);")
            complete_stage_lines.append(f"        Door_Open({exit_tag_b}, 16, 0);")
            complete_stage_lines.append("        finished = 1;")
            complete_stage_lines.append("        FadeTo(0.0, 1.0, 0.0, 0.45, 0.0);")
            complete_stage_lines.append("        Exit_Normal(0);")
            complete_stage_lines.append("    }")
        else:
            for idx in range(1, num_sections):
                keyword = "if" if idx == 1 else "else if"
                complete_stage_lines.append(f"    {keyword} (stage == {idx}) {{")
                complete_stage_lines.append(f"        Door_Open({transition_tags_a[idx]}, 16, 0);")
                complete_stage_lines.append(f"        Door_Open({transition_tags_b[idx]}, 16, 0);")
                complete_stage_lines.append(f"        current_stage = {idx + 1};")
                complete_stage_lines.append("    }")

            complete_stage_lines.append(f"    else if (stage == {num_sections}) {{")
            complete_stage_lines.append(f"        Door_Open({exit_tag_a}, 16, 0);")
            complete_stage_lines.append(f"        Door_Open({exit_tag_b}, 16, 0);")
            complete_stage_lines.append("        finished = 1;")
            complete_stage_lines.append("        FadeTo(0.0, 1.0, 0.0, 0.45, 0.0);")
            complete_stage_lines.append("        Exit_Normal(0);")
            complete_stage_lines.append("    }")

        complete_stage_lines.append("    ResetPending();")
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

function void TriggerFail(void) {
    if (failed || finished) {
        return;
    }

    failed = 1;
    FadeTo(1.0, 0.0, 0.0, 0.75, 0.0);
    Exit_Normal(0);
}
""",
                    "\n".join(get_stage_type_lines),
                    "\n".join(assign_stage_type_lines),
                    "\n".join(switch_x_a_lines),
                    "\n".join(switch_y_a_lines),
                    "\n".join(switch_x_b_lines),
                    "\n".join(switch_y_b_lines),
                    f"""
function int ShouldShowCueForPlayer(int player_num) {{
    int stage_type = GetStageType(current_stage);
    if (stage_type == 2) {{
        if (player_num == 0) return 1;
        return 0;
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
            """
ResetPending();
AssignStageTypes();
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

    env_signal = 1;
    tick_serial = tick_serial + 1;
    Delay(1);
    env_signal = 0;
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
""",
            number=903,
        )

        acs.add_script(
            ScriptType.ENTER,
            """
int pn = PlayerNumber();
int seen = tick_serial;
int prev_buttons = 0;

while (!failed && !finished) {
    int buttons = GetPlayerInput(-1, INPUT_BUTTONS);

    if ((buttons & BT_USE) && !(prev_buttons & BT_USE)) {
        TryUseSwitchForPlayer(pn);
    }
    prev_buttons = buttons;

    if (tick_serial != seen) {
        seen = tick_serial;

        if (ShouldShowCueForPlayer(pn)) {
            FadeTo(1.0, 0.0, 0.0, 0.35, 0.0);
            Delay(1);
            FadeTo(0.0, 0.0, 0.0, 0.0, 0.2);
        }
    }

    Delay(1);
}
""",
            number=904,
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
