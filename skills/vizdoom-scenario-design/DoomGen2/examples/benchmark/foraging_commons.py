import os
import random
from typing import Any, Dict

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class ForagingCommonsScenario(Scenario):

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "field_size": 1600,
            "cleanup_size": 384,
            "cleanup_distance": 1200,
            "corridor_width": 192,
            "num_agents": 2,
            "initial_spawn_rate": 80,
            "harvest_degrade": 2,
            "natural_decay_rate": 1,
            "cleanup_boost": 15,
            "cleanup_time": 72,
            "hp_drain": 2,
            "spots_per_room": 16,
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)
        random.seed(seed)

        field_size       = cfg["field_size"]
        cleanup_size     = cfg["cleanup_size"]
        cleanup_distance = cfg["cleanup_distance"]
        corridor_w       = cfg["corridor_width"]
        total_spots      = cfg["spots_per_room"]
        initial_sr       = cfg["initial_spawn_rate"]
        harvest_deg      = cfg["harvest_degrade"]
        decay_rate       = cfg["natural_decay_rate"]
        cleanup_boost    = cfg["cleanup_boost"]
        cleanup_time     = cfg["cleanup_time"]
        hp_drain         = cfg["hp_drain"]
        num_agents       = cfg["num_agents"]

        if num_agents < 2 or num_agents > 4:
            raise ValueError(f"ForagingCommonsScenario supports 2-4 agents, got {num_agents}")

        cleanup_half = cleanup_size // 2

        # Cleanup zone bounds
        cz_cx = 0
        cz_cy = -cleanup_distance
        cz_xmin = cz_cx - cleanup_half
        cz_xmax = cz_cx + cleanup_half
        cz_ymin = cz_cy - cleanup_half
        cz_ymax = cz_cy + cleanup_half

        far_edge = max(field_size // 2, cleanup_distance + cleanup_size // 2)
        pad      = 400
        bounds   = (-(far_edge + pad), -(far_edge + pad),
                      far_edge + pad,    far_edge + pad)

        builder = ProceduralMapBuilder(bounds=bounds, seed=seed)

        decorate_str = """
ACTOR ForageBonus : CustomInventory 15040 {
  +AUTOACTIVATE
  Inventory.MaxAmount 0
  States {
  Spawn:
    BON1 ABCDCB 6
    Loop
  Pickup:
    TNT1 A 0 ACS_ExecuteAlways(997, 0)
    Stop
  }
}
"""
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode("utf-8"))

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("fc_spawn_rate",     1, "int")
        acs.add_global_var("fc_total_harvest",  2, "int")
        acs.add_global_var("fc_total_cleanups", 3, "int")

        for i, slot in enumerate(range(4, 8), start=1):
            acs.add_global_var(f"fc_p{i}_harvests", slot, "int")
        for i, slot in enumerate(range(8, 12), start=1):
            acs.add_global_var(f"fc_p{i}_cleanups", slot, "int")
        for i, slot in enumerate(range(12, 16), start=1):
            acs.add_global_var(f"fc_p{i}_cleanup_progress", slot, "int")

        # Harvest Field — large open area
        builder.add_area(
            "Harvest Field",
            shape=(0, 0, field_size, field_size),
            floor_height=0,
            ceiling_height=256,
            floor_texture="CEIL5_2",
            ceiling_texture="CEIL3_5",
            wall_texture="BROWN96",
            light_level=180,
        )

        # Cleanup Station — south of field
        builder.add_area(
            "Cleanup Station",
            shape=(cz_cx, cz_cy, cleanup_size, cleanup_size),
            floor_height=0,
            ceiling_height=256,
            floor_texture="NUKAGE1",
            ceiling_texture="TLITE6_4",
            wall_texture="PIPE4",
            light_level=255,
            tag=200,
        )

        # Corridor connecting field to cleanup; when the station overlaps the
        # field there is no gap for a corridor, so open the shared edge instead.
        if builder.are_adjacent("Harvest Field", "Cleanup Station"):
            builder.connect_adjacent("Harvest Field", "Cleanup Station")
        else:
            builder.add_corridor(
                "Harvest Field", "Cleanup Station",
                width=corridor_w,
                floor_texture="CEIL5_2",
                ceiling_height=200,
                wall_texture="BROWN96",
                light_level=160,
            )

        # Player starts (field center)
        start_positions = [
            (-48, 0),
            (48, 0),
            (-48, 48),
            (48, 48),
        ]
        start_types = [
            ThingType.PLAYER1_START,
            ThingType.PLAYER2_START,
            ThingType.PLAYER3_START,
            ThingType.PLAYER4_START,
        ]
        for i in range(num_agents):
            sx, sy = start_positions[i]
            builder.add_thing(start_types[i], sx, sy, angle=0)

        # MAP_SPOTs in Harvest Field (TIDs 3000+)
        spot_base = 3000
        placed_spots = 0
        for i in range(total_spots):
            pt = builder.get_random_point_in_area("Harvest Field", safe=False)
            if pt:
                builder.add_thing(
                    "MAP_SPOT", pt[0], pt[1],
                    tid=spot_base + i,
                )
                placed_spots += 1

        spot_start = 0
        spot_end = placed_spots - 1

        # Script 1 (ENTER): Player setup
        script_setup = """
        int pn;
        pn = PlayerNumber();
        Thing_ChangeTID(0, 1000 + pn);
        ClearInventory();
        SetActorProperty(0, APROP_Health, 100);
        ACS_ExecuteAlways(10, 0, 0, 0, 0);
        """
        acs.add_script(ScriptType.ENTER,   script_setup, number=1)
        acs.add_script(ScriptType.RESPAWN, script_setup, number=2)

        # Script 997 (VOID): Harvest callback — degrade spawn rate, give HP
        harvest_cases = []
        for i in range(num_agents):
            harvest_cases.append(f"if (pn == {i}) fc_p{i + 1}_harvests = fc_p{i + 1}_harvests + 1;")
        harvest_cases_str = "\n        ".join(harvest_cases)

        script_harvest = f"""
        int pn;
        pn = PlayerNumber();
        fc_spawn_rate = fc_spawn_rate - {harvest_deg};
        if (fc_spawn_rate < 5) fc_spawn_rate = 5;
        fc_total_harvest = fc_total_harvest + 1;
        {harvest_cases_str}
        int hp;
        hp = GetActorProperty(0, APROP_Health);
        hp = hp + 2;
        if (hp > 200) hp = 200;
        SetActorProperty(0, APROP_Health, hp);
        """
        acs.add_script(ScriptType.VOID, script_harvest, number=997)

        # Script 3 (OPEN): Main spawn loop — spawn rate mechanic
        spawn_player_checks = []
        for i in range(num_agents):
            spawn_player_checks.append(f"if (PlayerInGame({i})) agents = agents + 1;")
        spawn_player_checks_str = "\n            ".join(spawn_player_checks)

        counter_resets = []
        for i in range(1, 5):
            counter_resets.append(f"fc_p{i}_harvests = 0;")
            counter_resets.append(f"fc_p{i}_cleanups = 0;")
            counter_resets.append(f"fc_p{i}_cleanup_progress = 0;")
        counter_resets_str = "\n        ".join(counter_resets)

        script_spawn = f"""
        fc_spawn_rate = {initial_sr};
        fc_total_harvest = 0;
        fc_total_cleanups = 0;
        {counter_resets_str}
        int decay_tick;
        decay_tick = 0;

        Delay(35 * 3);

        int agents;
        int att;

        while (TRUE) {{
            agents = 0;
            {spawn_player_checks_str}

            for (att = 0; att < agents; att = att + 1) {{
                if (ThingCount(T_NONE, 800) < 48) {{
                    if (fc_spawn_rate > Random(0, 100)) {{
                        SpawnSpot("ForageBonus", {spot_base} + Random({spot_start}, {spot_end}), 800, 0);
                    }}
                }}
            }}

            decay_tick = decay_tick + 1;
            if (decay_tick >= 3) {{
                fc_spawn_rate = fc_spawn_rate - {decay_rate};
                if (fc_spawn_rate < 5) fc_spawn_rate = 5;
                decay_tick = 0;
            }}

            Delay(15);
        }}
        """
        acs.add_script(ScriptType.OPEN, script_spawn, number=3)

        # Script 4 (OPEN): Cleanup detection — single station
        cleanup_state_vars = []
        cleanup_ready_vars = []
        cleanup_blocks = []
        for i in range(num_agents):
            cleanup_var = f"c_p{i}"
            cleanup_ready_var = f"ready_p{i}"
            player_tid = 1000 + i
            player_num = i + 1
            cleanup_state_vars.append(f"int {cleanup_var};")
            cleanup_ready_vars.append(f"int {cleanup_ready_var};")
            cleanup_blocks.append(
                f"""
            if (PlayerInGame({i}) && ThingCount(T_NONE, {player_tid}) > 0) {{
                px = GetActorX({player_tid}) >> 16;
                py = GetActorY({player_tid}) >> 16;
                if (px >= {cz_xmin} && px <= {cz_xmax} && py >= {cz_ymin} && py <= {cz_ymax}) {{
                    if ({cleanup_ready_var}) {{
                        {cleanup_var} = {cleanup_var} + 4;
                        fc_p{player_num}_cleanup_progress = {cleanup_var};
                        if ({cleanup_var} >= {cleanup_time}) {{
                            if (fc_spawn_rate < 100) {{
                                fc_spawn_rate = fc_spawn_rate + {cleanup_boost};
                                if (fc_spawn_rate > 100) fc_spawn_rate = 100;
                                fc_total_cleanups = fc_total_cleanups + 1;
                                fc_p{player_num}_cleanups = fc_p{player_num}_cleanups + 1;
                            }}
                            {cleanup_var} = 0;
                            {cleanup_ready_var} = 0;
                            fc_p{player_num}_cleanup_progress = 0;
                        }}
                    }} else {{
                        fc_p{player_num}_cleanup_progress = 0;
                    }}
                }} else {{
                    {cleanup_var} = 0;
                    {cleanup_ready_var} = 1;
                    fc_p{player_num}_cleanup_progress = 0;
                }}
            }} else {{
                {cleanup_var} = 0;
                {cleanup_ready_var} = 1;
                fc_p{player_num}_cleanup_progress = 0;
            }}"""
            )

        cleanup_state_vars_str = "\n        ".join(cleanup_state_vars + cleanup_ready_vars)
        cleanup_state_init = []
        for i in range(num_agents):
            cleanup_state_init.append(f"c_p{i} = 0;")
            cleanup_state_init.append(f"ready_p{i} = 1;")
        cleanup_state_init_str = "\n        ".join(cleanup_state_init)
        cleanup_blocks_str = "\n".join(cleanup_blocks)

        script_cleanup = f"""
        {cleanup_state_vars_str}
        int px;
        int py;
        {cleanup_state_init_str}

        Delay(35);

        while (TRUE) {{
            // Require agents to leave and re-enter after a successful cleanup.
            {cleanup_blocks_str}
            Delay(4);
        }}
        """
        acs.add_script(ScriptType.OPEN, script_cleanup, number=4)

        # Script 5 (OPEN): Health drain loop
        drain_blocks = []
        for i in range(num_agents):
            player_tid = 1000 + i
            drain_blocks.append(
                f"""
            if (PlayerInGame({i}) && ThingCount(T_NONE, {player_tid}) > 0) {{
                hp = GetActorProperty({player_tid}, APROP_Health);
                hp = hp - {hp_drain};
                SetActorProperty({player_tid}, APROP_Health, hp);
                if (hp <= 0) {{
                    Delay(1);
                    Exit_Normal(0);
                }}
            }}"""
            )
        drain_blocks_str = "\n".join(drain_blocks)

        script_drain = f"""
        int hp;

        Delay(35 * 3);

        while (TRUE) {{
            {drain_blocks_str}
            Delay(35);
        }}
        """
        acs.add_script(ScriptType.OPEN, script_drain, number=5)

        # Script 10 (VOID): Per-player HUD loop
        bar_cases = []
        for i in range(21):
            filled = "=" * i
            empty  = "-" * (20 - i)
            bar_cases.append(
                f'case {i}: bar_gfx = "\\cD{filled}\\cG{empty}"; break;'
            )
        bar_switch_str = "\n                ".join(bar_cases)

        script_hud = f"""
        int sr;
        int idx;
        str bar_gfx;

        while (TRUE) {{
            SetHudSize(320, 240, 1);
            SetFont("SMALLFONT");

            sr = fc_spawn_rate;
            idx = sr / 5;
            if (idx < 0) idx = 0;
            if (idx > 20) idx = 20;
            bar_gfx = "";

            switch (idx) {{
                {bar_switch_str}
            }}

            // HudMessage(s:bar_gfx, s:" ", d:sr, s:"%"; HUDMSG_PLAIN, 1, CR_WHITE, 160.0, 4.0, 0.0);
            HudMessage(s:bar_gfx; HUDMSG_PLAIN, 1, CR_WHITE, 160.0, 4.0, 0.0);

            Delay(4);
        }}
        """
        acs.add_script(ScriptType.VOID, script_hud, number=10)

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
    path = os.path.join(out_dir, "foraging_commons.wad")
    print(f"Generating {path}...")
    ForagingCommonsScenario().generate(path)
    print("Done.")
