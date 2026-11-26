import math
import os
import random
from typing import Any, Dict

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class StagHuntArenaScenario(Scenario):

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "arena_size": 2048,
            "num_alcoves": 12,
            "alcove_width": 192,
            "alcove_depth": 128,
            "stag_health": 500,
            "stag_regen_rate": 4,
            "stag_range": 555,
            "stag_spawn_delay": 175,
            "stag_respawn_killed": 500,
            "stag_despawn_time": 2450,
            "stag_respawn_despawn": 200,
            "rabbit_spawn_interval": 20,
            "max_rabbits": 12,
            "rabbit_speed": 24,
            "rabbit_kill_radius": 555,
            "rabbit_corpse_cleanup_delay": 105,
            "rabbit_kill_reward": 3,
            "stag_kill_health_reward": 50,
            "stag_kill_ammo_reward": 20,
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)
        random.seed(seed)

        arena_size = cfg["arena_size"]
        arena_hw = arena_size // 2
        num_alcoves = cfg["num_alcoves"]
        stag_health = cfg["stag_health"]
        stag_regen_rate = cfg["stag_regen_rate"]
        stag_range = cfg["stag_range"]
        stag_spawn_delay = cfg["stag_spawn_delay"]
        stag_respawn_killed = cfg["stag_respawn_killed"]
        stag_despawn_time = cfg["stag_despawn_time"]
        stag_respawn_despawn = cfg["stag_respawn_despawn"]
        rabbit_spawn_interval = cfg["rabbit_spawn_interval"]
        max_rabbits = cfg["max_rabbits"]
        rabbit_speed = cfg["rabbit_speed"]
        rabbit_kill_radius = cfg["rabbit_kill_radius"]
        rabbit_corpse_cleanup_delay = cfg["rabbit_corpse_cleanup_delay"]
        rabbit_kill_reward = cfg["rabbit_kill_reward"]
        stag_kill_health_reward = cfg["stag_kill_health_reward"]
        stag_kill_ammo_reward = cfg["stag_kill_ammo_reward"]

        pad = 500
        bound = arena_hw + pad
        bounds = (-bound, -bound, bound, bound)

        builder = ProceduralMapBuilder(bounds=bounds, seed=seed)

        decorate_str = f"""
ACTOR Stag : BaronOfHell 15030 {{
  Health {stag_health}
  +BOSS
  DropItem "None"
  States {{
  Death:
    BOSS H 0 ACS_ExecuteAlways(996, 0)
    Goto Super::Death
  }}
}}

ACTOR Rabbit : Zombieman 15031 {{
  Health 1
  Speed {rabbit_speed}
  +FRIGHTENED
  DropItem "None"
  States {{
  Spawn:
    POSS AB 4 A_Look
    POSS A 0 ACS_ExecuteAlways(995, 0)
    Loop
  See:
    POSS A 0 ACS_ExecuteAlways(995, 0)
    POSS AABBCCDD 2 A_Wander
    Loop
  Missile:
    Goto See
  Death:
    POSS H 0 ACS_ExecuteAlways(997, 0)
    POSS H 0 ACS_ExecuteAlways(998, 0)
    POSS I 3 A_Scream
    POSS J 3 A_NoBlocking
    POSS K 35
    Wait
  }}
}}
"""
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode("utf-8"))

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("sh_stag_health",     51, "int")
        acs.add_global_var("sh_stag_vulnerable", 52, "int")
        acs.add_global_var("sh_rabbit_kills",    53, "int")
        acs.add_global_var("sh_stag_kills",      54, "int")
        acs.add_global_var("sh_stag_alive",      55, "int")
        acs.add_global_var("sh_p1_rabbit_kills", 56, "int")
        acs.add_global_var("sh_p2_rabbit_kills", 57, "int")


        builder.add_area(
            "Arena",
            shape=(0, 0, arena_size, arena_size),
            floor_height=0,
            ceiling_height=384,
            floor_texture="FLOOR4_8",
            ceiling_texture="CEIL3_5",
            wall_texture="STONE2",
            light_level=200,
        )


        # Player starts (center-south of arena)
        builder.add_thing(ThingType.PLAYER1_START, -64, -arena_hw + 128, angle=90)
        builder.add_thing(ThingType.PLAYER2_START,  64, -arena_hw + 128, angle=90)
        builder.add_thing(ThingType.PLAYER3_START, -64, -arena_hw + 192, angle=90)
        builder.add_thing(ThingType.PLAYER4_START,  64, -arena_hw + 192, angle=90)

        # Stag spawn spot at center (TID 5000)
        builder.add_thing("MAP_SPOT", 0, 0, tid=5000)

        # Rabbit spawn spots at perimeter positions (TIDs 6000-6011)
        alcove_r = int(arena_hw * 0.85)
        for i in range(num_alcoves):
            angle = (2 * math.pi * i) / num_alcoves
            rx = int(math.cos(angle) * alcove_r)
            ry = int(math.sin(angle) * alcove_r)
            builder.add_thing("MAP_SPOT", rx, ry, tid=6000 + i)

        # Interior rabbit spawn spots at 50% radius (TIDs 6100+)
        interior_r = int(arena_hw * 0.50)
        interior_count = 8
        for i in range(interior_count):
            angle = (2 * math.pi * i) / interior_count + math.pi / interior_count  # offset angle
            rx = int(math.cos(angle) * interior_r)
            ry = int(math.sin(angle) * interior_r)
            builder.add_thing("MAP_SPOT", rx, ry, tid=6100 + i)

        # Script 1 (ENTER) + Script 2 (RESPAWN): Player setup
        script_setup = """
int pn = PlayerNumber();
Thing_ChangeTID(0, 1000 + pn);
SetActorProperty(0, APROP_Health, 100);
ClearInventory();
SetAmmoCapacity("Shell", 9999);
GiveInventory("Shotgun", 1);
GiveInventory("Shell", 9999);
SetWeapon("Shotgun");
ACS_ExecuteAlways(10, 0, 0, 0, 0);
"""
        acs.add_script(ScriptType.ENTER,   script_setup, number=1)
        acs.add_script(ScriptType.RESPAWN, script_setup, number=2)

        # Script 3 (OPEN): Main stag logic loop
        range_sq = stag_range * stag_range
        script_stag = f"""
int stag_alive = 0;
int stag_timer = 0;
int respawn_timer = {stag_spawn_delay};
    int TID_TEMP_SPAWN = 9100;
    int spawn_success = 0;

sh_stag_health = 0;
sh_stag_vulnerable = 0;
sh_rabbit_kills = 0;
sh_stag_kills = 0;
sh_stag_alive = 0;
sh_p1_rabbit_kills = 0;
sh_p2_rabbit_kills = 0;

while (TRUE) {{
    if (stag_alive) {{
        if (ThingCount(T_NONE, 3000) == 0) {{
            stag_alive = 0;
            respawn_timer = {stag_respawn_killed};
            sh_stag_alive = 0;
            sh_stag_health = 0;
            sh_stag_vulnerable = 0;
        }} else {{
            stag_timer = stag_timer + 4;
            if (stag_timer > {stag_despawn_time}) {{
                Thing_Remove(3000);
                stag_alive = 0;
                respawn_timer = {stag_respawn_despawn};
                sh_stag_alive = 0;
                sh_stag_health = 0;
                sh_stag_vulnerable = 0;
            }} else {{
                int sx = GetActorX(3000) >> 16;
                int sy = GetActorY(3000) >> 16;
                int p1_alive = ThingCount(T_NONE, 1000) > 0;
                int p2_alive = ThingCount(T_NONE, 1001) > 0;
                int d1sq = 2147483647;
                int d2sq = 2147483647;
                if (p1_alive) {{
                    int p1x = GetActorX(1000) >> 16;
                    int p1y = GetActorY(1000) >> 16;
                    int d1x = sx - p1x;
                    int d1y = sy - p1y;
                    d1sq = d1x * d1x + d1y * d1y;
                }}
                if (p2_alive) {{
                    int p2x = GetActorX(1001) >> 16;
                    int p2y = GetActorY(1001) >> 16;
                    int d2x = sx - p2x;
                    int d2y = sy - p2y;
                    d2sq = d2x * d2x + d2y * d2y;
                }}
                if (p1_alive && p2_alive && d1sq < {range_sq} && d2sq < {range_sq}) {{
                    SetActorProperty(3000, APROP_Invulnerable, 0);
                    sh_stag_vulnerable = 1;
                }} else {{
                    SetActorProperty(3000, APROP_Invulnerable, 1);
                    int cur_hp = GetActorProperty(3000, APROP_Health);
                    if (cur_hp < {stag_health}) {{
                        SetActorProperty(3000, APROP_Health, cur_hp + {stag_regen_rate});
                    }}
                    sh_stag_vulnerable = 0;
                }}
                sh_stag_health = GetActorProperty(3000, APROP_Health);
            }}
        }}
    }} else {{
        respawn_timer = respawn_timer - 4;
        if (respawn_timer <= 0) {{
            // Use a temp TID + success guard to avoid stale references/crashes
            // on failed spawns (blocked location, invalid spot TID).
            Thing_Remove(TID_TEMP_SPAWN);
            spawn_success = SpawnSpot("Stag", 5000, TID_TEMP_SPAWN, 0);
            if (spawn_success && ThingCount(T_NONE, TID_TEMP_SPAWN) > 0) {{
                Thing_ChangeTID(TID_TEMP_SPAWN, 3000);
                SetActorProperty(3000, APROP_Invulnerable, 1);
                stag_alive = 1;
                stag_timer = 0;
                sh_stag_alive = 1;
                sh_stag_health = {stag_health};
                sh_stag_vulnerable = 0;
            }} else {{
                // Back off briefly to avoid rapid spawn spam.
                respawn_timer = 35;
            }}
        }}
    }}
    Delay(4);
}}
"""
        acs.add_script(ScriptType.OPEN, script_stag, number=3)

        # Script 5 (OPEN): Rabbit spawner
        script_rabbits = f"""
int TID_RABBITS = 4000;
int TID_TEMP_SPAWN = 9101;
int spawn_success = 0;

while (TRUE) {{
    if (ThingCount(T_NONE, TID_RABBITS) < {max_rabbits}) {{
        int ring;
        int spot;
        ring = Random(0, 1);
        if (ring == 0) {{
            spot = 6000 + Random(0, {num_alcoves - 1});
        }} else {{
            spot = 6100 + Random(0, {interior_count - 1});
        }}
        // Use a temp TID + success guard to avoid stale references/crashes.
        Thing_Remove(TID_TEMP_SPAWN);
        spawn_success = SpawnSpotFacing("Rabbit", spot, TID_TEMP_SPAWN);
        if (spawn_success && ThingCount(T_NONE, TID_TEMP_SPAWN) > 0) {{
            Thing_ChangeTID(TID_TEMP_SPAWN, TID_RABBITS);
        }}
    }}
    Delay({rabbit_spawn_interval});
}}
"""
        acs.add_script(ScriptType.OPEN, script_rabbits, number=5)

        rabbit_kill_radius_sq = rabbit_kill_radius * rabbit_kill_radius
        script_rabbit_vulnerability = f"""
int rx = GetActorX(0) >> 16;
int ry = GetActorY(0) >> 16;
int p1_alive = ThingCount(T_NONE, 1000) > 0;
int p2_alive = ThingCount(T_NONE, 1001) > 0;
int d1sq = 2147483647;
int d2sq = 2147483647;
if (p1_alive) {{
    int p1x = GetActorX(1000) >> 16;
    int p1y = GetActorY(1000) >> 16;
    int d1x = rx - p1x;
    int d1y = ry - p1y;
    d1sq = d1x * d1x + d1y * d1y;
}}
if (p2_alive) {{
    int p2x = GetActorX(1001) >> 16;
    int p2y = GetActorY(1001) >> 16;
    int d2x = rx - p2x;
    int d2y = ry - p2y;
    d2sq = d2x * d2x + d2y * d2y;
}}
if ((p1_alive && d1sq <= {rabbit_kill_radius_sq}) || (p2_alive && d2sq <= {rabbit_kill_radius_sq})) {{
    SetActorProperty(0, APROP_Invulnerable, 0);
}} else {{
    SetActorProperty(0, APROP_Invulnerable, 1);
}}
"""
        acs.add_script(ScriptType.VOID, script_rabbit_vulnerability, number=995)

        # Script 996 (VOID): Stag kill reward — both players get health+shells
        script_stag_kill = f"""
sh_stag_kills = sh_stag_kills + 1;
int p;
for (p = 0; p < 2; p = p + 1) {{
    int player_tid = 1000 + p;
    if (PlayerInGame(p) && ThingCount(T_NONE, player_tid) > 0) {{
        GiveActorInventory(player_tid, "HealthBonus", {stag_kill_health_reward});
        GiveActorInventory(player_tid, "Shell", {stag_kill_ammo_reward});
    }}
}}
"""
        acs.add_script(ScriptType.VOID, script_stag_kill, number=996)

        # Script 997 (VOID): Rabbit kill reward
        script_rabbit_kill = f"""
SetActivatorToTarget(0);
int pn = PlayerNumber();
if (pn >= 0) {{
    GiveInventory("HealthBonus", {rabbit_kill_reward});
    sh_rabbit_kills = sh_rabbit_kills + 1;
    if (pn == 0) {{
        sh_p1_rabbit_kills = sh_p1_rabbit_kills + 1;
    }} else if (pn == 1) {{
        sh_p2_rabbit_kills = sh_p2_rabbit_kills + 1;
    }}
}}
"""
        acs.add_script(ScriptType.VOID, script_rabbit_kill, number=997)

        # Script 998 (VOID): Corpse cleanup
        acs.add_script(ScriptType.VOID, f"""
int cleanup_tid = UniqueTID(9200, 0);
if (cleanup_tid > 0) {{
    Thing_ChangeTID(0, cleanup_tid);
    Delay({rabbit_corpse_cleanup_delay});
    if (ThingCount(T_NONE, cleanup_tid) > 0) {{
        Thing_Remove(cleanup_tid);
    }}
}}
""", number=998)

        # Script 10 (VOID): Per-player HUD loop
        bar_cases = []
        for i in range(21):
            filled = "=" * i
            empty = "-" * (20 - i)
            bar_cases.append(
                f'case {i}: bar_gfx = "\\cD{filled}\\cG{empty}"; break;'
            )
        bar_switch_str = "\n            ".join(bar_cases)

        script_hud = f"""
while (TRUE) {{
    SetFont("SMALLFONT");
    int hp = sh_stag_health;
    int alive = sh_stag_alive;
    int vuln = sh_stag_vulnerable;
    int sk = sh_stag_kills;
    int rk = sh_rabbit_kills;

    if (alive) {{
        int hp_idx = (hp * 20) / {stag_health};
        if (hp_idx < 0) hp_idx = 0;
        if (hp_idx > 20) hp_idx = 20;
        str bar_gfx = "";
        switch (hp_idx) {{
            {bar_switch_str}
        }}
        HudMessage(s:"STAG ", s:bar_gfx;
            HUDMSG_PLAIN, 1, CR_WHITE, 0.5, 1.0, 2.0);

        if (vuln) {{
            HudMessage(s:"STAG: \\cDVULNERABLE";
                HUDMSG_PLAIN, 2, CR_GREEN, 0.5, 0.97, 2.0);
        }} else {{
            HudMessage(s:"STAG: \\cGSHIELDED";
                HUDMSG_PLAIN, 2, CR_RED, 0.5, 0.97, 2.0);
        }}
    }} else {{
        HudMessage(s:"STAG: \\cGNOT SPAWNED";
            HUDMSG_PLAIN, 1, CR_GREY, 0.5, 0.99, 2.0);
    }}

    Delay(1);
}}
"""
        acs.add_script(ScriptType.VOID, script_hud, number=10)

        try:
            builder.map_data.behavior = acs.compile()
        except Exception as e:
            print(f"ACS compile warning: {e}")
            if hasattr(e, "stderr") and e.stderr:
                print(f"ACC stderr: {e.stderr}")

        builder.map_data.scripts = acs.to_code()
        builder.build(output_path)


if __name__ == "__main__":
    out_dir = os.path.join("examples", "benchmark", "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stag_hunt_arena.wad")
    print(f"Generating {path}...")
    StagHuntArenaScenario().generate(path)
    print("Done.")
