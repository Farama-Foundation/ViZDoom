import random
from typing import Dict, Any
from doomgen.builder import ProceduralMapBuilder
from doomgen.abstraction.layout import ConnectionType
from doomgen.logic.acs_builder import ACSBuilder, ScriptType
from doomgen.batch.scenario import Scenario

class ArmorySiegeScenario(Scenario):
    """
    Parameterized version of Armory Siege Defense Benchmark.
    """

    def get_default_config(self) -> Dict[str, Any]:
        return {
            'distance': 600,             # Radius of supply rooms from center (Closer = Easier)
            'corridor_width': 192,       # Width of connecting corridors in map units (Wider = Easier)
            'door_timer': 300,           # Door open duration (tics) (Longer = Easier access)
            'core_health': 1000,         # Health of the defense core (Higher = Easier)
            'enemy_difficulty': 0.5,     # Multiplier for enemy counts (Lower = Easier)
            'max_players': 8,          # Number of cooperative player starts to emit
            'seed': 42,                # Random seed
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        seed = cfg['seed'] if cfg['seed'] is not None else random.randint(0, 999999)

        # Determine bounds based on distance
        dist = int(cfg['distance']) # Ensure int
        bound = dist + 1000 # Enough padding

        max_players = int(cfg.get('max_players', 8))
        if max_players < 1 or max_players > 8:
            raise ValueError(f"max_players must be in range [1, 8], got {max_players}")

        # 1. Initialize Builder
        builder = ProceduralMapBuilder(
            bounds=(-bound, -bound, bound, bound),
            seed=seed
        )

        # 0. Add DECORATE
        decorate_str = f"""
ACTOR HarmlessTargeter
{{
  Radius 3
  Height 1
  Speed 100
  Damage 1
  DamageType "Taunt"
  Projectile
  +BLOODLESSIMPACT
  +FORCEPAIN
  States
  {{
  Spawn:
    TNT1 A 1
    Loop
  Death:
    TNT1 A 1
    Stop
  }}
}}

ACTOR DefenseCore 15000
{{
  //$Category "Objectives"
  Health {int(cfg['core_health'])}
  Mass 100000
  Radius 16
  Height 50
  Speed 0
  +SOLID
  +SHOOTABLE
  +DONTTHRUST
  +NOBLOOD
  +FRIENDLY
  +ISMONSTER
  +LOOKALLAROUND
  +MISSILEMORE
  +MISSILEEVENMORE
  DamageFactor "Player", 0

  States
  {{
  Spawn:
    TLMP ABCD 4
    TLMP A 0 A_Look
    Loop
  See:
    TLMP A 1 A_Chase
    Loop
  Missile:
    TLMP A 0 A_CustomMissile("HarmlessTargeter", 32, 0, 0, 0, 0)
    TLMP A 2
    Goto See
  Death:
    TLMP A 1
    TLMP A 0 ACS_Execute(999, 0)
    Stop
  }}
}}

ACTOR PlaceablePistol : Pistol 15001
{{
  //$Category "Weapons"
  Weapon.SlotNumber 1
}}

ACTOR Slot2SSG : SuperShotgun 15005
{{
  //$Category "Weapons"
  Weapon.SlotNumber 2
}}

ACTOR MeleeRevenant : Revenant 15002
{{
  DamageFactor "Taunt", 0
  States
  {{
  Missile:
    SKEL A 0
    Goto See
  }}
}}

ACTOR MeleeZombieman : Zombieman 15003
{{
  DamageFactor "Taunt", 0
  States
  {{
  Missile:
    POSS A 0
    Goto See
  Melee:
    POSS E 10 A_FaceTarget
    POSS F 8 A_CustomMeleeAttack(10)
    POSS E 8
    Goto See
  Death:
    POSS H 0 ACS_ExecuteAlways(998, 0)
    Goto Super::Death
  }}
}}

ACTOR MeleeImp : DoomImp 15004
{{
  DamageFactor "Taunt", 0
  States
  {{
  Missile:
    TROO A 0
    Goto See
  Melee:
    TROO EF 8 A_FaceTarget
    TROO G 6 A_CustomMeleeAttack(3 * random(1, 8), "imp/melee")
    Goto See
  Death:
    TROO I 0 ACS_ExecuteAlways(998, 0)
    Goto Super::Death
  }}
}}

ACTOR SiegeDemon : Demon 15006
{{
  DamageFactor "Taunt", 0
  States
  {{
  Death:
    SARG I 0 ACS_ExecuteAlways(998, 0)
    Goto Super::Death
  }}
}}

ACTOR SiegeLostSoul : LostSoul 15007
{{
  DamageFactor "Taunt", 0
  States
  {{
  Death:
    SKUL F 0 ACS_ExecuteAlways(998, 0)
    Goto Super::Death
  }}
}}
"""
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode('utf-8'))

        # 2. Define Areas

        # Central Armory
        builder.add_area(
            name="Armory",
            shape=(0, 0, 512, 512),
            floor_height=0,
            ceiling_height=192,
            floor_texture="FLOOR4_8",
            ceiling_texture="TLITE6_4",
            wall_texture="STARTAN3"
        )

        # Calculate positions
        # Room A (North) - 90 deg
        pos_a = (0, dist)
        # Room B (South East) - 330 deg (-30)
        # x = r * cos(-30) = r * 0.866
        # y = r * sin(-30) = r * -0.5
        pos_b = (int(dist * 0.866), int(dist * -0.5))
        # Room C (South West) - 210 deg
        pos_c = (int(dist * -0.866), int(dist * -0.5))

        # Supply Room A (Heavy Weapons)
        builder.add_area(
            name="Room A",
            shape=(pos_a[0], pos_a[1], 384, 384),
            floor_height=0,
            ceiling_height=160,
            floor_texture="CEIL5_2",
            wall_texture="STONE2"
        )

        # Supply Room B (Ammo)
        builder.add_area(
            name="Room B",
            shape=(pos_b[0], pos_b[1], 384, 384),
            floor_height=0,
            ceiling_height=160,
            floor_texture="FLAT5_4",
            wall_texture="BROWN1"
        )

        # Supply Room C (Health/Armor)
        builder.add_area(
            name="Room C",
            shape=(pos_c[0], pos_c[1], 384, 384),
            floor_height=0,
            ceiling_height=160,
            floor_texture="MFLR8_1",
            wall_texture="STARGR1"
        )

        # 3. Connect via Corridors
        cw = int(cfg['corridor_width'])
        corr_a = builder.add_corridor("Armory", "Room A", width=cw, floor_texture="RROCK19")
        corr_b = builder.add_corridor("Armory", "Room B", width=cw, floor_texture="RROCK19")
        corr_c = builder.add_corridor("Armory", "Room C", width=cw, floor_texture="RROCK19")

        # 4. Add Doors
        TAG_DOOR_A = 10
        TAG_DOOR_B = 11
        TAG_DOOR_C = 12

        # Full corridor-width door rectangle so the door is the only passage.
        door_a_id = builder.add_boundary(
            "Room A", corr_a,
            name="Door A", connection_type=ConnectionType.DOOR,
            floor_height=0, ceiling_height=0,
            floor_texture="GATE1", ceiling_texture="GATE1", wall_texture="BIGDOOR2",
            tag=TAG_DOOR_A, door_width=cw, door_depth=16
        )

        door_b_id = builder.add_boundary(
            "Room B", corr_b,
            name="Door B", connection_type=ConnectionType.DOOR,
            floor_height=0, ceiling_height=0,
            floor_texture="GATE1", ceiling_texture="GATE1", wall_texture="BIGDOOR2",
            tag=TAG_DOOR_B, door_width=cw, door_depth=16
        )

        door_c_id = builder.add_boundary(
            "Room C", corr_c,
            name="Door C", connection_type=ConnectionType.DOOR,
            floor_height=0, ceiling_height=0,
            floor_texture="GATE1", ceiling_texture="GATE1", wall_texture="BIGDOOR2",
            tag=TAG_DOOR_C, door_width=cw, door_depth=16
        )

        # 5. Add Things (Spawners)
        # Players. ZDoom supports dedicated cooperative start thing types for
        # players 5-8 (4001-4004), so emit the full set when requested.
        player_starts = [
            ("PLAYER1_START", -128, -128, 45),
            ("PLAYER2_START", 0, -128, 90),
            ("PLAYER3_START", 128, -128, 135),
            ("PLAYER4_START", 128, 0, 180),
            ("PLAYER5_START", 128, 128, 225),
            ("PLAYER6_START", 0, 128, 270),
            ("PLAYER7_START", -128, 128, 315),
            ("PLAYER8_START", -128, 0, 0),
        ]
        for thing_type, x, y, angle in player_starts[:max_players]:
            builder.add_thing(thing_type, x, y, angle=angle)

        def place_spawn_cluster(area_name, base_tid, count=5):
            for i in range(count):
                pt = builder.get_random_point_in_area(area_name)
                if pt:
                    builder.add_thing("MAP_SPOT", pt[0], pt[1], tid=base_tid + i)

        place_spawn_cluster("Armory", 50, count=5) # Core spots - Critical
        place_spawn_cluster("Armory", 60, count=10) # Player respawn spots - Critical
        place_spawn_cluster("Room A", 100, count=5)
        place_spawn_cluster("Room B", 110, count=5)
        place_spawn_cluster("Room C", 120, count=5)
        place_spawn_cluster(corr_a, 200, count=5) # Enemy spawns A
        place_spawn_cluster(corr_b, 210, count=5)
        place_spawn_cluster(corr_c, 220, count=5)

        # Corridor Items
        place_spawn_cluster(corr_a, 300, count=5)
        place_spawn_cluster(corr_a, 310, count=5)
        place_spawn_cluster(corr_b, 320, count=5)
        place_spawn_cluster(corr_b, 330, count=5)
        place_spawn_cluster(corr_c, 340, count=5)
        place_spawn_cluster(corr_c, 350, count=5)

        # 6. Generate ACS
        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("core_health_global", 1, "int")
        acs.add_global_var("core_x_global", 2, "int")
        acs.add_global_var("core_y_global", 3, "int")

        acs.add_script(ScriptType.ENTER, """
            int p_tid = 1000 + PlayerNumber();
            Thing_ChangeTID(0, p_tid);
            SetActorProperty(0, APROP_Health, 100);
            ClearInventory();
            int dest_spot = 60 + Random(0, 9);
            SetActorPosition(0, GetActorX(dest_spot), GetActorY(dest_spot), GetActorZ(dest_spot), 0);
        """)

        acs.add_script(ScriptType.RESPAWN, """
            int p_tid = 1000 + PlayerNumber();
            Thing_ChangeTID(0, p_tid);
            SetActorProperty(0, APROP_Health, 100);
            ClearInventory();
            int dest_spot = 60 + Random(0, 9);
            SetActorPosition(0, GetActorX(dest_spot), GetActorY(dest_spot), GetActorZ(dest_spot), 0);
        """)

        # Dynamic Script Variables
        open_dur = int(cfg['door_timer'])
        # Door schedule: 3 doors open in a staggered rotating pattern across
        # a repeating cycle. The cycle length is derived from the door open
        # duration so that the schedule scales with the config parameter.
        # Window offsets are spaced at cycle/6 increments, preserving the
        # overlap pattern of the original hardcoded schedule (1200 tics with
        # 200-tic spacing when open_dur = 300).
        cycle_len = open_dur * 4
        # Stagger between different doors: cycle / 6.
        stagger = cycle_len // 6
        # Repeat gap between windows of the same door: 5 * cycle / 12.
        repeat_gap = 5 * cycle_len // 12
        # Pre-compute door window offsets for injection into ACS.
        # Door A: 3 windows (dominant/armory door), B and C: 2 windows each.
        # With defaults (open_dur=300, cycle_len=1200, stagger=200, repeat=500)
        # this reproduces: A@0,500,1000  B@200,700  C@400,900.
        off_a1 = 0
        off_a2 = repeat_gap                # 500 with defaults
        off_a3 = repeat_gap * 2            # 1000 with defaults
        off_b1 = stagger                   # 200 with defaults
        off_b2 = stagger + repeat_gap      # 700 with defaults
        off_c1 = stagger * 2               # 400 with defaults
        off_c2 = stagger * 2 + repeat_gap  # 900 with defaults

        # Calculate health multiplier
        diff = float(cfg['enemy_difficulty'])
        health_mult = diff if diff < 1.0 else 1.0
        health_mult_fixed = int(health_mult * 65536)

        # Pre-compute enemy difficulty as integer percentage for ACS.
        # ACS uses 16.16 fixed-point: a float literal like 0.5 compiles to
        # 32768, breaking integer arithmetic. Using integer division avoids
        # the fixed-point trap entirely.
        enemy_diff_pct = max(5, int(diff * 100))

        # Generate ACS Health Bar Switch
        bar_cases = []
        for i in range(21):
             filled = "=" * i
             empty = "-" * (20 - i)
             bar_cases.append(f'case {i}: bar_gfx = "\\cD{filled}\\cG{empty}"; break;')
        bar_switch_str = "\n                ".join(bar_cases)

        script_body = f"""
        int max_core_health = {int(cfg['core_health'])};
        core_health_global = max_core_health;
        int tick = 0;
        int cycle = 0;
        int wave = 0;
        int TID_ITEM_A = 401;
        int TID_ITEM_B = 402;
        int TID_ITEM_C = 403;
        int TID_ENEMY_UNIT = 500;
        int TID_CORE = 90;
        int TID_TEMP_SPAWN = 900;

        // retry spawn DefenseCore if fail
        int core_spawned = 0;
        int core_spawn_attempts = 0;
        while (!core_spawned && core_spawn_attempts < 5) {{
            int spot_idx = 50 + core_spawn_attempts;  // Try different spawn spots
            Thing_Remove(TID_CORE);  // Clear any stale TID
            int spawn_result = SpawnSpot("DefenseCore", spot_idx, TID_CORE, 0);
            if (spawn_result && ThingCount(T_NONE, TID_CORE) > 0) {{
                core_spawned = 1;
            }}
            core_spawn_attempts++;
        }}
        if (core_spawned) {{
            core_x_global = GetActorX(TID_CORE) >> 16;
            core_y_global = GetActorY(TID_CORE) >> 16;
        }} else {{
            // Fallback: use map center if spawn fails
            core_x_global = 0;
            core_y_global = 0;
        }}

        // Initial Items (Scaled maybe? For now static)
        SpawnSpot("PlaceablePistol", 300 + Random(0, 4), 0, 0);
        SpawnSpot("Clip", 310 + Random(0, 4), 0, 0);

        SpawnSpot("PlaceablePistol", 320 + Random(0, 4), 0, 0);
        SpawnSpot("Clip", 330 + Random(0, 4), 0, 0);

        SpawnSpot("PlaceablePistol", 340 + Random(0, 4), 0, 0);
        SpawnSpot("Clip", 350 + Random(0, 4), 0, 0);

        Sector_SetColor({TAG_DOOR_A}, 255, 0, 0);
        Sector_SetColor({TAG_DOOR_B}, 255, 0, 0);
        Sector_SetColor({TAG_DOOR_C}, 255, 0, 0);

        int door_a_open = 0;
        int door_b_open = 0;
        int door_c_open = 0;

        while (TRUE) {{
            cycle = tick % {cycle_len};
            wave = tick / ({cycle_len} - 50); // Difficulty scaling

            // Check against core being destroyed/missing
            if (ThingCount(T_NONE, TID_CORE) > 0) {{
                core_health_global = GetActorProperty(TID_CORE, APROP_HEALTH);
            }} else {{
                core_health_global = 0;  // Core destroyed or spawn failed
            }}

            // Health Bar
            int hp_idx = (core_health_global * 20) / max_core_health;
            if (hp_idx < 0) hp_idx = 0;
            if (hp_idx > 20) hp_idx = 20;
            str bar_gfx = "";
            switch(hp_idx) {{
                {bar_switch_str}
            }}
            SetFont("SMALLFONT");
            HudMessage(s:"", s:bar_gfx, s:"\\c-"; HUDMSG_PLAIN, 1, CR_WHITE, 0.5, 0.9, 0.1);
            SetFont("SMALLFONT");
            HudMessage(d:core_health_global, s:"/", d:max_core_health; HUDMSG_PLAIN, 2, CR_WHITE, 0.5, 0.93, 0.1);

            // Door Logic with parameterized duration
            // Durations are relative to open_dur.

            int dur = {open_dur};
            int should_open_a = (cycle >= {off_a1} && cycle < ({off_a1}+dur)) || (cycle >= {off_a2} && cycle < ({off_a2}+dur)) || (cycle >= {off_a3} && cycle < ({off_a3}+dur));
            int should_open_b = (cycle >= {off_b1} && cycle < ({off_b1}+dur)) || (cycle >= {off_b2} && cycle < ({off_b2}+dur));
            int should_open_c = (cycle >= {off_c1} && cycle < ({off_c1}+dur)) || (cycle >= {off_c2} && cycle < ({off_c2}+dur));

            if (should_open_a) {{
                if (!door_a_open) {{
                    Door_Open({TAG_DOOR_A}, 16, 0);
                    Sector_SetColor({TAG_DOOR_A}, 0, 255, 0);
                    door_a_open = 1;
                    if (ThingCount(T_NONE, TID_ITEM_A) == 0) {{
                        SpawnSpot("Slot2SSG", 100 + Random(0, 4), TID_ITEM_A, 0);
                    }}
                }}
                if (dur > 300 && tick % 300 == 0) {{
                    if (ThingCount(T_NONE, TID_ITEM_A) == 0) {{
                        SpawnSpot("Slot2SSG", 100 + Random(0, 4), TID_ITEM_A, 0);
                    }}
                }}
            }} else {{
                if (door_a_open) {{
                    Door_Close({TAG_DOOR_A}, 16, 0);
                    Sector_SetColor({TAG_DOOR_A}, 255, 0, 0);
                    door_a_open = 0;
                }}
            }}

             if (should_open_b) {{
                if (!door_b_open) {{
                    Door_Open({TAG_DOOR_B}, 16, 0);
                    Sector_SetColor({TAG_DOOR_B}, 0, 255, 0);
                    door_b_open = 1;
                    if (ThingCount(T_NONE, TID_ITEM_B) == 0) {{
                        SpawnSpot("ShellBox", 110 + Random(0, 4), TID_ITEM_B, 0);
                    }}
                }}
                if (dur > 300 && tick % 300 == 0) {{
                    if (ThingCount(T_NONE, TID_ITEM_B) == 0) {{
                        SpawnSpot("ShellBox", 110 + Random(0, 4), TID_ITEM_B, 0);
                    }}
                }}
            }} else {{
                if (door_b_open) {{
                    Door_Close({TAG_DOOR_B}, 16, 0);
                    Sector_SetColor({TAG_DOOR_B}, 255, 0, 0);
                    door_b_open = 0;
                }}
            }}

             if (should_open_c) {{
                if (!door_c_open) {{
                    Door_Open({TAG_DOOR_C}, 16, 0);
                    Sector_SetColor({TAG_DOOR_C}, 0, 255, 0);
                    door_c_open = 1;
                    if (ThingCount(T_NONE, TID_ITEM_C) == 0) {{
                        SpawnSpot("Medikit", 120 + Random(0, 4), TID_ITEM_C, 0);
                    }}
                }}
                if (dur > 300 && tick % 300 == 0) {{
                     if (ThingCount(T_NONE, TID_ITEM_C) == 0) {{
                        SpawnSpot("Medikit", 120 + Random(0, 4), TID_ITEM_C, 0);
                     }}
                }}
            }} else {{
                if (door_c_open) {{
                    Door_Close({TAG_DOOR_C}, 16, 0);
                    Sector_SetColor({TAG_DOOR_C}, 255, 0, 0);
                    door_c_open = 0;
                }}
            }}

            // Wave System
            int max_enemies = (20 + wave * 5) * {enemy_diff_pct} / 100;
            int spawn_interval = 200 - (wave * 20);
            if (spawn_interval < 40) spawn_interval = 40;

            if (tick % spawn_interval == 0 && tick > 0 && ThingCount(T_NONE, TID_ENEMY_UNIT) < max_enemies) {{
                int r = Random(0, 10);
                int spawn_base = 200;
                int r_spawn = Random(0, 2);
                if (r_spawn == 1) spawn_base = 210;
                if (r_spawn == 2) spawn_base = 220;

                int spawn_tid = spawn_base + Random(0, 4);
                str spawn_class = "MeleeZombieman";

                if (wave == 0) {{
                    spawn_class = "MeleeZombieman";
                }} else if (wave == 1) {{
                    if (r < 6) spawn_class = "MeleeZombieman";
                    else spawn_class = "MeleeImp";
                }} else if (wave < 3) {{
                    if (r < 4) spawn_class = "MeleeImp";
                    else spawn_class = "SiegeDemon";
                }} else {{
                    if (r < 5) spawn_class = "SiegeDemon";
                    else spawn_class = "SiegeLostSoul";
                }}

                // Check spawn failures (blocked location, invalid spot TID)
                // Clear any stale actors with this TID to prevent hash chain corruption
                Thing_Remove(TID_TEMP_SPAWN);
                int spawn_success = SpawnSpot(spawn_class, spawn_tid, TID_TEMP_SPAWN, 0);
                if (spawn_success && ThingCount(T_NONE, TID_TEMP_SPAWN) > 0) {{
                    int cur_h = GetActorProperty(TID_TEMP_SPAWN, APROP_HEALTH);
                    SetActorProperty(TID_TEMP_SPAWN, APROP_HEALTH, FixedMul(cur_h, {health_mult_fixed}));
                    Thing_Hate(TID_TEMP_SPAWN, TID_CORE, 0);
                    Thing_ChangeTID(TID_TEMP_SPAWN, TID_ENEMY_UNIT);
                }}
            }}

            Delay(1);
            tick++;
        }}
        """
        acs.add_script(ScriptType.OPEN, script_body)

        acs.add_script(ScriptType.VOID, "Delay(35*3); Thing_Remove(0);", number=998)
        acs.add_script(ScriptType.VOID, "Delay(35*3); Exit_Normal(0);", number=999)

        # Build logic
        try:
            print(f"Compiling ACS for {self.name}...")
            bytecode = acs.compile()
            if bytecode:
                builder.map_data.behavior = bytecode
            else:
                builder.map_data.behavior = b'ACSE\x08\x00\x00\x00\x00\x00\x00\x00'
        except Exception as e:
            print(f"ACS compile error: {e}")
            builder.map_data.behavior = b'ACSE\x08\x00\x00\x00\x00\x00\x00\x00'

        builder.map_data.scripts = acs.to_code()

        builder.build(output_path)
        print(f"WAD written to {output_path}")

if __name__ == "__main__":
    import os
    print("Generating default Armory Siege...")
    output = "examples/benchmark/output/armory_siege.wad"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    scenario = ArmorySiegeScenario()
    scenario.generate(output)
