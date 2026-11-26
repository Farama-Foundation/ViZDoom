"""
Smart Enemies Benchmark Scenario.
Generates an arena with custom enemy behavior governed by ACS.
"""

from typing import Dict, Any
import random
import math
import os
from shapely.geometry import box
from shapely.affinity import rotate
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.geometry.shapes import circle
from doomgen.logic.acs_builder import ACSBuilder, ScriptType
from doomgen.doom.things import ThingType

class SmartEnemiesScenario(Scenario):
    """
    A scenario featuring custom 'Smart' enemies that react to player density.

    Revised Requirements:
    - Circular Arena (Damaging floor everywhere).
    - Player Loadout: Shotgun + Ammo.
    - Enemy Behavior: Prioritize fleeing (Frightened, no attacks), drop health/ammo on kill.
    - Continuous enemy spawning at random safe positions.
    """

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "arena_radius": 800,    # Radius of the circular arena
            "num_enemies": 4,       # Max concurrent enemies
            "spawn_interval": 3,     # Seconds between spawn checks
            "player_count": 2,       # Number of player starts
            "proximity_radius": 512, # Distance to trigger "smart" behavior
            "num_obstacles": 30,     # Increased count for density
            "seed": 42
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        rad = cfg["arena_radius"]
        seed = cfg.get("seed", 42)

        random.seed(seed)

        builder = ProceduralMapBuilder(
            bounds=(-rad - 256, -rad - 256, rad + 256, rad + 256),
            seed=seed
        )

        arena_poly = circle(0, 0, rad)
        ARENA_TAG = 1

        builder.add_area(
            "Arena",
            shape=arena_poly,
            floor_height=0,
            ceiling_height=384,
            floor_texture="NUKAGE1",
            ceiling_texture="CEIL3_5",
            wall_texture="STONE2",
            light_level=255,
            tag=ARENA_TAG,
            special=0 # Set damage via ACS
        )

        obstacle_geoms = []

        for i in range(cfg["num_obstacles"]):
            # Random position within radius * 0.9
            r = math.sqrt(random.uniform(200**2, (rad * 0.9)**2))
            theta = random.uniform(0, 2 * math.pi)
            ox = r * math.cos(theta)
            oy = r * math.sin(theta)

            shape_type = random.choice(["circle", "rect", "rect", "l_shape", "cross"])

            obs_shape = None
            if shape_type == "circle":
                radius = random.randint(32, 96)
                obs_shape = circle(ox, oy, radius)
            elif shape_type == "l_shape":
                w1 = random.randint(96, 192)
                h1 = random.randint(32, 64)
                rect1 = box(ox, oy, ox + w1, oy + h1)

                w2 = random.randint(32, 64)
                h2 = random.randint(96, 192)
                rect2 = box(ox, oy, ox + w2, oy + h2)

                obs_shape = rect1.union(rect2)
                obs_shape = rotate(obs_shape, random.uniform(0, 360), origin=(ox, oy))

            elif shape_type == "cross":
                arm_len = random.randint(96, 160)
                thickness = random.randint(32, 48)

                rect1 = box(ox - arm_len, oy - thickness, ox + arm_len, oy + thickness)
                rect2 = box(ox - thickness, oy - arm_len, ox + thickness, oy + arm_len)
                obs_shape = rect1.union(rect2)
                obs_shape = rotate(obs_shape, random.uniform(0, 360), origin=(ox, oy))

            else:
                w = random.randint(64, 256)
                h = random.randint(64, 256)
                obs_shape = box(ox - w/2, oy - h/2, ox + w/2, oy + h/2)
                obs_shape = rotate(obs_shape, random.uniform(0, 360))

            obs_shape = obs_shape.intersection(arena_poly)
            if obs_shape.is_empty:
                continue

            builder.add_area(
                f"Obstacle_{i}",
                shape=obs_shape,
                floor_height=384, # Solid (floor to ceiling)
                ceiling_height=384,
                wall_texture="BIGBRIK1",
                mode="overwrite"
            )
            obstacle_geoms.append(obs_shape)

        # Portals (instead of solid walls) show the obstacle's BIGBRIK1 as a
        # full-height lower texture from the arena side.
        for i in range(cfg["num_obstacles"]):
            if f"Obstacle_{i}" in builder.name_to_id:
                builder.connect_adjacent("Arena", f"Obstacle_{i}", silent=True)

        decorate_script = """
        ACTOR ShortShotgun : Shotgun 15003
        {
            States
            {
            Fire:
                SHTG A 3
                SHTG A 0 A_FireBullets (5.6, 0, 7, 5, "BulletPuff", 1, 500)
                SHTG A 7 A_GunFlash
                SHTG BC 5
                SHTG D 4
                SHTG CB 5
                SHTG A 3
                SHTG A 7 A_ReFire
                Goto Ready
            }
        }

        ACTOR HuntedDemon : Demon 15001
        {
            Health 100
            Speed 21
            +FRIGHTENED
            +LOOKALLAROUND
            States
            {
            Spawn:
                SARG AB 5 A_Look
                Loop
            See:
                SARG AABBCCDD 2 A_Chase
                Loop
            Missile:
                Goto See
            Melee:
                Goto See
            Death:
                SARG I 8
                SARG J 8 A_Scream
                SARG K 6
                SARG L 6 A_NoBlocking
                SARG L 0 ACS_ExecuteAlways(997, 0)
                SARG M 0 ACS_ExecuteAlways(998, 0)
                SARG M -1
                Stop
            }
        }
        """
        builder.wad_writer.add_lump("DECORATE", decorate_script.encode('utf-8'))

        # Player starts: the four relaxed spots closest to the arena centre
        player_spots = sorted(
            builder.get_spots("Arena", spacing=64, clearance=32),
            key=lambda p: math.hypot(p[0], p[1])
        )[:4]
        player_starts = (
            ThingType.PLAYER1_START, ThingType.PLAYER2_START,
            ThingType.PLAYER3_START, ThingType.PLAYER4_START,
        )
        for thing_type, (px, py) in zip(player_starts, player_spots):
            builder.add_thing(thing_type, px, py, angle=0)

        # Spawn Points (MapSpots) for Continuous Spawning
        spawn_spot_base_tid = 9000
        num_spawn_spots = 50
        THING_MAPSPOT = 9001

        spots_placed = 0
        attempts = 0
        while spots_placed < num_spawn_spots and attempts < 1000:
            attempts += 1

            # Relaxed (centroidal Voronoi) spot within "Arena", clear of walls
            pt = builder.get_random_point_in_area("Arena", safe=True, clearance=32)
            if pt is None:
                break

            sx, sy = pt
            if all(math.hypot(sx - px, sy - py) >= 128 for px, py in player_spots):
                builder.add_thing(
                    THING_MAPSPOT,
                    sx, sy,
                    tid=spawn_spot_base_tid + spots_placed,
                    angle=0
                )
                spots_placed += 1

        acs = ACSBuilder()
        acs.add_global_var("se_alive_enemies_global", 31, "int")
        acs.add_global_var("se_fast_enemies_global", 32, "int")
        acs.add_global_var("se_fast_events_global", 33, "int")
        acs.add_global_var("se_spawn_events_global", 34, "int")

        # Logic constants
        radius_sq = cfg["proximity_radius"] * cfg["proximity_radius"]
        spawn_tics = cfg["spawn_interval"] * 35

        acs_global = f"""
        #include "zcommon.acs"

        #define MAX_ENEMIES {cfg['num_enemies']}
        #define MAX_PLAYERS {cfg['player_count']}
        #define PROXIMITY_SQ {radius_sq}
        #define ENEMY_BASE_TID {1000}
        #define ENEMY_TEMP_TID {9900}
        #define PLAYER_BASE_TID 100
        #define SPAWN_SPOT_BASE {spawn_spot_base_tid}
        #define NUM_SPOTS {spots_placed}
        #define ARENA_TAG {ARENA_TAG}

        // ENTER script runs for each player on spawn
        script 100 ENTER {{
            int pid = PlayerNumber();
            Thing_ChangeTID(0, PLAYER_BASE_TID + pid);
            SetActorProperty(0, APROP_Health, 100);

            // Give Loadout
            ClearInventory();
            GiveInventory("ShortShotgun", 1);
            GiveInventory("Shell", 20);
            SetWeapon("ShortShotgun");

            HudMessage(s:"WARNING: TOXIC FLOOR DETECTED"; HUDMSG_PLAIN, 1, CR_RED, 0.5, 0.8, 5.0);
        }}

        // SCRIPT 1: Smart Behavior Loop (OPEN)
        script 1 OPEN {{
            // Set Sector Damage via ACS
            // Tag, Damage Amount, Interval (0=Default), MOD_UNKNOWN(0)
            Sector_SetDamage(ARENA_TAG, 2, 0);
            se_alive_enemies_global = 0;
            se_fast_enemies_global = 0;
            se_fast_events_global = 0;
            se_spawn_events_global = 0;
            int fast_state[MAX_ENEMIES];
            int i;
            for (i = 0; i < MAX_ENEMIES; i++) {{
                fast_state[i] = 0;
            }}

            while (TRUE) {{
                int alive_count = 0;
                int fast_count = 0;

                // Iterate all potential enemies
                for (i = 0; i < MAX_ENEMIES; i++) {{
                    int etid = ENEMY_BASE_TID + i;

                    // Only process alive enemies
                    if (GetActorProperty(etid, APROP_Health) > 0) {{
                        alive_count++;
                        int nearby_players = 0;
                        int ex = GetActorX(etid) >> 16;
                        int ey = GetActorY(etid) >> 16;

                        // Check distance to all players
                        int closest_dist_sq = 2147483647;
                        int closest_pid = 0;
                        int p;

                        for (p = 0; p < MAX_PLAYERS; p++) {{
                            int ptid = PLAYER_BASE_TID + p;
                            if (GetActorProperty(ptid, APROP_Health) > 0) {{
                                int px = GetActorX(ptid) >> 16;
                                int py = GetActorY(ptid) >> 16;
                                int dx = ex - px;
                                int dy = ey - py;
                                int dist_sq = dx*dx + dy*dy;

                                if (dist_sq < closest_dist_sq) {{
                                    closest_dist_sq = dist_sq;
                                    closest_pid = ptid;
                                }}

                                if (dist_sq < PROXIMITY_SQ) {{
                                    nearby_players++;
                                }}
                            }}
                        }}

                        // Continuously force hate on closest player to ensure fleeing
                        if (closest_pid != 0) {{
                            Thing_Hate(etid, closest_pid, 1);
                        }}

                        int is_fast = 0;
                        if (nearby_players > 1) {{
                            is_fast = 1;
                        }}

                        if (is_fast) {{
                            fast_count++;
                            if (fast_state[i] == 0) {{
                                se_fast_events_global++;
                                fast_state[i] = 1;
                                SetActorProperty(etid, APROP_Health, GetActorProperty(etid, APROP_Health) + 400);
                            }}
                        }} else {{
                            if (fast_state[i] == 1) {{
                                int hp = GetActorProperty(etid, APROP_Health);
                                if (hp > 400) {{
                                    SetActorProperty(etid, APROP_Health, hp - 400);
                                }} else {{
                                    SetActorProperty(etid, APROP_Health, 1);
                                }}
                            }}
                            fast_state[i] = 0;
                        }}

                        // Apply movement behavior only (kept lightweight to reduce ACS overhead)
                        if (is_fast) {{
                            SetActorProperty(etid, APROP_Speed, 70.0);
                        }} else {{
                            SetActorProperty(etid, APROP_Speed, 21.0);
                        }}
                    }} else {{
                        fast_state[i] = 0;
                    }}
                }}

                se_alive_enemies_global = alive_count;
                se_fast_enemies_global = fast_count;
                Delay(8);
            }}
        }}

        // SCRIPT 2: Continuous Spawning (OPEN)
        script 2 OPEN {{
            while (TRUE) {{
                int i;
                // Try to spawn ONE enemy per cycle if slot available
                for (i = 0; i < MAX_ENEMIES; i++) {{
                    int etid = ENEMY_BASE_TID + i;

                    // Check if slot is free (Thing Count is 0)
                    if (ThingCount(T_NONE, etid) == 0) {{
                        // Pick random spawn spot
                        int spot_idx = Random(0, NUM_SPOTS - 1);
                        int spot_tid = SPAWN_SPOT_BASE + spot_idx;

                        // Spawn with temp TID, initialize, then move into slot TID.
                        // This avoids race conditions with delayed corpse cleanup.
                        if (SpawnSpotFacing("HuntedDemon", spot_tid, ENEMY_TEMP_TID)) {{
                            // Safety: enforce damageable state on fresh spawn
                            SetActorProperty(ENEMY_TEMP_TID, APROP_Invulnerable, 0);
                            TakeActorInventory(ENEMY_TEMP_TID, "PowerInvulnerable", 255);

                            // Wake up the enemy and make it hate the player
                            // Thing_Hate(hater, hated, type=1 (Hate enemies without sight check))
                            int rnd_player = PLAYER_BASE_TID + Random(0, MAX_PLAYERS-1);

                            // Use Type 1 to assign target even if not seen immediately.
                            // Type 6 forces "Hunt" which overrides +FRIGHTENED behavior.
                            Thing_Hate(ENEMY_TEMP_TID, rnd_player, 1);

                            // Force transition to See state so A_Chase runs and +FRIGHTENED takes effect
                            SetActorState(ENEMY_TEMP_TID, "See");

                            // Move actor from temp setup slot to tracked enemy slot
                            Thing_ChangeTID(ENEMY_TEMP_TID, etid);

                            // Spawn sound effect at spot
                            PlaySound(spot_tid, "misc/teleport", CHAN_BODY, 1.0, FALSE, ATTN_NORM);
                            se_spawn_events_global++;

                            break; // Only spawn one per interval
                        }}
                    }}
                }}
                Delay({spawn_tics});
            }}
        }}
        // SCRIPT 997: Reward all players (VOID)
        script 997 (VOID) {{
            int i;
            for (i = 0; i < MAX_PLAYERS; i++) {{
                int ptid = PLAYER_BASE_TID + i;
                GiveActorInventory(ptid, "HealthBonus", 7);
                GiveActorInventory(ptid, "Shell", 5);
            }}
        }}
        """

        acs.add_global_code(acs_global)

        acs.add_script(
            ScriptType.VOID,
            "int dead_tid = ActivatorTID(); int cleanup_tid = dead_tid + 5000; Thing_ChangeTID(0, cleanup_tid); Delay(35*2); Thing_Remove(cleanup_tid);",
            number=998
        )
        acs.add_script(ScriptType.DEATH, "Delay(1); Exit_Normal(0);", number=999)

        builder.map_data.scripts = acs.to_code()
        try:
            builder.map_data.behavior = acs.compile()
        except Exception as e:
            if hasattr(e, 'stderr') and e.stderr:
                print(f"ACS Compilation Warning:\n{e.stderr}")
            else:
                print(f"ACS Compilation Warning: {e}")

        builder.build(output_path)

if __name__ == "__main__":
    scenario = SmartEnemiesScenario(config={"seed": random.randint(0, 99999)})
    os.makedirs("examples/benchmark/output", exist_ok=True)
    scenario.generate("examples/benchmark/output/smart_enemies.wad")
