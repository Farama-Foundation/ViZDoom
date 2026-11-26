import random
from typing import Any, Dict
import io
from PIL import Image

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder


class LavaMazeSimpleScenario(Scenario):
    """
    Simplified Two-player cooperative maze scenario.

    Player 1 navigates a 1D straight-line lava maze.
    Player 2 observes from elevated stands and signals directions via
    colored screen flashes (only Green/Red) triggered by weapon fire (Pistol/Shotgun).
    """

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "physical_size": 18,
            "cell_size": 128,
            "initial_maze_size": 4,
            "max_maze_size": 14,
            "lava_damage": 35,
            "lava_depth": 64,
            "step_tics": 12,
            "step_cooldown_tics": 4,
            "level_clear_grace_tics": 12,
            "seed": 42,
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        PHYSICAL_SIZE = int(cfg["physical_size"])
        CELL_SIZE = int(cfg["cell_size"])
        INITIAL_MAZE_SIZE = int(cfg["initial_maze_size"])
        MAX_MAZE_SIZE = int(cfg.get("max_maze_size", 14))
        LAVA_DAMAGE = int(cfg["lava_damage"])
        LAVA_DEPTH = int(cfg.get("lava_depth", 64))
        STEP_TICS = int(cfg.get("step_tics", 12))
        STEP_COOLDOWN_TICS = int(cfg.get("step_cooldown_tics", 4))
        LEVEL_CLEAR_GRACE_TICS = int(cfg.get("level_clear_grace_tics", 12))
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)

        MAZE_WIDTH_UNITS = PHYSICAL_SIZE * CELL_SIZE
        OFFSET_X = -(MAZE_WIDTH_UNITS // 2)
        OFFSET_Y = -(MAZE_WIDTH_UNITS // 2)

        builder = ProceduralMapBuilder(
            bounds=(-2400, -2400, 2400, 2400),
            seed=seed,
        )

        # Generate signal image lumps using PIL
        def create_color_lump(name: str, color_rgb: tuple[int, int, int]):
            img = Image.new('RGB', (1920, 1080), color=color_rgb)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            builder.wad_writer.add_lump(name, buf.getvalue())

        lump_files = {
            "S_GREEN": (0, 255, 0),
            "S_RED":   (255, 0, 0),
            "S_YELLOW":(255, 255, 0),
            "S_BLUE":  (0, 0, 255),
            "S_BLACK": (0, 0, 0),
            "S_WALL":  (200, 200, 200),
            "F_START": (0, 255, 255),  # Cyan
            "F_END":   (255, 0, 255),  # Magenta
        }
        builder.wad_writer.add_lump("TX_START", b"")
        for lump_name, color in lump_files.items():
            create_color_lump(lump_name, color)
        builder.wad_writer.add_lump("TX_END", b"")

        decorate_str = """
ACTOR LMPistol : Pistol replaces Pistol
{
  Weapon.SlotNumber 1
  States
  {
  Select:
    PISG A 0 A_Raise
    PISG A 0 A_Raise
    PISG A 0 A_Raise
    PISG A 0 A_Raise
    PISG A 0 A_Raise
    PISG A 0 A_Raise
    PISG A 0 A_Raise
    PISG A 0 A_Raise
    PISG A 1 A_Raise
    Loop
  Deselect:
    PISG A 0 A_Lower
    PISG A 0 A_Lower
    PISG A 0 A_Lower
    PISG A 0 A_Lower
    PISG A 0 A_Lower
    PISG A 0 A_Lower
    PISG A 0 A_Lower
    PISG A 0 A_Lower
    PISG A 1 A_Lower
    Loop
  Fire:
    PISG A 4
    PISG B 0 ACS_ExecuteAlways(910, 0, 1)
    PISG B 6 A_FirePistol
    PISG C 4
    PISG B 5
    PISG A 20
    PISG A 1 A_ReFire
    Goto Ready
  }
}

ACTOR LMShotgun : Shotgun replaces Shotgun
{
  Weapon.SlotNumber 2
  States
  {
  Select:
    SHTG A 0 A_Raise
    SHTG A 0 A_Raise
    SHTG A 0 A_Raise
    SHTG A 0 A_Raise
    SHTG A 0 A_Raise
    SHTG A 0 A_Raise
    SHTG A 0 A_Raise
    SHTG A 0 A_Raise
    SHTG A 1 A_Raise
    Loop
  Deselect:
    SHTG A 0 A_Lower
    SHTG A 0 A_Lower
    SHTG A 0 A_Lower
    SHTG A 0 A_Lower
    SHTG A 0 A_Lower
    SHTG A 0 A_Lower
    SHTG A 0 A_Lower
    SHTG A 0 A_Lower
    SHTG A 1 A_Lower
    Loop
  Fire:
    SHTG A 3
    SHTG A 0 ACS_ExecuteAlways(910, 0, 2)
    SHTG A 0 A_FireShotgun
    SHTG A 4
    SHTG B 5
    SHTG C 5
    SHTG D 4
    SHTG C 5
    SHTG B 5
    SHTG A 20
    SHTG A 7 A_ReFire
    Goto Ready
  }
}
"""
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode('utf-8'))

        builder.add_area(
            "LavaPit",
            shape=(0, 0, 4400, 4400),
            floor_height=-LAVA_DEPTH,
            ceiling_height=800,
            floor_texture="LAVA1",
            mode="overwrite",
            light_level=255,
            tag=99,
        )

        STANDS_HEIGHT = 350
        STANDS_THICKNESS = 100
        STANDS_INNER = (MAZE_WIDTH_UNITS // 2) + 64

        pos_y = STANDS_INNER + STANDS_THICKNESS / 2
        stands_width = 2 * STANDS_INNER + 2 * STANDS_THICKNESS
        builder.add_area("Stands_N", shape=(0, pos_y, stands_width, STANDS_THICKNESS),   floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=201, wall_texture="S_YELLOW",  light_level=255)
        builder.add_area("Stands_S", shape=(0, -pos_y, stands_width, STANDS_THICKNESS),  floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=203, wall_texture="S_BLUE", light_level=255)

        pos_x = STANDS_INNER + STANDS_THICKNESS / 2
        h_span = 2 * STANDS_INNER + 2 * STANDS_THICKNESS
        builder.add_area("Stands_E", shape=( pos_x, 0, STANDS_THICKNESS, h_span), floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=200, wall_texture="S_GREEN", light_level=255)
        builder.add_area("Stands_W", shape=(-pos_x, 0, STANDS_THICKNESS, h_span), floor_height=STANDS_HEIGHT, ceiling_height=1024, mode="overwrite", tag=202, wall_texture="S_RED",  light_level=255)

        base_tag = 1000
        center_idx = PHYSICAL_SIZE // 2
        for py in range(PHYSICAL_SIZE):
            for px in range(PHYSICAL_SIZE):
                cx = OFFSET_X + (px * CELL_SIZE) + (CELL_SIZE // 2)
                cy = OFFSET_Y + (py * CELL_SIZE) + (CELL_SIZE // 2)
                tag = base_tag + (py * PHYSICAL_SIZE + px)

                if px == center_idx and py == center_idx:
                    f_tex, f_h = "FLOOR4_8", 0
                else:
                    f_tex, f_h = "LAVA1", -LAVA_DEPTH

                builder.add_area(
                    f"Grid_{px}_{py}",
                    shape=(cx, cy, CELL_SIZE, CELL_SIZE),
                    floor_height=f_h,
                    ceiling_height=1024,
                    tag=tag,
                    mode="overwrite",
                    floor_texture=f_tex,
                    wall_texture="S_WALL",
                    light_level=255,
                )

        print("Building Connectivity Graph...")

        builder.connect_adjacent("Stands_N", "Stands_E", ConnectionType.OPEN)
        builder.connect_adjacent("Stands_E", "Stands_S", ConnectionType.OPEN)
        builder.connect_adjacent("Stands_S", "Stands_W", ConnectionType.OPEN)
        builder.connect_adjacent("Stands_W", "Stands_N", ConnectionType.OPEN)

        barrier_args = {"blocking": True, "middle_texture": "-"}
        for stand in ("Stands_N", "Stands_S", "Stands_E", "Stands_W"):
            builder.connect_adjacent(stand, "LavaPit", ConnectionType.WINDOW, **barrier_args)

        for py in range(PHYSICAL_SIZE):
            for px in range(PHYSICAL_SIZE):
                curr = f"Grid_{px}_{py}"
                if px < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px+1}_{py}", ConnectionType.OPEN, silent=True)
                if py < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px}_{py+1}", ConnectionType.OPEN, silent=True)
                if px < PHYSICAL_SIZE - 1 and py < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px+1}_{py+1}", ConnectionType.OPEN, silent=True)
                if px > 0 and py < PHYSICAL_SIZE - 1:
                    builder.connect_adjacent(curr, f"Grid_{px-1}_{py+1}", ConnectionType.OPEN, silent=True)
                builder.connect_adjacent(curr, "LavaPit", ConnectionType.OPEN, silent=True)

        p1_x = OFFSET_X + (center_idx * CELL_SIZE) + (CELL_SIZE // 2)
        p1_y = OFFSET_Y + (center_idx * CELL_SIZE) + (CELL_SIZE // 2)
        builder.add_thing(ThingType.PLAYER1_START, x=p1_x, y=p1_y, angle=0)

        p2_y = -(STANDS_INNER + (STANDS_THICKNESS // 2))
        builder.add_thing(ThingType.PLAYER2_START, x=0, y=p2_y, angle=90)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")

        acs.add_map_var("g_flash_timer", initial=0)
        acs.add_map_var("g_flash_state", initial=0)
        acs.add_map_var("g_flash_lump", var_type="str", initial='"S_BLACK"')
        acs.add_map_var("level_clear_lock", initial=1)
        acs.add_global_var("lm_maze_size_global",        11, "int")
        acs.add_global_var("lm_goal_grid_x_global",      13, "int")
        acs.add_global_var("lm_p1_grid_x_global",        14, "int")
        acs.add_global_var("lm_p1_grid_y_global",        15, "int")
        acs.add_global_var("lm_flash_signal_global",     16, "int")
        acs.add_global_var("lm_flash_active_global",     17, "int")
        acs.add_global_var("lm_goal_grid_y_global",      18, "int")
        acs.add_global_var("lm_maze_bits_0_global",      19, "int")
        acs.add_global_var("lm_maze_bits_1_global",      20, "int")
        acs.add_global_var("lm_maze_bits_2_global",      21, "int")
        acs.add_global_var("lm_levels_completed_global", 22, "int")
        acs.add_global_var("lm_can_see_global",          23, "int")

        acs.scripts = []

        STANDS_RADIUS_UNITS = int(STANDS_INNER + STANDS_THICKNESS / 2)

        maze_logic_script = f"""
    #define MAP_WIDTH {PHYSICAL_SIZE}
    #define CELL_SIZE {CELL_SIZE}
    #define BASE_TAG {base_tag}
    #define MAX_TILES {MAX_MAZE_SIZE}
    #define APROP_PainChance 21
    #define PLATFORM_RADIUS {STANDS_RADIUS_UNITS}
    #define LM_STEP_TICS {STEP_TICS}
    #define LM_STEP_COOLDOWN_TICS {STEP_COOLDOWN_TICS}
    #define LEVEL_CLEAR_GRACE_TICS {LEVEL_CLEAR_GRACE_TICS}

    int max_unlocked_size = {INITIAL_MAZE_SIZE};
    int current_maze_size = {INITIAL_MAZE_SIZE};
    int current_start_tag = 0;
    int current_end_tag = 0;

    int g_start_x = MAP_WIDTH / 2;
    int g_start_y = MAP_WIDTH / 2;

    int maze_bits_chunk[3];

    function int get_tag(int x, int y) {{
        return BASE_TAG + (y * MAP_WIDTH + x);
    }}

    function int get_world_x(int gx) {{
        return {OFFSET_X} + (gx * {CELL_SIZE}) + ({CELL_SIZE} / 2);
    }}

    function int get_world_y(int gy) {{
        return {OFFSET_Y} + (gy * {CELL_SIZE}) + ({CELL_SIZE} / 2);
    }}

    function int cell_is_safe(int x, int y) {{
        if (x < 0 || x >= MAP_WIDTH || y < 0 || y >= MAP_WIDTH) return FALSE;
        if (y != lm_goal_grid_y_global) return FALSE;

        int bit_idx = x;
        int chunk = bit_idx / 27;
        int shift = bit_idx % 27;
        int mask = 1 << shift;

        if (chunk == 0) return ((lm_maze_bits_0_global & mask) != 0);
        if (chunk == 1) return ((lm_maze_bits_1_global & mask) != 0);
        if (chunk == 2) return ((lm_maze_bits_2_global & mask) != 0);
        return FALSE;
    }}

    function void reset_map(int keep_tag) {{
        int t;
        int i;
        for (i = 0; i < MAP_WIDTH * MAP_WIDTH; i++) {{
            t = BASE_TAG + i;
            if (t != keep_tag) {{
                Floor_MoveToValue(t, 2048, -{LAVA_DEPTH}, 0);
                ChangeFloor(t, "LAVA1");
                Sector_SetDamage(t, {LAVA_DAMAGE}, 14);
                Sector_SetColor(t, 255, 255, 255);
            }}
        }} // END FOR
    }}

    function void carve(int x, int y) {{
        int tag = get_tag(x, y);
        Floor_MoveToValue(tag, 2048, 0, 0);
        ChangeFloor(tag, "FLOOR4_8");
        Sector_SetDamage(tag, 0, 0);
        Light_ChangeToValue(tag, 255);
        if (x >= 0 && x < MAP_WIDTH && y >= 0 && y < MAP_WIDTH) {{
            // Since this is a 1D maze, we only encode the X-axis presence
            // so it fits properly within the 3 available 27-bit global variables.
            int bit_idx = x;
            int chunk = bit_idx / 27;
            int shift = bit_idx % 27;
            if (chunk < 3) maze_bits_chunk[chunk] = maze_bits_chunk[chunk] | (1 << shift);
        }}
    }}

    // Unused helper functions removed for simplicity

    function int IsInFOV(int observer, int target) {{
        if (!CheckSight(observer, target, 0)) return FALSE;

        int dx = GetActorX(target) - GetActorX(observer);
        int dy = GetActorY(target) - GetActorY(observer);

        int obsZ = GetActorZ(observer) + GetActorViewHeight(observer);
        int targZ = GetActorZ(target) + (GetActorProperty(target, APROP_Height) / 2);
        int dz = targZ - obsZ;

        int angToTarget = VectorAngle(dx, dy);
        int observerAng = GetActorAngle(observer);
        int yawDiff = angToTarget - observerAng;

        while (yawDiff <= -0.5) yawDiff += 1.0;
        while (yawDiff > 0.5) yawDiff -= 1.0;
        if (yawDiff < 0) yawDiff = -yawDiff;

        if (yawDiff > 0.125) return FALSE;

        int xy_dist = VectorLength(dx, dy);
        int pitchToTarget = VectorAngle(xy_dist, dz);

        if (pitchToTarget > 0.5) pitchToTarget -= 1.0;
        pitchToTarget = -pitchToTarget;

        int observerPitch = GetActorPitch(observer);
        int pitchDiff = pitchToTarget - observerPitch;

        while (pitchDiff <= -0.5) pitchDiff += 1.0;
        while (pitchDiff > 0.5) pitchDiff -= 1.0;
        if (pitchDiff < 0) pitchDiff = -pitchDiff;

        if (pitchDiff > 0.10) return FALSE;

        return TRUE;
    }}

    script 1 OPEN {{
        Sector_SetDamage(99, {LAVA_DAMAGE}, 14);

        g_start_x = Random(0, MAP_WIDTH - 1);
        g_start_y = Random(0, MAP_WIDTH - 1);

        current_start_tag = get_tag(g_start_x, g_start_y);
        lm_maze_size_global        = MAP_WIDTH;
        lm_goal_grid_x_global      = -1;
        lm_goal_grid_y_global      = -1;
        lm_p1_grid_x_global        = -1;
        lm_p1_grid_y_global        = -1;
        lm_flash_signal_global     = 0;
        lm_flash_active_global     = 0;
        lm_maze_bits_0_global      = 0;
        lm_maze_bits_1_global      = 0;
        lm_maze_bits_2_global      = 0;
        lm_levels_completed_global = 0;
        level_clear_lock           = 1;

        ACS_NamedExecute("GenerateMaze", 0, g_start_x, g_start_y, current_maze_size);

        while (TRUE) {{
            lm_can_see_global = IsInFOV(1001, 1000);
            Delay(5);
        }}
    }}

    script "GenerateMaze" (int sx, int sy, int size) {{
        int start_tag = get_tag(sx, sy);
        int i;
        reset_map(start_tag);
        Delay(1);

        for (i = 0; i < 3; i++) maze_bits_chunk[i] = 0;

        carve(sx, sy);
        ChangeFloor(start_tag, "F_START");
        Sector_SetColor(start_tag, 255, 255, 255);
        Light_ChangeToValue(start_tag, 255);
        Sector_SetFade(start_tag, 0, 0, 0);

        // Simple straight line generation (1D in X-axis direction)
        int dir = 1;
        int can_pos = (sx + size - 1 < MAP_WIDTH);
        int can_neg = (sx - size + 1 >= 0);

        if (can_pos && can_neg) {{
            dir = (Random(0, 1) * 2) - 1; // +1 or -1
        }} else if (can_pos) {{
            dir = 1;
        }} else {{
            dir = -1;
        }}

        int current_x = sx;
        int last_visited_tag = start_tag;

        for (i = 1; i < size; i++) {{
            current_x = current_x + dir;
            carve(current_x, sy);
            last_visited_tag = get_tag(current_x, sy);
        }} // END FOR

        current_end_tag = last_visited_tag;

        lm_maze_size_global        = MAP_WIDTH;
        lm_goal_grid_x_global      = current_x;
        lm_goal_grid_y_global      = sy;
        lm_maze_bits_0_global      = maze_bits_chunk[0];
        lm_maze_bits_1_global      = maze_bits_chunk[1];
        lm_maze_bits_2_global      = maze_bits_chunk[2];

        ChangeFloor(current_end_tag, "F_END");
        Sector_SetColor(current_end_tag, 255, 255, 255);
        Light_ChangeToValue(current_end_tag, 255);
        Sector_SetFade(current_end_tag, 0, 0, 0);

        level_clear_lock = 0;
    }}

    script "EndGame" (void) {{
        Delay(1);
        Exit_Normal(0);
    }}

    script 910 (int signal) {{
        if (signal < 1 || signal > 2) terminate;

        lm_flash_signal_global = signal;
        if (signal == 1) {{ g_flash_lump = "S_GREEN"; }}
        if (signal == 2) {{ g_flash_lump = "S_RED";   }}

        g_flash_timer = 25;
    }}

    script 999 DEATH {{
        if (PlayerNumber() != -1) {{
            ACS_NamedExecuteAlways("EndGame", 0);
        }}
    }}

    script 2 ENTER {{
        if (PlayerNumber() != 0) terminate;

        Thing_ChangeTID(0, 1000);
        SetActorProperty(0, APROP_Health, 100);
        SetPlayerProperty(0, 1, 0); // PROP_FROZEN

        int wx = get_world_x(g_start_x) << 16;
        int wy = get_world_y(g_start_y) << 16;

        SetActorPosition(0, wx, wy, 0, 0);
        SetActorProperty(0, APROP_ViewHeight, 110.0);
        SetActorProperty(0, APROP_Mass, 0x7FFFFFFF);
        SetActorProperty(0, APROP_ScaleX, 5.5);
        SetActorProperty(0, APROP_ScaleY, 5.5);
        SetActorProperty(0, 21, 0);

        SetHudSize(320, 200, 1);
        SetFont("S_BLACK");
        HudMessage(s:"A"; HUDMSG_PLAIN | HUDMSG_LAYER_UNDERHUD, 1, CR_UNTRANSLATED, 160.0, 100.0, 0.0);
        ClearInventory();

        int p1_cell_x = g_start_x;
        int p1_cell_y = g_start_y;
        int step_active = 0;
        int step_timer = 0;
        int step_from_x = g_start_x;
        int step_to_x = g_start_x;
        int step_cooldown = 0;
        int step_into_lava = 0;

        while (TRUE) {{
            if (GetActorProperty(0, APROP_HEALTH) <= 0) {{
                ACS_NamedExecute("EndGame", 0);
                terminate;
            }}

            if (step_active) {{
                step_timer++;
                int from_world_x = get_world_x(step_from_x);
                int to_world_x = get_world_x(step_to_x);
                int interp_world_x = from_world_x + (((to_world_x - from_world_x) * step_timer) / LM_STEP_TICS);
                SetActorPosition(0, interp_world_x << 16, get_world_y(p1_cell_y) << 16, 0, 0);
                SetActorVelocity(0, 0, 0, 0, FALSE, FALSE);

                if (step_timer >= LM_STEP_TICS) {{
                    p1_cell_x = step_to_x;
                    SetActorPosition(0, get_world_x(p1_cell_x) << 16, get_world_y(p1_cell_y) << 16, 0, 0);
                    SetActorVelocity(0, 0, 0, 0, FALSE, FALSE);
                    step_active = 0;
                    step_timer = 0;
                    step_cooldown = LM_STEP_COOLDOWN_TICS;
                    if (step_into_lava) {{
                        SetActorProperty(0, APROP_Health, 0);
                    }}
                    step_into_lava = 0;
                }}
            }} else if (step_cooldown > 0) {{
                step_cooldown--;
            }} else if (level_clear_lock == 0) {{
                int btns = GetPlayerInput(-1, INPUT_BUTTONS);
                int desired_dir = 0;
                if ((btns & BT_FORWARD) && !(btns & BT_BACK)) desired_dir = 1;
                if ((btns & BT_BACK) && !(btns & BT_FORWARD)) desired_dir = -1;

                if (desired_dir != 0) {{
                    int target_x = p1_cell_x + desired_dir;
                    step_into_lava = !cell_is_safe(target_x, p1_cell_y);
                    step_active = 1;
                    step_timer = 0;
                    step_from_x = p1_cell_x;
                    step_to_x = target_x;
                }}
            }}

            if (!step_active) {{
                SetActorPosition(0, get_world_x(p1_cell_x) << 16, get_world_y(p1_cell_y) << 16, 0, 0);
                SetActorVelocity(0, 0, 0, 0, FALSE, FALSE);
            }}

            if (p1_cell_x >= 0 && p1_cell_x < MAP_WIDTH && p1_cell_y >= 0 && p1_cell_y < MAP_WIDTH) {{
                lm_p1_grid_x_global = p1_cell_x;
                lm_p1_grid_y_global = p1_cell_y;
                int my_tag = get_tag(p1_cell_x, p1_cell_y);

                if (my_tag == current_end_tag && level_clear_lock == 0) {{
                    level_clear_lock = 1;
                    lm_levels_completed_global++;
                    max_unlocked_size += 2;
                    if (max_unlocked_size > MAX_TILES) max_unlocked_size = MAX_TILES;

                    int next_size = Random({INITIAL_MAZE_SIZE}, max_unlocked_size);
                    current_maze_size = next_size;
                    lm_maze_size_global = MAP_WIDTH;

                    ACS_NamedExecute("GenerateMaze", 0, p1_cell_x, p1_cell_y, current_maze_size);
                }}
            }} else {{
                lm_p1_grid_x_global = -1;
                lm_p1_grid_y_global = -1;
            }}

            if (g_flash_timer > 0) {{
                g_flash_timer--;
                if (g_flash_state == 0 || g_flash_timer > 18) {{
                     SetHudSize(320, 200, 1);
                     SetFont(g_flash_lump);
                     HudMessage(s:"A"; HUDMSG_PLAIN | HUDMSG_LAYER_UNDERHUD, 1, CR_UNTRANSLATED, 160.0, 100.0, 0.0);
                     g_flash_state = 1;
                     lm_flash_active_global = 1;
                }}
            }} else {{
                if (g_flash_state == 1) {{
                    SetHudSize(320, 200, 1);
                    SetFont("S_BLACK");
                    HudMessage(s:"A"; HUDMSG_PLAIN | HUDMSG_LAYER_UNDERHUD, 1, CR_UNTRANSLATED, 160.0, 100.0, 0.0);
                    g_flash_state = 0;
                    lm_flash_active_global = 0;
                    lm_flash_signal_global = 0;
                }}
            }}

            SetActorAngle(1000, 0);
            SetActorPitch(1000, 0);

            Delay(1);
        }} // END WHILE
    }}

    script 3 ENTER {{
        if (PlayerNumber() != 1) terminate;

        Thing_ChangeTID(0, 1001);
        SetActorProperty(0, APROP_Health, 100);
        SetPlayerProperty(0, 1, 0); // PROP_FROZEN

        ClearInventory();
        GiveInventory("LMPistol",      1);
        GiveInventory("LMShotgun",     1);
        GiveInventory("Clip",   200);
        GiveInventory("Shell",   50);
        SetWeapon("LMPistol");

        int r_fix = PLATFORM_RADIUS << 16;

        while(TRUE) {{
            GiveInventory("Clip",  1);
            GiveInventory("Shell", 1);

            int px = GetActorX(1000);
            int py = GetActorY(1000);

            // Move parallel on the further stand (North or South)
            int obs_x = px;
            int obs_y = -r_fix;
            if (py <= 0) {{
                obs_y = r_fix;
            }}
            int obs_z = GetActorZ(0);

            // Keep observer smoothly clamped within the actual stand endpoints length
            int bound = r_fix - (128 << 16);
            if (obs_x > bound) obs_x = bound;
            if (obs_x < -bound) obs_x = -bound;

            SetActorPosition(0, obs_x, obs_y, obs_z, 0);

            int dx = GetActorX(1000) - GetActorX(0);
            int dy = GetActorY(1000) - GetActorY(0);
            int dz = (GetActorZ(1000) + (GetActorProperty(1000, APROP_Height) / 2)) - (GetActorZ(0) + GetActorViewHeight(0));

            int target_angle = VectorAngle(dx, dy);
            int xy_dist = VectorLength(dx, dy);
            int target_pitch = VectorAngle(xy_dist, dz);

            if (target_pitch > 0.5) {{
                target_pitch -= 1.0;
            }}
            target_pitch = -target_pitch;

            SetActorAngle(0, target_angle);
            SetActorPitch(0, target_pitch);

            Delay(1);
        }} // END WHILE
    }}
    """

        acs.add_global_code(maze_logic_script)

        source_acs = acs.to_code()
        builder.map_data.scripts = source_acs
        try:
            builder.map_data.behavior = acs.compile()
        except Exception as e:
            print(f"ACS Compile Warning: {e}")
            if hasattr(e, "stderr") and e.stderr:
                print(f"ACC Stderr: {e.stderr}")

        builder.build(output_path)
        print(f"Built: {output_path}")


if __name__ == "__main__":
    LavaMazeSimpleScenario().generate("../comrad/scenarios/lava_maze_simple.wad")
