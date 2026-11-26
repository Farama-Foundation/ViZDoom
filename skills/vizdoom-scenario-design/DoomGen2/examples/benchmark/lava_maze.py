import random
from typing import Any, Dict
import io
from PIL import Image

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.logic.acs_builder import ACSBuilder


class LavaMazeScenario(Scenario):
    """
    Two-player cooperative maze scenario.

    Player 1 navigates a procedurally-generated lava maze from inside.
    Player 2 observes from elevated stands and signals directions via
    colored screen flashes triggered by weapon fire.
    """

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "physical_size": 8,
            "cell_size": 128,
            "initial_maze_size": 8,
            "max_maze_size": 20,
            "lava_damage": 35,
            "lava_depth": 64,
            "seed": 42,
        }

    def generate(self, output_path: str) -> None:
        cfg = self.config
        PHYSICAL_SIZE = int(cfg["physical_size"])
        CELL_SIZE = int(cfg["cell_size"])
        INITIAL_MAZE_SIZE = int(cfg["initial_maze_size"])
        MAX_MAZE_SIZE = int(cfg.get("max_maze_size", 20))
        LAVA_DAMAGE = int(cfg["lava_damage"])
        LAVA_DEPTH = int(cfg.get("lava_depth", 64))
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 999999)

        MAZE_WIDTH_UNITS = PHYSICAL_SIZE * CELL_SIZE
        OFFSET_X = -(MAZE_WIDTH_UNITS // 2)
        OFFSET_Y = -(MAZE_WIDTH_UNITS // 2)

        builder = ProceduralMapBuilder(
            bounds=(-1600, -1600, 1600, 1600),
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

ACTOR LMChaingun : Chaingun replaces Chaingun
{
  Weapon.SlotNumber 3
  States
  {
  Select:
    CHGG A 0 A_Raise
    CHGG A 0 A_Raise
    CHGG A 0 A_Raise
    CHGG A 0 A_Raise
    CHGG A 0 A_Raise
    CHGG A 0 A_Raise
    CHGG A 0 A_Raise
    CHGG A 0 A_Raise
    CHGG A 1 A_Raise
    Loop
  Deselect:
    CHGG A 0 A_Lower
    CHGG A 0 A_Lower
    CHGG A 0 A_Lower
    CHGG A 0 A_Lower
    CHGG A 0 A_Lower
    CHGG A 0 A_Lower
    CHGG A 0 A_Lower
    CHGG A 0 A_Lower
    CHGG A 1 A_Lower
    Loop
  Fire:
    CHGG A 0 ACS_ExecuteAlways(910, 0, 3)
    CHGG A 4 A_FireCGun
    CHGG B 4 A_FireCGun
    CHGG B 20
    CHGG B 0 A_ReFire
    Goto Ready
  }
}

ACTOR LMPlasmaRifle : PlasmaRifle replaces PlasmaRifle
{
  Weapon.SlotNumber 4
  States
  {
  Select:
    PLSG A 0 A_Raise
    PLSG A 0 A_Raise
    PLSG A 0 A_Raise
    PLSG A 0 A_Raise
    PLSG A 0 A_Raise
    PLSG A 0 A_Raise
    PLSG A 0 A_Raise
    PLSG A 0 A_Raise
    PLSG A 1 A_Raise
    Loop
  Deselect:
    PLSG A 0 A_Lower
    PLSG A 0 A_Lower
    PLSG A 0 A_Lower
    PLSG A 0 A_Lower
    PLSG A 0 A_Lower
    PLSG A 0 A_Lower
    PLSG A 0 A_Lower
    PLSG A 0 A_Lower
    PLSG A 1 A_Lower
    Loop
  Fire:
    PLSG A 0 ACS_ExecuteAlways(910, 0, 4)
    PLSG A 3 A_FirePlasma
    PLSG B 24
    PLSG B 0 A_ReFire
    Goto Ready
  }
}
"""
        builder.wad_writer.add_lump("DECORATE", decorate_str.encode('utf-8'))

        builder.add_area(
            "LavaPit",
            shape=(0, 0, 2400, 2400),
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

        base_tag = 100
        for py in range(PHYSICAL_SIZE):
            for px in range(PHYSICAL_SIZE):
                cx = OFFSET_X + (px * CELL_SIZE) + (CELL_SIZE // 2)
                cy = OFFSET_Y + (py * CELL_SIZE) + (CELL_SIZE // 2)
                tag = base_tag + (py * PHYSICAL_SIZE + px)
                if px == 1 and py == 1:
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

        p1_x = OFFSET_X + (1 * CELL_SIZE) + (CELL_SIZE // 2)
        p1_y = OFFSET_Y + (1 * CELL_SIZE) + (CELL_SIZE // 2)
        builder.add_thing(ThingType.PLAYER1_START, x=p1_x, y=p1_y, angle=0)

        p2_y = -(STANDS_INNER + (STANDS_THICKNESS // 2))
        builder.add_thing(ThingType.PLAYER2_START, x=0, y=p2_y, angle=90)

        acs = ACSBuilder()
        acs.add_include("zcommon.acs")

        acs.add_map_var("g_flash_timer", initial=0)
        acs.add_map_var("g_flash_state", initial=0)
        acs.add_map_var("g_flash_lump", var_type="str", initial='"S_BLACK"')
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

    int current_maze_size = {INITIAL_MAZE_SIZE};
    int current_start_tag = 0;
    int current_end_tag = 0;

    int g_start_x = 1;
    int g_start_y = 1;

    int visited[MAP_WIDTH * MAP_WIDTH];
    int stack[100];
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
        }}
    }}

    function void carve(int x, int y) {{
        int tag = get_tag(x, y);
        Floor_MoveToValue(tag, 2048, 0, 0);
        ChangeFloor(tag, "FLOOR4_8");
        Sector_SetDamage(tag, 0, 0);
        Light_ChangeToValue(tag, 255);
        if (x >= 0 && x < MAP_WIDTH && y >= 0 && y < MAP_WIDTH) {{
            int bit_idx = y * MAP_WIDTH + x;
            int chunk = bit_idx / 27;
            int shift = bit_idx % 27;
            maze_bits_chunk[chunk] = maze_bits_chunk[chunk] | (1 << shift);
        }}
    }}

    function int RingDistToX(int s_fix, int r_fix) {{
        while (s_fix < 0) s_fix += 8 * r_fix;
        s_fix = s_fix % (8 * r_fix);
        if (s_fix < 2 * r_fix) return r_fix - s_fix;
        if (s_fix < 4 * r_fix) return -r_fix;
        if (s_fix < 6 * r_fix) return (s_fix - 4 * r_fix) - r_fix;
        return r_fix;
    }}

    function int RingDistToY(int s_fix, int r_fix) {{
        while (s_fix < 0) s_fix += 8 * r_fix;
        s_fix = s_fix % (8 * r_fix);
        if (s_fix < 2 * r_fix) return r_fix;
        if (s_fix < 4 * r_fix) return r_fix - (s_fix - 2 * r_fix);
        if (s_fix < 6 * r_fix) return -r_fix;
        return (s_fix - 6 * r_fix) - r_fix;
    }}

    function int IsInFOV(int observer, int target) {{
        // Ubobstructed Line of Sight Check
        if (!CheckSight(observer, target, 0)) {{
            return FALSE;
        }}

        // Get 3D Coordinates (Adjusting for camera eye level and target center)
        int dx = GetActorX(target) - GetActorX(observer);
        int dy = GetActorY(target) - GetActorY(observer);

        int obsZ = GetActorZ(observer) + GetActorViewHeight(observer);
        int targZ = GetActorZ(target) + (GetActorProperty(target, APROP_Height) / 2);
        int dz = targZ - obsZ;

        // Horizontal Angle Check
        int angToTarget = VectorAngle(dx, dy);
        int observerAng = GetActorAngle(observer);
        int yawDiff = angToTarget - observerAng;

        while (yawDiff <= -0.5) yawDiff += 1.0;
        while (yawDiff > 0.5) yawDiff -= 1.0;
        if (yawDiff < 0) yawDiff = -yawDiff;

        // 0.125 fixed point = 45 degrees (90-degree horizontal FOV total)
        if (yawDiff > 0.125) {{
            return FALSE;
        }}

        // Vertical Pitch Check
        int xy_dist = VectorLength(dx, dy);
        int pitchToTarget = VectorAngle(xy_dist, dz);

        // VectorAngle returns standard math angles. We must invert it to match
        // ZDoom's pitch system where looking down is positive.
        if (pitchToTarget > 0.5) {{
            pitchToTarget -= 1.0;
        }}
        pitchToTarget = -pitchToTarget;

        int observerPitch = GetActorPitch(observer);
        int pitchDiff = pitchToTarget - observerPitch;

        while (pitchDiff <= -0.5) pitchDiff += 1.0;
        while (pitchDiff > 0.5) pitchDiff -= 1.0;
        if (pitchDiff < 0) pitchDiff = -pitchDiff;

        // Vertical FOV cone check. (0.10 roughly equals ~72 degrees vertical FOV,
        // which matches standard 4:3 displays with 90 horizontal FOV).
        if (pitchDiff > 0.10) {{
            return FALSE;
        }}

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

        for (i = 0; i < MAP_WIDTH * MAP_WIDTH; i++) visited[i] = 0;

        int stack_ptr = 0;
        int cx = sx;
        int cy = sy;
        int count_carved = 1;

        visited[cy * MAP_WIDTH + cx] = 1;
        stack[stack_ptr++] = (cx << 8) | cy;

        int neighbors[4];
        int n_count;
        int next_packed, nx, ny, wall_x, wall_y;
        int last_visited_tag = start_tag;
        int last_x = sx;
        int last_y = sy;

        while (count_carved < size && stack_ptr > 0) {{
            int packed = stack[stack_ptr-1];
            cx = packed >> 8;
            cy = packed & 0xFF;

            n_count = 0;

            if (cy + 2 < MAP_WIDTH && visited[(cy+2)*MAP_WIDTH + cx] == 0)
                neighbors[n_count++] = (cx << 8) | (cy+2);
            if (cy - 2 >= 0 && visited[(cy-2)*MAP_WIDTH + cx] == 0)
                neighbors[n_count++] = (cx << 8) | (cy-2);
            if (cx + 2 < MAP_WIDTH && visited[cy*MAP_WIDTH + (cx+2)] == 0)
                neighbors[n_count++] = ((cx+2) << 8) | cy;
            if (cx - 2 >= 0 && visited[cy*MAP_WIDTH + (cx-2)] == 0)
                neighbors[n_count++] = ((cx-2) << 8) | cy;

            if (n_count > 0) {{
                int r = Random(0, n_count-1);
                next_packed = neighbors[r];
                nx = next_packed >> 8;
                ny = next_packed & 0xFF;

                wall_x = (cx + nx) / 2;
                wall_y = (cy + ny) / 2;

                carve(wall_x, wall_y);
                carve(nx, ny);

                visited[ny * MAP_WIDTH + nx] = 1;
                count_carved += 2;
                last_visited_tag = get_tag(nx, ny);
                last_x = nx;
                last_y = ny;

                stack[stack_ptr++] = next_packed;
            }} else {{
                stack_ptr--;
            }}
        }}

        current_end_tag = last_visited_tag;
        lm_maze_size_global        = MAP_WIDTH;
        lm_goal_grid_x_global      = last_x;
        lm_goal_grid_y_global      = last_y;
        lm_maze_bits_0_global      = maze_bits_chunk[0];
        lm_maze_bits_1_global      = maze_bits_chunk[1];
        lm_maze_bits_2_global      = maze_bits_chunk[2];

        ChangeFloor(current_end_tag, "F_END");
        Sector_SetColor(current_end_tag, 255, 255, 255);
        Light_ChangeToValue(current_end_tag, 255);
        Sector_SetFade(current_end_tag, 0, 0, 0);
        //Print(s:"Level Up! Size: ", d:size);
    }}

    script "EndGame" (void) {{
        Delay(1);
        Exit_Normal(0);
    }}

    script 910 (int signal) {{
        if (signal < 1 || signal > 4) terminate;

        lm_flash_signal_global = signal;
        if (signal == 1) {{ g_flash_lump = "S_GREEN";  }}
        if (signal == 2) {{ g_flash_lump = "S_RED";    }}
        if (signal == 3) {{ g_flash_lump = "S_YELLOW"; }}
        if (signal == 4) {{ g_flash_lump = "S_BLUE";   }}

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

        while (TRUE) {{
            if (GetActorProperty(0, APROP_HEALTH) <= 0) {{
                ACS_NamedExecute("EndGame", 0);
                terminate;
            }}

            int x = GetActorX(0);
            int y = GetActorY(0);

            int offset = (MAP_WIDTH * CELL_SIZE) / 2;
            int gx = ((x >> 16) + offset) / CELL_SIZE;
            int gy = ((y >> 16) + offset) / CELL_SIZE;

            if (gx >= 0 && gx < MAP_WIDTH && gy >= 0 && gy < MAP_WIDTH) {{
                lm_p1_grid_x_global = gx;
                lm_p1_grid_y_global = gy;
                int my_tag = get_tag(gx, gy);

                if (my_tag == current_end_tag) {{
                    lm_levels_completed_global++;
                    current_maze_size += 2;
                    if (current_maze_size > MAX_TILES) current_maze_size = MAX_TILES;
                    if (current_maze_size < {INITIAL_MAZE_SIZE}) current_maze_size = {INITIAL_MAZE_SIZE};
                    lm_maze_size_global = MAP_WIDTH;

                    ACS_NamedExecute("GenerateMaze", 0, gx, gy, current_maze_size);
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
        }}
    }}

    script 3 ENTER {{
        if (PlayerNumber() != 1) terminate;

        Thing_ChangeTID(0, 1001);
        SetActorProperty(0, APROP_Health, 100);
        SetPlayerProperty(0, 1, 0); // PROP_FROZEN (Disables manual movement but allows weapons/looking)

        ClearInventory();
        GiveInventory("LMPistol",      1);
        GiveInventory("LMShotgun",     1);
        GiveInventory("LMChaingun",    1);
        GiveInventory("LMPlasmaRifle", 1);
        GiveInventory("Clip",   200);
        GiveInventory("Shell",   50);
        GiveInventory("Cell",   200);
        SetWeapon("LMPistol");

        int r_fix = PLATFORM_RADIUS << 16;
        int current_s = 5 * r_fix; // Start at middle of South edge

        while(TRUE) {{
            GiveInventory("Clip",  1);
            GiveInventory("Shell", 1);
            GiveInventory("Cell",  1);

            // Calculate optimal target on the square ring (opposite to player 1)
            int px = GetActorX(1000);
            int py = GetActorY(1000);
            int tx = -px;
            int ty = -py;
            int abs_tx = tx; if (abs_tx < 0) abs_tx = -abs_tx;
            int abs_ty = ty; if (abs_ty < 0) abs_ty = -abs_ty;

            int target_s = current_s; // fallback if centered

            if (abs_tx > 655360 || abs_ty > 655360) {{ // > 10 units away from center
                if (abs_tx > abs_ty) {{
                    ty = FixedMul(ty, FixedDiv(r_fix, abs_tx));
                    if (tx > 0) {{ target_s = r_fix + ty + 6 * r_fix; }}
                    else        {{ target_s = r_fix - ty + 2 * r_fix; }}
                }} else {{
                    tx = FixedMul(tx, FixedDiv(r_fix, abs_ty));
                    if (ty > 0) {{ target_s = r_fix - tx; }}
                    else        {{ target_s = tx + r_fix + 4 * r_fix; }}
                }}
            }}

            int p_len = 8 * r_fix;
            int diff = (target_s - current_s) % p_len;
            if (diff < 0) diff += p_len;
            if (diff > p_len / 2) diff -= p_len;

            int max_speed = 32 << 16; // maximum orbital units per tic
            int move_amt = diff / 10; // smooth ease-in/ease-out
            if (move_amt > max_speed) move_amt = max_speed;
            if (move_amt < -max_speed) move_amt = -max_speed;

            current_s = (current_s + move_amt) % p_len;
            if (current_s < 0) current_s += p_len;

            int obs_x = RingDistToX(current_s, r_fix);
            int obs_y = RingDistToY(current_s, r_fix);
            int obs_z = GetActorZ(0); // stay grounded

            SetActorPosition(0, obs_x, obs_y, obs_z, 0);

            // Always look at Player 1 (TID 1000)
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
        }}
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
    LavaMazeScenario().generate("examples/benchmark/output/lava_maze.wad")
