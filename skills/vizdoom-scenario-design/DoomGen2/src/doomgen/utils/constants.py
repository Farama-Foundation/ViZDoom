"""Doom engine constants and scale factors."""

# Doom units per real-world meter (approximately)
UNITS_PER_METER = 32

# Standard grid sizes for snapping
GRID_SIZE_FINE = 1
GRID_SIZE_SMALL = 8
GRID_SIZE_MEDIUM = 16
GRID_SIZE_LARGE = 32
GRID_SIZE_HUGE = 64

# Default grid for snapping
DEFAULT_GRID_SIZE = 8

# Coordinate limits (Doom uses signed 16-bit for vanilla)
MAP_MIN_COORD = -32768
MAP_MAX_COORD = 32767

# Safe bounds (leaving margin for nodes/BSP)
SAFE_MIN_COORD = -16384
SAFE_MAX_COORD = 16384

# Maximum number of various structures (vanilla limits)
MAX_VERTICES = 65535
MAX_LINEDEFS = 65535
MAX_SIDEDEFS = 65535
MAX_SECTORS = 65535
MAX_THINGS = 65535

# Standard heights
DEFAULT_FLOOR_HEIGHT = 0
DEFAULT_CEILING_HEIGHT = 128

# Player height (for clearance calculations)
PLAYER_HEIGHT = 56

# Minimum passable height
MIN_PASSABLE_HEIGHT = 56

# Step-up height (maximum step player can climb)
MAX_STEP_HEIGHT = 24

# Minimum corridor width (player diameter is ~32)
MIN_CORRIDOR_WIDTH = 64

# Recommended corridor width
DEFAULT_CORRIDOR_WIDTH = 96

# Player collision radius
PLAYER_RADIUS = 16

# Minimum sector area (avoid degenerate sectors)
MIN_SECTOR_AREA = 64

# Light level range
MIN_LIGHT_LEVEL = 0
MAX_LIGHT_LEVEL = 255

# Common light levels
LIGHT_PITCH_BLACK = 0
LIGHT_VERY_DARK = 64
LIGHT_DARK = 96
LIGHT_DIM = 128
LIGHT_NORMAL = 160
LIGHT_BRIGHT = 192
LIGHT_VERY_BRIGHT = 224
LIGHT_FULLBRIGHT = 255

# Linedef flag bits
LF_BLOCKING = 1          # Blocks players and monsters
LF_BLOCKMONSTERS = 2     # Blocks monsters only
LF_TWOSIDED = 4          # Two-sided line
LF_UPPERUNPEGGED = 8     # Upper texture unpegged
LF_LOWERUNPEGGED = 16    # Lower texture unpegged
LF_SECRET = 32           # Secret (shows as 1-sided on automap)
LF_BLOCKSOUND = 64       # Blocks sound propagation
LF_DONTDRAW = 128        # Never appears on automap
LF_MAPPED = 256          # Already on automap

# Common flag combinations
FLAGS_SOLID_WALL = LF_BLOCKING
FLAGS_IMPASSABLE = LF_BLOCKING
FLAGS_MONSTER_BARRIER = LF_BLOCKMONSTERS
FLAGS_TWO_SIDED = LF_TWOSIDED
FLAGS_TWO_SIDED_PASSABLE = LF_TWOSIDED  # No LF_BLOCKING
FLAGS_SECRET_WALL = LF_SECRET | LF_BLOCKING

# Thing skill appearance flags
TF_SKILL_EASY = 1        # Appears on skills 1-2
TF_SKILL_MEDIUM = 2      # Appears on skill 3
TF_SKILL_HARD = 4        # Appears on skills 4-5
TF_DEAF = 8              # Deaf/ambush flag
TF_MULTIPLAYER = 16      # Multiplayer only

# Common flag combinations
THING_ALL_SKILLS = TF_SKILL_EASY | TF_SKILL_MEDIUM | TF_SKILL_HARD
THING_HARD_ONLY = TF_SKILL_HARD
THING_EASY_MEDIUM = TF_SKILL_EASY | TF_SKILL_MEDIUM
THING_AMBUSH = TF_SKILL_EASY | TF_SKILL_MEDIUM | TF_SKILL_HARD | TF_DEAF

# Wall textures (Doom 2)
TEX_STARTAN2 = "STARTAN2"
TEX_BROWN1 = "BROWN1"
TEX_BROWNGRN = "BROWNGRN"
TEX_STONE2 = "STONE2"
TEX_METAL = "METAL"
TEX_SUPPORT2 = "SUPPORT2"
TEX_DOORSTOP = "DOORSTOP"
TEX_DOORTRAK = "DOORTRAK"

# Floor textures
TEX_FLOOR4_8 = "FLOOR4_8"
TEX_FLOOR5_1 = "FLOOR5_1"
TEX_FLAT5_4 = "FLAT5_4"
TEX_NUKAGE1 = "NUKAGE1"

# Ceiling textures
TEX_CEIL3_5 = "CEIL3_5"
TEX_FLAT1 = "FLAT1"
TEX_F_SKY1 = "F_SKY1"

# Game tics per second
TICS_PER_SECOND = 35

# Common timing values
DOOR_WAIT_TICS = 150      # ~4.3 seconds
DOOR_SPEED = 16
LIFT_WAIT_TICS = 105      # ~3 seconds
LIFT_SPEED = 32

# Reserved script number ranges
SCRIPT_MIN = 1
SCRIPT_MAX = 999
SCRIPT_SYSTEM_START = 900  # Reserve 900-999 for system scripts
