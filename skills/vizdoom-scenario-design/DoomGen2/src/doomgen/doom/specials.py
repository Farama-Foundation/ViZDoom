"""Doom linedef actions and sector specials."""

from enum import IntEnum, IntFlag
from typing import Optional


class LinedefFlags(IntFlag):
    """Linedef flags controlling rendering and collision."""
    NONE = 0
    BLOCKING = 1           # Blocks players and monsters
    BLOCK_MONSTERS = 2     # Blocks monsters only
    TWO_SIDED = 4          # Renders both sides
    UPPER_UNPEGGED = 8     # Upper texture anchored to ceiling
    LOWER_UNPEGGED = 16    # Lower texture anchored to floor
    SECRET = 32            # Shows as solid on automap
    BLOCK_SOUND = 64       # Blocks sound propagation
    NOT_ON_MAP = 128       # Hidden on automap
    ALREADY_ON_MAP = 256   # Always visible on automap


class DoorAction(IntEnum):
    """Door linedef actions. Format: Trigger + Behavior."""
    # Manual doors (D = "use" on the door linedef itself)
    DR_OPEN_WAIT_CLOSE = 1          # Repeatable manual door
    D1_OPEN_STAY = 31               # One-shot manual door, stays open

    # Remote doors (S = switch, W = walk-over, G = gun/shoot)
    SR_OPEN_WAIT_CLOSE = 63         # Repeatable switch door
    S1_OPEN_WAIT_CLOSE = 29         # One-shot switch door
    WR_OPEN_WAIT_CLOSE = 86         # Repeatable walk-trigger door
    W1_OPEN_WAIT_CLOSE = 4          # One-shot walk-trigger door

    # Fast doors (Doom II)
    DR_OPEN_WAIT_CLOSE_FAST = 117   # Fast repeatable manual door
    D1_OPEN_STAY_FAST = 118         # Fast one-shot manual door

    # Keyed doors (D1 = one-shot, stay open)
    D1_BLUE_DOOR = 32
    D1_RED_DOOR = 33
    D1_YELLOW_DOOR = 34
    DR_BLUE_DOOR = 26               # Repeatable blue key door
    DR_RED_DOOR = 28                # Repeatable red key door
    DR_YELLOW_DOOR = 27             # Repeatable yellow key door


class LiftAction(IntEnum):
    """Lift/platform linedef actions. Lifts lower floor to lowest neighbor then raise back."""
    # Standard lifts (down, wait, up)
    SR_LIFT = 62                    # Repeatable switch-activated lift
    S1_LIFT = 21                    # One-shot switch lift
    WR_LIFT = 88                    # Repeatable walk-trigger lift
    W1_LIFT = 10                    # One-shot walk-trigger lift

    # Fast/turbo lifts
    SR_LIFT_FAST = 123              # Fast repeatable lift
    WR_LIFT_FAST = 120              # Fast walk-trigger lift

    # Perpetual platforms (continuous up/down motion)
    SR_PERPETUAL_RAISE = 87         # Start perpetual raise
    S1_STOP_PERPETUAL = 89          # Stop perpetual platform


class FloorAction(IntEnum):
    """Floor movement actions."""
    S1_FLOOR_LOWER_TO_LOWEST = 23
    SR_FLOOR_LOWER_TO_LOWEST = 60
    W1_FLOOR_LOWER_TO_LOWEST = 38
    WR_FLOOR_LOWER_TO_LOWEST = 36

    S1_FLOOR_RAISE_TO_NEXT = 18
    SR_FLOOR_RAISE_TO_NEXT = 69
    W1_FLOOR_RAISE_TO_NEXT = 119

    S1_FLOOR_RAISE_24 = 15
    W1_FLOOR_RAISE_24 = 58

    S1_FLOOR_RAISE_TO_CEILING = 20


class CeilingAction(IntEnum):
    """Ceiling movement actions."""
    S1_CEILING_LOWER_TO_FLOOR = 43
    SR_CEILING_LOWER_TO_FLOOR = 44
    W1_CEILING_LOWER_TO_FLOOR = 41
    WR_CEILING_LOWER_TO_FLOOR = 72


class CrusherAction(IntEnum):
    """Crusher ceiling actions."""
    W1_CRUSHER_SLOW = 6
    WR_CRUSHER_SLOW = 73
    S1_CRUSHER_SLOW = 49
    SR_CRUSHER_SLOW = 77

    W1_CRUSHER_FAST = 25
    WR_CRUSHER_FAST = 74

    W1_CRUSHER_STOP = 57
    S1_CRUSHER_STOP = 168

    W1_CRUSHER_SILENT = 141


class PolyobjAction(IntEnum):
    """Hexen/ZDoom polyobject actions."""
    START_LINE = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    MOVE = 4
    EXPLICIT_LINE = 5
    MOVE_TIMES8 = 6
    DOOR_SWING = 7
    DOOR_SLIDE = 8
    OR_ROTATE_LEFT = 10
    OR_ROTATE_RIGHT = 11
    OR_MOVE = 12
    OR_MOVE_TIMES8 = 13
    OR_MOVE_TO = 14
    OR_MOVE_TO_SPOT = 15
    STOP = 16


class StairAction(IntEnum):
    """Stair building actions."""
    S1_STAIRS_8 = 7                 # Build stairs, 8 unit steps
    W1_STAIRS_8 = 8
    S1_STAIRS_16 = 127              # Build stairs, 16 unit steps (fast, crushing)
    W1_STAIRS_16_FAST = 100


class TeleportAction(IntEnum):
    """Teleporter actions. Target is thing type 14 in tagged sector."""
    W1_TELEPORT = 39                # One-shot walk teleport
    WR_TELEPORT = 97                # Repeatable walk teleport
    SR_TELEPORT = 195               # Switch teleport (Boom)

    W1_TELEPORT_MONSTER_ONLY = 126  # Monsters only
    WR_TELEPORT_MONSTER_ONLY = 125

    W1_SILENT_TELEPORT = 207        # Silent teleport (Boom)
    WR_SILENT_TELEPORT = 208


class ExitAction(IntEnum):
    """Level exit actions."""
    S1_EXIT = 11                    # Switch exit to next level
    W1_EXIT = 52                    # Walk exit
    G1_EXIT = 197                   # Gun/shoot exit (Boom)

    S1_SECRET_EXIT = 51             # Switch secret exit
    W1_SECRET_EXIT = 124            # Walk secret exit


class LightAction(IntEnum):
    """Lighting effects."""
    W1_LIGHT_TO_HIGHEST = 13        # Light to highest adjacent
    WR_LIGHT_TO_HIGHEST = 81
    S1_LIGHT_TO_HIGHEST = 138

    W1_LIGHT_TO_LOWEST = 35         # Light to lowest adjacent
    WR_LIGHT_TO_LOWEST = 79

    W1_LIGHT_TO_255 = 12            # Maximum brightness
    WR_LIGHT_TO_255 = 80

    W1_LIGHT_TO_35 = 104            # Minimum brightness
    S1_LIGHT_BLINK = 17             # Start blinking


class ScrollerAction(IntEnum):
    """Scrolling texture effects."""
    SCROLL_WALL_LEFT = 48           # Scroll first sidedef left
    SCROLL_WALL_RIGHT = 85          # Scroll first sidedef right (Boom)
    SCROLL_BY_OFFSETS = 255         # Scroll using sidedef offsets (Boom)


class SectorSpecial(IntEnum):
    """Sector special types (sector type field, not linedef)."""
    NORMAL = 0
    BLINK_RANDOM = 1
    BLINK_HALF_SEC = 2
    BLINK_1_SEC = 3
    DAMAGE_20_AND_BLINK = 4
    DAMAGE_10 = 5
    DAMAGE_5 = 7
    OSCILLATE_LIGHT = 8
    SECRET = 9
    DOOR_CLOSE_30_SEC = 10
    DAMAGE_20_END = 11
    BLINK_SYNC_1_SEC = 12
    BLINK_SYNC_HALF = 13
    DOOR_OPEN_5_MIN = 14
    DAMAGE_20 = 16
    FLICKER = 17


# Trigger type descriptions for documentation
TRIGGER_TYPES = {
    'W1': 'Walk over linedef once',
    'WR': 'Walk over linedef (repeatable)',
    'S1': 'Switch/use once',
    'SR': 'Switch/use (repeatable)',
    'G1': 'Gun/shoot once',
    'GR': 'Gun/shoot (repeatable)',
    'D1': 'Door push once (manual door)',
    'DR': 'Door push (repeatable manual door)',
}
