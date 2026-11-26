"""Doom 2 thing type definitions (enemies, items, decorations)."""

from enum import IntEnum


class ThingType(IntEnum):
    """Standard Doom 2 thing type IDs."""

    # Player starts
    PLAYER1_START = 1
    PLAYER2_START = 2
    PLAYER3_START = 3
    PLAYER4_START = 4
    PLAYER5_START = 4001
    PLAYER6_START = 4002
    PLAYER7_START = 4003
    PLAYER8_START = 4004
    DEATHMATCH_START = 11

    # Enemies - Easy/Medium
    ZOMBIEMAN = 3004
    SHOTGUN_GUY = 9
    CHAINGUNNER = 65
    IMP = 3001
    DEMON = 3002
    SPECTRE = 58
    LOST_SOUL = 3006

    # Enemies - Hard
    CACODEMON = 3005
    HELL_KNIGHT = 69
    BARON_OF_HELL = 3003
    REVENANT = 66
    MANCUBUS = 67
    ARACHNOTRON = 68
    PAIN_ELEMENTAL = 71
    ARCH_VILE = 64

    # Bosses
    SPIDER_MASTERMIND = 7
    CYBERDEMON = 16
    ICON_OF_SIN = 89

    # Weapons
    CHAINSAW = 2005
    SHOTGUN = 2001
    SUPER_SHOTGUN = 82
    CHAINGUN = 2002
    ROCKET_LAUNCHER = 2003
    PLASMA_RIFLE = 2004
    BFG9000 = 2006

    # Ammo - Small
    CLIP = 2007
    SHELLS = 2008
    ROCKET = 2010
    CELL = 2047

    # Ammo - Large
    BOX_OF_BULLETS = 2048
    BOX_OF_SHELLS = 2049
    BOX_OF_ROCKETS = 2046
    ENERGY_CELL_PACK = 17

    # Health
    STIMPACK = 2011
    MEDIKIT = 2012
    HEALTH_BONUS = 2014
    SOULSPHERE = 2013
    MEGASPHERE = 83

    # Armor
    ARMOR_BONUS = 2015
    GREEN_ARMOR = 2018
    BLUE_ARMOR = 2019

    # Powerups
    BERSERK = 2023
    INVULNERABILITY = 2022
    PARTIAL_INVISIBILITY = 2024
    COMPUTER_MAP = 2026
    LIGHT_AMP_GOGGLES = 2045
    RADIATION_SUIT = 2025

    # Keys
    BLUE_KEYCARD = 5
    YELLOW_KEYCARD = 6
    RED_KEYCARD = 13
    BLUE_SKULL_KEY = 40
    YELLOW_SKULL_KEY = 39
    RED_SKULL_KEY = 38

    # Decorations - Small
    CANDLE = 34
    CANDELABRA = 35
    FLOOR_LAMP = 2028
    TECH_LAMP = 85
    TECH_LAMP2 = 86
    TALL_GREEN_PILLAR = 30
    SHORT_GREEN_PILLAR = 31
    TALL_RED_PILLAR = 32
    SHORT_RED_PILLAR = 33

    # Decorations - Gore
    BLOODY_MESS = 10
    BLOODY_MESS2 = 12
    DEAD_PLAYER = 15
    DEAD_ZOMBIE = 18
    DEAD_DEMON = 21
    HANGING_VICTIM = 49
    HANGING_PAIR = 50
    HANGING_LEG = 51

    # Teleport
    TELEPORT_LANDING = 14

    # ZDoom/GZDoom special actors
    MAP_SPOT = 9001  # Invisible spawn point used by ACS SpawnSpot.


# Convenience groupings
PLAYER_STARTS = [
    ThingType.PLAYER1_START,
    ThingType.PLAYER2_START,
    ThingType.PLAYER3_START,
    ThingType.PLAYER4_START,
    ThingType.PLAYER5_START,
    ThingType.PLAYER6_START,
    ThingType.PLAYER7_START,
    ThingType.PLAYER8_START,
]

EASY_ENEMIES = [
    ThingType.ZOMBIEMAN,
    ThingType.SHOTGUN_GUY,
    ThingType.IMP,
]

MEDIUM_ENEMIES = [
    ThingType.CHAINGUNNER,
    ThingType.DEMON,
    ThingType.SPECTRE,
    ThingType.LOST_SOUL,
]

HARD_ENEMIES = [
    ThingType.CACODEMON,
    ThingType.HELL_KNIGHT,
    ThingType.REVENANT,
    ThingType.MANCUBUS,
    ThingType.ARACHNOTRON,
]

BOSS_ENEMIES = [
    ThingType.BARON_OF_HELL,
    ThingType.PAIN_ELEMENTAL,
    ThingType.ARCH_VILE,
    ThingType.SPIDER_MASTERMIND,
    ThingType.CYBERDEMON,
]

ALL_WEAPONS = [
    ThingType.CHAINSAW,
    ThingType.SHOTGUN,
    ThingType.SUPER_SHOTGUN,
    ThingType.CHAINGUN,
    ThingType.ROCKET_LAUNCHER,
    ThingType.PLASMA_RIFLE,
    ThingType.BFG9000,
]

ALL_KEYS = [
    ThingType.BLUE_KEYCARD,
    ThingType.YELLOW_KEYCARD,
    ThingType.RED_KEYCARD,
    ThingType.BLUE_SKULL_KEY,
    ThingType.YELLOW_SKULL_KEY,
    ThingType.RED_SKULL_KEY,
]

HEALTH_ITEMS = [
    ThingType.STIMPACK,
    ThingType.MEDIKIT,
    ThingType.HEALTH_BONUS,
    ThingType.SOULSPHERE,
    ThingType.MEGASPHERE,
]

ARMOR_ITEMS = [
    ThingType.ARMOR_BONUS,
    ThingType.GREEN_ARMOR,
    ThingType.BLUE_ARMOR,
]


def get_thing_radius(thing_type: ThingType) -> int:
    """Return an approximate collision radius in map units."""
    large_enemies = [
        ThingType.MANCUBUS,
        ThingType.ARACHNOTRON,
        ThingType.SPIDER_MASTERMIND,
        ThingType.CYBERDEMON,
    ]

    medium_enemies = [
        ThingType.CACODEMON,
        ThingType.PAIN_ELEMENTAL,
        ThingType.BARON_OF_HELL,
        ThingType.HELL_KNIGHT,
        ThingType.REVENANT,
    ]

    if thing_type in large_enemies:
        return 48
    elif thing_type in medium_enemies:
        return 31
    elif thing_type in [ThingType.DEMON, ThingType.SPECTRE]:
        return 30
    else:
        return 16
