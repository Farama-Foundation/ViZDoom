"""Doom-specific structures and WAD helpers."""

from doomgen.doom.config import (
    SectorConfig,
    WallConfig,
    PrefabDoomConfig,
)

from doomgen.doom.wad import WADWriter, run_zdbsp

from doomgen.doom.translator import MapTranslator

from doomgen.doom.things import ThingType

from doomgen.doom.tags import TagRegistry

from doomgen.doom.specials import (
    LinedefFlags,
    DoorAction,
    LiftAction,
    TeleportAction,
    FloorAction,
    CeilingAction,
    CrusherAction,
    StairAction,
    ExitAction,
    LightAction,
    SectorSpecial,
    PolyobjAction,
)

from doomgen.doom.polyobjects import (
    PolyobjectBuilder,
    PolyobjectConfig,
)

__all__ = [
    "SectorConfig",
    "WallConfig",
    "PrefabDoomConfig",
    "WADWriter",
    "run_zdbsp",
    "MapTranslator",
    "ThingType",
    "TagRegistry",
    "LinedefFlags",
    "DoorAction",
    "LiftAction",
    "TeleportAction",
    "FloorAction",
    "CeilingAction",
    "CrusherAction",
    "StairAction",
    "ExitAction",
    "LightAction",
    "SectorSpecial",
    "PolyobjAction",
    "PolyobjectBuilder",
    "PolyobjectConfig",
]
