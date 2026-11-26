"""
Doom sector and wall configuration.

Dataclasses for configuring Doom sectors and wall textures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SectorConfig:
    """Sector properties: heights, textures, lighting."""
    floor_height: int = 0
    ceiling_height: int = 128
    floor_texture: str = "FLOOR4_8"
    ceiling_texture: str = "CEIL3_5"
    light_level: int = 160
    special: int = 0
    tag: int = 0


@dataclass
class WallConfig:
    """Wall texture configuration."""
    upper_texture: str = "-"
    middle_texture: str = "STARTAN2"
    lower_texture: str = "-"
    x_offset: int = 0
    y_offset: int = 0


@dataclass
class PrefabDoomConfig:
    """
    Complete Doom configuration for a prefab.

    Encapsulates all Doom-specific settings needed to apply a prefab
    to a sector: heights, textures, linedef actions, and flags.
    """
    sector_config: SectorConfig
    linedef_action: int = 0
    linedef_flags: int = 0
    sector_tag: int = 0

    # Texture overrides for linedefs
    upper_texture: Optional[str] = None
    lower_texture: Optional[str] = None
    middle_texture: Optional[str] = None
