"""
Observation wrappers for ViZDoom Gymnasium environments.
"""

from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, SupportsInt, cast

import gymnasium as gym
import numpy as np

import vizdoom.vizdoom as vzd

from .base_gymnasium_env import VizdoomEnv


INVISIBLE_CATEGORIES = [
    "Internal",
    "ScriptThing",
    "DynamicLight",
    "Power",
    "Token",
    "MapSpot",
]
DEFAULT_CATEGORY_MAPPING = MappingProxyType(
    {
        # 1: Wall
        "Wall": 1,
        # 2: Progression / special pickups
        "Key": 2,
        "Artifact": 2,
        "QuestItem": 2,
        "PuzzleItem": 2,
        # 3: Temporary power modifiers
        "Powerup": 3,
        # 4: Hazards / damage-causing objects
        "Explosive": 4,
        "Hazard": 4,
        # 5: Transient visual/audio effects and remains
        "SFX": 5,
        "Gibs": 5,
        "Gore": 5,
        # 6: Static world props / scenery
        "Vegetation": 6,
        "Decoration": 6,
        "LightSource": 6,
        "Bridge": 6,
        # 7: Interactive / destructible world objects
        "Breakable": 7,
        "InteractiveObject": 7,
        # 8-10: Characters / agents
        "Self": 8,
        "Player": 9,
        "Monster": 10,
        # 11-12: Armor / health
        "Armor": 11,
        "Health": 12,
        # 13: Ammo
        "Ammo": 13,
        # 14: Weapons
        "Weapon": 14,
    }
)


def _copy_category_label_mapping(
    mapping: Any, is_category: bool = False
) -> Mapping[str, int]:
    if mapping is None:
        return DEFAULT_CATEGORY_MAPPING if is_category else {}
    if not isinstance(mapping, Mapping):
        raise ValueError("Not a valid mapping")
    if is_category:
        unknown = set(mapping.keys()) - set(vzd.get_default_categories())
        if unknown:
            raise ValueError(f"Unknown categories: {sorted(unknown)}")
    if not all(
        isinstance(v, SupportsInt) and 0 <= int(v) <= 255 for v in mapping.values()
    ):
        raise ValueError("Label values must be unsigned 8-bit integer")
    return dict(mapping)


class SegmentationWrapper(gym.ObservationWrapper):
    """Segmentation Wrapper"""

    def __init__(
        self,
        env: VizdoomEnv,
        category_mapping: Optional[Dict[str, int]] = None,
        overrides: Optional[Dict[str, int]] = None,
    ):
        super().__init__(env)
        if "labels" not in env.observation_space.spaces:
            raise ValueError(
                "SegmentationColormapWrapper requires a 'labels' observation; "
                "set labels_buffer_enabled in the ViZDoom config."
            )
        self._category2label = _copy_category_label_mapping(
            category_mapping, is_category=True
        )
        self._name2label = _copy_category_label_mapping(overrides, is_category=False)

        h, w, _ = env.observation_space["labels"].shape
        spaces = dict(env.observation_space.spaces)
        spaces["segmentation"] = gym.spaces.Box(0, 255, (h, w), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)
        self.empty_obs = np.zeros((h, w), dtype=np.uint8)

    def observation(self, observation: Dict[str, Any]):
        state = cast(VizdoomEnv, self.env.unwrapped).state
        if state is None:
            return observation | {"segmentation": self.empty_obs.copy()}

        labels_arr = np.zeros_like(state.labels_buffer)
        wall_label = self._category2label.get("Wall")
        if wall_label is not None:
            labels_arr[state.labels_buffer == 1] = wall_label
        for label in state.labels or []:
            label_val = self._name2label.get(label.object_name)
            if label_val is None:
                label_val = self._category2label.get(label.object_category)
            if label_val is not None:
                labels_arr[state.labels_buffer == label.value] = label_val

        return observation | {"segmentation": labels_arr}
