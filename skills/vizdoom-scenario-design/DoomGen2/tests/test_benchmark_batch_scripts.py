from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = REPO_ROOT / "DoomGen" / "examples" / "benchmark" / "batches"

if str(BATCH_DIR) not in sys.path:
    sys.path.insert(0, str(BATCH_DIR))

from _batch_utils import count_axis_combinations, iter_axis_configs


FULL_BATCH_MODULES = [
    "generate_stag_hunt_arena_batch",
    "generate_rhythm_sync_batch",
    "generate_foraging_commons_batch",
    "generate_coop_puzzle_batch",
    "generate_platform_chain_batch",
    "generate_armory_siege_batch",
    "generate_coop_health_gathering_batch",
    "generate_lava_pit_batch",
    "generate_smart_enemies_batch",
    "generate_dumb_enemies_batch",
    "generate_ammo_carrier_batch",
    "generate_stealth_labyrinth_batch",
    "generate_lava_maze_batch",
]

ALL_BATCH_MODULES = FULL_BATCH_MODULES + [
    "generate_platform_chain_batch_small",
    "generate_armory_siege_batch_small",
    "generate_lava_maze_batch_small",
]


@pytest.mark.parametrize("module_name", FULL_BATCH_MODULES)
def test_full_batch_counts_match_declared_sizes(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert count_axis_combinations(module.AXIS_OPTIONS) == module.EXPECTED_COUNT


@pytest.mark.parametrize("module_name", ALL_BATCH_MODULES)
def test_batch_axes_only_use_real_scenario_config_keys(module_name: str) -> None:
    module = importlib.import_module(module_name)
    default_cfg_keys = set(module.SCENARIO_CLS().config.keys())

    seen = 0
    for config, _axis_choices in iter_axis_configs(module.AXIS_OPTIONS):
        seen += 1
        assert set(config).issubset(default_cfg_keys)
        module.SCENARIO_CLS(config, name="test_batch_cfg")

    assert seen == module.EXPECTED_COUNT
