"""End-to-end smoke test: every benchmark scenario generates a structurally valid WAD."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from doomgen.utils.constants import MAP_MAX_COORD, MAP_MIN_COORD
from doomgen.utils.validation import validate_wad

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# (module path, scenario class name)
SCENARIOS = [
    ("examples.benchmark.deathmatch_duel", "DeathmatchDuelScenario"),
    ("examples.benchmark.deathmatch_duel_1000", "DeathmatchDuelScenario"),
    ("examples.benchmark.hide_and_seek_1000", "HideAndSeekScenario"),
    ("examples.benchmark.stag_hunt_arena", "StagHuntArenaScenario"),
    ("examples.benchmark.coop_health_gathering", "CoopHealthGatheringScenario"),
    ("examples.benchmark.dumb_enemies", "DumbEnemiesScenario"),
    ("examples.benchmark.smart_enemies", "SmartEnemiesScenario"),
    ("examples.benchmark.simple_tag", "SimpleTagScenario"),
    ("examples.benchmark.simple_tag_audio", "SimpleTagAudioScenario"),
    ("examples.misc.resource_greed", "ResourceGreedScenario"),
    ("examples.benchmark.ammo_carrier", "AmmoCarrierScenario"),
    ("examples.benchmark.armory_siege", "ArmorySiegeScenario"),
    ("examples.benchmark.foraging_commons", "ForagingCommonsScenario"),
    ("examples.benchmark.coop_puzzle", "CoopPuzzleScenario"),
    ("examples.benchmark.coop_relay", "CoopRelayScenario"),
    ("examples.benchmark.rhythm_sync", "RhythmSyncScenario"),
    ("examples.rhythm_sync_fadeto", "RhythmSyncScenario"),
    ("examples.benchmark.lavapit", "LavaPitScenario"),
    ("examples.benchmark.platform_chain", "PlatformChainScenario"),
    ("examples.benchmark.platform_chain_easy", "PlatformChainScenario"),
    ("examples.benchmark.lava_maze", "LavaMazeScenario"),
    ("examples.benchmark.lava_maze_simple", "LavaMazeSimpleScenario"),
    ("examples.benchmark.stealth_labyrinth", "StealthLabyrinthScenario"),
    ("examples.benchmark.text_maze", "TextMazeScenario"),
    ("examples.benchmark.simple_communication", "SimpleCommunicationScenario"),
]


def check_wad_structure(path: Path) -> None:
    """Assert Doom-level invariants on the first map of a WAD."""
    from omg import WAD, MapEditor

    ok, issues = validate_wad(path)
    assert ok, issues

    wad = WAD(str(path))
    name = next(iter(wad.maps))
    editor = MapEditor(wad.maps[name])

    assert editor.linedefs, "map has no linedefs"
    for v in editor.vertexes:
        assert MAP_MIN_COORD <= v.x <= MAP_MAX_COORD
        assert MAP_MIN_COORD <= v.y <= MAP_MAX_COORD

    n_sides = len(editor.sidedefs)
    for ld in editor.linedefs:
        a, b = editor.vertexes[ld.vx_a], editor.vertexes[ld.vx_b]
        assert (a.x, a.y) != (b.x, b.y), "zero-length linedef"
        assert 0 <= ld.front < n_sides, "linedef without a front sidedef"
        assert ld.back == 0xFFFF or ld.back == -1 or 0 <= ld.back < n_sides

    referenced = {sd.sector for sd in editor.sidedefs}
    assert referenced == set(range(len(editor.sectors))), "unreferenced sector"


@pytest.mark.parametrize("module_name,class_name", SCENARIOS,
                         ids=[f"{m.rsplit('.', 1)[-1]}" for m, _ in SCENARIOS])
def test_scenario_generates_valid_wad(module_name: str, class_name: str, tmp_path: Path) -> None:
    module = importlib.import_module(module_name)
    scenario = getattr(module, class_name)()
    out = tmp_path / f"{module_name.rsplit('.', 1)[-1]}.wad"
    scenario.generate(str(out))
    assert out.exists()
    check_wad_structure(out)
