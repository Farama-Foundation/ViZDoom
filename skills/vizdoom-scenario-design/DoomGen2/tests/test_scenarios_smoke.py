"""End-to-end smoke test: every example generates a structurally valid WAD."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from doomgen.utils.constants import MAP_MAX_COORD, MAP_MIN_COORD
from doomgen.utils.paths import resolve_compiler
from doomgen.utils.validation import validate_wad

requires_acc = pytest.mark.skipif(
    resolve_compiler("acc") is None,
    reason="no runnable acc compiler found (install acc on PATH on Linux)",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def test_geometry_and_placement_example(tmp_path: Path) -> None:
    from examples.geometry_and_placement import build_geometry_and_placement

    out = tmp_path / "geometry_and_placement.wad"
    build_geometry_and_placement(out)
    check_wad_structure(out)


@requires_acc
@pytest.mark.parametrize(
    "module_name,class_name,config",
    [
        ("examples.key_maze", "KeyMazeScenario", {}),
        ("examples.coop_relay", "CoopRelayScenario", {}),
        ("examples.simple_tag", "SimpleTagScenario", {"audio": False}),
        ("examples.simple_tag", "SimpleTagScenario", {"audio": True}),
        ("examples.deathmatch_arena", "DeathmatchArenaScenario", {}),
    ],
    ids=["key_maze", "coop_relay", "simple_tag", "simple_tag_audio", "deathmatch_arena"],
)
def test_example_scenario_generates_valid_wad(
    module_name: str, class_name: str, config: dict, tmp_path: Path
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    scenario = getattr(module, class_name)(config)
    out = tmp_path / f"{module_name.rsplit('.', 1)[-1]}.wad"
    scenario.generate(str(out))
    assert out.exists()
    check_wad_structure(out)


@requires_acc
def test_deathmatch_arena_batch_generation(tmp_path: Path) -> None:
    from examples.deathmatch_arena import generate_batch

    wads = generate_batch(tmp_path / "batch")
    assert len(wads) == 4
    for wad in wads:
        check_wad_structure(wad)
