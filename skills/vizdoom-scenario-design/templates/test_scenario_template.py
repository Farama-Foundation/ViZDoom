"""Regression test for one ViZDoom scenario, wrapping ``validate_scenario.validate``.

Copy to ``tests/test_<scenario>_scenario.py``, edit the constants below to match the
scenario spec (``templates/scenario_spec.md``) and run ``pytest -s tests/test_<scenario>_scenario.py``.
Set ``VIZDOOM_SKILL_DIR`` if the skill folder is not found automatically.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest


# --- edit for your scenario -------------------------------------------------
CFG = "scenarios/simple_tag.cfg"  # relative to the repository root, or absolute
PLAYERS = 4  # networked players (1 = single player)
EPISODES = 3  # resets to probe
EXPECT = ["USER50=0", "USER51=0", "USER52=0"]  # must hold right after every reset
EXPECT_IDLE = ["USER50=0"]  # must hold while all players stand still
IDLE_STEPS = 16
CHECK_TIMEOUT = False  # True: run every episode to its end (slower)
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()


def _find_skill_dir() -> Path:
    if "VIZDOOM_SKILL_DIR" in os.environ:
        return Path(os.environ["VIZDOOM_SKILL_DIR"])
    for parent in HERE.parents:
        if (parent / "scripts" / "validate_scenario.py").is_file():
            return parent
        candidate = parent / "skills" / "vizdoom-scenario-design"
        if (candidate / "scripts" / "validate_scenario.py").is_file():
            return candidate
    raise FileNotFoundError(
        "vizdoom-scenario-design skill not found; set VIZDOOM_SKILL_DIR"
    )


def _cfg_path() -> Path:
    cfg = Path(CFG)
    if cfg.is_absolute():
        return cfg
    for parent in HERE.parents:
        if (parent / cfg).is_file():
            return parent / cfg
    return Path.cwd() / cfg


def _load_validator():
    path = _find_skill_dir() / "scripts" / "validate_scenario.py"
    spec = importlib.util.spec_from_file_location("validate_scenario", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = (
        module  # dataclasses resolve annotations through sys.modules
    )
    spec.loader.exec_module(module)
    return module


def _port() -> int:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    return 47600 + 20 * int(worker[2:] or 0)


@pytest.fixture(scope="module")
def validator():
    return _load_validator()


def test_scenario_static(validator):
    """Map lumps, spawns, DECORATE parents and ACS calls (no engine)."""
    report = validator.Report(str(_cfg_path()))
    validator.static_checks(_cfg_path(), report, None, PLAYERS)
    assert report.ok, report.render()


def test_scenario_contract(validator):
    """Engine load plus the USER contract over several resets with PLAYERS players."""
    report = validator.validate(
        _cfg_path(),
        players=PLAYERS,
        episodes=EPISODES,
        expect=EXPECT,
        expect_idle=EXPECT_IDLE,
        idle_steps=IDLE_STEPS,
        check_timeout=CHECK_TIMEOUT,
        port=_port(),
    )
    print(report.render())
    assert report.ok, report.render()
