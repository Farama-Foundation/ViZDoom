# Tests for the vizdoom-scenario-design skill: vendored DoomGen2, condensed
# examples, deterministic static checks, native netgame validation, snapshot
# capture, playtest launcher, reference-pack freshness and doc path integrity.

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = REPO_ROOT / "skills" / "vizdoom-scenario-design"
DOOMGEN = SKILL / "DoomGen2"
ACC = DOOMGEN / "acc" / "acc"


def _has_doomgen_deps() -> bool:
    for mod in ("omg", "shapely", "scipy", "networkx"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


def _acc_runs() -> bool:
    try:
        subprocess.run([str(ACC)], capture_output=True, timeout=10)
    except OSError:
        return False
    return True


requires_doomgen = pytest.mark.skipif(
    not _has_doomgen_deps(),
    reason="DoomGen2 dependencies (omgifol, shapely, scipy, networkx) not installed",
)
requires_acc = pytest.mark.skipif(
    not _acc_runs(), reason="bundled acc does not run on this platform"
)


def _worker_port(offset: int) -> int:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker_id = (
        int(worker[2:]) if worker.startswith("gw") and worker[2:].isdigit() else 0
    )
    return 47800 + 40 * worker_id + offset


@pytest.fixture(scope="module")
def skill_modules():
    paths = [str(SKILL / "scripts"), str(DOOMGEN / "src")]
    for path in paths:
        sys.path.insert(0, path)
    try:
        yield (
            importlib.import_module("validate_scenario"),
            importlib.import_module("build_probe_wads"),
        )
    finally:
        for path in paths:
            if path in sys.path:
                sys.path.remove(path)


@pytest.fixture(scope="module")
def probes(skill_modules, tmp_path_factory):
    if not _has_doomgen_deps() or not _acc_runs():
        pytest.skip("DoomGen2 deps and runnable acc are required for probe WADs")
    _, probe = skill_modules
    out = tmp_path_factory.mktemp("probes")
    probe.build_all(out, str(ACC))
    return out


@requires_doomgen
def test_doomgen2_suite():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=DOOMGEN,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


@requires_doomgen
@requires_acc
@pytest.mark.parametrize(
    "probe, needle",
    [
        ("bad_parent", "PowerupNone"),
        ("unsupported", "IsNetworkGame"),
        ("nodm", "deathmatch starts"),
        ("int_reward", "integer literals"),
        ("zscript", "ZSCRIPT"),
        ("bad_playsound", "PlaySound"),
        ("dynamic_spawn", "non-literal class name"),
        ("uncompiled_scripts", "BEHAVIOR"),
        ("missing_wad", "WAD not found"),
    ],
)
def test_static_checks_name_the_cause(skill_modules, probes, probe, needle):
    vs, _ = skill_modules
    report = vs.Report(probe)
    vs.static_checks(probes / f"{probe}.cfg", report, None, 2)
    assert needle in " ".join(report.failures), report.render()


@requires_doomgen
@requires_acc
def test_ok_probe_passes_static_and_engine_load(skill_modules, probes):
    vs, _ = skill_modules
    report = vs.Report("ok")
    assert vs.static_checks(probes / "ok.cfg", report, None, 2) is not None
    vs.engine_load(probes / "ok.cfg", report, tics=35)
    assert report.ok, report.render()


@requires_doomgen
@requires_acc
def test_validate_scenario_netgame_native(skill_modules, probes):
    """Native netgame probe passes on ok.cfg and catches global persistence on noreset.cfg."""
    vs, _ = skill_modules
    ok_report = vs.validate(
        probes / "ok.cfg",
        players=2,
        episodes=2,
        expect=["USER10=0"],
        idle_steps=8,
        port=_worker_port(0),
    )
    assert ok_report.ok, ok_report.render()

    leak_report = vs.validate(
        probes / "noreset.cfg",
        players=2,
        episodes=2,
        expect=["USER10=0"],
        idle_steps=40,
        port=_worker_port(4),
    )
    assert not leak_report.ok, leak_report.render()
    assert "USER10=1, expected 0" in " ".join(
        leak_report.failures
    ), leak_report.render()


@requires_doomgen
@requires_acc
@pytest.mark.parametrize(
    "script, extra_args, game_args, players",
    [
        ("geometry_and_placement.py", [], "", 1),
        ("key_maze.py", [], "", 1),
        ("coop_relay.py", [], "", 2),
        ("simple_tag.py", [], "-deathmatch +sv_norespawn 1", 4),
        ("simple_tag.py", ["--audio"], "-deathmatch +sv_norespawn 1", 4),
        ("deathmatch_arena.py", [], "-deathmatch +sv_norespawn 1", 2),
    ],
    ids=[
        "geometry_and_placement",
        "key_maze",
        "coop_relay",
        "simple_tag",
        "simple_tag_audio",
        "deathmatch_arena",
    ],
)
def test_all_examples_pass_validator(
    skill_modules, tmp_path, script, extra_args, game_args, players
):
    vs, _ = skill_modules
    wad = tmp_path / "example.wad"
    env = dict(os.environ, PYTHONPATH=str(DOOMGEN / "src"))
    subprocess.run(
        [
            sys.executable,
            str(DOOMGEN / "examples" / script),
            *extra_args,
            "--output",
            str(wad),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    cfg = tmp_path / "example.cfg"
    cfg.write_text(
        "doom_scenario_path = example.wad\n"
        "screen_resolution = RES_160X120\n"
        "episode_timeout = 350\n"
        "available_buttons = { MOVE_FORWARD TURN_LEFT }\n"
        "mode = PLAYER\n" + (f"game_args = {game_args}\n" if game_args else "")
    )
    report = vs.Report(script)
    assert vs.static_checks(cfg, report, None, players) is not None, report.render()
    assert report.ok, report.render()
    vs.engine_load(cfg, report, tics=35)
    assert report.ok, report.render()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["example.cfg", "example.wad"]


@requires_doomgen
@requires_acc
def test_capture_multiagent_scenario_runs(probes, tmp_path):
    snap_dir = tmp_path / "snap"
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts" / "capture_multiagent_scenario.py"),
            str(probes / "ok.cfg"),
            "--agents",
            "2",
            "--tics",
            "8",
            "--port",
            str(_worker_port(12)),
            "--out",
            str(snap_dir),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    summary = json.loads((snap_dir / "summary.json").read_text())
    assert len(summary["players"]) == 2
    assert (snap_dir / "snapshot.pkl").is_file()


@requires_doomgen
@requires_acc
def test_play_netgame_dry_run(probes):
    result = subprocess.run(
        [
            "bash",
            str(SKILL / "scripts" / "play_netgame.sh"),
            "--dry-run",
            "--players",
            "4",
            "--mode",
            "deathmatch",
            "--colors",
            "ff 00 00,ff 00 00,ff 00 00,00 ff 00",
            str(probes / "ok.wad"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "-host 4" in result.stdout
    assert "-join 127.0.0.1:" in result.stdout
    assert "Player3" in result.stdout


@requires_doomgen
@requires_acc
def test_cfg_warning_caught_by_engine_load(skill_modules, probes, tmp_path):
    vs, _ = skill_modules
    cfg = tmp_path / "bad_cfg.cfg"
    cfg.write_text(
        f"doom_scenario_path = {probes / 'ok.wad'}\n"
        "screen_resolution = RES_160X120\n"
        "audio_sampling_rate = 22050\n"
        "available_buttons = { MOVE_FORWARD }\n"
        "mode = PLAYER\n"
    )
    report = vs.Report("bad_cfg")
    vs.engine_load(cfg, report, tics=10)
    assert not report.ok
    assert "WARNING! Loading config from:" in " ".join(report.failures)


def test_integer_reward_lint(skill_modules):
    vs, _ = skill_modules
    source = "global int 0:reward;\nscript 1 OPEN { reward = 0; reward += 1.0; }"
    report = vs.Report("fixed")
    vs.check_integer_reward("src", source, report)
    assert report.ok
    for bad_stmt in ("reward -= 2;", "reward++;", "++reward;"):
        report = vs.Report("integer")
        vs.check_integer_reward(
            "src", f"{source}\nscript 2 OPEN {{ {bad_stmt} }}", report
        )
        assert not report.ok, bad_stmt


def test_parse_cfg_append(skill_modules, tmp_path):
    vs, _ = skill_modules
    cfg = tmp_path / "a.cfg"
    cfg.write_text(
        "game_args = -deathmatch\n"
        "game_args += +sv_norespawn 1\n"
        "doom_scenario_path = foo.wad # inline comment\n"
    )
    parsed = vs.parse_cfg(cfg)
    assert parsed["gameargs"] == "-deathmatch +sv_norespawn 1"
    assert parsed["doomscenariopath"] == "foo.wad"


def test_udmf_high_special_and_decorate_replaces(skill_modules):
    vs, _ = skill_modules
    report = vs.Report("udmf")
    vs.check_map_specials(
        {"TEXTMAP": b"linedef { v1 = 0; v2 = 1; special = 277; }"}, report
    )
    assert not report.ok and "277" in " ".join(report.failures)

    report = vs.Report("replaces")
    vs.check_decorate(
        [("DECORATE", b"ACTOR CustomImp : DoomImp replaces MissingMonster {}")],
        {"doomimp"},
        report,
    )
    assert not report.ok and "MissingMonster" in " ".join(report.failures)


def test_package_copy_check_missing_and_stale(skill_modules, tmp_path):
    vs, _ = skill_modules
    (tmp_path / "CMakeLists.txt").write_text("")
    src_scenarios = tmp_path / "scenarios"
    pkg_scenarios = tmp_path / "pkg_scenarios"
    src_scenarios.mkdir()
    pkg_scenarios.mkdir()
    cfg = src_scenarios / "demo.cfg"
    wad = src_scenarios / "demo.wad"
    cfg.write_text("doom_scenario_path = demo.wad\n")
    wad.write_bytes(b"PWAD_NEW")

    with patch.dict(
        sys.modules,
        {"vizdoom": type("M", (), {"scenarios_path": str(pkg_scenarios)})()},
    ):
        missing_report = vs.Report("missing")
        vs.check_package_copy(cfg, wad, missing_report)
        assert not missing_report.ok
        assert "missing in" in " ".join(missing_report.failures)

        (pkg_scenarios / "demo.cfg").write_text("doom_scenario_path = demo.wad\n")
        (pkg_scenarios / "demo.wad").write_bytes(b"PWAD_OLD")
        stale_report = vs.Report("stale")
        vs.check_package_copy(cfg, wad, stale_report)
        assert not stale_report.ok
        assert "differs from" in " ".join(stale_report.failures)

        (pkg_scenarios / "demo.wad").write_bytes(b"PWAD_NEW")
        ok_report = vs.Report("ok")
        vs.check_package_copy(cfg, wad, ok_report)
        assert ok_report.ok


def test_reference_pack_up_to_date(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts" / "build_reference_pack.py"),
            "--vizdoom-root",
            str(REPO_ROOT),
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    for name in (
        "acs_functions.md",
        "acs_unsupported.md",
        "line_specials.md",
        "actor_classes.md",
    ):
        assert (tmp_path / name).read_text() == (
            SKILL / "references" / name
        ).read_text(), f"{name} drifted from engine sources; regenerate with build_reference_pack.py"


def test_skill_markdown_paths_exist():
    """Every relative file path referenced in the skill's hand-written docs must exist."""
    docs = [
        SKILL / "SKILL.md",
        SKILL / "pitfalls.md",
        SKILL / "references" / "README.md",
        SKILL / "references" / "tooling.md",
        SKILL / "references" / "vizdoom_scenario_api.md",
        SKILL / "references" / "zdoom_wiki_index.md",
        DOOMGEN / "README.md",
        DOOMGEN / "examples" / "README.md",
    ]
    prefixes = (
        "DoomGen2/",
        "references/",
        "scripts/",
        "templates/",
        "examples/",
        "docs/",
        "src/",
        "scenarios/",
        "tests/",
    )
    placeholders = ("my_scenario", "my_map", "x.wad", "output/", "player<i>", "<", "*")
    missing = []
    for doc in docs:
        base_dirs = [doc.parent, SKILL, DOOMGEN, REPO_ROOT]
        for token in re.findall(r"`([^`\n]+)`", doc.read_text()):
            candidate = token.split()[0].rstrip(".,:;)")
            if not candidate.startswith(prefixes):
                continue
            if any(p in candidate for p in placeholders):
                continue
            if not any((base / candidate).exists() for base in base_dirs):
                missing.append(f"{doc.relative_to(REPO_ROOT)}: {candidate}")
    assert not missing, "Broken relative paths in skill docs:\n" + "\n".join(missing)
