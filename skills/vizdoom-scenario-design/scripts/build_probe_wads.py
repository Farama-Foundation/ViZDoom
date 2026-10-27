#!/usr/bin/env python3
"""Build small probe WADs that reproduce the engine-level pitfalls, and check them.

Each probe is a DoomGen arena (4 player starts) with marker globals exported as
``USER10``-``USER13``:

* ``ok``          - correct scenario; must pass every validation level;
* ``bad_parent``  - DECORATE ``ACTOR PowerTagInvisibility : PowerupNone`` (pitfall 4);
* ``unsupported`` - ``m1 = 1; SetActivatorToPlayer(0); m2 = 1; IsNetworkGame(); m3 = 1;`` (pitfalls 1, 2);
* ``noreset``     - a global set mid-episode and never reset (pitfall 10), coop and deathmatch;
* ``nodm``        - ``-deathmatch`` on a map without deathmatch starts (pitfall 23);
* ``missing_wad`` - cfg pointing to a WAD that does not exist.

``--check`` runs ``validate_scenario.py`` on every probe and compares the observed
behaviour with the expectation; the exit code is non-zero on any mismatch.

Usage (from a ViZDoom checkout with DoomGen and acc):
    python build_probe_wads.py --out /tmp/vizdoom_probes --acc DoomGen/acc/acc --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
CFG = """doom_scenario_path = {wad}
episode_timeout = 700
living_reward = 0
screen_resolution = RES_160X120
render_hud = false
available_buttons = {{ MOVE_FORWARD TURN_LEFT }}
available_game_variables = {{ HEALTH USER10 USER11 USER12 USER13 }}
mode = PLAYER
"""


def build_wad(
    out: Path,
    name: str,
    acc: str,
    globals_,
    body: str,
    decorate: str | None = None,
    dm_starts: bool = False,
):
    from doomgen.builder import ProceduralMapBuilder
    from doomgen.logic.acs_builder import ACSBuilder, ScriptType

    builder = ProceduralMapBuilder(
        bounds=(-640, -640, 640, 640), num_seeds=2000, seed=42
    )
    builder.add_exact_room(
        (0, 0, 960, 960),
        ceiling_height=192,
        floor_texture="FLOOR0_3",
        ceiling_texture="F_SKY1",
        wall_texture="ROCK1",
    )
    for i, (x, y) in enumerate(
        [(-300, -300), (300, -300), (-300, 300), (300, 300)], start=1
    ):
        builder.add_thing(f"PLAYER{i}_START", x, y, angle=0)
        if dm_starts:
            builder.add_thing("DEATHMATCH_START", x // 2, y // 2, angle=0)
    acs = ACSBuilder()
    acs.add_include("zcommon.acs")
    for var, index in globals_:
        acs.add_global_var(var, index, "int")
    acs.add_script(ScriptType.OPEN, body)
    builder.map_data.behavior = acs.compile(acc_path=acc)
    builder.map_data.scripts = acs.to_code()
    if decorate:
        builder.wad_writer.add_lump("DECORATE", decorate.encode())
    builder.build(str(out / f"{name}.wad"))


def write_cfg(out: Path, name: str, wad: str, game_args: str = "") -> Path:
    path = out / f"{name}.cfg"
    path.write_text(
        CFG.format(wad=wad) + (f"game_args = {game_args}\n" if game_args else "")
    )
    return path


def build_all(out: Path, acc: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    flag = [("flag", 10)]
    build_wad(out, "ok", acc, flag, "flag = 0;")
    build_wad(
        out,
        "bad_parent",
        acc,
        flag,
        "flag = 0;",
        decorate="ACTOR PowerTagInvisibility : PowerupNone\n{\n}\n",
    )
    build_wad(
        out,
        "unsupported",
        acc,
        [("m1", 11), ("m2", 12), ("m3", 13)],
        "m1 = 1;\nSetActivatorToPlayer(0);\nm2 = 1;\nIsNetworkGame();\nm3 = 1;",
    )
    build_wad(out, "noreset", acc, flag, "Delay(105);\nflag = 1;", dm_starts=True)
    for name in ("ok", "bad_parent", "unsupported", "noreset"):
        write_cfg(out, name, f"{name}.wad")
    write_cfg(out, "noreset_dm", "noreset.wad", "-deathmatch +sv_norespawn 1")
    write_cfg(out, "nodm", "ok.wad", "-deathmatch")
    write_cfg(out, "missing_wad", "does_not_exist.wad")


def check_all(out: Path) -> bool:
    sys.path.insert(0, str(SCRIPTS))
    import validate_scenario as vs

    results = []

    def record(probe: str, expectation: str, observed: str, ok: bool) -> None:
        results.append((probe, expectation, observed, ok))

    report = vs.validate(out / "ok.cfg", players=2, expect=["USER10=0"], port=47700)
    record(
        "ok",
        "all levels pass (2 players)",
        "pass" if report.ok else "; ".join(report.failures),
        report.ok,
    )

    report = vs.validate(out / "missing_wad.cfg")
    record(
        "missing_wad", "static failure", "; ".join(report.failures)[:90], not report.ok
    )

    for probe, needle in (
        ("bad_parent", "PowerupNone"),
        ("unsupported", "IsNetworkGame"),
        ("nodm", "deathmatch"),
    ):
        report = vs.Report(probe)
        vs.static_checks(out / f"{probe}.cfg", report, None, 2)
        record(
            probe,
            "static check names the cause",
            "; ".join(report.failures)[:90],
            needle in " ".join(report.failures),
        )

    for probe, needle in (
        ("bad_parent", "Parent type 'PowerupNone'"),
        ("nodm", "exit code"),
        ("unsupported", "Unknown P-Code 118"),
    ):
        report = vs.Report(probe)
        vs.engine_load(out / f"{probe}.cfg", report, tics=35)
        text = " ".join(report.failures)
        record(
            probe,
            f"engine run shows '{needle}'",
            text[:90] or "no failure",
            needle in text,
        )

    env = vs.SinglePlayerEnv(out / "unsupported.cfg", 4, 0, [])
    env.reset()
    for _ in range(10):
        infos = env.step({})[4]
    env.close()
    markers = tuple(int(infos["agent_0"][f"USER1{i}"]) for i in (1, 2, 3))
    record(
        "unsupported",
        "markers (1, 1, 0): continues past SetActivatorToPlayer, stops at IsNetworkGame",
        str(markers),
        markers == (1, 1, 0),
    )

    for probe, port in (("noreset", 47720), ("noreset_dm", 47740)):
        report = vs.validate(
            out / f"{probe}.cfg",
            players=2,
            episodes=3,
            expect=["USER10=0"],
            idle_steps=40,
            port=port,
        )
        record(
            probe,
            "global set mid-episode is 0 again after each netgame reset",
            "pass" if report.ok else "; ".join(report.failures)[:90],
            report.ok,
        )

    width = max(len(r[0]) for r in results)
    for probe, expectation, observed, ok in results:
        print(
            f"{'OK ' if ok else 'BAD'} {probe:<{width}}  {expectation}\n      observed: {observed}"
        )
    return all(r[3] for r in results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for the probe WADs and cfgs",
    )
    parser.add_argument("--acc", default="DoomGen/acc/acc", help="ACC 1.60 binary")
    parser.add_argument(
        "--doomgen-src",
        type=Path,
        default=Path("DoomGen/src"),
        help="DoomGen source dir if not installed",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the probes and compare with the expectations",
    )
    args = parser.parse_args()
    if args.doomgen_src.is_dir():
        sys.path.insert(0, str(args.doomgen_src.resolve()))
    build_all(args.out, args.acc)
    print(f"probe WADs written to {args.out}")
    if args.check:
        sys.exit(0 if check_all(args.out) else 1)


if __name__ == "__main__":
    main()
