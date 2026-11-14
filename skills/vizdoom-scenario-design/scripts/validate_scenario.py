#!/usr/bin/env python3
"""Validate a ViZDoom scenario (.cfg + .wad) before training on it.

Three levels, each stops the run on failure:

1. static: the cfg and WAD exist, the map lumps are present, DECORATE parents
   exist in ViZDoom's ZDoom 2.8.1 (``references/actor_classes.md``) and the ACS
   source in the ``SCRIPTS`` lump (or ``--acs``) calls nothing from
   ``references/acs_unsupported.md``;
2. load: one single-player engine instance loads the scenario in a subprocess
   (so an actor-load crash cannot kill the validator) and runs a few tics; the
   engine output is scanned for load and script errors;
3. contract: ``--players`` instances play a networked game (``vizdoom.pettingzoo_wrapper``
   for N > 1). After every ``new_episode`` the ``--expect`` values must hold; during
   ``--idle-steps`` no-op steps the ``--expect-idle`` values must hold and no player
   may die; with ``--check-timeout`` every episode must end within ``episode_timeout``.

Example:
    python validate_scenario.py --cfg scenarios/simple_tag_audio.cfg --players 4 \
        --episodes 3 --expect USER50=0 --expect USER51=0 --expect-idle USER50=0
"""

from __future__ import annotations

import argparse
import math
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
REFERENCES = SKILL_DIR / "references"
# Engine console messages (printed because the load stage enables the console).
ERROR_PATTERNS = [
    r"Parent type '[^']*' not found in \w+",
    r"Unknown P-Code \d+ in script \S+",
    r'Script error, "[^"]*" line \d+',
    r"Execution could not continue",
    r"Unknown actor class \S+",
    r"ViZDoomErrorException",
]
OPENAL_DIRS = ["/opt/homebrew/opt/openal-soft/lib", "/usr/local/opt/openal-soft/lib"]


@dataclass
class Report:
    cfg: str
    passed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def check(self, condition: bool, ok_msg: str, fail_msg: str) -> bool:
        (self.passed if condition else self.failures).append(
            ok_msg if condition else fail_msg
        )
        return condition

    def render(self) -> str:
        lines = [f"Scenario: {self.cfg}"]
        lines += [f"  PASS  {m}" for m in self.passed]
        lines += [f"  NOTE  {m}" for m in self.notes]
        lines += [f"  FAIL  {m}" for m in self.failures]
        lines.append(
            "RESULT: "
            + ("PASS" if self.ok else f"FAIL ({len(self.failures)} problem(s))")
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Static checks
# --------------------------------------------------------------------------


def parse_cfg(path: Path) -> dict[str, str]:
    """Single-valued keys of a ViZDoom cfg (lower-case, underscores removed)."""
    values = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line and "{" not in line:
            key, value = line.split("=", 1)
            values[key.strip().lower().replace("_", "")] = value.strip()
    return values


def read_lumps(wad: Path) -> list[tuple[str, bytes]]:
    data = wad.read_bytes()
    magic, count, offset = struct.unpack("<4sii", data[:12])
    if magic not in (b"PWAD", b"IWAD"):
        raise ValueError(f"{wad} is not a WAD file (magic {magic!r})")
    lumps = []
    for i in range(count):
        pos, size, name = struct.unpack(
            "<ii8s", data[offset + 16 * i : offset + 16 * (i + 1)]
        )
        lumps.append(
            (name.rstrip(b"\0").decode("ascii", "replace"), data[pos : pos + size])
        )
    return lumps


PLAYER_STARTS = [1, 2, 3, 4, 4001, 4002, 4003, 4004]  # editor numbers, players 1-8
DEATHMATCH_START = 11


def map_thing_types(lumps: list[tuple[str, bytes]], map_name: str) -> list[int]:
    """Editor numbers of the things of one map (UDMF, Hexen or Doom format)."""
    names = [n.upper() for n, _ in lumps]
    start = names.index(map_name) + 1
    data: dict[str, bytes] = {}
    for name, payload in lumps[start : start + 12]:
        data.setdefault(name.upper(), payload)
    if "TEXTMAP" in data:
        text = strip_acs(data["TEXTMAP"].decode("latin-1"))
        return [
            int(t)
            for t in re.findall(r"thing\s*\{[^}]*?\btype\s*=\s*(\d+)", text, flags=re.I)
        ]
    if "THINGS" in data:
        raw = data["THINGS"]
        size, offset = (
            (20, 10) if "BEHAVIOR" in data else (10, 6)
        )  # Hexen vs Doom layout
        return [
            struct.unpack_from("<h", raw, i * size + offset)[0]
            for i in range(len(raw) // size)
        ]
    return []


def strip_acs(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def table_names(path: Path) -> dict[str, str]:
    """First backticked column -> last column of the Markdown tables in a reference file."""
    rows = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|(.*)\|\s*$", line)
        if match:
            rows[match.group(1)] = match.group(2).split("|")[-1].strip()
    return rows


def check_spawns(
    values: dict[str, str], types: list[int], players: int, report: Report
) -> None:
    if "-deathmatch" in values.get("gameargs", "").lower():
        count = types.count(DEATHMATCH_START)
        report.check(
            count > 0,
            f"deathmatch: {count} deathmatch start(s)",
            "game_args has -deathmatch but the map has no deathmatch starts (editor number 11); "
            "the engine crashes while loading",
        )
        if 0 < count < players:
            report.notes.append(
                f"only {count} deathmatch start(s) for {players} players: spawns may telefrag"
            )
        return
    if PLAYER_STARTS[0] not in types:
        report.failures.append("the map has no player 1 start (editor number 1)")
        return
    missing = [n for n in PLAYER_STARTS[: min(players, 8)] if n not in types]
    if missing:
        report.notes.append(
            f"cooperative game without player starts {missing} for {players} players"
        )
    else:
        report.passed.append(f"player starts present for {min(players, 8)} player(s)")


def static_checks(
    cfg: Path, report: Report, acs_file: Path | None, players: int = 1
) -> Path | None:
    if not report.check(cfg.is_file(), f"cfg exists ({cfg})", f"cfg not found: {cfg}"):
        return None
    values = parse_cfg(cfg)
    if "doomscenariopath" not in values:
        report.failures.append("cfg has no doom_scenario_path")
        return None
    wad = (cfg.parent / values["doomscenariopath"]).resolve()
    if not report.check(
        wad.is_file(),
        f"WAD exists ({wad.name})",
        f"WAD not found: {wad} (doom_scenario_path is relative to the cfg)",
    ):
        return None
    try:
        lumps = read_lumps(wad)
    except (ValueError, struct.error) as error:
        report.failures.append(f"cannot read WAD: {error}")
        return None
    names = [n.upper() for n, _ in lumps]
    map_name = values.get("doommap", "map01").upper()
    if not report.check(
        map_name in names,
        f"map {map_name} present",
        f"map {map_name} not in WAD (lumps: {', '.join(names[:12])} ...)",
    ):
        return wad
    map_lumps = names[names.index(map_name) + 1 :][:12]
    if map_lumps and map_lumps[0] == "TEXTMAP":
        report.passed.append("UDMF map (TEXTMAP)")
    elif "LINEDEFS" in map_lumps and "BEHAVIOR" in map_lumps:
        report.passed.append("Hexen-format map (LINEDEFS + BEHAVIOR)")
    elif "LINEDEFS" in map_lumps:
        report.notes.append(
            "Doom-format map: it cannot run ACS; use UDMF or Hexen format for scripts"
        )
    if "BEHAVIOR" not in names:
        report.notes.append("no BEHAVIOR lump: the map has no compiled ACS")
    check_spawns(values, map_thing_types(lumps, map_name), players, report)

    known = set(table_names(REFERENCES / "actor_classes.md"))
    known_lower = {k.lower() for k in known}
    for name, data in lumps:
        if name.upper() != "DECORATE":
            continue
        text = strip_acs(data.decode("latin-1"))
        headers = re.findall(
            r"^\s*actor\s+([\w.]+)(?:\s*:\s*([\w.]+))?", text, flags=re.I | re.M
        )
        local = {h[0].lower() for h in headers}
        missing = sorted(
            {p for _, p in headers if p and p.lower() not in known_lower | local}
        )
        report.check(
            not missing,
            f"DECORATE: {len(headers)} actor(s), all parents exist in ZDoom 2.8.1",
            "DECORATE parent class(es) not in ZDoom 2.8.1: "
            + ", ".join(missing)
            + " (see references/actor_classes.md; the engine fails while loading actors)",
        )

    sources = [(n, d.decode("latin-1")) for n, d in lumps if n.upper() == "SCRIPTS"]
    if acs_file is not None:
        sources.append((str(acs_file), acs_file.read_text(errors="replace")))
    if not sources:
        report.notes.append(
            "no ACS source (SCRIPTS lump or --acs): unsupported-function check skipped"
        )
    unsupported = table_names(REFERENCES / "acs_unsupported.md")
    by_lower = {k.lower(): (k, v) for k, v in unsupported.items()}
    for label, source in sources:
        calls = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", strip_acs(source)))
        bad = sorted({by_lower[c.lower()][0] for c in calls if c.lower() in by_lower})
        report.check(
            not bad,
            f"ACS source ({label}): no unsupported functions",
            f"ACS source ({label}) calls functions ViZDoom does not run: "
            + "; ".join(f"{b} ({by_lower[b.lower()][1]})" for b in bad),
        )
    return wad


# --------------------------------------------------------------------------
# Engine load (subprocess) and contract probe
# --------------------------------------------------------------------------


def ensure_openal() -> None:
    """On macOS, restart the CLI with OpenAL Soft on DYLD_LIBRARY_PATH.

    dyld reads the variable only at process start; setting os.environ later does not
    reach the engine processes of the in-process contract probe (silent audio).
    """
    env = openal_env()
    if env.get("DYLD_LIBRARY_PATH") and "DYLD_LIBRARY_PATH" not in os.environ:
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def openal_env() -> dict[str, str]:
    env = dict(os.environ)
    if sys.platform == "darwin" and "DYLD_LIBRARY_PATH" not in env:
        for libdir in OPENAL_DIRS:
            if Path(libdir, "libopenal.1.dylib").is_file():
                env["DYLD_LIBRARY_PATH"] = libdir
                break
    return env


def load_only(cfg: str, tics: int) -> None:
    """Run inside the subprocess: load the scenario single-player and advance a few tics."""
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.load_config(cfg)
    game.set_window_visible(False)
    game.set_console_enabled(True)  # engine console -> stdout: actor and script errors
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    game.new_episode()
    buttons = game.get_available_buttons_size()
    for _ in range(tics):
        if game.is_episode_finished():
            break
        game.make_action([0.0] * buttons)
    print(f"LOAD_OK tics={tics} finished={game.is_episode_finished()}", flush=True)
    game.close()


def engine_load(cfg: Path, report: Report, tics: int = 70, timeout: int = 120) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_load-only",
        str(cfg),
        "--_tics",
        str(tics),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=openal_env()
        )
    except subprocess.TimeoutExpired:
        report.failures.append(f"engine load did not finish within {timeout} s")
        return
    output = result.stdout + result.stderr
    errors = sorted(
        {m.group(0) for p in ERROR_PATTERNS for m in re.finditer(p, output)}
    )
    crashed = result.returncode != 0 or "LOAD_OK" not in result.stdout
    if crashed:
        code = result.returncode
        signal = (
            f" (signal {-code})" if code < 0 else (" (segfault)" if code == 139 else "")
        )
        cause = "; ".join(errors) or (
            "no engine message; see pitfalls.md (e.g. a missing DECORATE parent or missing deathmatch starts)"
        )
        tail = "\n        ".join(output.strip().splitlines()[-8:])
        report.failures.append(
            f"engine load failed, exit code {code}{signal}: {cause}; last output:\n        {tail}"
        )
    elif errors:
        report.failures.append(
            "engine console reports errors (scenario still ran): "
            + "; ".join(errors)
            + " (see pitfalls.md)"
        )
    else:
        report.passed.append(
            f"engine loads the scenario single-player and runs {tics} tics without errors"
        )


def parse_expectations(items: list[str]) -> dict[str, float]:
    expected = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expectation must be NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        expected[name.strip().upper()] = float(value)
    return expected


def compare(values: dict[str, float], expected: dict[str, float]) -> list[str]:
    problems = []
    for name, want in expected.items():
        if name not in values:
            problems.append(
                f"{name} is not exported (add it to available_game_variables in the cfg)"
            )
        elif not math.isclose(values[name], want):
            problems.append(f"{name}={values[name]:g}, expected {want:g}")
    return problems


class SinglePlayerEnv:
    """Minimal PettingZoo-like adapter over one DoomGame, used for ``--players 1``.

    Single-player ``new_episode()`` starts a new game (ACS variables are cleared),
    which differs from a netgame map change, so one player is not run as a netgame.
    """

    def __init__(self, cfg: Path, skip: int, seed: int, extra_vars: list[str]):
        import vizdoom as vzd

        game = vzd.DoomGame()
        game.load_config(str(cfg))
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_seed(seed)
        have = {v.name for v in game.get_available_game_variables()}
        for name in extra_vars:
            if name not in have and hasattr(vzd.GameVariable, name):
                game.add_available_game_variable(getattr(vzd.GameVariable, name))
        game.init()
        self.game = game
        self.skip = skip
        self.agents = ["agent_0"]
        self.buttons = game.get_available_buttons_size()

    def _infos(self) -> dict[str, dict]:
        game = self.game
        info: dict = {
            v.name: float(game.get_game_variable(v))
            for v in game.get_available_game_variables()
        }
        info["player_dead"] = bool(game.is_player_dead())
        return {"agent_0": info}

    def reset(self, seed: int | None = None):
        self.game.new_episode()
        return None, self._infos()

    def step(self, actions: dict):
        self.game.make_action([0.0] * self.buttons, self.skip)
        done = self.game.is_episode_finished()
        truncated = done and self.game.is_episode_timeout_reached()
        return (
            None,
            None,
            {"agent_0": done and not truncated},
            {"agent_0": truncated},
            self._infos(),
        )

    def close(self) -> None:
        self.game.close()


def contract_probe(
    cfg: Path,
    report: Report,
    players: int,
    episodes: int,
    expect: dict[str, float],
    expect_idle: dict[str, float],
    idle_steps: int,
    check_timeout: bool,
    allow_death: bool,
    skip: int,
    port: int,
    seed: int,
) -> None:
    timeout_tics = int(parse_cfg(cfg).get("episodetimeout", "0") or 0)
    if players == 1:
        env = SinglePlayerEnv(cfg, skip, seed, sorted(set(expect) | set(expect_idle)))
    else:
        from vizdoom.pettingzoo_wrapper import make

        env = make(
            config_file=str(cfg),
            num_agents=players,
            resolution="160X120",
            frame_stack=1,
            skip_frames=skip,
            enable_video=False,
            seed=seed,
            port=port,
        )
    failures_before = len(report.failures)
    try:
        for episode in range(episodes):
            _, infos = env.reset(seed=seed + episode)
            report.check(
                len(infos) == players,
                f"episode {episode}: {players} player(s) joined",
                f"episode {episode}: {len(infos)} of {players} players joined",
            )
            for agent, info in infos.items():
                for problem in compare(info, expect):
                    report.failures.append(
                        f"episode {episode}, {agent}, after reset: {problem}"
                    )
            ended = False
            for step in range(idle_steps):
                _, _, term, trunc, infos = env.step(dict.fromkeys(env.agents, 0))
                for agent, info in infos.items():
                    for problem in compare(info, expect_idle):
                        report.failures.append(
                            f"episode {episode}, {agent}, idle step {step}: {problem}"
                        )
                    if not allow_death and info.get("player_dead"):
                        report.failures.append(
                            f"episode {episode}, {agent}: dead during idle step {step}"
                        )
                if any(term.values()) or any(trunc.values()):
                    report.failures.append(
                        f"episode {episode}: ended during idle step {step}"
                    )
                    ended = True
                    break
            if len(report.failures) > failures_before:
                break
            if check_timeout and not ended:
                if timeout_tics <= 0:
                    report.notes.append(
                        "no episode_timeout in the cfg: timeout check skipped"
                    )
                else:
                    limit = math.ceil(timeout_tics / skip) + 8 - idle_steps
                    steps = 0
                    while env.agents and steps < limit:
                        _, _, term, trunc, _ = env.step(dict.fromkeys(env.agents, 0))
                        steps += 1
                        if all(term.get(a) or trunc.get(a) for a in term):
                            break
                    report.check(
                        steps < limit,
                        f"episode {episode}: ends within episode_timeout ({timeout_tics} tics)",
                        f"episode {episode}: still running after {timeout_tics} tics (episode_timeout)",
                    )
    finally:
        env.close()
    if len(report.failures) == failures_before:
        checked = ", ".join(sorted(expect)) or "none"
        report.passed.append(
            f"{episodes} episode(s) x {players} player(s): reset contract holds ({checked}); "
            f"{idle_steps} idle steps clean"
        )


def validate(
    cfg: str | Path,
    players: int = 1,
    episodes: int = 3,
    expect: list[str] | None = None,
    expect_idle: list[str] | None = None,
    idle_steps: int = 16,
    check_timeout: bool = False,
    allow_death: bool = False,
    acs: str | Path | None = None,
    skip: int = 4,
    port: int = 47400,
    seed: int = 0,
    load_tics: int = 70,
) -> Report:
    """Run all levels and return a :class:`Report` (``report.ok``, ``report.failures``)."""
    cfg = Path(cfg).resolve()
    report = Report(str(cfg))
    if not 1 <= players <= 16:
        report.failures.append(f"--players must be 1-16, got {players}")
        return report
    try:
        expected = parse_expectations(expect or [])
        expected_idle = parse_expectations(expect_idle or [])
    except ValueError as error:
        report.failures.append(str(error))
        return report
    if (
        static_checks(cfg, report, Path(acs) if acs else None, players) is None
        or not report.ok
    ):
        return report
    engine_load(cfg, report, tics=load_tics)
    if not report.ok:
        return report
    contract_probe(
        cfg,
        report,
        players,
        episodes,
        expected,
        expected_idle,
        idle_steps,
        check_timeout,
        allow_death,
        skip,
        port,
        seed,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cfg", help="scenario .cfg (doom_scenario_path is resolved relative to it)"
    )
    parser.add_argument(
        "--players", type=int, default=1, help="number of networked players (1-16)"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="episodes (resets) to probe"
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="must hold right after every reset, e.g. USER50=0",
    )
    parser.add_argument(
        "--expect-idle",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="must hold during the idle steps",
    )
    parser.add_argument(
        "--idle-steps", type=int, default=16, help="no-op steps after each reset"
    )
    parser.add_argument(
        "--check-timeout",
        action="store_true",
        help="run each episode to its end and require it to end by episode_timeout",
    )
    parser.add_argument(
        "--allow-death",
        action="store_true",
        help="do not fail if a player dies while idle",
    )
    parser.add_argument(
        "--acs", help="ACS source to lint in addition to the SCRIPTS lump"
    )
    parser.add_argument("--skip", type=int, default=4, help="tics per step")
    parser.add_argument("--port", type=int, default=47400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--_load-only", dest="load_only", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_tics", dest="tics", type=int, default=70, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.load_only:
        load_only(args.load_only, args.tics)
        return
    if not args.cfg:
        parser.error("--cfg is required")
    ensure_openal()
    report = validate(
        args.cfg,
        players=args.players,
        episodes=args.episodes,
        expect=args.expect,
        expect_idle=args.expect_idle,
        idle_steps=args.idle_steps,
        check_timeout=args.check_timeout,
        allow_death=args.allow_death,
        acs=args.acs,
        skip=args.skip,
        port=args.port,
        seed=args.seed,
    )
    print(report.render())
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
