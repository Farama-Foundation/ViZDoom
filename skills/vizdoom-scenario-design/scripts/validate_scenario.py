#!/usr/bin/env python3
"""Validate a ViZDoom scenario (.cfg + .wad) before training on it.

Three levels, each stops the run on failure:

1. static: the cfg and WAD exist, map lumps and spawns are valid, no ZSCRIPT
   lump or uncompiled SCRIPTS lump is present, UDMF linedef specials are within
   ZDoom 2.8.1's 256-entry table, DECORATE parents/replaces exist in
   ``references/actor_classes.md``, ACS source calls nothing from
   ``references/acs_unsupported.md``, uses fixed-point reward literals, valid
   ``PlaySound`` argument order and literal ``SpawnSpot`` class names, and any
   ``scenarios/`` cfg/WAD matches its package copy;
2. load: one single-player engine instance loads the scenario in a subprocess
   (so an actor-load crash cannot kill the validator) and runs a few tics; the
   engine output is scanned for load and script errors;
3. contract: ``--players`` instances run a native ViZDoom game (single-player
   for N = 1, networked host + joiners for N > 1). After every reset the
   ``--expect`` values must hold; during ``--idle-steps`` no-op steps the
   ``--expect-idle`` values must hold and no player may die; with
   ``--check-timeout`` every episode must end within ``episode_timeout``.

Example:
    python validate_scenario.py --cfg scenarios/my_scenario.cfg --players 4 \\
        --episodes 3 --expect USER50=0 --expect USER51=0 --expect-idle USER50=0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import select
import signal
import struct
import subprocess
import sys
import threading
import time
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
    r"WARNING! Loading config from:[^\n]+",
    r"ViZDoomErrorException",
]
OPENAL_DIRS = ["/opt/homebrew/opt/openal-soft/lib", "/usr/local/opt/openal-soft/lib"]
PLAYER_STARTS = [1, 2, 3, 4, 4001, 4002, 4003, 4004]  # editor numbers, players 1-8
DEATHMATCH_START = 11
BINARY_MAP_LUMPS = {
    "THINGS",
    "LINEDEFS",
    "SIDEDEFS",
    "VERTEXES",
    "SEGS",
    "SSECTORS",
    "NODES",
    "SECTORS",
    "REJECT",
    "BLOCKMAP",
    "BEHAVIOR",
    "SCRIPTS",
    "ZNODES",
}
SPAWNSPOT_FUNCS = (
    "SpawnSpot",
    "SpawnSpotFacing",
    "SpawnSpotForced",
    "SpawnSpotFacingForced",
)


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
    values: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "{" in line or "=" not in line:
            continue
        if "+=" in line:
            key, value = line.split("+=", 1)
            norm = key.strip().lower().replace("_", "")
            val = value.strip()
            values[norm] = f"{values[norm]} {val}".strip() if norm in values else val
        else:
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


def map_lumps_for(
    lumps: list[tuple[str, bytes]], map_name: str
) -> list[tuple[str, bytes]]:
    """Return the lumps belonging strictly to ``map_name``."""
    names = [n.upper() for n, _ in lumps]
    start = names.index(map_name) + 1
    out: list[tuple[str, bytes]] = []
    if start < len(lumps) and names[start] == "TEXTMAP":
        for name, payload in lumps[start:]:
            upper = name.upper()
            out.append((upper, payload))
            if upper == "ENDMAP":
                break
        return out
    for name, payload in lumps[start:]:
        upper = name.upper()
        if upper not in BINARY_MAP_LUMPS:
            break
        out.append((upper, payload))
    return out


def strip_comments(text: str) -> str:
    """Strip C/ACS comments while keeping string literals intact."""
    pattern = r'("(?:\\.|[^"\\])*")|/\*.*?\*/|//[^\n]*'
    return re.sub(pattern, lambda m: m.group(1) or " ", text, flags=re.S)


def strip_acs(text: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', strip_comments(text))


def split_call_args(text: str, open_paren: int) -> list[str] | None:
    """Split comma-separated arguments starting at ``text[open_paren] == '('``."""
    args: list[str] = []
    depth = 0
    in_str = False
    escape = False
    start = open_paren + 1
    for idx in range(open_paren, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                tail = text[start:idx].strip()
                if tail or args:
                    args.append(tail)
                return args
        elif ch == "," and depth == 1:
            args.append(text[start:idx].strip())
            start = idx + 1
    return None


def map_thing_types(map_data: dict[str, bytes]) -> list[int]:
    """Editor numbers of the things of one map (UDMF, Hexen or Doom format)."""
    if "TEXTMAP" in map_data:
        text = strip_acs(map_data["TEXTMAP"].decode("latin-1"))
        return [
            int(t)
            for t in re.findall(r"thing\s*\{[^}]*?\btype\s*=\s*(\d+)", text, flags=re.I)
        ]
    if "THINGS" in map_data:
        raw = map_data["THINGS"]
        size, offset = (
            (20, 10) if "BEHAVIOR" in map_data else (10, 6)
        )  # Hexen vs Doom layout
        return [
            struct.unpack_from("<h", raw, i * size + offset)[0]
            for i in range(len(raw) // size)
        ]
    return []


def check_map_specials(map_data: dict[str, bytes], report: Report) -> None:
    """Pitfall 3: ZDoom 2.8.1 only has a 256-entry LineSpecials table."""
    if "TEXTMAP" not in map_data:
        return
    text = strip_acs(map_data["TEXTMAP"].decode("latin-1"))
    high = sorted(
        {
            int(s)
            for s in re.findall(
                r"linedef\s*\{[^}]*?\bspecial\s*=\s*(\d+)", text, flags=re.I
            )
            if int(s) > 255
        }
    )
    if high:
        report.failures.append(
            "UDMF linedef special(s) > 255: "
            + ", ".join(map(str, high))
            + " (ZDoom 2.8.1 has only a 256-entry LineSpecials table; see references/line_specials.md, pitfall 3)"
        )


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
            "the engine crashes while loading (pitfall 23)",
        )
        if players > 1 and "sv_norespawn" not in values.get("gameargs", "").lower():
            report.notes.append(
                "deathmatch without +sv_norespawn 1: dead players respawn; if a latched "
                "outcome ends the round, add it to game_args (pitfall 13)"
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


def check_decorate(
    lumps: list[tuple[str, bytes]], known_lower: set[str], report: Report
) -> None:
    """Pitfall 4: DECORATE parents and replaced classes must exist in ZDoom 2.8.1."""
    decorate_lumps = [d for n, d in lumps if n.upper() == "DECORATE"]
    if not decorate_lumps:
        return
    headers: list[tuple[str, str, str]] = []
    for data in decorate_lumps:
        text = strip_acs(data.decode("latin-1"))
        headers.extend(
            re.findall(
                r"^\s*actor\s+([\w.]+)(?:\s*:\s*([\w.]+))?(?:\s+replaces\s+([\w.]+))?",
                text,
                flags=re.I | re.M,
            )
        )
    local = {name.lower() for name, _, _ in headers}
    valid = known_lower | local
    missing_parents = sorted({p for _, p, _ in headers if p and p.lower() not in valid})
    missing_replaces = sorted(
        {r for _, _, r in headers if r and r.lower() not in valid}
    )
    report.check(
        not missing_parents,
        f"DECORATE: {len(headers)} actor(s), all parents exist in ZDoom 2.8.1",
        "DECORATE parent class(es) not in ZDoom 2.8.1: "
        + ", ".join(missing_parents)
        + " (see references/actor_classes.md; the engine fails while loading actors, pitfall 4)",
    )
    if missing_replaces:
        report.failures.append(
            "DECORATE replaces class(es) not in ZDoom 2.8.1: "
            + ", ".join(missing_replaces)
            + " (see references/actor_classes.md, pitfall 4)"
        )


def check_integer_reward(label: str, source: str, report: Report) -> None:
    """Pitfall 9: ``global int 0`` is fixed point, so ``reward += 1`` is 1/65536."""
    text = strip_acs(source)
    names = re.findall(r"\bglobal\s+int\s+0\s*:\s*(\w+)", text)
    if not names:
        return
    bad = []
    for name in names:
        for value in re.findall(rf"\b{name}\s*[-+]?=\s*([^;]+);", text):
            literal = value.strip()
            if re.fullmatch(r"-?\s*\d+", literal) and int(literal.replace(" ", "")):
                bad.append(f"{name} ... {literal}")
        for match in re.findall(
            rf"(?:\+\+|--)\s*{name}\b|\b{name}\s*(?:\+\+|--)", text
        ):
            bad.append(match.strip())
    report.check(
        not bad,
        f"ACS source ({label}): reward uses fixed-point literals",
        f"ACS source ({label}) assigns integer literals to the reward (global int 0): "
        + ", ".join(sorted(set(bad)))
        + "; write 1.0, not 1 (1 is 1/65536 in fixed point, pitfall 9)",
    )


def check_playsound_args(label: str, source: str, report: Report) -> None:
    """Pitfall 5: ``PlaySound(tid, sound, channel, volume, looping, attenuation)``."""
    text = strip_comments(source)
    swapped = []
    for match in re.finditer(r"\bPlaySound\s*\(", text, flags=re.I):
        args = split_call_args(text, match.end() - 1)
        if not args or len(args) < 3:
            continue
        is_str = lambda a: bool(re.fullmatch(r'"(?:\\.|[^"\\])*"', a))  # noqa: E731
        if not is_str(args[1]) and is_str(args[2]):
            swapped.append(f"PlaySound({args[0]}, {args[1]}, {args[2]}, ...)")
    if swapped:
        report.failures.append(
            f"ACS source ({label}) swaps PlaySound sound/channel arguments: "
            + "; ".join(swapped)
            + " — use PlaySound(tid, sound, channel, volume, looping, attenuation) (pitfall 5)"
        )


def check_spawnspot_literal(label: str, source: str, report: Report) -> None:
    """Pitfall 6: runtime string variables in ``SpawnSpot*`` are unreliable in 2.8.1."""
    text = strip_comments(source)
    str_defines = set(
        re.findall(
            r'^\s*#define\s+([A-Za-z_]\w*)\s+"(?:\\.|[^"\\])*"', text, flags=re.M
        )
    )
    pattern = r"\b(" + "|".join(SPAWNSPOT_FUNCS) + r")\s*\("
    bad = []
    for match in re.finditer(pattern, text, flags=re.I):
        args = split_call_args(text, match.end() - 1)
        if not args:
            continue
        cls = args[0]
        if re.fullmatch(r'"(?:\\.|[^"\\])*"', cls) or cls in str_defines:
            continue
        bad.append(f"{match.group(1)}({cls}, ...)")
    if bad:
        report.failures.append(
            f"ACS source ({label}) passes a non-literal class name to SpawnSpot: "
            + ", ".join(sorted(set(bad)))
            + "; branch to literal class names in ZDoom 2.8.1 (pitfall 6)"
        )


def check_open_resets(label: str, source: str, report: Report) -> None:
    """Pitfall 10: remind if exported ``global int 1..60`` vars are not assigned in OPEN."""
    text = strip_acs(source)
    exports = [
        name
        for idx, name in re.findall(r"\bglobal\s+int\s+(\d+)\s*:\s*(\w+)", text)
        if 1 <= int(idx) <= 60
    ]
    if not exports:
        return
    open_bodies: list[str] = []
    for match in re.finditer(r"\bscript\s+\S+\s+OPEN\b[^{]*\{", text, flags=re.I):
        depth = 1
        pos = match.end()
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        open_bodies.append(text[match.end() : pos - 1])
    joined = "\n".join(open_bodies)
    unreset = [name for name in exports if not re.search(rf"\b{name}\s*=", joined)]
    if unreset:
        report.notes.append(
            f"ACS source ({label}): exported global(s) {', '.join(unreset)} not assigned "
            "directly in OPEN; reset all USER1-60 exports in OPEN (pitfall 10)"
        )


def check_package_copy(cfg: Path, wad: Path, report: Report) -> None:
    """Pitfall 15: scenarios/ edits reach training only after ``make -j``."""
    if (
        cfg.parent.name != "scenarios"
        or not (cfg.parent.parent / "CMakeLists.txt").is_file()
    ):
        return
    try:
        import vizdoom
    except ImportError:
        return
    package = Path(vizdoom.scenarios_path)
    if not package.is_dir() or package.resolve() == cfg.parent.resolve():
        return
    missing = [path.name for path in (cfg, wad) if not (package / path.name).is_file()]
    stale = [
        path.name
        for path in (cfg, wad)
        if (package / path.name).is_file()
        and (package / path.name).resolve() != path.resolve()
        and (package / path.name).read_bytes() != path.read_bytes()
    ]
    problems = []
    if missing:
        problems.append(f"missing in {package}: {', '.join(missing)}")
    if stale:
        problems.append(f"differs from {package}: {', '.join(stale)}")
    report.check(
        not problems,
        "package copy of the scenario is up to date",
        "package copy out of sync ("
        + "; ".join(problems)
        + "): run make -j (training by scenario name loads the package copy, pitfall 15)",
    )


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
    if "ZSCRIPT" in names:
        report.failures.append(
            "WAD contains a ZSCRIPT lump, which ViZDoom's ZDoom 2.8.1 engine ignores; "
            "write actors in DECORATE and logic in ACS"
        )
    map_name = values.get("doommap", "map01").upper()
    if not report.check(
        map_name in names,
        f"map {map_name} present",
        f"map {map_name} not in WAD (lumps: {', '.join(names[:12])} ...)",
    ):
        return wad
    scoped = map_lumps_for(lumps, map_name)
    map_names = [n for n, _ in scoped]
    map_data: dict[str, bytes] = {}
    for name, payload in scoped:
        map_data.setdefault(name, payload)

    if map_names and map_names[0] == "TEXTMAP":
        report.passed.append("UDMF map (TEXTMAP)")
    elif "LINEDEFS" in map_data and "BEHAVIOR" in map_data:
        report.passed.append("Hexen-format map (LINEDEFS + BEHAVIOR)")
    elif "LINEDEFS" in map_data:
        report.notes.append(
            "Doom-format map: it cannot run ACS; use UDMF or Hexen format for scripts"
        )
    if "BEHAVIOR" not in map_data:
        report.notes.append("no BEHAVIOR lump: the map has no compiled ACS")

    check_map_specials(map_data, report)
    check_spawns(values, map_thing_types(map_data), players, report)

    known_lower = {k.lower() for k in table_names(REFERENCES / "actor_classes.md")}
    check_decorate(lumps, known_lower, report)

    sources = [
        ("SCRIPTS", map_data["SCRIPTS"].decode("latin-1").rstrip("\x00"))
        for _ in ([1] if "SCRIPTS" in map_data else [])
    ]
    if not sources:
        sources = [
            (n, d.decode("latin-1").rstrip("\x00"))
            for n, d in lumps
            if n.upper() == "SCRIPTS"
        ]
    if acs_file is not None:
        sources.append((str(acs_file), acs_file.read_text(errors="replace")))
    has_nonempty_scripts = any(src.strip() for _, src in sources)
    if has_nonempty_scripts and len(map_data.get("BEHAVIOR", b"")) < 8:
        report.failures.append(
            "map has ACS source (SCRIPTS) without a compiled BEHAVIOR lump; compile with acc"
        )
    if not sources:
        report.notes.append(
            "no ACS source (SCRIPTS lump or --acs): ACS static checks skipped"
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
        check_integer_reward(label, source, report)
        check_playsound_args(label, source, report)
        check_spawnspot_literal(label, source, report)
        check_open_resets(label, source, report)
    check_package_copy(cfg, wad, report)
    return wad


# --------------------------------------------------------------------------
# Engine load (subprocess) and native contract probe
# --------------------------------------------------------------------------


def openal_env() -> dict[str, str]:
    env = dict(os.environ)
    if sys.platform == "darwin" and "DYLD_LIBRARY_PATH" not in env:
        for libdir in OPENAL_DIRS:
            if Path(libdir, "libopenal.1.dylib").is_file():
                env["DYLD_LIBRARY_PATH"] = libdir
                break
    return env


def ensure_openal() -> None:
    """On macOS, restart the CLI with OpenAL Soft on DYLD_LIBRARY_PATH."""
    env = openal_env()
    if env.get("DYLD_LIBRARY_PATH") and "DYLD_LIBRARY_PATH" not in os.environ:
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def load_only(cfg: str, tics: int) -> None:
    """Run inside a subprocess: load the scenario single-player and advance a few tics."""
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
        signal_name = (
            f" (signal {-code})" if code < 0 else (" (segfault)" if code == 139 else "")
        )
        cause = "; ".join(errors) or (
            "no engine message; see pitfalls.md (e.g. a missing DECORATE parent or missing deathmatch starts)"
        )
        tail = "\n        ".join(output.strip().splitlines()[-8:])
        report.failures.append(
            f"engine load failed, exit code {code}{signal_name}: {cause}; last output:\n        {tail}"
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


def _read_game_info(game) -> dict:
    info: dict = {
        v.name: float(game.get_game_variable(v))
        for v in game.get_available_game_variables()
    }
    info["player_dead"] = bool(game.is_player_dead())
    return info


class SinglePlayerEnv:
    """Minimal single-player adapter over one DoomGame (``--players 1``)."""

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

    def reset(self, seed: int | None = None):
        self.game.new_episode()
        return None, {"agent_0": _read_game_info(self.game)}

    def step(self, actions: dict):
        self.game.make_action([0.0] * self.buttons, self.skip)
        done = self.game.is_episode_finished()
        truncated = done and self.game.is_episode_timeout_reached()
        return (
            None,
            None,
            {"agent_0": done and not truncated},
            {"agent_0": truncated},
            {"agent_0": _read_game_info(self.game)},
        )

    def close(self) -> None:
        self.game.close()


def netgame_worker(config_json: str) -> None:
    """Single subprocess hosting N lockstep ViZDoom netgame peers via threads.

    Duplicates fd 1 for clean JSON-line IPC before redirecting stdout to
    ``/dev/null`` so ViZDoom's handshake output never interleaves with IPC.
    A ``threading.Barrier`` keeps all peers on the exact same tic during
    ``advance_action(1)`` and ``new_episode()``, eliminating UDP stall timers.
    """
    import vizdoom as vzd

    ipc_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    ipc_out = os.fdopen(ipc_fd, "w", buffering=1)

    cfg = json.loads(config_json)
    players = int(cfg["players"])
    skip = int(cfg["skip"])
    timeout = float(cfg["timeout"])
    step_barrier = threading.Barrier(players)
    cmd_ready = threading.Barrier(players + 1)
    cmd_done = threading.Barrier(players + 1)
    state: dict = {"cmd": None, "replies": [None] * players, "error": None}

    def send(payload: dict) -> None:
        ipc_out.write(json.dumps(payload) + "\n")
        ipc_out.flush()

    def peer(idx: int) -> None:
        try:
            game = vzd.DoomGame()
            game.load_config(cfg["cfg"])
            game.set_window_visible(False)
            game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
            game.set_mode(vzd.Mode.PLAYER)
            game.set_seed(int(cfg["seed"]))
            have = {v.name for v in game.get_available_game_variables()}
            for name in cfg["extra_vars"]:
                if name not in have and hasattr(vzd.GameVariable, name):
                    game.add_available_game_variable(getattr(vzd.GameVariable, name))
            if idx == 0:
                game.add_game_args(
                    f"-host {players} -port {cfg['port']} -netmode 0 "
                    "+timelimit 0 +viz_connect_timeout 30 +name Player0"
                )
            else:
                time.sleep(0.15)
                game.add_game_args(
                    f"-join 127.0.0.1:{cfg['port']} -netmode 0 "
                    f"+viz_connect_timeout 30 +name Player{idx}"
                )
            game.init()
            buttons = [0.0] * game.get_available_buttons_size()
            first_reset = True
            step_barrier.wait(timeout=timeout)
            cmd_done.wait(timeout=timeout)

            while True:
                cmd_ready.wait(timeout=timeout)
                cmd = state["cmd"]
                if cmd == "close":
                    break
                if cmd == "reset":
                    if first_reset:
                        first_reset = False
                    else:
                        step_barrier.wait(timeout=timeout)
                        game.new_episode()
                        step_barrier.wait(timeout=timeout)
                    state["replies"][idx] = {"info": _read_game_info(game)}
                elif cmd == "step":
                    game.set_action(buttons)
                    for tic in range(skip):
                        game.advance_action(1, tic == skip - 1)
                        step_barrier.wait(timeout=timeout)
                    done = game.is_episode_finished()
                    truncated = done and game.is_episode_timeout_reached()
                    state["replies"][idx] = {
                        "term": done and not truncated,
                        "trunc": truncated,
                        "info": _read_game_info(game),
                    }
                cmd_done.wait(timeout=timeout)
        except Exception as exc:
            if state["error"] is None:
                state["error"] = f"player {idx}: {type(exc).__name__}: {exc}"
            step_barrier.abort()
            cmd_ready.abort()
            cmd_done.abort()

    threads = [
        threading.Thread(target=peer, args=(i,), daemon=True) for i in range(players)
    ]
    for t in threads:
        t.start()

    try:
        cmd_done.wait(timeout=timeout)
        if state["error"]:
            send({"error": state["error"]})
            return
        send({"ok": True})
        for raw in sys.stdin:
            cmd = json.loads(raw)["cmd"]
            state["cmd"] = cmd
            cmd_ready.wait(timeout=timeout)
            if cmd == "close":
                break
            cmd_done.wait(timeout=timeout)
            if state["error"]:
                send({"error": state["error"]})
                break
            send({"replies": state["replies"]})
    except (threading.BrokenBarrierError, Exception) as exc:
        send({"error": state["error"] or f"{type(exc).__name__}: {exc}"})
    time.sleep(60)


class NetgameEnv:
    """Native N-player ViZDoom netgame runner using one isolated worker subprocess.

    Works on upstream ViZDoom without any PettingZoo wrapper, propagates
    ``openal_env()``, keeps peers in per-tic lockstep, and survives dynamic
    imports (``spec_from_file_location``).
    """

    def __init__(
        self,
        cfg: Path,
        players: int,
        skip: int,
        port: int,
        seed: int,
        extra_vars: list[str],
        timeout: float = 60.0,
    ):
        self.agents = [f"agent_{i}" for i in range(players)]
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        spec = json.dumps(
            {
                "cfg": str(cfg),
                "players": players,
                "port": port,
                "skip": skip,
                "seed": seed,
                "extra_vars": extra_vars,
                "timeout": timeout,
            }
        )
        try:
            self.proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_netgame-worker",
                    spec,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=openal_env(),
                start_new_session=True,
            )
            init_msg = self._readline()
            if "error" in init_msg:
                raise RuntimeError(init_msg["error"])
        except Exception:
            self.close()
            raise

    def _readline(self) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        if sys.platform != "win32":
            ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
            if not ready:
                raise TimeoutError(
                    f"netgame worker timed out after {self.timeout:.0f} s"
                )
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(
                f"netgame worker exited unexpectedly: {err.strip()[-400:]}"
            )
        return json.loads(line)

    def _rpc(self, cmd: str) -> list[dict]:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"cmd": cmd}) + "\n")
        self.proc.stdin.flush()
        reply = self._readline()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["replies"]

    def reset(self, seed: int | None = None):
        replies = self._rpc("reset")
        return None, {
            agent: reply["info"] for agent, reply in zip(self.agents, replies)
        }

    def step(self, actions: dict):
        replies = self._rpc("step")
        term = {agent: reply["term"] for agent, reply in zip(self.agents, replies)}
        trunc = {agent: reply["trunc"] for agent, reply in zip(self.agents, replies)}
        infos = {agent: reply["info"] for agent, reply in zip(self.agents, replies)}
        return None, None, term, trunc, infos

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write('{"cmd":"close"}\n')
                self.proc.stdin.flush()
        except OSError:
            pass
        try:
            if sys.platform != "win32":
                os.killpg(self.proc.pid, signal.SIGKILL)
            else:
                self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None


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
    extra_vars = sorted(set(expect) | set(expect_idle))
    if players == 1:
        env: SinglePlayerEnv | NetgameEnv = SinglePlayerEnv(cfg, skip, seed, extra_vars)
    else:
        env = NetgameEnv(cfg, players, skip, port, seed, extra_vars)
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
    parser.add_argument(
        "--_netgame-worker", dest="netgame_worker", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.load_only:
        load_only(args.load_only, args.tics)
        return
    if args.netgame_worker:
        netgame_worker(args.netgame_worker)
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
