#!/usr/bin/env python3
"""Capture one synchronised multi-agent snapshot of a ViZDoom scenario.

Generalised copy of the repository's ``scripts/capture_multiagent_scenario.py``:
the scenario can be a ``.cfg`` path or a scenario name (looked up in
``vizdoom.scenarios_path`` and ``./scenarios``), and the WAD for the top-down map
is taken from the cfg's ``doom_scenario_path`` (or ``--wad``).

Every player runs in its own process with the training game setup
(``vizdoom.pettingzoo_wrapper.base_env_common.configure_doom_game``), plays a
seeded scripted policy for T steps and saves its egocentric screen, pose,
notifications and audio buffer; the map lines (plus pits and gates for UDMF maps)
are read from the WAD.

Output:
* ``--out snapshot.pkl``: one pickle (same layout as the original script);
* ``--out DIR``: ``DIR/snapshot.pkl`` plus files a multimodal agent can read:
  ``summary.json`` (poses, notifications, objects, audio peaks), ``player<i>.png``,
  ``player<i>_audio.wav`` (if the scenario has audio) and ``topdown.png``.

Usage:
    python capture_multiagent_scenario.py scenarios/simple_tag.cfg --agents 4 --tics 40 --out /tmp/tag
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import queue as queue_module
import re
import signal
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np


OPENAL_DIRS = ["/opt/homebrew/opt/openal-soft/lib", "/usr/local/opt/openal-soft/lib"]
TICRATE = 35


def ensure_openal():
    """On macOS, restart this script with OpenAL Soft on DYLD_LIBRARY_PATH (audio capture needs it).

    dyld reads the variable only at process start, so setting os.environ here is too
    late for the engine processes (observed: silent buffers); re-exec instead.
    """
    if sys.platform != "darwin" or "DYLD_LIBRARY_PATH" in os.environ:
        return
    for libdir in OPENAL_DIRS:
        if Path(libdir, "libopenal.1.dylib").is_file():
            env = dict(os.environ, DYLD_LIBRARY_PATH=libdir)
            os.execve(sys.executable, [sys.executable, *sys.argv], env)


def resolve_cfg(scenario: str) -> Path:
    path = Path(scenario)
    if path.suffix == ".cfg" and path.is_file():
        return path.resolve()
    import vizdoom as vzd

    for base in (Path(vzd.scenarios_path), Path.cwd() / "scenarios"):
        candidate = base / f"{path.stem}.cfg"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"scenario {scenario!r}: not a .cfg file and not found in vizdoom.scenarios_path or ./scenarios"
    )


def resolve_wad(cfg: Path, wad: str | None) -> Path:
    if wad:
        return Path(wad).resolve()
    for raw in cfg.read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0]
        key, _, value = line.partition("=")
        if key.strip().lower().replace("_", "") == "doomscenariopath":
            return (cfg.parent / value.strip()).resolve()
    raise ValueError(f"{cfg} has no doom_scenario_path; pass --wad")


def sampling_rate_hz(rate) -> int:
    """Hz from ``get_audio_sampling_rate()``: an int in this fork, a ``SamplingRate`` enum upstream."""
    return int(rate.name.split("_")[1]) if hasattr(rate, "name") else int(rate)


def scripted_action(n_buttons, rng):
    # random button presses so players spread out; seeded so snapshots reproduce
    return (rng.random(n_buttons) < 0.3).astype(float).tolist()


def player(idx, args, queue):
    # own process group, so the parent can stop the engine processes this player starts
    os.setpgrp()
    import vizdoom as vzd
    from vizdoom.pettingzoo_wrapper.base_env_common import configure_doom_game

    game = configure_doom_game(
        config_path=str(args.cfg),
        resolution=args.resolution,
        ticrate=vzd.DEFAULT_TICRATE,
        async_mode=False,
        timeout=None,
        seed=args.seed,
        is_host=idx == 0,
        num_agents=args.agents,
        host_address="127.0.0.1",
        port=args.port,
        netmode=0,
        agent_idx=idx,
    )
    for var in (
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.ANGLE,
    ):
        game.add_available_game_variable(var)
    game.set_objects_info_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.init()
    rate = (
        sampling_rate_hz(game.get_audio_sampling_rate())
        if game.is_audio_buffer_enabled()
        else None
    )
    rng = np.random.default_rng(args.seed * 100 + idx)
    n = game.get_available_buttons_size()
    frames = []  # per step, so audio plots can pick a step where a source is audible
    for t in range(args.tics):
        if args.say and idx == 0 and t == args.tics - 6:
            game.send_game_command(f"say {args.say}")
        frozen = args.still is not None and idx in args.still
        game.make_action([0.0] * n if frozen else scripted_action(n, rng), args.skip)
        if game.is_episode_finished():  # a random USE can end it: keep the last frame
            break
        state = game.get_state()
        frames.append(
            {
                "screen": state.screen_buffer.copy(),
                "pos": (
                    game.get_game_variable(vzd.GameVariable.POSITION_X),
                    game.get_game_variable(vzd.GameVariable.POSITION_Y),
                ),
                "angle": game.get_game_variable(vzd.GameVariable.ANGLE),
                "notifications": state.notifications_buffer,
                "audio": (
                    None if state.audio_buffer is None else state.audio_buffer.copy()
                ),
                "objects": [
                    (o.name, o.position_x, o.position_y) for o in state.objects
                ],
            }
        )
    out = dict(frames[-1], idx=idx, frames=frames, audio_rate=rate)
    queue.put(out)
    # no game.close(): in a netgame it can block holding the GIL before the
    # queue feeder has sent the snapshot; the parent terminates this process
    time.sleep(3600)


def udmf_map(text):
    """Lines, pits (death-sector outlines) and gates (player-blocking lines) of a UDMF map.

    Read from the file rather than the engine: scripts may toggle line flags at
    runtime, so a separate engine instance would not match the netgame.
    """
    text = re.sub(r"//[^\n]*", "", text)
    blocks = {}
    for kind, body in re.findall(r"(\w+)\s*\{([^}]*)\}", text):
        fields = [f.split("=", 1) for f in body.split(";") if "=" in f]
        blocks.setdefault(kind, []).append(
            {k.strip(): v.strip().strip('"') for k, v in fields}
        )
    verts = [(float(v["x"]), float(v["y"])) for v in blocks["vertex"]]
    side_sector = [int(s["sector"]) for s in blocks["sidedef"]]
    pit_sectors = {
        i for i, s in enumerate(blocks["sector"]) if s.get("special") == "115"
    }  # instant death
    lines, gates, pit_edges = [], [], {}
    for line in blocks["linedef"]:
        v1, v2 = int(line["v1"]), int(line["v2"])
        front = side_sector[int(line["sidefront"])]
        back = side_sector[int(line["sideback"])] if "sideback" in line else None
        seg = (*verts[v1], *verts[v2])
        if line.get("blockplayers") == "true" and front == back:
            gates.append(seg)
            continue
        lines.append((*seg, line.get("blocking") == "true" or back is None))
        for sector in {front, back} & pit_sectors:
            pit_edges.setdefault(sector, []).append((v1, v2))
    pits = []
    for edges in pit_edges.values():  # chain the outline into one closed ring
        nbrs = {}
        for a, b in edges:
            nbrs.setdefault(a, []).append(b)
            nbrs.setdefault(b, []).append(a)
        ring, prev = [edges[0][0]], None
        while True:
            nxt = next(v for v in nbrs[ring[-1]] if v != prev)
            if nxt == ring[0]:
                break
            prev = ring[-1]
            ring.append(nxt)
        pits.append([verts[v] for v in ring])
    return lines, pits, gates


def read_lumps(wad_path):
    data = Path(wad_path).read_bytes()
    count, offset = struct.unpack("<4xii", data[:12])
    lumps = {}
    for i in range(count):
        pos, size, name = struct.unpack(
            "<ii8s", data[offset + 16 * i : offset + 16 * i + 16]
        )
        lumps.setdefault(name.rstrip(b"\0").decode(), (pos, size))
    return data, lumps


def wad_lines(wad_path):
    """(x1, y1, x2, y2, blocking) from a binary-format map, or None for UDMF."""
    data, lumps = read_lumps(wad_path)
    if "LINEDEFS" not in lumps or "VERTEXES" not in lumps:
        return None
    pos, size = lumps["VERTEXES"]
    verts = [
        struct.unpack("<hh", data[pos + 4 * i : pos + 4 * i + 4])
        for i in range(size // 4)
    ]
    pos, size = lumps["LINEDEFS"]
    hexen = (
        "BEHAVIOR" in lumps
    )  # Hexen linedefs: v1 v2 flags special args[5] front back
    fmt, step = ("<HHHB5BHH", 16) if hexen else ("<HHHHHHH", 14)
    lines = []
    for i in range(size // step):
        fields = struct.unpack(fmt, data[pos + step * i : pos + step * (i + 1)])
        v1, v2, flags, back = fields[0], fields[1], fields[2], fields[-1]
        lines.append((*verts[v1], *verts[v2], bool(flags & 1) or back == 0xFFFF))
    return lines


def map_geometry(wad: Path):
    """Snapshot fields for the top-down view: lines, plus pits and gates for UDMF maps."""
    lines = wad_lines(wad)
    if lines is not None:
        return {"lines": lines}
    data, lumps = read_lumps(wad)
    if "TEXTMAP" in lumps:
        pos, size = lumps["TEXTMAP"]
        lines, pits, gates = udmf_map(data[pos : pos + size].decode())
        return {"lines": lines, "pits": pits, "gates": gates}
    return {"lines": []}


def write_exports(out_dir: Path, snapshot: dict) -> list[str]:
    """Agent-readable files next to the pickle; returns notes about skipped outputs."""
    notes = []
    summary = {"scenario": snapshot["scenario"], "wad": snapshot["wad"], "players": []}
    for snap in snapshot["players"]:
        idx = snap["idx"]
        entry = {
            "idx": idx,
            "pos": [round(float(v), 1) for v in snap["pos"]],
            "angle": round(float(snap["angle"]), 1),
            "notifications": snap["notifications"],
            "objects": [
                (name, round(float(x)), round(float(y)))
                for name, x, y in snap["objects"]
            ],
            "steps": len(snap["frames"]),
        }
        audio = snap.get("audio")
        if audio is not None and snap.get("audio_rate"):
            # each buffer holds the last N tics, so consecutive buffers overlap: keep the newest `skip` tics
            new = snapshot.get("skip", 1) * snap["audio_rate"] // TICRATE
            parts = [
                np.asarray(f["audio"], dtype=np.int16)[-new:]
                for f in snap["frames"]
                if f["audio"] is not None
            ]
            clip = np.concatenate(parts).reshape(-1, 2)
            with wave.open(str(out_dir / f"player{idx}_audio.wav"), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(int(snap["audio_rate"]))
                wav.writeframes(clip.tobytes())
            peaks = np.abs(clip.astype(np.int32)).max(axis=0)
            entry["audio_peak_left_right"] = [int(peaks[0]), int(peaks[1])]
            entry["audio_seconds"] = round(len(clip) / snap["audio_rate"], 2)
        summary["players"].append(entry)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        from PIL import Image

        for snap in snapshot["players"]:
            Image.fromarray(np.asarray(snap["screen"], dtype=np.uint8)).save(
                out_dir / f"player{snap['idx']}.png"
            )
    except ImportError:
        notes.append("Pillow not installed: player PNGs skipped")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        for x1, y1, x2, y2, blocking in snapshot["lines"]:
            ax.plot([x1, x2], [y1, y2], color="black" if blocking else "0.7", lw=1)
        for pit in snapshot.get("pits", []):
            ax.fill(*zip(*pit), color="tab:red", alpha=0.3)
        for x1, y1, x2, y2 in snapshot.get("gates", []):
            ax.plot([x1, x2], [y1, y2], color="tab:orange", ls="--")
        for snap in snapshot["players"]:
            (x, y), angle = snap["pos"], np.deg2rad(snap["angle"])
            ax.plot(x, y, "o", ms=8)
            ax.arrow(x, y, 48 * np.cos(angle), 48 * np.sin(angle), width=4)
            ax.annotate(
                f"P{snap['idx']}", (x, y), textcoords="offset points", xytext=(6, 6)
            )
        ax.set_aspect("equal")
        steps = len(snapshot["players"][0]["frames"])
        ax.set_title(f"{snapshot['scenario']}: player poses after {steps} steps")
        fig.savefig(out_dir / "topdown.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        notes.append("matplotlib not installed: topdown.png skipped")
    return notes


def collect(queue, procs, timeout):
    """One snapshot per player; raise at once if a player process died (e.g. a Python error)."""
    snaps, deadline = [], time.monotonic() + timeout
    while len(snaps) < len(procs):
        try:
            snaps.append(queue.get(timeout=1.0))
            continue
        except queue_module.Empty:
            pass
        dead = [i for i, p in enumerate(procs) if p.exitcode not in (None, 0)]
        if dead:
            raise RuntimeError(
                f"player process(es) {dead} exited with an error; see the traceback above"
            )
        if time.monotonic() > deadline:
            raise queue_module.Empty
    return snaps


def stop_players(procs):
    """Kill every player with its engine processes (netgame close() can block once peers left).

    ``Process.terminate()`` alone leaves the engine processes running; they keep
    the parent's stdout open, so a shell pipe on this script would never close.
    """
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
    for p in procs:
        p.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scenario", help="scenario .cfg path or scenario name")
    parser.add_argument(
        "--wad",
        help="WAD for the top-down map (default: doom_scenario_path of the cfg)",
    )
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument(
        "--tics", type=int, default=20, help="scripted steps per player"
    )
    parser.add_argument("--skip", type=int, default=4, help="tics per step")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=5130)
    parser.add_argument("--resolution", default="160X120")
    parser.add_argument(
        "--say", help="chat message the host sends 6 steps before the end"
    )
    parser.add_argument("--still", type=int, nargs="*", help="players that stay idle")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="snapshot.pkl, or a directory for pickle + PNG/WAV/JSON",
    )
    args = parser.parse_args()
    if not 1 <= args.agents <= 16:
        parser.error("--agents must be 1-16")
    args.cfg = resolve_cfg(args.scenario)
    wad = resolve_wad(args.cfg, args.wad)
    ensure_openal()

    ctx = mp.get_context("spawn")
    snaps = None
    for attempt in range(
        6
    ):  # netgame handshakes occasionally stall: retry on a new port
        queue = ctx.Queue()
        procs = [
            ctx.Process(target=player, args=(i, args, queue), daemon=True)
            for i in range(args.agents)
        ]
        for p in procs:  # host first, then the joiners
            p.start()
            time.sleep(1.0)
        try:
            snaps = sorted(collect(queue, procs, timeout=90), key=lambda s: s["idx"])
        except queue_module.Empty:
            print(f"attempt {attempt} stalled, retrying", flush=True)
            args.port += 7
        except RuntimeError:
            stop_players(procs)
            raise
        stop_players(procs)
        if snaps is not None:
            break
    if snaps is None:
        raise RuntimeError("could not start the netgame")

    snapshot = {
        "scenario": args.cfg.stem,
        "cfg": str(args.cfg),
        "wad": str(wad),
        "skip": args.skip,
        "players": snaps,
        **map_geometry(wad),
    }
    if args.out.suffix == ".pkl":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(pickle.dumps(snapshot))
        written = str(args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "snapshot.pkl").write_bytes(pickle.dumps(snapshot))
        for note in write_exports(args.out, snapshot):
            print(f"note: {note}")
        written = f"{args.out}/ (snapshot.pkl, summary.json, player*.png, topdown.png)"
    poses = ", ".join(
        f"P{s['idx']}@({s['pos'][0]:.0f},{s['pos'][1]:.0f})" for s in snaps
    )
    print(f"saved {written}: {poses}")


if __name__ == "__main__":
    main()
