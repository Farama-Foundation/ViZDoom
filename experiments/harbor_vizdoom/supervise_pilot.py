#!/usr/bin/env python3
"""Monitor a running pilot, guard recording memory, and validate final artifacts.

This does not launch original trials: run it alongside run_pilot.py. Optional
memory management pauses only agent containers belonging to this run, never the
game servers. Pauses affect wall time, not simulated time, and are logged.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_pilot import EXPERIMENT_DIR, _protocol_tasks
from summarize_pilot import _valid_trial


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # The runner can be appending its final line while we read the file.
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if line.strip() and line.endswith("\n")
    ]


def _cell(record: dict[str, Any]) -> tuple[str, int, str]:
    return record["model"], record["seed"], record["task"]


def _log(run_dir: Path, event: str, **fields: Any) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with (run_dir / "supervision.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    _stdout(json.dumps(record))


def _stdout(message: str) -> None:
    """Keep supervision running if its optional stdout consumer disconnects."""
    try:
        print(message, flush=True)
    except OSError as error:
        if not isinstance(error, BrokenPipeError) and error.errno != errno.EPIPE:
            raise


def _resume_owned_pauses(run_dir: Path, owned_pauses: set[str]) -> None:
    """Attempt every unpause before reporting a durable logging failure."""
    logging_errors = []
    for name in sorted(owned_pauses):
        try:
            _docker("unpause", name)
        except (subprocess.SubprocessError, OSError) as error:
            event, fields = "cleanup_warning", {"detail": str(error)}
        else:
            event, fields = "resume", {"reason": "supervisor_exit"}
        try:
            _log(run_dir, event, container=name, **fields)
        except OSError as error:
            logging_errors.append(error)
    if logging_errors:
        raise logging_errors[0]


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=30
    ).stdout


def _owned_containers(run_dir: Path) -> dict[str, dict[str, Any]]:
    names = {
        f"{path.parent.name.lower()}__env-{service}-1"
        for path in run_dir.glob("jobs/**/trial.log")
        for service in ("main", "vizdoom")
    }
    present = set(_docker("ps", "--format", "{{.Names}}").splitlines()) & names
    if not present:
        return {}
    return {
        entry["Name"].lstrip("/"): entry
        for entry in json.loads(_docker("inspect", *sorted(present)))
        if entry["State"]["Running"]
    }


def _memory_mib(meminfo: str) -> float:
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise ValueError("Docker VM did not report MemAvailable")


def _pause_plan(
    agents: dict[str, dict[str, Any]],
    owned_pauses: set[str],
    progress: dict[str, int],
    available_mib: float,
    conserving: bool,
    reserve_mib: int,
    resume_mib: int,
) -> tuple[bool, set[str], set[str]]:
    """Choose a single furthest-progressed episode under memory pressure."""
    eligible = {
        name
        for name, state in agents.items()
        if not state["Paused"] or name in owned_pauses
    }
    if available_mib < reserve_mib:
        conserving = True
    elif available_mib >= resume_mib:
        conserving = False
    if not eligible:
        return conserving, set(), set()
    keep = max(eligible, key=lambda name: (progress.get(name, 0), name))
    pause = {
        name
        for name in eligible
        if conserving and name != keep and not agents[name]["Paused"]
    }
    resume = {
        name
        for name in eligible & owned_pauses
        if agents[name]["Paused"] and (not conserving or name == keep)
    }
    return conserving, pause, resume


def _run_helper(name: str, run_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / name), str(run_dir)], check=True
    )


def _validate_videos(run_dir: Path, protocol: dict[str, Any]) -> None:
    import imageio_ffmpeg

    records = _jsonl(run_dir / "runs.jsonl") + _jsonl(run_dir / "retry_attempts.jsonl")
    validated = []
    selected = set()
    for record in records:
        cell = _cell(record)
        if cell in selected or not _valid_trial(record, record["task"]):
            continue
        results = Path(record["trial_dir"]) / "artifacts/vizdoom-results"
        result = json.loads((results / "result.json").read_text(encoding="utf-8"))
        expected_cap = (
            protocol.get("task_settings", {})
            .get(record["task"], {})
            .get("episode_timeout_seconds", 0)
        )
        if result.get("episode_timeout_seconds", 0) != expected_cap:
            raise RuntimeError(f"Mismatched game-time limit: {cell}")
        if expected_cap and result["elapsed_game_tics"] > expected_cap * 35:
            raise RuntimeError(f"Game-time limit exceeded: {cell}")
        if not result.get("video_finalized"):
            raise RuntimeError(f"Video was not finalized: {cell}")
        decoded = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(results / "recording.mp4"),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        if decoded.stderr.strip():
            raise RuntimeError(f"Video decode errors for {cell}: {decoded.stderr}")
        selected.add(cell)
        validated.append(
            {
                "model": cell[0],
                "seed": cell[1],
                "task": cell[2],
                "trial_dir": record["trial_dir"],
                "video_decode_ok": True,
            }
        )
    expected = {
        (model, seed, task)
        for model in protocol["models"]
        for seed in protocol["seeds"]
        for task in _protocol_tasks(protocol)
    }
    if selected != expected:
        raise RuntimeError(
            f"Missing validated artifacts: {sorted(expected - selected)}"
        )
    (run_dir / "artifact_validation.json").write_text(
        json.dumps({"validated_trials": validated}, indent=2) + "\n", encoding="utf-8"
    )


def _supervise(args: argparse.Namespace, run_dir: Path) -> None:
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    expected = {
        (model, seed, task)
        for model in protocol["models"]
        for seed in protocol["seeds"]
        for task in _protocol_tasks(protocol)
    }
    owned_pauses: set[str] = set()
    for event in _jsonl(run_dir / "supervision.jsonl"):
        if event["event"] == "pause":
            owned_pauses.add(event["container"])
        elif event["event"] == "resume":
            owned_pauses.discard(event["container"])
    conserving = bool(owned_pauses)
    previous_count = -1
    try:
        _log(
            run_dir,
            "started",
            manage_memory=args.manage_memory,
            reserve_mib=args.reserve_mib,
            resume_mib=args.resume_mib,
            supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        while True:
            records = [r for r in _jsonl(run_dir / "runs.jsonl") if r.get("trial_dir")]
            if len(records) != previous_count:
                _run_helper("summarize_pilot.py", run_dir)
                previous_count = len(records)
            if expected <= {_cell(record) for record in records}:
                break
            try:
                containers = _owned_containers(run_dir)
                owned_pauses.intersection_update(containers)
                servers = [
                    name for name in containers if name.endswith("__env-vizdoom-1")
                ]
                progress = {}
                finished = set()
                available = None
                if servers:
                    available = _memory_mib(
                        _docker("exec", servers[0], "cat", "/proc/meminfo")
                    )
                for server in servers:
                    result = json.loads(
                        _docker("exec", server, "cat", "/vizdoom-results/result.json")
                    )
                    agent = server.replace("__env-vizdoom-1", "__env-main-1")
                    progress[agent] = result.get("decisions", 0)
                    if result.get("status") not in (None, "running"):
                        finished.add(agent)
                if args.manage_memory and available is not None:
                    agents = {
                        name: info["State"]
                        for name, info in containers.items()
                        if name.endswith("__env-main-1") and name not in finished
                    }
                    conserving, pause, resume = _pause_plan(
                        agents,
                        owned_pauses,
                        progress,
                        available,
                        conserving,
                        args.reserve_mib,
                        args.resume_mib,
                    )
                    # Let terminal episodes finish artifact collection promptly.
                    resume.update(finished & owned_pauses)
                    for name in sorted(pause):
                        _docker("pause", name)
                        owned_pauses.add(name)
                        _log(run_dir, "pause", container=name, available_mib=available)
                    for name in sorted(resume):
                        _docker("unpause", name)
                        owned_pauses.discard(name)
                        _log(run_dir, "resume", container=name, available_mib=available)
                _log(
                    run_dir,
                    "status",
                    indexed=len(records),
                    expected=len(expected),
                    available_mib=available,
                    paused=sorted(owned_pauses),
                    decisions=progress,
                )
            except (
                subprocess.SubprocessError,
                json.JSONDecodeError,
                ValueError,
            ) as error:
                # Containers can disappear during artifact collection. Retry discovery.
                _log(run_dir, "monitor_warning", detail=str(error))
            time.sleep(args.poll_seconds)
    finally:
        # Never leave a live container paused when supervision is interrupted.
        _resume_owned_pauses(run_dir, owned_pauses)
    _run_helper("recover_pilot.py", run_dir)
    _validate_videos(run_dir, protocol)
    _log(run_dir, "complete", validated_trials=len(expected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manage-memory", action="store_true")
    parser.add_argument("--reserve-mib", type=int, default=2048)
    parser.add_argument("--resume-mib", type=int, default=3584)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1 or not 0 < args.reserve_mib < args.resume_mib:
        parser.error("Require poll-seconds > 0 and 0 < reserve-mib < resume-mib")
    run_dir = args.run_dir.expanduser().resolve()
    if not (run_dir / "protocol.json").is_file():
        parser.error("run_dir must contain protocol.json")

    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with (run_dir / "supervisor.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            _supervise(args, run_dir)
        except Exception as error:
            _log(run_dir, "failed", detail=str(error))
            raise


if __name__ == "__main__":
    main()
