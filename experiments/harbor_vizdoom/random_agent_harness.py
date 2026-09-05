#!/usr/bin/env python3
"""Smoke-test ViZDoom tasks with the shared server's random-agent mode."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_tasks import load_task_definitions, select_task_definitions

EXPERIMENT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = EXPERIMENT_DIR / "template"
SERVER_PATH = TEMPLATE_DIR / "common/environment/vizdoom-server/server.py"
GRADER_PATH = TEMPLATE_DIR / "common/tests/grade.py"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "random_agent_runs"


def _unsigned_seed(raw_seed: str | None) -> int:
    if raw_seed is None:
        return secrets.randbits(32)
    seed = int(raw_seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", help="Unsigned 32-bit seed; random by default")
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Test only this manifest task slug; repeat to select several",
    )
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--button-probability", type=float, default=0.15)
    parser.add_argument(
        "--memory-every",
        type=int,
        default=4,
        help="Exercise store/retrieve every N actions; 0 disables it",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Artifact root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if not 0 <= args.button_probability <= 1:
        parser.error("--button-probability must be between 0 and 1")
    if args.memory_every < 0:
        parser.error("--memory-every must be non-negative")
    return args


def _task_environment(
    task: dict[str, Any], results_dir: Path, seed: int
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "VIZDOOM_ENVIRONMENT_ID": task["environment_id"],
            "VIZDOOM_IWAD_PATH": "",
            "VIZDOOM_MAX_DECISIONS": str(task["max_decisions"]),
            "VIZDOOM_OBJECTIVE": task["objective"],
            "VIZDOOM_REQUIRE_AUTOMAP": "1" if task["require_automap"] else "0",
            "VIZDOOM_RESULTS_DIR": str(results_dir),
            "VIZDOOM_SEED": str(seed),
        }
    )
    iwad = task["iwad"]
    if iwad is not None:
        environment["VIZDOOM_IWAD_PATH"] = str(
            (TEMPLATE_DIR / iwad["template_file"]).resolve()
        )
    return environment


def _run_task(
    task: dict[str, Any],
    run_dir: Path,
    *,
    seed: int,
    max_actions: int,
    button_probability: float,
    memory_every: int,
) -> dict[str, Any]:
    task_dir = run_dir / task["slug"]
    results_dir = task_dir / "vizdoom-results"
    task_dir.mkdir(parents=True)
    command = [
        sys.executable,
        str(SERVER_PATH),
        "--random-agent",
        "--max-actions",
        str(max_actions),
        "--button-probability",
        str(button_probability),
        "--policy-seed",
        str(seed),
        "--memory-every",
        str(memory_every),
    ]
    server_completed = subprocess.run(
        command,
        env=_task_environment(task, results_dir, seed),
        text=True,
        capture_output=True,
        check=False,
    )
    (task_dir / "server.stdout.log").write_text(
        server_completed.stdout, encoding="utf-8"
    )
    (task_dir / "server.stderr.log").write_text(
        server_completed.stderr, encoding="utf-8"
    )

    verifier_log_dir = task_dir / "verifier"
    verifier_environment = os.environ.copy()
    verifier_environment.update(
        {
            "EXPECTED_ENVIRONMENT_ID": task["environment_id"],
            "EXPECTED_MAX_DECISIONS": str(task["max_decisions"]),
            "VERIFIER_LOG_DIR": str(verifier_log_dir),
            "VIZDOOM_RESULTS_DIR": str(results_dir),
        }
    )
    verifier_completed = subprocess.run(
        [sys.executable, str(GRADER_PATH)],
        env=verifier_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (task_dir / "verifier.stdout.log").write_text(
        verifier_completed.stdout, encoding="utf-8"
    )
    (task_dir / "verifier.stderr.log").write_text(
        verifier_completed.stderr, encoding="utf-8"
    )

    result_path = results_dir / "result.json"
    result: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                result = loaded
        except json.JSONDecodeError:
            pass
    required_artifacts = ("result.json", "events.jsonl", "recording.mp4")
    missing_artifacts = [
        name for name in required_artifacts if not (results_dir / name).is_file()
    ]
    metrics_path = verifier_log_dir / "metrics.json"
    metrics: dict[str, Any] | None = None
    if metrics_path.is_file():
        try:
            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metrics = loaded
        except json.JSONDecodeError:
            pass
    passed = (
        server_completed.returncode == 0
        and verifier_completed.returncode == 0
        and not missing_artifacts
    )
    return {
        "task": task["slug"],
        "environment": task["environment_id"],
        "passed": passed,
        "server_returncode": server_completed.returncode,
        "verifier_returncode": verifier_completed.returncode,
        "gameplay_reward": metrics.get("reward") if metrics else None,
        "missing_artifacts": missing_artifacts,
        "status": result.get("status") if result else None,
        "success": result.get("success") if result else None,
        "decisions": result.get("decisions") if result else None,
        "episode_return": result.get("episode_return") if result else None,
        "results_dir": str(results_dir),
    }


def main() -> None:
    args = _parse_args()
    try:
        seed = _unsigned_seed(args.seed)
    except ValueError as exc:
        raise SystemExit(f"invalid --seed: {args.seed!r}") from exc
    try:
        tasks = select_task_definitions(load_task_definitions(), args.tasks)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = (
        args.output_root.expanduser().resolve()
        / "agent-random"
        / "reasoning-none"
        / f"seed-{seed}"
        / timestamp
    )
    run_dir.mkdir(parents=True)
    config = {
        "agent": "random",
        "reasoning": None,
        "seed": seed,
        "max_actions": args.max_actions,
        "button_probability": args.button_probability,
        "memory_every": args.memory_every,
        "tasks": [task["slug"] for task in tasks],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summaries = []
    print(f"seed={seed}")
    print(f"run_dir={run_dir}")
    for task in tasks:
        summary = _run_task(
            task,
            run_dir,
            seed=seed,
            max_actions=args.max_actions,
            button_probability=args.button_probability,
            memory_every=args.memory_every,
        )
        summaries.append(summary)
        print(
            f"{summary['task']}: passed={summary['passed']} "
            f"status={summary['status']} decisions={summary['decisions']}"
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all(summary["passed"] for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
