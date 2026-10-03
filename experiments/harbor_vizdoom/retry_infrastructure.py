#!/usr/bin/env python3
"""Retry one incomplete pilot trial without hiding the original attempt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_tasks import load_task_definitions
from run_pilot import EXPERIMENT_DIR, MODELS, _collect_job, _protocol_tasks
from summarize_pilot import _valid_trial


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--task",
        choices=[task["slug"] for task in load_task_definitions()],
        required=True,
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=int, default=45)
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if not 0 <= args.backoff_seconds <= 60:
        parser.error("--backoff-seconds must be between 0 and 60")
    return args


def _original_failure(
    run_dir: Path, model: str, seed: int, task: str
) -> dict[str, Any]:
    index_path = run_dir / "runs.jsonl"
    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        record
        for record in records
        if (record["model"], record["seed"], record["task"]) == (model, seed, task)
        and record.get("trial_dir") is not None
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one indexed original trial")
    original = matches[0]
    if _valid_trial(original, task):
        raise RuntimeError("Original trial is already a valid natural ending")
    trial_dir = original.get("trial_dir")
    if not trial_dir:
        raise RuntimeError("Original trial has no Harbor trial directory")
    trial_result = json.loads(
        (Path(trial_dir) / "result.json").read_text(encoding="utf-8")
    )
    exception_type = (trial_result.get("exception_info") or {}).get("exception_type")
    if exception_type == "ApiOverloadedError":
        reason = exception_type
    elif original["status"] == "agent_stopped" and exception_type is None:
        reason = "EarlyAgentExit"
    else:
        raise RuntimeError(
            "Original trial is not an eligible incomplete attempt: "
            f"status={original['status']} exception={exception_type}"
        )
    return {**original, "retry_reason": reason}


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if args.seed not in protocol["seeds"] or args.model not in protocol["models"]:
        raise RuntimeError("Model and seed must belong to this pilot")
    if args.task not in _protocol_tasks(protocol):
        raise RuntimeError("Task must belong to this pilot")
    original = _original_failure(run_dir, args.model, args.seed, args.task)
    retry_index = run_dir / "retry_attempts.jsonl"
    prior = (
        [
            json.loads(line)
            for line in retry_index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if retry_index.is_file()
        else []
    )
    prior_matches = [
        record
        for record in prior
        if (record["model"], record["seed"], record["task"])
        == (args.model, args.seed, args.task)
    ]
    if any(_valid_trial(record, args.task) for record in prior_matches):
        print("A validated natural-ending retry already exists; no new trial launched.")
        return
    start_attempt = len(prior_matches) + 1
    for attempt in range(start_attempt, start_attempt + args.max_attempts):
        attempt_dir = (
            run_dir
            / "infrastructure_retries"
            / f"{args.model.replace('/', '_')}__seed-{args.seed}__{args.task}__attempt-{attempt}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(EXPERIMENT_DIR / "run_experiment.py"),
            "--agent",
            "codex",
            "--agent-auth",
            protocol["auth"],
            "--codex-model",
            args.model,
            "--codex-reasoning",
            protocol["reasoning"],
            "--seed",
            str(args.seed),
            "--jobs-root",
            str(attempt_dir / "jobs"),
            "--concurrent-trials",
            "1",
            "--task",
            args.task,
        ]
        if protocol.get("codex_agent_version"):
            command.extend(("--codex-version", protocol["codex_agent_version"]))
        print(
            f"retry model={args.model} seed={args.seed} task={args.task} attempt={attempt}",
            flush=True,
        )
        with (attempt_dir / "harbor.log").open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=EXPERIMENT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = next(
            record
            for record in _collect_job(
                attempt_dir, protocol, args.model, args.seed, process.returncode
            )
            if record["task"] == args.task
        )
        record.update(
            {
                "attempt": attempt,
                "retry_of_trial_dir": original["trial_dir"],
                "retry_reason": original["retry_reason"],
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        with retry_index.open("a", encoding="utf-8") as index_file:
            index_file.write(json.dumps(record, sort_keys=True) + "\n")
            index_file.flush()
        if _valid_trial(record, args.task):
            print(f"natural_end={record['status']} attempt={attempt}", flush=True)
            return
        print(f"retry_incomplete status={record['status']}", flush=True)
        if attempt < start_attempt + args.max_attempts - 1:
            time.sleep(args.backoff_seconds)
    raise SystemExit("No retry reached a validated natural ending")


if __name__ == "__main__":
    main()
