#!/usr/bin/env python3
"""Retry eligible interrupted pilot trials and require five natural ends each."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from retry_infrastructure import _original_failure
from run_pilot import EXPERIMENT_DIR, _protocol_tasks
from summarize_pilot import _valid_trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    indexed = [
        json.loads(line)
        for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_records = [
        record for record in indexed if record.get("trial_dir") is not None
    ]
    originals = {
        (record["model"], record["seed"], record["task"]): record
        for record in actual_records
    }
    if len(originals) != len(actual_records):
        raise RuntimeError("Multiple launched originals share a model/seed/task cell")
    expected = {
        (model, seed, task)
        for model in protocol["models"]
        for seed in protocol["seeds"]
        for task in _protocol_tasks(protocol)
    }
    missing = expected - originals.keys()
    if missing:
        raise RuntimeError(
            f"Main pilot has {len(missing)} unlaunched episodes; "
            "finish run_pilot.py before retrying interruptions"
        )

    retry_order = {
        "basic": 0,
        "defend-the-center": 1,
        "my-way-home": 2,
        "health-gathering": 3,
    }
    for model, seed, task in sorted(
        expected, key=lambda cell: (cell[0], retry_order.get(cell[2], 4), cell[1])
    ):
        record = originals[(model, seed, task)]
        if _valid_trial(record, task):
            continue
        failure = _original_failure(run_dir, model, seed, task)
        print(
            f"recovering model={model} seed={seed} task={task} "
            f"reason={failure['retry_reason']}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(EXPERIMENT_DIR / "retry_infrastructure.py"),
                str(run_dir),
                "--model",
                model,
                "--seed",
                str(seed),
                "--task",
                task,
                "--max-attempts",
                str(args.max_attempts),
            ],
            cwd=EXPERIMENT_DIR,
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "summarize_pilot.py"),
            str(run_dir),
        ],
        cwd=EXPERIMENT_DIR,
        check=True,
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    incomplete = [
        (model, task, values["valid_natural"])
        for model, tasks in summary["conditions"].items()
        for task, values in tasks.items()
        if values["valid_natural"] != len(protocol["seeds"])
    ]
    if incomplete:
        raise RuntimeError(f"Pilot still has incomplete conditions: {incomplete}")
    print(f"All {len(expected)} model/task/seed cells have validated natural endings.")


if __name__ == "__main__":
    main()
