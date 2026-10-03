#!/usr/bin/env python3
"""Summarize validated, indexed episodes from a Harbor pilot."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_pilot import (
    NATURAL_STATUSES,
    _codex_log_metrics,
    _collect_job,
    _job_root,
    _protocol_tasks,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--include-completed-trials",
        action="store_true",
        help="Include finished valid trials not yet journaled; write live_summary.json",
    )
    return parser.parse_args()


def _usage(record: dict[str, Any]) -> dict[str, float | int | None]:
    trial_dir = record.get("trial_dir")
    trajectory_path = Path(trial_dir) / "agent/trajectory.json" if trial_dir else None
    log_metrics = _codex_log_metrics(
        Path(trial_dir) / "agent/codex.txt" if trial_dir else None
    )
    if trajectory_path is None or not trajectory_path.is_file():
        return {"model_calls": None, "cost_usd": None, **log_metrics}
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    agent_steps = [
        step
        for step in trajectory.get("steps", [])
        if isinstance(step, dict) and step.get("source") == "agent"
    ]
    metrics = trajectory.get("final_metrics") or {}
    return {
        "model_calls": sum(step.get("llm_call_count", 0) for step in agent_steps),
        "cost_usd": metrics.get("total_cost_usd"),
        **log_metrics,
    }


def _mean(values: list[float | int]) -> float | None:
    return statistics.mean(values) if values else None


def _valid_trial(record: dict[str, Any], task: str) -> bool:
    trial_dir = record.get("trial_dir")
    trial_result_path = Path(trial_dir) / "result.json" if trial_dir else None
    if trial_result_path is None or not trial_result_path.is_file():
        return False
    trial_result = json.loads(trial_result_path.read_text(encoding="utf-8"))
    if trial_result.get("exception_info") is not None:
        return False
    if not all(
        record.get("artifacts", {}).get(name) is True
        for name in ("result.json", "events.jsonl", "recording.mp4")
    ):
        return False
    if record.get("trajectory_present") is not True:
        return False
    if not _codex_log_metrics(Path(trial_dir) / "agent/codex.txt")[
        "image_tool_results"
    ]:
        return False
    if not isinstance(record.get("verifier_reward"), (int, float)):
        return False
    if record.get("status") not in NATURAL_STATUSES:
        return False
    if task == "defend-the-center":
        return isinstance(record.get("episode_return"), (int, float))
    if task == "health-gathering":
        return isinstance(record.get("survival_seconds"), (int, float))
    return True


def _completed_trial_records(
    run_dir: Path,
    protocol: dict[str, Any],
    journal_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read completed original trials without changing the runner's journal."""
    selected = {
        (record["model"], record["seed"], record["task"]) for record in journal_records
    }
    discovered = []
    skipped = []
    for model in protocol["models"]:
        for seed in protocol["seeds"]:
            jobs = sorted(_job_root(run_dir, model, protocol, seed).glob("vizdoom-*"))
            if not jobs:
                continue
            for task in _protocol_tasks(protocol):
                cell = model, seed, task
                if cell in selected:
                    continue
                candidates = sorted(jobs[-1].glob(f"{task}__*"))
                if not candidates:
                    continue
                trial_dir = candidates[-1]
                trial_result_path = trial_dir / "result.json"
                if not trial_result_path.is_file():
                    continue
                candidate = {
                    "model": model,
                    "seed": seed,
                    "task": task,
                    "trial_dir": str(trial_dir),
                }
                try:
                    trial_result = json.loads(
                        trial_result_path.read_text(encoding="utf-8")
                    )
                    if not trial_result.get("finished_at"):
                        continue
                    record = _collect_job(
                        run_dir, {**protocol, "tasks": [task]}, model, seed, 0
                    )[0]
                    if not _valid_trial(record, task):
                        skipped.append(
                            {**candidate, "reason": "finished_trial_not_valid_natural"}
                        )
                        continue
                except (OSError, json.JSONDecodeError) as error:
                    skipped.append(
                        {
                            **candidate,
                            "reason": "transient_evidence_read_error",
                            "error_type": type(error).__name__,
                        }
                    )
                    continue
                record["trial_finished_at"] = trial_result["finished_at"]
                # The individual trial finished, but its encompassing Harbor
                # job can still be running other tasks; no job exit is observed.
                record["harbor_returncode"] = None
                discovered.append(record)
                selected.add(cell)
    return discovered, skipped


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    index_path = run_dir / "runs.jsonl"
    indexed_records = (
        [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if line.strip()
            and (not args.include_completed_trials or line.endswith("\n"))
        ]
        if index_path.is_file()
        else []
    )
    setup_failures = [
        record for record in indexed_records if record.get("trial_dir") is None
    ]
    records = [
        record for record in indexed_records if record.get("trial_dir") is not None
    ]
    journal_indexed_trials = len(records)
    discovered_records = []
    skipped_candidates = []
    if args.include_completed_trials:
        discovered_records, skipped_candidates = _completed_trial_records(
            run_dir, protocol, records
        )
        records.extend(discovered_records)
    retry_index = run_dir / "retry_attempts.jsonl"
    retry_records = (
        [
            json.loads(line)
            for line in retry_index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if retry_index.is_file()
        else []
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    retries_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["task"])].append(record)
    for record in retry_records:
        retries_by_key[(record["model"], record["seed"], record["task"])].append(record)

    summary: dict[str, Any] = {
        "protocol": protocol,
        "indexed_trials": len(records),
        "prior_setup_failed_jobs": sorted(
            {(record["model"], record["seed"]) for record in setup_failures}
        ),
        "incomplete_retry_attempts": len(retry_records),
        "expected_trials": len(protocol["seeds"])
        * len(protocol["models"])
        * len(_protocol_tasks(protocol)),
        "conditions": {},
    }
    if args.include_completed_trials:
        summary.update(
            {
                "snapshot_mode": "read_only_completed_trials",
                "journal_indexed_trials": journal_indexed_trials,
                "discovered_completed_trials": len(discovered_records),
                "discovered_completed_records": discovered_records,
                "skipped_live_candidates": skipped_candidates,
            }
        )
    for model in protocol["models"]:
        model_summary = summary["conditions"].setdefault(model, {})
        for task in _protocol_tasks(protocol):
            task_records = grouped[(model, task)]
            seed_counts = [record["seed"] for record in task_records]
            if len(seed_counts) != len(set(seed_counts)):
                raise RuntimeError(f"Duplicate indexed trial for {model}/{task}")
            valid_records = []
            recovered_seeds = []
            for record in task_records:
                if _valid_trial(record, task):
                    valid_records.append(record)
                    continue
                retries = sorted(
                    retries_by_key[(model, record["seed"], task)],
                    key=lambda retry: retry["attempt"],
                )
                recovered = next(
                    (retry for retry in retries if _valid_trial(retry, task)), None
                )
                if recovered is not None:
                    valid_records.append(recovered)
                    recovered_seeds.append(record["seed"])
            for record in valid_records:
                record.update(_usage(record))
            goal = task not in ("defend-the-center", "health-gathering")
            primary_values = [
                (
                    float(record.get("success") is True)
                    if goal
                    else float(
                        record["episode_return"]
                        if task == "defend-the-center"
                        else record["survival_seconds"]
                    )
                )
                for record in valid_records
            ]
            model_summary[task] = {
                "indexed": len(task_records),
                "expected": len(protocol["seeds"]),
                "valid_natural": len(valid_records),
                "recovered_incomplete_seeds": recovered_seeds,
                "retry_attempts": sum(
                    len(retries_by_key[(model, record["seed"], task)])
                    for record in task_records
                ),
                "original_invalid_or_interrupted": [
                    {"seed": record["seed"], "status": record.get("status")}
                    for record in task_records
                    if not _valid_trial(record, task)
                ],
                "unresolved_invalid": [
                    {"seed": record["seed"], "status": record.get("status")}
                    for record in task_records
                    if not _valid_trial(record, task)
                    and record["seed"] not in recovered_seeds
                ],
                "primary_measure": (
                    "success_rate"
                    if goal
                    else (
                        "episode_return"
                        if task == "defend-the-center"
                        else "survival_seconds"
                    )
                ),
                "primary_mean": _mean(primary_values),
                "successes": (
                    sum(record.get("success") is True for record in valid_records)
                    if goal
                    else None
                ),
                "mean_decisions": _mean(
                    [record["decisions"] for record in valid_records]
                ),
                "mean_model_calls": _mean(
                    [
                        record["model_calls"]
                        for record in valid_records
                        if record["model_calls"] is not None
                    ]
                ),
                "mean_cli_turns": _mean(
                    [
                        record["cli_turns"]
                        for record in valid_records
                        if record.get("cli_turns") is not None
                    ]
                ),
                "mean_image_tool_results": _mean(
                    [
                        record["image_tool_results"]
                        for record in valid_records
                        if record.get("image_tool_results") is not None
                    ]
                ),
                "mean_cost_usd": _mean(
                    [
                        record["cost_usd"]
                        for record in valid_records
                        if record["cost_usd"] is not None
                    ]
                ),
                "mean_narration_coverage": _mean(
                    [
                        record["narration_coverage"]
                        for record in valid_records
                        if isinstance(record.get("narration_coverage"), (int, float))
                    ]
                ),
                "seeds": {
                    str(record["seed"]): record.get("status")
                    for record in valid_records
                },
                "per_seed_primary": {
                    str(record["seed"]): (
                        float(record.get("success") is True)
                        if goal
                        else float(
                            record["episode_return"]
                            if task == "defend-the-center"
                            else record["survival_seconds"]
                        )
                    )
                    for record in valid_records
                },
            }
    output_path = run_dir / (
        "live_summary.json" if args.include_completed_trials else "summary.json"
    )
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for model, conditions in summary["conditions"].items():
        for task, values in conditions.items():
            print(
                f"{model} {task}: {values['valid_natural']}/{values['expected']} "
                f"natural; {values['primary_measure']}={values['primary_mean']}"
            )
    print(f"summary={output_path}")


if __name__ == "__main__":
    main()
