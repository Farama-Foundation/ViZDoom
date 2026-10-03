#!/usr/bin/env python3
"""Export validated pilot summaries as a paper table and its supporting data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_pilot import BASELINE_MODELS, MODELS, N_REPEATS, NATURAL_STATUSES, TASKS


def _read_summary(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read once so the data and source hash describe the same snapshot."""
    raw = path.read_bytes()
    return raw, json.loads(raw)


LABELS = {
    "basic": "Basic (goal)",
    "my-way-home": "My Way Home (goal)",
    "defend-the-center": "Defend the Center (return)",
    "health-gathering": "Health Gathering (s)",
    "doom-e1m1": "Doom E1M1: Hangar (exit)",
    "doom-e1m2": "Doom E1M2: Nuclear Plant (exit)",
    "doom2-map01": "Doom II MAP01: Entryway (exit)",
    "doom2-map02": "Doom II MAP02: Underhalls (exit)",
}
LEVELS = ("doom-e1m1", "doom-e1m2", "doom2-map01", "doom2-map02")
MATCHED_FIELDS = (
    "seeds",
    "tasks",
    "reasoning",
    "auth",
    "frame_skip",
    "max_action_repeat",
)
GAMEPLAY_FIELDS = (
    "environment_id",
    "outcome_kind",
    "episode_timeout_seconds",
    "max_decisions",
    "require_automap",
    "iwad",
)
MODEL_LABELS = (
    r"\shortstack{5.6\\Luna}",
    r"\shortstack{5.6\\Terra}",
    r"\shortstack{5.6\\Sol}",
    r"\shortstack{6\\Luna}",
    r"\shortstack{6\\Sol}",
    r"\shortstack{6.1\\Sol}",
)
# The initial standard batches predate task_settings in protocol.json. These
# are their published environment/budget controls, not reconstructed hashes of
# a source snapshot that those batches did not save.
LEGACY_TASK_SETTINGS = {
    task: {
        "environment_id": environment,
        "outcome_kind": outcome,
        "episode_timeout_seconds": 0,
        "max_decisions": budget,
        "require_automap": False,
        "iwad": None,
    }
    for task, environment, outcome, budget in (
        ("basic", "VizdoomBasic-MultiBinary-v1", "goal", 75),
        ("my-way-home", "VizdoomMyWayHome-MultiBinary-v1", "goal", 525),
        ("defend-the-center", "VizdoomDefendCenter-MultiBinary-v1", "combat", 525),
        ("health-gathering", "VizdoomHealthGathering-MultiBinary-v1", "survival", 525),
    )
}


def _gameplay_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        field: settings.get(field, 0 if field == "episode_timeout_seconds" else None)
        for field in GAMEPLAY_FIELDS
    }


def _source(path: Path, raw: bytes, summary: dict[str, Any]) -> dict[str, Any]:
    protocol = summary["protocol"]
    return {
        "run_id": path.parent.name,
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "models": protocol["models"],
        "tasks": protocol["tasks"],
        "created_at": protocol["created_at"],
        "indexed_trials": summary["indexed_trials"],
        "expected_trials": summary["expected_trials"],
        "retry_attempts": summary["incomplete_retry_attempts"],
        "source_sha256": protocol.get("source_sha256", {}),
        "iwad_sha256": protocol.get("iwad_sha256", {}),
        "reused_outcomes": protocol.get("reused_outcomes", []),
        "codex_agent_version": protocol.get("codex_agent_version"),
        "host_codex_version": protocol.get("codex_version"),
        "harbor_version": protocol.get("harbor_version"),
        "git_revision": protocol.get("git_revision"),
        "harness_changes": protocol.get("harness_changes", []),
        "task_settings": protocol.get("task_settings", {}),
        "legacy_task_controls": "task_settings" not in protocol,
        "snapshot_mode": summary.get("snapshot_mode", "journal_index"),
        "journal_indexed_trials": summary.get(
            "journal_indexed_trials", summary["indexed_trials"]
        ),
        "discovered_completed_trials": summary.get("discovered_completed_trials", 0),
        "discovered_completed_records": summary.get("discovered_completed_records", []),
    }


def _partial_result(
    result: dict[str, Any], seeds: set[str], model: str, task: str
) -> dict[str, Any]:
    outcomes = result["per_seed_primary"]
    statuses = result["seeds"]
    n = result["valid_natural"]
    if not (
        result["expected"] == N_REPEATS
        and 0 <= n <= N_REPEATS
        and len(outcomes) == n
        and set(outcomes) == set(statuses)
        and set(outcomes) <= seeds
    ):
        raise ValueError(f"Invalid completed sample: {model}/{task}")
    if not set(statuses.values()) <= NATURAL_STATUSES:
        raise ValueError(f"Non-terminal outcome: {model}/{task}")
    values = list(outcomes.values())
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite outcomes: {model}/{task}")
    goal = task not in ("defend-the-center", "health-gathering")
    if goal and (
        not all(value in (0, 1) for value in values)
        or sum(values) != result["successes"]
        or any(
            outcomes[seed] != (status == "success") for seed, status in statuses.items()
        )
    ):
        raise ValueError(f"Invalid success counts: {model}/{task}")
    if (
        n
        and not math.isclose(
            statistics.mean(values), result["primary_mean"], abs_tol=1e-12
        )
    ) or (not n and result["primary_mean"] is not None):
        raise ValueError(f"Invalid outcome mean: {model}/{task}")
    return {
        **result,
        "partial": n < N_REPEATS,
        "sample_standard_deviation": statistics.stdev(values) if n > 1 else None,
    }


def combine_summaries(paths: list[Path]) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "sources": [],
        "conditions": {},
        "condition_sources": {},
        "standard_task_settings": {},
    }
    reference: dict[str, Any] | None = None
    for path in paths:
        raw, summary = _read_summary(path)
        protocol = summary["protocol"]
        settings = {key: protocol[key] for key in MATCHED_FIELDS}
        if reference is None:
            reference = settings
        elif settings != reference:
            raise ValueError(f"Unmatched seeds or experimental settings: {path}")
        seeds = {str(seed) for seed in protocol["seeds"]}
        if (
            len(seeds) != N_REPEATS
            or len(protocol["seeds"]) != N_REPEATS
            or set(protocol["tasks"]) != set(TASKS)
        ):
            raise ValueError(f"Expected five seeds on all four tasks: {path}")
        expected = N_REPEATS * len(TASKS) * len(protocol["models"])
        if (
            summary["indexed_trials"] != expected
            or summary["expected_trials"] != expected
        ):
            raise ValueError(f"Incomplete trial index: {path}")
        if set(summary["conditions"]) != set(protocol["models"]):
            raise ValueError(f"Model conditions disagree with protocol: {path}")
        source = _source(path, raw, summary)
        combined["sources"].append(source)
        for task in TASKS:
            controls = _gameplay_settings(
                protocol.get("task_settings", {}).get(task, LEGACY_TASK_SETTINGS[task])
            )
            prior = combined["standard_task_settings"].get(task)
            if prior is not None and controls != prior:
                raise ValueError(f"Unmatched standard task settings: {task}")
            combined["standard_task_settings"][task] = controls
        for model, tasks in summary["conditions"].items():
            if model not in BASELINE_MODELS or model in combined["conditions"]:
                raise ValueError(f"Unsupported or duplicate model: {model}")
            if set(tasks) != set(TASKS):
                raise ValueError(f"Missing task conditions for {model}")
            for task, result in tasks.items():
                if result["valid_natural"] != N_REPEATS or result["unresolved_invalid"]:
                    raise ValueError(f"Incomplete natural endings: {model}/{task}")
                if (
                    set(result["per_seed_primary"]) != seeds
                    or set(result["seeds"]) != seeds
                ):
                    raise ValueError(f"Incomplete seed outcomes: {model}/{task}")
                if not set(result["seeds"].values()) <= NATURAL_STATUSES:
                    raise ValueError(f"Non-terminal seed outcomes: {model}/{task}")
                values = list(result["per_seed_primary"].values())
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite outcomes: {model}/{task}")
                mean = statistics.mean(values)
                if not math.isclose(mean, result["primary_mean"], abs_tol=1e-12):
                    raise ValueError(
                        f"Mean disagrees with seed outcomes: {model}/{task}"
                    )
                if task in ("basic", "my-way-home") and (
                    not all(value in (0, 1) for value in values)
                    or sum(values) != result["successes"]
                    or any(
                        result["per_seed_primary"][seed] != (status == "success")
                        for seed, status in result["seeds"].items()
                    )
                ):
                    raise ValueError(f"Invalid success counts: {model}/{task}")
                result["sample_standard_deviation"] = statistics.stdev(values)
            combined["conditions"][model] = tasks
            combined["condition_sources"][model] = {task: source for task in tasks}
    if set(combined["conditions"]) != set(BASELINE_MODELS):
        raise ValueError("The baseline table requires all three GPT-5.6 models")
    combined["matched_settings"] = reference
    return combined


def add_additional_summaries(data: dict[str, Any], paths: list[Path]) -> None:
    """Add partial standard-task extensions without weakening baseline checks."""
    for path in paths:
        raw, summary = _read_summary(path)
        protocol = summary["protocol"]
        tasks = protocol["tasks"]
        models = protocol["models"]
        seeds = {str(seed) for seed in protocol["seeds"]}
        if (
            len(seeds) != N_REPEATS
            or len(protocol["seeds"]) != N_REPEATS
            or not tasks
            or not set(tasks) <= set(TASKS)
            or len(tasks) != len(set(tasks))
            or len(models) != len(set(models))
        ):
            raise ValueError(f"Expected five paired seeds on standard tasks: {path}")
        for field in MATCHED_FIELDS:
            if field != "tasks" and protocol[field] != data["matched_settings"][field]:
                raise ValueError(
                    f"Unmatched additional control setting {field}: {path}"
                )
        expected = N_REPEATS * len(tasks) * len(models)
        if summary["expected_trials"] != expected or not (
            0 <= summary["indexed_trials"] <= expected
        ):
            raise ValueError(f"Invalid additional trial counts: {path}")
        if set(summary["conditions"]) != set(models):
            raise ValueError(f"Model conditions disagree with protocol: {path}")
        for task in tasks:
            controls = protocol.get("task_settings", {}).get(task)
            if (
                controls is None
                or _gameplay_settings(controls) != data["standard_task_settings"][task]
            ):
                raise ValueError(f"Unmatched additional task settings: {task}")
        source = _source(path, raw, summary)
        for model, conditions in summary["conditions"].items():
            if model not in MODELS or set(conditions) != set(tasks):
                raise ValueError(f"Unsupported model or standard conditions: {model}")
            target = data["conditions"].setdefault(model, {})
            provenance = data["condition_sources"].setdefault(model, {})
            for task, result in conditions.items():
                if task in target:
                    raise ValueError(f"Duplicate standard condition: {model}/{task}")
                target[task] = _partial_result(result, seeds, model, task)
                provenance[task] = source
        data["sources"].append(source)


def add_level_summaries(data: dict[str, Any], paths: list[Path]) -> None:
    """Include validated level outcomes without turning pending trials into losses.

    Seeds are paired across models within a level, not across different task
    suites. The standard-scenario export remains strictly complete.
    """
    levels: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    settings_by_task: dict[str, Any] = {}
    for path in paths:
        raw, summary = _read_summary(path)
        protocol = summary["protocol"]
        seeds = {str(seed) for seed in protocol["seeds"]}
        tasks = protocol["tasks"]
        models = protocol["models"]
        if (
            len(seeds) != N_REPEATS
            or len(protocol["seeds"]) != N_REPEATS
            or not tasks
            or not set(tasks) <= set(LEVELS)
        ):
            raise ValueError(f"Expected five paired seeds on supported levels: {path}")
        if len(tasks) != len(set(tasks)) or len(models) != len(set(models)):
            raise ValueError(f"Duplicate task or model settings: {path}")
        for field in MATCHED_FIELDS[2:]:
            if protocol[field] != data["matched_settings"][field]:
                raise ValueError(f"Unmatched level control setting {field}: {path}")
        expected = len(seeds) * len(tasks) * len(models)
        if summary["expected_trials"] != expected or not (
            0 <= summary["indexed_trials"] <= expected
        ):
            raise ValueError(f"Invalid level trial counts: {path}")
        if set(summary["conditions"]) != set(models):
            raise ValueError(f"Model conditions disagree with level protocol: {path}")
        for task in tasks:
            settings = {
                "seeds": protocol["seeds"],
                "task_settings": _gameplay_settings(protocol["task_settings"][task]),
            }
            if task in settings_by_task and settings_by_task[task] != settings:
                raise ValueError(f"Unmatched paired level settings: {task}")
            settings_by_task[task] = settings
        source = _source(path, raw, summary)
        for model, conditions in summary["conditions"].items():
            if model not in MODELS or set(conditions) != set(tasks):
                raise ValueError(f"Unsupported model or level conditions: {model}")
            target = levels.setdefault(model, {})
            condition_sources = provenance.setdefault(model, {})
            for task, result in conditions.items():
                if task in target:
                    raise ValueError(f"Duplicate level condition: {model}/{task}")
                target[task] = _partial_result(result, seeds, model, task)
                condition_sources[task] = source
        data["sources"].append(source)
    data["level_conditions"] = levels
    data["level_settings"] = settings_by_task
    data["level_condition_sources"] = provenance
    data["exported_at"] = datetime.now(timezone.utc).isoformat()


def render_table(data: dict[str, Any]) -> str:
    rows = [
        "% Generated by experiments/harbor_vizdoom/export_paper_results.py",
        r"\begin{tabularx}{\columnwidth}{@{}Y*{6}{c}@{}}",
        r"\toprule",
        "Task (measure) & " + " & ".join(MODEL_LABELS) + r" \\",
        r"\midrule",
    ]
    for task in TASKS:
        cells = [LABELS[task]]
        for model in MODELS:
            result = data["conditions"].get(model, {}).get(task)
            cells.append(_render_cell(task, result))
        rows.append(" & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    for task in LEVELS:
        cells = [LABELS[task]]
        for model in MODELS:
            result = data.get("level_conditions", {}).get(model, {}).get(task)
            cells.append(_render_cell(task, result))
        rows.append(" & ".join(cells) + r" \\")
    rows.extend((r"\bottomrule", r"\end{tabularx}"))
    return "\n".join(rows) + "\n"


def _render_cell(task: str, result: dict[str, Any] | None) -> str:
    if result is None:
        return r"\pending"
    n = result["valid_natural"]
    if not n:
        cell = r"\pending{}"
    elif task not in ("defend-the-center", "health-gathering"):
        cell = f"{result['successes']}/{n}"
    else:
        deviation = result.get("sample_standard_deviation")
        cell = (
            f"${result['primary_mean']:.1f} " + rf"\pm {deviation:.1f}$"
            if deviation is not None
            else f"${result['primary_mean']:.1f}$"
        )
    if n < N_REPEATS:
        cell += f" [{n}/{N_REPEATS}]"
    return cell


def render_status(data: dict[str, Any]) -> str:
    standard = sum(
        result["valid_natural"]
        for tasks in data["conditions"].values()
        for result in tasks.values()
    )
    levels = [
        result
        for tasks in data.get("level_conditions", {}).values()
        for result in tasks.values()
    ]
    snapshot = datetime.fromisoformat(data["exported_at"]).strftime(
        "%d %B %Y, %H:%M UTC"
    )
    return "\n".join(
        [
            "% Generated with the results table; counts include validated outcomes only.",
            rf"\newcommand{{\standardEpisodeCount}}{{{standard}}}",
            rf"\newcommand{{\levelEpisodeCount}}{{{sum(r['valid_natural'] for r in levels)}}}",
            rf"\newcommand{{\levelExpectedCount}}{{{sum(r['expected'] for r in levels)}}}",
            rf"\newcommand{{\resultSnapshotTime}}{{{snapshot}}}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--additional-summaries",
        nargs="+",
        type=Path,
        default=[],
        help="Partial or completed new-model standard-task summaries",
    )
    parser.add_argument(
        "--level-summaries",
        nargs="+",
        type=Path,
        default=[],
        help="Level summaries; unfinished cells are explicitly labeled",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = combine_summaries(args.summaries)
    add_additional_summaries(data, args.additional_summaries)
    add_level_summaries(data, args.level_summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "llm_pilot_results.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "llm_pilot_table.tex").write_text(
        render_table(data), encoding="utf-8"
    )
    (args.output_dir / "llm_pilot_status.tex").write_text(
        render_status(data), encoding="utf-8"
    )
    print(f"Exported validated standard and level outcomes to {args.output_dir}")


if __name__ == "__main__":
    main()
