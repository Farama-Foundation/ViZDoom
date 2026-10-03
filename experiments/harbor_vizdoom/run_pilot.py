#!/usr/bin/env python3
"""Run a paired five-seed Harbor pilot and index its artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import secrets
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_tasks import load_task_definitions, select_task_definitions
from run_experiment import EXPERIMENT_DIR, _slug

TASKS = ("basic", "my-way-home", "defend-the-center", "health-gathering")
BASELINE_MODELS = ("openai/gpt-5.6-luna", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol")
MODELS = BASELINE_MODELS + (
    "openai/gpt-6-luna",
    "openai/gpt-6-sol",
    "openai/gpt-6.1-sol",
)
N_REPEATS = 5
NATURAL_STATUSES = {"success", "dead", "timeout", "episode_ended"}
DEFAULT_ROOT = EXPERIMENT_DIR / "pilot_runs"
FROZEN_SOURCES = (
    "environments.json",
    "generate_tasks.py",
    "run_experiment.py",
    "template/common/environment/vizdoom-server/server.py",
    "template/common/environment/vizdoom-server/streaming_video.py",
    "template/common/environment/vizdoom-server/Dockerfile",
    "template/common/environment/vizdoom-server/iwads/.gitkeep",
    "template/common/environment/Dockerfile",
    "template/common/tests/grade.py",
    "template/common/tests/test.sh",
    "template/common/tests/Dockerfile",
    "template/task.toml.template",
    "template/docker-compose.yaml.template",
    "template/run.yaml.template",
    "harbor_no_redaction/sitecustomize.py",
    "harbor_no_redaction/episode_continuation.py",
    *(f"template/instructions/{task}.md.template" for task in TASKS),
    "template/instructions/doom.md.template",
)


def _protocol_tasks(protocol: dict[str, Any]) -> tuple[str, ...]:
    return tuple(protocol.get("tasks", TASKS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-dir", type=Path, help="Resume an existing pilot")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[task["slug"] for task in load_task_definitions()],
        help="Manifest tasks to run (default: the four-scenario pilot)",
    )
    parser.add_argument("--codex-version", help="Pin the Codex CLI installed in trials")
    parser.add_argument(
        "--reuse-completed-from",
        nargs="+",
        type=Path,
        default=[],
        help="Reuse validated same-seed natural outcomes; preserve their provenance",
    )
    parser.add_argument(
        "--seeds",
        nargs=N_REPEATS,
        type=int,
        help="Five distinct paired unsigned 32-bit seeds; random by default",
    )
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument(
        "--auth", choices=("subscription", "api-key"), default="subscription"
    )
    parser.add_argument("--concurrent-trials", type=int, default=2)
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Independent Harbor jobs in flight (default: 1)",
    )
    parser.add_argument(
        "--skip-in-progress",
        action="store_true",
        help="On resume, leave an existing unfinished Harbor job running",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save the protocol and source archive without launching any trials",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run persistently with runner/supervisor output redirected to files",
    )
    args = parser.parse_args()
    if args.resume_dir and args.seeds:
        parser.error("--resume-dir and --seeds cannot be combined")
    if args.resume_dir and (args.models or args.codex_version or args.tasks):
        parser.error(
            "A resumed pilot keeps its recorded models, tasks and Codex version"
        )
    if args.resume_dir and args.reuse_completed_from:
        parser.error("A resumed pilot keeps its recorded reused outcomes")
    if args.tasks and len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must be distinct")
    if args.models and len(set(args.models)) != len(args.models):
        parser.error("models must be distinct")
    if args.skip_in_progress and not args.resume_dir:
        parser.error("--skip-in-progress requires --resume-dir")
    if args.concurrent_trials < 1:
        parser.error("--concurrent-trials must be positive")
    if args.parallel_jobs < 1:
        parser.error("--parallel-jobs must be positive")
    if args.seeds and (
        len(set(args.seeds)) != N_REPEATS
        or any(not 0 <= seed <= 2**32 - 1 for seed in args.seeds)
    ):
        parser.error("seeds must be five distinct unsigned 32-bit integers")
    return args


def _new_protocol(args: argparse.Namespace) -> dict[str, Any]:
    task_definitions = select_task_definitions(
        load_task_definitions(), args.tasks or TASKS
    )
    seeds = args.seeds
    if seeds is None:
        selected: set[int] = set()
        while len(selected) < N_REPEATS:
            selected.add(secrets.randbits(32))
        seeds = list(selected)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=EXPERIMENT_DIR,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    codex_version = subprocess.run(
        ["codex", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    docker_version = subprocess.run(
        ["docker", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "harbor_version": importlib.metadata.version("harbor"),
        "codex_version": codex_version,
        "codex_agent_version": args.codex_version,
        "docker_version": docker_version,
        "source_sha256": {
            source: hashlib.sha256((EXPERIMENT_DIR / source).read_bytes()).hexdigest()
            for source in FROZEN_SOURCES
        },
        "iwad_sha256": {
            task["iwad"]["template_file"]: hashlib.sha256(
                (
                    EXPERIMENT_DIR / "template" / task["iwad"]["template_file"]
                ).read_bytes()
            ).hexdigest()
            for task in load_task_definitions()
            if task["iwad"] is not None
        },
        "models": args.models or BASELINE_MODELS,
        "tasks": [task["slug"] for task in task_definitions],
        "task_settings": {task["slug"]: task for task in task_definitions},
        "seeds": seeds,
        "reasoning": args.reasoning,
        "auth": args.auth,
        "concurrent_trials": args.concurrent_trials,
        "parallel_jobs": args.parallel_jobs,
        "frame_skip": 4,
        "max_action_repeat": 2,
        "image_transport": "PNG bytes in MCP response; no frame PNG files",
        "note": "Pilot of Codex harness, not a direct-API or harness ablation.",
        "video_transport": "Streaming Gymnasium RecordVideo; bounded RGB pipe to FFmpeg",
        "frozen_execution": True,
    }


def _reuse_records(
    protocol: dict[str, Any], sources: list[Path]
) -> list[dict[str, Any]]:
    """Never replay a completed natural failure to improve its score."""
    from summarize_pilot import _valid_trial

    records = []
    selected = set()
    provenance = []
    controls = (
        "environment_id",
        "outcome_kind",
        "episode_timeout_seconds",
        "max_decisions",
        "require_automap",
        "iwad",
    )
    for source_dir in sources:
        source_dir = source_dir.expanduser().resolve()
        source_path = source_dir / "protocol.json"
        raw = source_path.read_bytes()
        original = json.loads(raw)
        for control in ("reasoning", "auth", "frame_skip", "max_action_repeat"):
            if original.get(control) != protocol.get(control):
                raise ValueError(f"Unmatched reused control {control}: {source_dir}")
        index_path = source_dir / "runs.jsonl"
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            model, seed, task = record["model"], record["seed"], record["task"]
            cell = model, seed, task
            if not (
                model in protocol["models"]
                and seed in protocol["seeds"]
                and task in _protocol_tasks(protocol)
            ) or not _valid_trial(record, task):
                continue
            if cell in selected:
                raise ValueError(f"Duplicate reused completed outcome: {cell}")
            old_task = original.get("task_settings", {}).get(task)
            new_task = protocol["task_settings"][task]
            if old_task is None or any(
                old_task.get(key) != new_task.get(key) for key in controls
            ):
                raise ValueError(f"Unmatched reused gameplay settings: {cell}")
            source_info = {
                "model": model,
                "seed": seed,
                "task": task,
                "run_dir": str(source_dir),
                "trial_dir": record["trial_dir"],
                "protocol_sha256": hashlib.sha256(raw).hexdigest(),
                "codex_agent_version": original.get("codex_agent_version"),
                "source_sha256": original.get("source_sha256", {}),
            }
            provenance.append(source_info)
            records.append({**record, "reused_from": source_info})
            selected.add(cell)
    protocol["reused_outcomes"] = provenance
    return records


def _status(message: str) -> None:
    """Console output is optional; a disconnected client must not stop a batch."""
    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass


def _freeze_sources(run_dir: Path) -> None:
    """Archive executable task sources, excluding credentials and commercial IWADs."""
    with tarfile.open(run_dir / "source_snapshot.tar.gz", "w:gz") as snapshot:
        for source in FROZEN_SOURCES:
            snapshot.add(EXPERIMENT_DIR / source, arcname=source, recursive=False)


def _source_root(run_dir: Path, protocol: dict[str, Any]) -> Path:
    """Execute the archived source tree, not mutable workspace templates."""
    archive = run_dir / "source_snapshot.tar.gz"
    if not archive.is_file():
        raise RuntimeError("Missing source snapshot; do not silently change a protocol")
    source_root = run_dir / "frozen_sources"
    if not source_root.is_dir():
        source_root.mkdir()
        with tarfile.open(archive, "r:gz") as snapshot:
            snapshot.extractall(source_root, filter="data")
    for source, expected in protocol["source_sha256"].items():
        actual = hashlib.sha256((source_root / source).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {source}")
    # Licensed game assets stay outside the archive and are read from the
    # user-provided template. Neither auth files nor secrets are copied.
    manifest = json.loads((source_root / "environments.json").read_text())
    for task in manifest["tasks"]:
        iwad = task.get("iwad")
        if iwad is not None:
            original = EXPERIMENT_DIR / "template" / iwad["template_file"]
            expected = protocol.get("iwad_sha256", {}).get(iwad["template_file"])
            if (
                expected
                and hashlib.sha256(original.read_bytes()).hexdigest() != expected
            ):
                raise RuntimeError(f"IWAD hash mismatch: {original.name}")
            target = source_root / "template" / iwad["template_file"]
            if not target.exists():
                target.symlink_to(original)
    return source_root


def _detach(run_dir: Path, args: argparse.Namespace) -> None:
    """No child inherits the tool client's stdout pipe or stdin."""
    commands = {
        "runner": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--resume-dir",
            str(run_dir),
            "--parallel-jobs",
            str(args.parallel_jobs),
        ],
        "supervisor": [
            sys.executable,
            str(EXPERIMENT_DIR / "supervise_pilot.py"),
            str(run_dir),
        ],
    }
    if args.skip_in_progress:
        commands["runner"].append("--skip-in-progress")
    launched = {}
    for role, command in commands.items():
        with (run_dir / f"{role}.log").open("ab") as output:
            process = subprocess.Popen(
                command,
                cwd=EXPERIMENT_DIR,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched[role] = {"pid": process.pid, "command": command}
    (run_dir / "launcher.json").write_text(
        json.dumps({"at": datetime.now(timezone.utc).isoformat(), **launched}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _status(f"detached runner_pid={launched['runner']['pid']} run_dir={run_dir}")


def _job_root(run_dir: Path, model: str, protocol: dict[str, Any], seed: int) -> Path:
    return (
        run_dir
        / "jobs"
        / f"models-codex-{_slug(model)}"
        / f"reasoning-codex-{_slug(protocol['reasoning'])}"
        / f"auth-codex-{_slug(protocol['auth'])}"
        / f"seed-{seed}"
    )


def _codex_log_metrics(path: Path | None) -> dict[str, int | None]:
    if path is None or not path.is_file():
        return {"cli_turns": None, "image_tool_results": None}
    cli_turns = 0
    image_tool_results = 0
    with path.open(encoding="utf-8") as codex_log:
        for line in codex_log:
            cli_turns += line.startswith('{"type":"turn.completed"')
            image_tool_results += (
                line.startswith('{"type":"item.completed"') and '"type":"image"' in line
            )
    return {"cli_turns": cli_turns, "image_tool_results": image_tool_results}


def _collect_job(
    run_dir: Path,
    protocol: dict[str, Any],
    model: str,
    seed: int,
    returncode: int,
) -> list[dict[str, Any]]:
    job_root = _job_root(run_dir, model, protocol, seed)
    job_dirs = sorted(job_root.glob("vizdoom-*"))
    job_dir = job_dirs[-1] if job_dirs else None
    records = []
    for task in _protocol_tasks(protocol):
        candidates = sorted(job_dir.glob(f"{task}__*")) if job_dir else []
        trial_dir = candidates[-1] if candidates else None
        results_dir = trial_dir / "artifacts/vizdoom-results" if trial_dir else None
        result_path = results_dir / "result.json" if results_dir else None
        metrics_path = trial_dir / "verifier/metrics.json" if trial_dir else None
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path and result_path.is_file()
            else {}
        )
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path and metrics_path.is_file()
            else {}
        )
        status = result.get("status")
        trajectory_path = trial_dir / "agent/trajectory.json" if trial_dir else None
        codex_log_path = trial_dir / "agent/codex.txt" if trial_dir else None
        log_metrics = _codex_log_metrics(codex_log_path)
        cli_turns = log_metrics["cli_turns"]
        image_tool_results = log_metrics["image_tool_results"]
        trajectory = (
            json.loads(trajectory_path.read_text(encoding="utf-8"))
            if trajectory_path and trajectory_path.is_file()
            else {}
        )
        agent_steps = [
            step
            for step in trajectory.get("steps", [])
            if isinstance(step, dict) and step.get("source") == "agent"
        ]
        usage = trajectory.get("final_metrics") or {}
        event_path = results_dir / "events.jsonl" if results_dir else None
        events = (
            [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if event_path and event_path.is_file()
            else []
        )
        artifacts = {
            name: bool(results_dir and (results_dir / name).is_file())
            for name in ("result.json", "events.jsonl", "recording.mp4")
        }
        records.append(
            {
                "model": model,
                "reasoning": protocol["reasoning"],
                "auth": protocol["auth"],
                "seed": seed,
                "task": task,
                "harbor_returncode": returncode,
                "job_dir": str(job_dir) if job_dir else None,
                "trial_dir": str(trial_dir) if trial_dir else None,
                "artifacts": artifacts,
                "trajectory_present": bool(trajectory),
                "agent_version": (trajectory.get("agent") or {}).get("version"),
                "model_calls": sum(
                    step.get("llm_call_count", 0) for step in agent_steps
                ),
                "cli_turns": cli_turns,
                "continuations": (
                    max(0, cli_turns - 1) if cli_turns is not None else None
                ),
                "image_tool_results": image_tool_results,
                "tool_calls": sum(
                    len(step.get("tool_calls") or []) for step in agent_steps
                ),
                "input_tokens": usage.get("total_prompt_tokens"),
                "cached_input_tokens": usage.get("total_cached_tokens"),
                "output_tokens": usage.get("total_completion_tokens"),
                "cost_usd": usage.get("total_cost_usd"),
                "state_reads": sum(
                    event.get("event") == "get_current_state" for event in events
                ),
                "memory_stores": sum(
                    event.get("event") == "store_state" for event in events
                ),
                "memory_retrievals": sum(
                    event.get("event") == "get_stored_state" for event in events
                ),
                "status": status,
                "natural_end": status in NATURAL_STATUSES,
                "success": result.get("success"),
                "episode_return": result.get("episode_return"),
                "elapsed_game_tics": result.get("elapsed_game_tics"),
                "elapsed_game_tics_exact": result.get("elapsed_game_tics_exact"),
                "episode_timeout_seconds": result.get("episode_timeout_seconds", 0),
                "survival_seconds": metrics.get("survival_seconds"),
                "decisions": result.get("decisions"),
                "narration_coverage": metrics.get("narration_coverage"),
                "verifier_reward": metrics.get("reward"),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return records


def _completed_existing_job(
    run_dir: Path, protocol: dict[str, Any], model: str, seed: int
) -> bool:
    job_dirs = sorted(_job_root(run_dir, model, protocol, seed).glob("vizdoom-*"))
    if not job_dirs:
        return False
    result_path = job_dirs[-1] / "result.json"
    if not result_path.is_file():
        return False
    job_result = json.loads(result_path.read_text(encoding="utf-8"))
    stats = job_result.get("stats") or {}
    return bool(job_result.get("finished_at")) and stats.get(
        "n_completed_trials", 0
    ) >= len(_protocol_tasks(protocol))


def _run_job(
    run_dir: Path,
    protocol: dict[str, Any],
    logs_dir: Path,
    model: str,
    seed: int,
) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / "run_experiment.py"),
        "--agent",
        "codex",
        "--agent-auth",
        protocol["auth"],
        "--codex-model",
        model,
        "--codex-reasoning",
        protocol["reasoning"],
        "--seed",
        str(seed),
        "--jobs-root",
        str(run_dir / "jobs"),
        "--concurrent-trials",
        str(protocol["concurrent_trials"]),
    ]
    if protocol.get("frozen_execution"):
        source_root = _source_root(run_dir, protocol)
        command[1] = str(source_root / "run_experiment.py")
        command.extend(("--source-root", str(source_root)))
    if protocol.get("codex_agent_version"):
        command.extend(("--codex-version", protocol["codex_agent_version"]))
    for task in _protocol_tasks(protocol):
        command.extend(("--task", task))
    log_path = logs_dir / f"{_slug(model)}__seed-{seed}.log"
    _status(f"starting model={model} seed={seed}")
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=EXPERIMENT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not _completed_existing_job(run_dir, protocol, model, seed):
        raise RuntimeError(
            f"Harbor did not finish model={model} seed={seed}; "
            f"exit={process.returncode}. See {log_path}. "
            "No episode records were indexed."
        )
    return _collect_job(run_dir, protocol, model, seed, process.returncode)


def _index_records(
    index_path: Path,
    indexed: set[tuple[str, int, str]],
    records: list[dict[str, Any]],
) -> None:
    with index_path.open("a", encoding="utf-8") as index_file:
        for record in records:
            index_file.write(json.dumps(record, sort_keys=True) + "\n")
            index_file.flush()
            indexed.add((record["model"], record["seed"], record["task"]))
    for record in records:
        _status(
            f"result model={record['model']} seed={record['seed']} "
            f"task={record['task']} status={record['status']} "
            f"natural_end={record['natural_end']}",
        )


def _run_pending(
    run_dir: Path,
    protocol: dict[str, Any],
    logs_dir: Path,
    pending: list[tuple[str, int]],
    index_path: Path,
    indexed: set[tuple[str, int, str]],
    parallel_jobs: int,
) -> None:
    """Keep completed outcomes if another worker fails; stop unstarted jobs."""
    failures = []

    def job_protocol(model: str, seed: int) -> dict[str, Any]:
        return {
            **protocol,
            "tasks": [
                task
                for task in _protocol_tasks(protocol)
                if (model, seed, task) not in indexed
            ],
        }

    def failed(model: str, seed: int, error: Exception) -> None:
        failure = {
            "at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "seed": seed,
            "error": str(error),
        }
        with (run_dir / "launch_failures.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(failure) + "\n")
        failures.append(failure)
        _status(f"job_failed model={model} seed={seed}: {error}")

    if parallel_jobs == 1:
        for model, seed in pending:
            try:
                records = _run_job(
                    run_dir, job_protocol(model, seed), logs_dir, model, seed
                )
            except Exception as error:
                failed(model, seed, error)
                break
            _index_records(index_path, indexed, records)
    else:
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {
                executor.submit(
                    _run_job, run_dir, job_protocol(model, seed), logs_dir, model, seed
                ): (
                    model,
                    seed,
                )
                for model, seed in pending
            }
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    records = future.result()
                except Exception as error:
                    failed(*futures[future], error)
                    for other in futures:
                        other.cancel()
                    continue
                _index_records(index_path, indexed, records)
    if failures:
        raise RuntimeError(
            f"{len(failures)} Harbor job(s) failed; preserved completed outcomes. "
            f"See {run_dir / 'launch_failures.jsonl'}"
        )


def main() -> None:
    args = _parse_args()
    if args.resume_dir:
        run_dir = args.resume_dir.expanduser().resolve()
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        if not args.dry_run and args.parallel_jobs != protocol.get("parallel_jobs", 1):
            protocol["parallel_jobs"] = args.parallel_jobs
            protocol.setdefault("orchestration_changes", []).append(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "parallel_jobs": args.parallel_jobs,
                    "note": "Only job scheduling changed; per-trial settings remain fixed.",
                }
            )
            (run_dir / "protocol.json").write_text(
                json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
            )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_root.expanduser().resolve() / timestamp
        protocol = _new_protocol(args)
        reused = _reuse_records(protocol, args.reuse_completed_from)
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "protocol.json").write_text(
                json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
            )
            _freeze_sources(run_dir)
            if reused:
                _index_records(run_dir / "runs.jsonl", set(), reused)
    _status(f"run_dir={run_dir}")
    _status(f"seeds={protocol['seeds']}")
    if args.dry_run:
        return
    if args.prepare_only:
        if protocol.get("frozen_execution"):
            _source_root(run_dir, protocol)
        _status("prepared; no trials launched")
        return
    if protocol.get("frozen_execution"):
        _source_root(run_dir, protocol)
    if args.detach:
        _detach(run_dir, args)
        return

    index_path = run_dir / "runs.jsonl"
    indexed = set()
    if index_path.is_file():
        indexed = {
            (item["model"], item["seed"], item["task"])
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
            if item.get("trial_dir") is not None
        }
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    pending: list[tuple[str, int]] = []
    in_progress: list[tuple[str, int]] = []
    for seed in protocol["seeds"]:
        for model in protocol["models"]:
            if all(
                (model, seed, task) in indexed for task in _protocol_tasks(protocol)
            ):
                _status(f"already_indexed model={model} seed={seed}")
                continue
            remaining_protocol = {
                **protocol,
                "tasks": [
                    task
                    for task in _protocol_tasks(protocol)
                    if (model, seed, task) not in indexed
                ],
            }
            if _completed_existing_job(run_dir, remaining_protocol, model, seed):
                _status(f"indexing_existing model={model} seed={seed}")
                records = _collect_job(run_dir, remaining_protocol, model, seed, 0)
                _index_records(index_path, indexed, records)
            elif args.skip_in_progress and list(
                _job_root(run_dir, model, protocol, seed).glob("vizdoom-*")
            ):
                _status(f"leaving_in_progress model={model} seed={seed}")
                in_progress.append((model, seed))
            else:
                pending.append((model, seed))
    _run_pending(
        run_dir,
        protocol,
        logs_dir,
        pending,
        index_path,
        indexed,
        args.parallel_jobs,
    )
    for model, seed in in_progress:
        if _completed_existing_job(run_dir, protocol, model, seed):
            records = _collect_job(run_dir, protocol, model, seed, 0)
            _index_records(index_path, indexed, records)
        else:
            _status(f"still_in_progress model={model} seed={seed}")


if __name__ == "__main__":
    main()
