#!/usr/bin/env python3
"""Run the four-map Harbor eval with reproducible, descriptive output paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
TASK_PATHS = (
    "experiments/harbor_vizdoom/tasks/doom-e1m1",
    "experiments/harbor_vizdoom/tasks/doom-e1m2",
    "experiments/harbor_vizdoom/tasks/doom2-map01",
    "experiments/harbor_vizdoom/tasks/doom2-map02",
)


def _unsigned_seed(raw_seed: str | None) -> int:
    if raw_seed is None:
        return secrets.randbits(32)
    seed = int(raw_seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return seed


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _require_iwad(variable: str) -> str:
    raw_path = os.environ.get(variable, "").strip()
    if not raw_path:
        raise SystemExit(f"{variable} is required")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{variable} does not point to a file: {path}")
    return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", help="Unsigned 32-bit seed; random by default")
    parser.add_argument(
        "--agent",
        choices=("both", "codex", "claude-code"),
        default="both",
    )
    parser.add_argument("--codex-model", default="openai/gpt-5.6-luna")
    parser.add_argument("--codex-reasoning", default="medium")
    parser.add_argument("--claude-model", default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--claude-reasoning", default="medium")
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=EXPERIMENT_DIR / "jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Harbor config without running it",
    )
    return parser.parse_args()


def _agent_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    agents = []
    if args.agent in ("both", "codex"):
        agents.append(
            {
                "name": "codex",
                "model_name": args.codex_model,
                "env": {"OPENAI_API_KEY": "${OPENAI_API_KEY}"},
                "kwargs": {
                    "reasoning_effort": args.codex_reasoning,
                    "web_search": "disabled",
                },
            }
        )
    if args.agent in ("both", "claude-code"):
        agents.append(
            {
                "name": "claude-code",
                "model_name": args.claude_model,
                "env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"},
                "kwargs": {
                    "reasoning_effort": args.claude_reasoning,
                    "disallowed_tools": "WebSearch,WebFetch",
                },
            }
        )
    return agents


def main() -> None:
    args = _parse_args()
    try:
        seed = _unsigned_seed(args.seed)
    except ValueError as exc:
        raise SystemExit(f"invalid --seed: {args.seed!r}") from exc

    agents = _agent_configs(args)
    model_part = "__".join(
        f"{_slug(agent['name'])}-{_slug(agent['model_name'])}" for agent in agents
    )
    reasoning_part = "__".join(
        f"{_slug(agent['name'])}-{_slug(agent['kwargs']['reasoning_effort'])}"
        for agent in agents
    )
    jobs_dir = (
        args.jobs_root.expanduser().resolve()
        / f"models-{model_part}"
        / f"reasoning-{reasoning_part}"
        / f"seed-{seed}"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {
        "job_name": f"vizdoom-doom1-doom2-{timestamp}",
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "quiet": False,
        "environment": {"type": "docker", "force_build": True, "delete": True},
        "agents": agents,
        "tasks": [{"path": str(REPOSITORY_ROOT / path)} for path in TASK_PATHS],
    }

    print(f"seed={seed}")
    print(f"jobs_dir={jobs_dir}")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    child_environment = os.environ.copy()
    child_environment["VIZDOOM_SEED"] = str(seed)
    child_environment["DOOM_WAD_PATH"] = _require_iwad("DOOM_WAD_PATH")
    child_environment["DOOM2_WAD_PATH"] = _require_iwad("DOOM2_WAD_PATH")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8"
    ) as config_file:
        json.dump(config, config_file, indent=2)
        config_file.flush()
        subprocess.run(
            ["harbor", "run", "-c", config_file.name],
            cwd=REPOSITORY_ROOT,
            env=child_environment,
            check=True,
        )


if __name__ == "__main__":
    main()
