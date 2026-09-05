#!/usr/bin/env python3
"""Generate and run the ViZDoom Harbor eval."""

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

from generate_tasks import (
    generate_tasks,
    load_task_definitions,
    select_task_definitions,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
AUTH_CHOICES = ("api-key", "subscription")


def _unsigned_seed(raw_seed: str | None) -> int:
    if raw_seed is None:
        return secrets.randbits(32)
    seed = int(raw_seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return seed


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", help="Unsigned 32-bit seed; random by default")
    parser.add_argument(
        "--agent",
        choices=("both", "codex", "claude-code"),
        default="both",
    )
    parser.add_argument(
        "--agent-auth",
        choices=AUTH_CHOICES,
        default="api-key",
        help="Authentication mode for selected agents (default: api-key)",
    )
    parser.add_argument(
        "--codex-auth",
        choices=AUTH_CHOICES,
        help="Override --agent-auth for Codex",
    )
    parser.add_argument(
        "--claude-auth",
        choices=AUTH_CHOICES,
        help="Override --agent-auth for Claude Code",
    )
    parser.add_argument(
        "--codex-auth-json",
        type=Path,
        help="Codex subscription auth.json; defaults to ~/.codex/auth.json",
    )
    parser.add_argument("--codex-model", default="openai/gpt-5.6-luna")
    parser.add_argument("--codex-reasoning", default="medium")
    parser.add_argument("--claude-model", default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--claude-reasoning", default="medium")
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Run only this manifest task slug; repeat to select several",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available manifest tasks without running Harbor",
    )
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


def _auth_modes(args: argparse.Namespace) -> dict[str, str]:
    modes: dict[str, str] = {}
    if args.agent in ("both", "codex"):
        modes["codex"] = args.codex_auth or args.agent_auth
    if args.agent in ("both", "claude-code"):
        modes["claude-code"] = args.claude_auth or args.agent_auth
    if args.codex_auth_json is not None:
        if "codex" not in modes:
            raise SystemExit("--codex-auth-json requires Codex to be selected")
        if modes["codex"] != "subscription":
            raise SystemExit(
                "--codex-auth-json requires --codex-auth subscription "
                "or --agent-auth subscription"
            )
        auth_path = args.codex_auth_json.expanduser().resolve()
        if not auth_path.is_file():
            raise SystemExit(f"--codex-auth-json does not exist: {auth_path}")
        args.codex_auth_json = auth_path
    return modes


def _agent_configs(
    args: argparse.Namespace, auth_modes: dict[str, str]
) -> list[dict[str, Any]]:
    agents = []
    if args.agent in ("both", "codex"):
        if auth_modes["codex"] == "subscription":
            codex_environment = (
                {"CODEX_AUTH_JSON_PATH": str(args.codex_auth_json)}
                if args.codex_auth_json is not None
                else {"CODEX_FORCE_AUTH_JSON": "1"}
            )
        else:
            codex_environment = {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
        agents.append(
            {
                "name": "codex",
                "model_name": args.codex_model,
                "env": codex_environment,
                "kwargs": {
                    "reasoning_effort": args.codex_reasoning,
                    "web_search": "disabled",
                },
            }
        )
    if args.agent in ("both", "claude-code"):
        if auth_modes["claude-code"] == "subscription":
            claude_environment = {
                "CLAUDE_CODE_OAUTH_TOKEN": "${CLAUDE_CODE_OAUTH_TOKEN}",
                "CLAUDE_FORCE_OAUTH": "1",
            }
        else:
            claude_environment = {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
        agents.append(
            {
                "name": "claude-code",
                "model_name": args.claude_model,
                "env": claude_environment,
                "kwargs": {
                    "reasoning_effort": args.claude_reasoning,
                    "disallowed_tools": "WebSearch,WebFetch",
                },
            }
        )
    return agents


def main() -> None:
    args = _parse_args()
    definitions = load_task_definitions()
    if args.list_tasks:
        for task in definitions:
            print(f"{task['slug']}: {task['display_name']}")
        return
    try:
        selected_definitions = select_task_definitions(definitions, args.tasks)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        seed = _unsigned_seed(args.seed)
    except ValueError as exc:
        raise SystemExit(f"invalid --seed: {args.seed!r}") from exc

    auth_modes = _auth_modes(args)
    agents = _agent_configs(args, auth_modes)
    model_part = "__".join(
        f"{_slug(agent['name'])}-{_slug(agent['model_name'])}" for agent in agents
    )
    reasoning_part = "__".join(
        f"{_slug(agent['name'])}-{_slug(agent['kwargs']['reasoning_effort'])}"
        for agent in agents
    )
    auth_part = "__".join(
        f"{_slug(agent_name)}-{_slug(auth_mode)}"
        for agent_name, auth_mode in auth_modes.items()
    )
    jobs_dir = (
        args.jobs_root.expanduser().resolve()
        / f"models-{model_part}"
        / f"reasoning-{reasoning_part}"
        / f"auth-{auth_part}"
        / f"seed-{seed}"
    )
    print(f"seed={seed}")
    print(f"jobs_dir={jobs_dir}")
    print(
        "auth="
        + ",".join(f"{agent_name}:{mode}" for agent_name, mode in auth_modes.items())
    )
    print("tasks=" + ",".join(task["slug"] for task in selected_definitions))

    with tempfile.TemporaryDirectory(prefix="harbor-vizdoom-tasks-") as temporary:
        generated_paths = generate_tasks(
            Path(temporary) / "tasks",
            [task["slug"] for task in selected_definitions],
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config = {
            "job_name": f"vizdoom-{timestamp}",
            "jobs_dir": str(jobs_dir),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "quiet": False,
            "environment": {
                "type": "docker",
                "force_build": True,
                "delete": True,
            },
            "agents": agents,
            "tasks": [{"path": str(path)} for path in generated_paths],
        }
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return

        child_environment = os.environ.copy()
        child_environment["VIZDOOM_SEED"] = str(seed)

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
