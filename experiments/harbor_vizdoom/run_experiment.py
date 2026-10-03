#!/usr/bin/env python3
"""Generate and run the ViZDoom Harbor eval."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
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
HARBOR_NO_REDACTION_DIR = EXPERIMENT_DIR / "harbor_no_redaction"
AUTH_CHOICES = ("api-key", "subscription")
DOCKER_BASE_IMAGE = "python:3.12-slim"
DOCKER_DESKTOP_BINARY = Path("/Applications/Docker.app/Contents/Resources/bin/docker")


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
    parser.add_argument("--codex-version", help="Pin the Codex CLI installed in trials")
    parser.add_argument("--claude-model", default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--claude-reasoning", default="medium")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Archived experiment source directory containing manifest, templates, and harness patches",
    )
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
        "--concurrent-trials",
        type=int,
        default=1,
        help="Maximum Harbor trials in flight (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Harbor config without running it",
    )
    parser.add_argument(
        "--docker-pull-retries",
        type=int,
        default=3,
        help="Attempts to cache a missing Docker base image (default: 3)",
    )
    parser.add_argument(
        "--skip-docker-preflight",
        action="store_true",
        help="Do not check or pull the Docker base image before Harbor starts",
    )
    args = parser.parse_args()
    if args.docker_pull_retries < 1:
        parser.error("--docker-pull-retries must be positive")
    if args.concurrent_trials < 1:
        parser.error("--concurrent-trials must be positive")
    return args


def _source_paths(source_root: Path | None) -> tuple[Path, Path, Path]:
    """Resolve per-run sources without changing generator or auth globals."""
    root = (source_root or EXPERIMENT_DIR).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Experiment source root does not exist: {root}")
    manifest_path = root / "environments.json"
    template_dir = root / "template"
    harness_dir = root / "harbor_no_redaction"
    for path in (template_dir, template_dir / "common", harness_dir):
        if not path.is_dir():
            raise SystemExit(f"Missing experiment source directory: {path}")
    for path in (
        manifest_path,
        template_dir / "task.toml.template",
        template_dir / "docker-compose.yaml.template",
        template_dir / "run.yaml.template",
        harness_dir / "sitecustomize.py",
        harness_dir / "episode_continuation.py",
    ):
        if not path.is_file():
            raise SystemExit(f"Missing experiment source file: {path}")
    return manifest_path, template_dir, harness_dir


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
                    **({"version": args.codex_version} if args.codex_version else {}),
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


def _docker_command(environment: dict[str, str]) -> str:
    executable = shutil.which("docker", path=environment.get("PATH"))
    if executable is not None:
        return executable
    if DOCKER_DESKTOP_BINARY.is_file():
        docker_bin = str(DOCKER_DESKTOP_BINARY.parent)
        current_path = environment.get("PATH", "")
        environment["PATH"] = os.pathsep.join(
            entry for entry in (docker_bin, current_path) if entry
        )
        return str(DOCKER_DESKTOP_BINARY)
    raise SystemExit("docker was not found on PATH and Docker Desktop is unavailable")


def _ensure_docker_base_image(environment: dict[str, str], pull_retries: int) -> None:
    docker = _docker_command(environment)
    daemon = subprocess.run(
        [docker, "info", "--format", "{{.ServerVersion}}"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if daemon.returncode != 0:
        raise SystemExit(
            "Cannot access the Docker daemon; check Docker Desktop and socket "
            f"permissions before launching Harbor.\n{daemon.stderr.strip()}"
        )
    inspected = subprocess.run(
        [docker, "image", "inspect", DOCKER_BASE_IMAGE],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if inspected.returncode == 0:
        print(f"docker_base_image={DOCKER_BASE_IMAGE} (cached)")
        return

    last_error = ""
    for attempt in range(1, pull_retries + 1):
        print(
            f"docker_base_image={DOCKER_BASE_IMAGE} "
            f"pull_attempt={attempt}/{pull_retries}"
        )
        pulled = subprocess.run(
            [docker, "pull", DOCKER_BASE_IMAGE],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if pulled.returncode == 0:
            print(f"docker_base_image={DOCKER_BASE_IMAGE} (ready)")
            return
        last_error = (pulled.stderr or pulled.stdout).strip()
        if attempt < pull_retries:
            time.sleep(min(2**attempt, 8))
    raise SystemExit(
        f"Could not pull {DOCKER_BASE_IMAGE} after {pull_retries} attempts. "
        f"Docker reported:\n{last_error}"
    )


def main() -> None:
    args = _parse_args()
    manifest_path, template_dir, harness_dir = _source_paths(args.source_root)
    try:
        definitions = load_task_definitions(manifest_path, template_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
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
            manifest_path=manifest_path,
            template_dir=template_dir,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config = {
            "job_name": f"vizdoom-{timestamp}",
            "jobs_dir": str(jobs_dir),
            "n_attempts": 1,
            "n_concurrent_trials": args.concurrent_trials,
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
        existing_pythonpath = child_environment.get("PYTHONPATH", "")
        child_environment["PYTHONPATH"] = os.pathsep.join(
            entry for entry in (str(harness_dir), existing_pythonpath) if entry
        )
        if not args.skip_docker_preflight:
            _ensure_docker_base_image(
                child_environment, pull_retries=args.docker_pull_retries
            )

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
