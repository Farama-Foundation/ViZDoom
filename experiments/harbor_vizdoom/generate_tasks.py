#!/usr/bin/env python3
"""Generate self-contained Harbor tasks from the ViZDoom task manifest."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

EXPERIMENT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = EXPERIMENT_DIR / "environments.json"
TEMPLATE_DIR = EXPERIMENT_DIR / "template"
COMMON_DIR = TEMPLATE_DIR / "common"
INSTRUCTIONS_DIR = TEMPLATE_DIR / "instructions"
TASK_CONFIG_TEMPLATE = TEMPLATE_DIR / "task.toml.template"
COMPOSE_TEMPLATE = TEMPLATE_DIR / "docker-compose.yaml.template"
RUN_CONFIG_TEMPLATE = TEMPLATE_DIR / "run.yaml.template"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "generated_tasks"
GENERATED_MARKER = ".generated-by-harbor-vizdoom"
PLACEHOLDER_PATTERN = re.compile(r"{{([a-z_]+)}}")
SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")

REQUIRED_FIELDS = {
    "slug",
    "task_name",
    "environment_id",
    "display_name",
    "family",
    "level_name",
    "description",
    "objective",
    "outcome_kind",
    "instruction_template",
    "difficulty",
    "keywords",
    "max_decisions",
    "agent_timeout_sec",
    "require_automap",
    "iwad",
}


def load_task_definitions(
    manifest_path: Path | None = None, template_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Load and validate task definitions in manifest order."""
    manifest_path = (manifest_path or MANIFEST_PATH).expanduser().resolve()
    template_dir = (template_dir or TEMPLATE_DIR).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 2:
        raise RuntimeError("environments.json must use schema_version 2")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("environments.json must contain a non-empty tasks list")

    validated: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    seen_names: set[str] = set()
    seen_environments: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise RuntimeError(f"Task at index {index} must be an object")
        missing = REQUIRED_FIELDS - task.keys()
        if missing:
            raise RuntimeError(f"Task at index {index} is missing {sorted(missing)}")
        slug = task["slug"]
        if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
            raise RuntimeError(f"Invalid task slug {slug!r}")
        if slug in seen_slugs:
            raise RuntimeError(f"Duplicate task slug {slug!r}")
        seen_slugs.add(slug)

        task_name = task["task_name"]
        environment_id = task["environment_id"]
        if not isinstance(task_name, str) or task_name in seen_names:
            raise RuntimeError(f"Invalid or duplicate task name {task_name!r}")
        if not isinstance(environment_id, str) or environment_id in seen_environments:
            raise RuntimeError(
                f"Invalid or duplicate environment ID {environment_id!r}"
            )
        seen_names.add(task_name)
        seen_environments.add(environment_id)

        for integer_field in ("max_decisions", "agent_timeout_sec"):
            value = task[integer_field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise RuntimeError(f"{slug}.{integer_field} must be positive")
        timeout = task.get("episode_timeout_seconds", 0)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 0:
            raise RuntimeError(f"{slug}.episode_timeout_seconds must be non-negative")
        if timeout and task["max_decisions"] * 4 < timeout * 35:
            raise RuntimeError(f"{slug}.max_decisions cannot cover the game-time limit")
        if not isinstance(task["require_automap"], bool):
            raise RuntimeError(f"{slug}.require_automap must be boolean")
        if task["outcome_kind"] not in {"goal", "combat", "survival"}:
            raise RuntimeError(f"{slug}.outcome_kind must be goal, combat, or survival")
        if not isinstance(task["keywords"], list) or not all(
            isinstance(keyword, str) for keyword in task["keywords"]
        ):
            raise RuntimeError(f"{slug}.keywords must be a list of strings")

        instruction_path = template_dir / "instructions" / task["instruction_template"]
        if not instruction_path.is_file():
            raise RuntimeError(f"Missing instruction template {instruction_path}")
        if task["family"] == "doom":
            if task["instruction_template"] != "doom.md.template":
                raise RuntimeError(
                    f"Commercial DOOM task {slug} must use doom.md.template"
                )
            for field in ("game_title", "map_id"):
                if not isinstance(task.get(field), str) or not task[field]:
                    raise RuntimeError(f"{slug}.{field} must be a non-empty string")

        iwad = task["iwad"]
        if task["family"] == "doom" and iwad is None:
            raise RuntimeError(f"Commercial DOOM task {slug} must select an IWAD")
        if iwad is not None:
            if not isinstance(iwad, dict):
                raise RuntimeError(f"{slug}.iwad must be an object or null")
            if set(iwad) != {"template_file", "container_path"}:
                raise RuntimeError(f"{slug}.iwad has unexpected fields")
            template_file = iwad["template_file"]
            if (
                not isinstance(template_file, str)
                or Path(template_file).name != template_file
            ):
                raise RuntimeError(f"{slug}.iwad.template_file must be a file name")
            if not (template_dir / template_file).is_file():
                raise RuntimeError(
                    f"Missing bundled IWAD {template_dir / template_file}"
                )
            container_path = iwad["container_path"]
            if not isinstance(container_path, str):
                raise RuntimeError(f"{slug}.iwad.container_path must be a string")
            expected_path = Path("/iwads") / template_file
            if Path(container_path) != expected_path:
                raise RuntimeError(
                    f"{slug}.iwad.container_path must be {expected_path}"
                )
        validated.append(task)
    return validated


def select_task_definitions(
    tasks: Iterable[dict[str, Any]], selected_slugs: Iterable[str] | None
) -> list[dict[str, Any]]:
    """Return selected definitions while retaining manifest order."""
    task_list = list(tasks)
    if selected_slugs is None:
        return task_list
    selected = set(selected_slugs)
    known = {task["slug"] for task in task_list}
    unknown = selected - known
    if unknown:
        raise RuntimeError(f"Unknown task slug(s): {', '.join(sorted(unknown))}")
    return [task for task in task_list if task["slug"] in selected]


def _render(template_path: Path, values: dict[str, str]) -> str:
    template = template_path.read_text(encoding="utf-8")
    placeholders = set(PLACEHOLDER_PATTERN.findall(template))
    missing = placeholders - values.keys()
    if missing:
        raise RuntimeError(
            f"Missing values {sorted(missing)} for template {template_path}"
        )
    rendered = PLACEHOLDER_PATTERN.sub(lambda match: values[match.group(1)], template)
    remaining = PLACEHOLDER_PATTERN.findall(rendered)
    if remaining:
        raise RuntimeError(f"Unresolved placeholders in {template_path}: {remaining}")
    return rendered


def _quoted(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _iwad_path(task: dict[str, Any]) -> str:
    iwad = task["iwad"]
    if iwad is None:
        return _quoted("")
    return _quoted(iwad["container_path"])


def _template_values(task: dict[str, Any]) -> dict[str, str]:
    return {
        "task_name": _quoted(task["task_name"]),
        "environment_id": _quoted(task["environment_id"]),
        "display_name": _quoted(task["display_name"]),
        "family": _quoted(task["family"]),
        "level_name": _quoted(task["level_name"]),
        "description": _quoted(task["description"]),
        "objective": _quoted(task["objective"]),
        "outcome_kind": _quoted(task["outcome_kind"]),
        "difficulty": _quoted(task["difficulty"]),
        "keywords": _quoted(task["keywords"]),
        "max_decisions": str(task["max_decisions"]),
        "episode_timeout_seconds": str(task.get("episode_timeout_seconds", 0)),
        "time_limit_instruction": (
            f"The episode has a {task['episode_timeout_seconds']}-second in-game "
            "time limit. Time advances only when you make actions, not while you "
            "think. Keep playing until done=true; the server enforces the limit."
            if task.get("episode_timeout_seconds")
            else "The server enforces the episode's time and decision limits."
        ),
        "agent_timeout_sec": str(task["agent_timeout_sec"]),
        "require_automap": "1" if task["require_automap"] else "0",
        "iwad_container_path": _iwad_path(task),
        "game_title": str(task.get("game_title", "")),
        "map_id": str(task.get("map_id", "")),
    }


def generate_tasks(
    output_dir: Path,
    selected_slugs: Iterable[str] | None = None,
    manifest_path: Path | None = None,
    template_dir: Path | None = None,
) -> list[Path]:
    """Atomically generate selected self-contained task directories."""
    template_dir = (template_dir or TEMPLATE_DIR).expanduser().resolve()
    common_dir = template_dir / "common"
    task_config_template = template_dir / "task.toml.template"
    compose_template = template_dir / "docker-compose.yaml.template"
    run_config_template = template_dir / "run.yaml.template"
    if not common_dir.is_dir():
        raise RuntimeError(f"Missing common task template directory {common_dir}")
    for path in (task_config_template, compose_template, run_config_template):
        if not path.is_file():
            raise RuntimeError(f"Missing task template {path}")
    tasks = select_task_definitions(
        load_task_definitions(manifest_path, template_dir), selected_slugs
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        for task in tasks:
            task_dir = staging_dir / task["slug"]
            shutil.copytree(common_dir, task_dir)
            iwad = task["iwad"]
            if iwad is not None:
                source = template_dir / iwad["template_file"]
                destination = task_dir / "environment/vizdoom-server/iwads"
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination / source.name)
            values = _template_values(task)
            (task_dir / "task.toml").write_text(
                _render(task_config_template, values), encoding="utf-8"
            )
            (task_dir / "environment/docker-compose.yaml").write_text(
                _render(compose_template, values), encoding="utf-8"
            )
            instruction_template = (
                template_dir / "instructions" / task["instruction_template"]
            )
            (task_dir / "instruction.md").write_text(
                _render(instruction_template, values), encoding="utf-8"
            )
        task_paths = "\n".join(
            f"  - path: {json.dumps(str(output_dir / task['slug']))}" for task in tasks
        )
        (staging_dir / "run.yaml").write_text(
            _render(run_config_template, {"task_paths": task_paths}),
            encoding="utf-8",
        )
        (staging_dir / GENERATED_MARKER).write_text(
            "Generated by generate_tasks.py; do not edit.\n", encoding="utf-8"
        )

        if output_dir.exists():
            marker = output_dir / GENERATED_MARKER
            if not marker.is_file():
                raise RuntimeError(
                    f"Refusing to replace unmarked directory {output_dir}"
                )
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return [output_dir / task["slug"] for task in tasks]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Generated task directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Generate only this task slug; repeat to select several",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available task slugs without generating them",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    definitions = load_task_definitions()
    if args.list:
        for task in definitions:
            print(f"{task['slug']}: {task['display_name']}")
        return
    generated = generate_tasks(args.output_dir, args.tasks)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
