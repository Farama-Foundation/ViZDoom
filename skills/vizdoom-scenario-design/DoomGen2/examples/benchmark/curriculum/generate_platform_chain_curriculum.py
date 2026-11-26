import json
import sys
from dataclasses import dataclass
from pathlib import Path


DOOMGEN_ROOT = Path(__file__).resolve().parents[3]


def _find_repo_root() -> Path:
    candidates = [
        DOOMGEN_ROOT.parent,
        DOOMGEN_ROOT.parent / "COMRAD",
    ]
    for candidate in candidates:
        if (candidate / "comrad" / "train_all.py").is_file():
            return candidate
    raise FileNotFoundError("Could not locate COMRAD repo root from DoomGen batch generator path")


REPO_ROOT = _find_repo_root()

for path in (DOOMGEN_ROOT, DOOMGEN_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from examples.benchmark.platform_chain import PlatformChainScenario
from examples.benchmark.platform_chain_easy import PlatformChainScenario as PlatformChainEasyScenario


BATCH_DIR = REPO_ROOT / "comrad" / "scenarios" / "batch_platform_chain_curriculum"
INTERESTINGNESS_FILENAME = "platform_chain_interestingness.json"


@dataclass(frozen=True)
class CurriculumTask:
    name: str
    stage: str
    scenario_cls: type
    config: dict


def curriculum_tasks() -> list[CurriculumTask]:
    return [
        CurriculumTask(
            name="platform_chain_easy_intro",
            stage="easy_intro",
            scenario_cls=PlatformChainEasyScenario,
            config={
                "seed": 42,
                "height_step": 10,
                "jump_gap": 24,
                "polyobject_count": 2,
                "chain_max_len": 300,
                "lava_damage": 20,
            },
        ),
        CurriculumTask(
            name="platform_chain_easy_movers",
            stage="easy_movers",
            scenario_cls=PlatformChainEasyScenario,
            config={
                "seed": 43,
                "height_step": 10,
                "jump_gap": 32,
                "polyobject_count": 4,
                "chain_max_len": 285,
                "lava_damage": 30,
            },
        ),
        CurriculumTask(
            name="platform_chain_easy_spacing",
            stage="easy_spacing",
            scenario_cls=PlatformChainEasyScenario,
            config={
                "seed": 44,
                "height_step": 11,
                "jump_gap": 36,
                "polyobject_count": 6,
                "chain_max_len": 270,
                "lava_damage": 40,
            },
        ),
        CurriculumTask(
            name="platform_chain_easy_mastery",
            stage="easy_mastery",
            scenario_cls=PlatformChainEasyScenario,
            config={
                "seed": 45,
                "height_step": 12,
                "jump_gap": 40,
                "polyobject_count": 8,
                "chain_max_len": 250,
                "lava_damage": 50,
            },
        ),
        CurriculumTask(
            name="platform_chain_bridge_intro",
            stage="bridge_intro",
            scenario_cls=PlatformChainScenario,
            config={
                "seed": 42,
                "level_count": 24,
                "height_step": 10,
                "jump_gap": 60,
                "polyobject_count": 4,
                "safe_start_levels_min": 6,
                "safe_start_levels_max": 6,
                "chain_max_len": 290,
                "lava_damage": 25,
            },
        ),
        CurriculumTask(
            name="platform_chain_bridge_progress",
            stage="bridge_progress",
            scenario_cls=PlatformChainScenario,
            config={
                "seed": 43,
                "level_count": 28,
                "height_step": 11,
                "jump_gap": 64,
                "polyobject_count": 6,
                "safe_start_levels_min": 5,
                "safe_start_levels_max": 5,
                "chain_max_len": 275,
                "lava_damage": 35,
            },
        ),
        CurriculumTask(
            name="platform_chain_bridge_mastery",
            stage="bridge_mastery",
            scenario_cls=PlatformChainScenario,
            config={
                "seed": 44,
                "level_count": 32,
                "height_step": 12,
                "jump_gap": 68,
                "polyobject_count": 6,
                "safe_start_levels_min": 4,
                "safe_start_levels_max": 4,
                "chain_max_len": 260,
                "lava_damage": 45,
            },
        ),
        CurriculumTask(
            name="platform_chain_hard_intro",
            stage="hard_intro",
            scenario_cls=PlatformChainScenario,
            config={
                "seed": 42,
                "level_count": 40,
                "height_step": 11,
                "jump_gap": 64,
                "polyobject_count": 6,
                "safe_start_levels_min": 4,
                "safe_start_levels_max": 4,
                "chain_max_len": 265,
                "lava_damage": 35,
            },
        ),
        CurriculumTask(
            name="platform_chain_hard_default",
            stage="hard_default",
            scenario_cls=PlatformChainScenario,
            config={"seed": 42},
        ),
        CurriculumTask(
            name="platform_chain_hard_tighter",
            stage="hard_tighter",
            scenario_cls=PlatformChainScenario,
            config={
                "seed": 43,
                "level_count": 48,
                "height_step": 12,
                "jump_gap": 70,
                "polyobject_count": 8,
                "safe_start_levels_min": 2,
                "safe_start_levels_max": 2,
                "chain_max_len": 240,
                "lava_damage": 50,
            },
        ),
        CurriculumTask(
            name="platform_chain_hard_extended",
            stage="hard_extended",
            scenario_cls=PlatformChainScenario,
            config={
                "seed": 44,
                "level_count": 56,
                "height_step": 12,
                "jump_gap": 72,
                "polyobject_count": 10,
                "safe_start_levels_min": 2,
                "safe_start_levels_max": 2,
                "chain_max_len": 230,
                "lava_damage": 60,
            },
        ),
        CurriculumTask(
            name="platform_chain_hard_endgame",
            stage="hard_endgame",
            scenario_cls=PlatformChainScenario,
            config={
                "seed": 45,
                "level_count": 60,
                "height_step": 13,
                "jump_gap": 76,
                "polyobject_count": 12,
                "safe_start_levels_min": 1,
                "safe_start_levels_max": 1,
                "chain_max_len": 220,
                "lava_damage": 70,
            },
        ),
    ]


def _generate_task(output_dir: Path, task: CurriculumTask) -> dict:
    wad_name = f"{task.name}.wad"
    wad_path = output_dir / wad_name
    if wad_path.exists():
        wad_path.unlink()

    scenario = task.scenario_cls(task.config, name=task.name)
    scenario.generate(str(wad_path))
    level_count = int(scenario.config.get("level_count", 1))

    return {
        "id": task.name,
        "filename": wad_name,
        "config": scenario.config,
        "curriculum_stage": task.stage,
        "curriculum_metric_scale": float(max(1, level_count - 1)),
    }


def _interestingness_graph(tasks: list[CurriculumTask]) -> dict[str, dict[str, bool]]:
    ordered_ids = [task.name for task in tasks]
    graph: dict[str, dict[str, bool]] = {}
    for idx, task_id in enumerate(ordered_ids):
        graph[task_id] = {
            other_id: other_idx > idx
            for other_idx, other_id in enumerate(ordered_ids)
        }
    return graph


def main() -> None:
    output_dir = BATCH_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_wad in output_dir.glob("*.wad"):
        old_wad.unlink()
    registry_path = output_dir / "batch_registry.json"
    if registry_path.exists():
        registry_path.unlink()
    interestingness_path = output_dir / INTERESTINGNESS_FILENAME
    if interestingness_path.exists():
        interestingness_path.unlink()

    tasks = curriculum_tasks()
    print(f"Generating {len(tasks)} Platform Chain curriculum scenarios into {output_dir}...")

    registry = []
    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] {task.name}.wad")
        registry.append(_generate_task(output_dir, task))

    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    interestingness_path.write_text(json.dumps(_interestingness_graph(tasks), indent=2), encoding="utf-8")
    print(f"Platform Chain curriculum batch generation complete: {registry_path}")


if __name__ == "__main__":
    main()
