"""Generate the immutable DoomGen variants for the deathmatch aiming campaign.

The ladder is intentionally described in one place.  A run records the JSON
manifest beside its WAD, while the ViZDoom checkout supplies the action-space
and training configuration.  ``--rung`` defaults to the foundation rung so a
bare invocation cannot accidentally generate or submit the whole campaign.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.benchmark.deathmatch_duel_1000 import DeathmatchDuelScenario  # noqa: E402


_BASE_MAP = {
    "seed": 42,
    "arena_size": 1000,
    "crate_density_pct": 0,
    "wall_density_pct": 0,
    "obstacle_pool_fill": 0.0,
    "fixed_spawn_pair": [[-128, 0], [128, 0]],
    "spawn_yaw_offset_deg": 0.0,
    "randomize_spawn_pair": False,
}


def _preset(
    *,
    rung: int,
    name: str,
    actions: list[str],
    angle_degrees_per_action: list[float],
    map_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scenario_config = copy.deepcopy(_BASE_MAP)
    if map_overrides:
        scenario_config.update(copy.deepcopy(map_overrides))
    return {
        "rung": rung,
        "name": name,
        "scenario_config": scenario_config,
        "action_buttons": actions,
        "angle_degrees_per_action": angle_degrees_per_action,
    }


RUNG_PRESETS: Dict[str, Dict[str, Any]] = {
    "aim_r00_attack_only": _preset(
        rung=0,
        name="aim_r00_attack_only",
        actions=["ATTACK"],
        angle_degrees_per_action=[2.0, 6.0, 18.0, 50.0],
    ),
    "aim_r01_yaw3": _preset(
        rung=1,
        name="aim_r01_yaw3",
        actions=["TURN_LEFT_RIGHT_DELTA", "ATTACK"],
        angle_degrees_per_action=[2.0],
        map_overrides={"spawn_yaw_offset_deg": 30.0},
    ),
    "aim_r02_yaw9": _preset(
        rung=2,
        name="aim_r02_yaw9",
        actions=["TURN_LEFT_RIGHT_DELTA", "ATTACK"],
        angle_degrees_per_action=[2.0, 6.0, 18.0, 50.0],
        map_overrides={"spawn_yaw_offset_deg": 30.0},
    ),
    "aim_r03_binary": _preset(
        rung=3,
        name="aim_r03_binary",
        actions=["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
        angle_degrees_per_action=[],
        map_overrides={"spawn_yaw_offset_deg": 30.0},
    ),
    "aim_r04_strafe": _preset(
        rung=4,
        name="aim_r04_strafe",
        actions=["TURN_LEFT_RIGHT_DELTA", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        angle_degrees_per_action=[2.0],
        map_overrides={"spawn_yaw_offset_deg": 30.0},
    ),
    "aim_r05_full_movement": _preset(
        rung=5,
        name="aim_r05_full_movement",
        actions=[
            "TURN_LEFT_RIGHT_DELTA",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "ATTACK",
        ],
        angle_degrees_per_action=[2.0],
        map_overrides={"spawn_yaw_offset_deg": 30.0},
    ),
    "aim_r06_long_range": _preset(
        rung=6,
        name="aim_r06_long_range",
        actions=[
            "TURN_LEFT_RIGHT_DELTA",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "ATTACK",
        ],
        angle_degrees_per_action=[2.0],
        map_overrides={"fixed_spawn_pair": [[-316, 0], [316, 0]]},
    ),
    "aim_r07_random_spawns": _preset(
        rung=7,
        name="aim_r07_random_spawns",
        actions=[
            "TURN_LEFT_RIGHT_DELTA",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "ATTACK",
        ],
        angle_degrees_per_action=[2.0],
        map_overrides={
            "fixed_spawn_pair": None,
            "randomize_spawn_pair": True,
            "min_spawn_separation": 632,
        },
    ),
    "aim_r08_crates": _preset(
        rung=8,
        name="aim_r08_crates",
        actions=[
            "TURN_LEFT_RIGHT_DELTA",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "ATTACK",
        ],
        angle_degrees_per_action=[2.0],
        map_overrides={
            "fixed_spawn_pair": None,
            "randomize_spawn_pair": True,
            "min_spawn_separation": 632,
            "obstacle_pool_fill": 0.45,
            "crate_density_pct": 60,
        },
    ),
    "aim_r09_walls": _preset(
        rung=9,
        name="aim_r09_walls",
        actions=[
            "TURN_LEFT_RIGHT_DELTA",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "ATTACK",
        ],
        angle_degrees_per_action=[2.0],
        map_overrides={
            "fixed_spawn_pair": None,
            "randomize_spawn_pair": True,
            "min_spawn_separation": 632,
            "obstacle_pool_fill": 0.45,
            "crate_density_pct": 60,
            "wall_density_pct": 50,
        },
    ),
}


# Diagnostic scenarios are immutable campaign controls, not additional ladder
# rungs. Keep them separate so the numbered curriculum remains exactly r00-r09.
DIAGNOSTIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "aim_r04_strafe_balanced": _preset(
        rung=4,
        name="aim_r04_strafe_balanced",
        actions=["TURN_LEFT_RIGHT_DELTA", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        angle_degrees_per_action=[2.0],
        map_overrides={
            "fixed_spawn_pair": None,
            "randomize_spawn_pair": True,
            "spawn_pair_layout": "parallel_line",
            "spawn_yaw_offset_deg": 30.0,
        },
    ),
    "aim_r04_strafe_seat_swap": _preset(
        rung=4,
        name="aim_r04_strafe_seat_swap",
        actions=["TURN_LEFT_RIGHT_DELTA", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        angle_degrees_per_action=[2.0],
        map_overrides={
            "fixed_spawn_pair": [[128, 0], [-128, 0]],
            "spawn_yaw_offset_deg": 30.0,
        },
    ),
}


def preset_config(rung: str) -> Dict[str, Any]:
    """Return a defensive copy of a named rung preset."""
    try:
        return copy.deepcopy({**RUNG_PRESETS, **DIAGNOSTIC_PRESETS}[rung])
    except KeyError as exc:
        raise ValueError(f"unknown deathmatch aiming rung: {rung}") from exc


def manifest_json(rung: str) -> str:
    """Return deterministic JSON for the rung's generation manifest."""
    return json.dumps(preset_config(rung), indent=2, sort_keys=True) + "\n"


def manifest_json_for_wad(rung: str, wad_sha256: str) -> str:
    """Return the deterministic manifest augmented with the generated WAD hash."""
    manifest = preset_config(rung)
    manifest["wad_sha256"] = wad_sha256
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def generate_rung(rung: str, output_dir: str | Path) -> tuple[Path, Path]:
    """Generate one WAD and its manifest, returning both paths."""
    preset = preset_config(rung)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    wad_path = output_path / f"{rung}.wad"
    manifest_path = output_path / f"{rung}.json"

    DeathmatchDuelScenario(preset["scenario_config"], name=rung).generate(
        str(wad_path)
    )
    digest = hashlib.sha256(wad_path.read_bytes()).hexdigest()
    manifest_path.write_text(manifest_json_for_wad(rung, digest), encoding="utf-8")
    return wad_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rung",
        choices=sorted({**RUNG_PRESETS, **DIAGNOSTIC_PRESETS}),
        default="aim_r00_attack_only",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/benchmark/output/deathmatch_aiming",
        help="directory for the selected WAD and deterministic JSON manifest",
    )
    args = parser.parse_args()
    wad_path, manifest_path = generate_rung(args.rung, args.output_dir)
    print(f"Generated {wad_path}")
    print(f"Manifest {manifest_path}")


if __name__ == "__main__":
    main()
