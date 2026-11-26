from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DOOMGEN_ROOT = Path(__file__).resolve().parents[3]

for path in (DOOMGEN_ROOT, DOOMGEN_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)


BATCH_OUTPUT_ROOT = DOOMGEN_ROOT / "examples" / "benchmark" / "output"

AxisOptions = Mapping[str, Sequence[dict[str, Any]]]


def _option_payload(option: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in option.items() if not key.startswith("_")}


def count_axis_combinations(axis_options: AxisOptions) -> int:
    total = 1
    for options in axis_options.values():
        total *= len(options)
    return total


def iter_axis_configs(axis_options: AxisOptions):
    axis_names = list(axis_options.keys())
    axis_values = [axis_options[name] for name in axis_names]

    for selections in itertools.product(*axis_values):
        config: dict[str, Any] = {}
        axis_choices: dict[str, Any] = {}

        for axis_name, selection in zip(axis_names, selections):
            payload = _option_payload(selection)
            overlap = set(config).intersection(payload)
            if overlap:
                overlap_list = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Axis '{axis_name}' overlaps existing config keys: {overlap_list}"
                )

            config.update(payload)
            axis_choices[axis_name] = selection.get("_label", payload)

        yield config, axis_choices


def generate_batch(
    scenario_cls,
    *,
    axis_options: AxisOptions,
    base_name: str,
    output_subdir: str,
) -> None:
    total = count_axis_combinations(axis_options)
    output_dir = BATCH_OUTPUT_ROOT / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {total} scenarios into {output_dir}...")

    registry = []
    for idx, (config, axis_choices) in enumerate(iter_axis_configs(axis_options), start=1):
        scenario_name = f"{base_name}_{idx - 1:04d}"
        scenario = scenario_cls(config, name=scenario_name)
        wad_name = f"{scenario_name}.wad"
        wad_path = output_dir / wad_name

        print(f"[{idx}/{total}] {wad_name}")
        scenario.generate(str(wad_path))

        registry.append(
            {
                "id": scenario_name,
                "filename": wad_name,
                "config": scenario.config,
                "axis_choices": axis_choices,
            }
        )

    registry_path = output_dir / "batch_registry.json"
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)

    print(f"Batch generation complete. Registry saved to {registry_path}")
