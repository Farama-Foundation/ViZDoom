import os
import json

import _batch_utils  # noqa: F401  (puts the repo root and src/ on sys.path)
from doomgen.batch import BatchGenerator
from examples.benchmark.lava_maze import LavaMazeScenario


SCENARIO_CLS = LavaMazeScenario
BASE_NAME = "lava_maze"
OUTPUT_SUBDIR = "batch_lava_maze"

PARAM_GRIDS = [
    {
        "physical_size":     [4],
        "initial_maze_size": [3, 5],
        "max_maze_size":     [4],
        "lava_depth":        [24, 64],
        "lava_damage":       [10, 20],
    },
    {
        "physical_size":     [6],
        "initial_maze_size": [6, 8],
        "max_maze_size":     [12],
        "lava_depth":        [24],
        "lava_damage":       [20, 35],
    },

    {
        "physical_size":     [6, 8],
        "initial_maze_size": [6, 8, 10],
        "max_maze_size":     [12, 20],
        "lava_depth":        [64],
        "lava_damage":       [35],
    }
]

for _grid in PARAM_GRIDS:
    _grid["cell_size"] = [128]
    _grid["seed"] = list(range(3))

# The small batch is a union of grids, so it is exposed as one axis whose
# options are the exact configs main() generates, in the same order.
AXIS_OPTIONS = {
    "grid_config": [
        {"_label": f"grid{grid_idx}_{cfg_idx}", **config}
        for grid_idx, grid in enumerate(PARAM_GRIDS)
        for cfg_idx, config in enumerate(BatchGenerator(SCENARIO_CLS, param_grid=grid).generate_configs())
    ],
}
EXPECTED_COUNT = 72


def main():
    output_dir = f"examples/benchmark/output/{OUTPUT_SUBDIR}"
    os.makedirs(output_dir, exist_ok=True)
    batch_metadata = []
    total_generated = 0

    for grid_idx, param_grid in enumerate(PARAM_GRIDS):
        generator = BatchGenerator(SCENARIO_CLS, param_grid=param_grid)

        for scenario in generator.generate_scenarios(base_name=BASE_NAME):
            unique_name = f"{BASE_NAME}_{total_generated}"
            scenario.name = unique_name

            total_generated += 1

            filename = f"{unique_name}.wad"
            path = os.path.join(output_dir, filename)
            try:
                scenario.generate(path)
                meta = {"id": unique_name, "filename": filename, "config": scenario.config}
                batch_metadata.append(meta)
            except Exception as e:
                print(f"Failed to generate {filename}: {e}")

    registry_path = os.path.join(output_dir, "batch_registry.json")
    with open(registry_path, "w") as f:
        json.dump(batch_metadata, f, indent=4)

    print(f"Done. Total scenarios: {total_generated}. Registry: {registry_path}")

if __name__ == "__main__":
    main()
