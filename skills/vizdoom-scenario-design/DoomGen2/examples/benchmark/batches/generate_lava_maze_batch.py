from _batch_utils import generate_batch
from examples.benchmark.lava_maze import LavaMazeScenario


SCENARIO_CLS = LavaMazeScenario
BASE_NAME = "lava_maze"
OUTPUT_SUBDIR = "batch_lava_maze"
EXPECTED_COUNT = 81
AXIS_OPTIONS = {
    "physical_grid_size": [
        {"_label": "small", "physical_size": 4},
        {"_label": "medium", "physical_size": 6},
        {"_label": "large", "physical_size": 8},
    ],
    "cell_size": [
        {"_label": "tight_cells", "cell_size": 96},
        {"_label": "standard_cells", "cell_size": 128},
        {"_label": "wide_cells", "cell_size": 160},
    ],
    "maze_growth_schedule": [
        {"_label": "short_growth", "initial_maze_size": 4, "max_maze_size": 8},
        {"_label": "medium_growth", "initial_maze_size": 6, "max_maze_size": 14},
        {"_label": "long_growth", "initial_maze_size": 8, "max_maze_size": 20},
    ],
    "lava_severity": [
        {"_label": "shallow_mild", "lava_damage": 20, "lava_depth": 24},
        {"_label": "baseline", "lava_damage": 35, "lava_depth": 64},
        {"_label": "deep_harsh", "lava_damage": 50, "lava_depth": 96},
    ],
}


def main() -> None:
    generate_batch(
        SCENARIO_CLS,
        axis_options=AXIS_OPTIONS,
        base_name=BASE_NAME,
        output_subdir=OUTPUT_SUBDIR,
    )


if __name__ == "__main__":
    main()
