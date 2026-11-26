from _batch_utils import generate_batch
from examples.benchmark.ammo_carrier import AmmoCarrierScenario


SCENARIO_CLS = AmmoCarrierScenario
BASE_NAME = "ammo_carrier"
OUTPUT_SUBDIR = "batch_ammo_carrier"
EXPECTED_COUNT = 81
AXIS_OPTIONS = {
    "map_radius": [
        {"_label": "compact", "map_radius": 1000},
        {"_label": "standard", "map_radius": 1200},
        {"_label": "wide", "map_radius": 1400},
    ],
    "depot_count": [
        {"_label": "single", "num_depots": 1},
        {"_label": "double", "num_depots": 2},
        {"_label": "triple", "num_depots": 3},
    ],
    "enemy_density": [
        {"_label": "light", "enemy_density": 0.75},
        {"_label": "standard", "enemy_density": 1.0},
        {"_label": "heavy", "enemy_density": 1.25},
    ],
    "layout_seed": [
        {"_label": "seed_41", "seed": 41},
        {"_label": "seed_42", "seed": 42},
        {"_label": "seed_43", "seed": 43},
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
