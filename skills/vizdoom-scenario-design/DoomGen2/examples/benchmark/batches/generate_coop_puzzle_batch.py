from _batch_utils import generate_batch
from examples.benchmark.coop_puzzle import CoopPuzzleScenario


SCENARIO_CLS = CoopPuzzleScenario
BASE_NAME = "coop_puzzle"
OUTPUT_SUBDIR = "batch_coop_puzzle"
EXPECTED_COUNT = 162
AXIS_OPTIONS = {
    "zone_count": [
        {"_label": "short", "num_zones": 8},
        {"_label": "standard", "num_zones": 10},
        {"_label": "long", "num_zones": 12},
    ],
    "lane_geometry": [
        {"_label": "tight", "zone_length": 320, "lane_height": 220, "lane_gap": 0},
        {"_label": "standard", "zone_length": 384, "lane_height": 240, "lane_gap": 32},
        {"_label": "wide", "zone_length": 448, "lane_height": 280, "lane_gap": 64},
    ],
    "plate_candidates": [
        {"_label": "few_large", "plates_per_side": 1, "plate_size": 96},
        {"_label": "balanced", "plates_per_side": 2, "plate_size": 72},
        {"_label": "many_small", "plates_per_side": 3, "plate_size": 56},
    ],
    "sync_frequency": [
        {"_label": "frequent", "and_frequency_pairs": 2},
        {"_label": "standard", "and_frequency_pairs": 3},
        {"_label": "rare", "and_frequency_pairs": 4},
    ],
    "affordance_color_theme": [
        {
            "_label": "amber_cyan",
            "plate_texture": "CEIL5_1",
            "highlight_texture_a": "AQF044",
            "highlight_texture_b": "AQF038",
        },
        {
            "_label": "cyan_amber",
            "plate_texture": "CEIL5_2",
            "highlight_texture_a": "AQF038",
            "highlight_texture_b": "AQF044",
        },
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
