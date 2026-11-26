from _batch_utils import generate_batch
from examples.benchmark.platform_chain import PlatformChainScenario


SCENARIO_CLS = PlatformChainScenario
BASE_NAME = "platform_chain"
OUTPUT_SUBDIR = "batch_platform_chain"
EXPECTED_COUNT = 243
AXIS_OPTIONS = {
    "route_length": [
        {"_label": "short", "level_count": 36},
        {"_label": "standard", "level_count": 48},
        {"_label": "long", "level_count": 60},
    ],
    "opening_safety_margin": [
        {"_label": "minimal", "safe_start_levels_min": 1, "safe_start_levels_max": 1},
        {"_label": "moderate", "safe_start_levels_min": 2, "safe_start_levels_max": 2},
        {"_label": "generous", "safe_start_levels_min": 4, "safe_start_levels_max": 4},
    ],
    "mover_density": [
        {"_label": "light", "polyobject_count": 4},
        {"_label": "standard", "polyobject_count": 8},
        {"_label": "dense", "polyobject_count": 12},
    ],
    "tether_slack": [
        {"_label": "tight", "chain_max_len": 220},
        {"_label": "standard", "chain_max_len": 250},
        {"_label": "loose", "chain_max_len": 280},
    ],
    "lava_severity": [
        {"_label": "mild", "lava_damage": 35},
        {"_label": "standard", "lava_damage": 50},
        {"_label": "harsh", "lava_damage": 70},
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
