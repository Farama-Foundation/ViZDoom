from _batch_utils import generate_batch
from examples.benchmark.coop_health_gathering import CoopHealthGatheringScenario


SCENARIO_CLS = CoopHealthGatheringScenario
BASE_NAME = "coop_health_gathering"
OUTPUT_SUBDIR = "batch_coop_health_gathering"
EXPECTED_COUNT = 729
AXIS_OPTIONS = {
    "map_radius": [
        {"_label": "compact", "map_radius": 1600},
        {"_label": "standard", "map_radius": 2000},
        {"_label": "expansive", "map_radius": 2400},
    ],
    "tether_slack": [
        {"_label": "tight", "chain_max_len": 320},
        {"_label": "standard", "chain_max_len": 400},
        {"_label": "loose", "chain_max_len": 480},
    ],
    "toxic_floor_damage": [
        {"_label": "light", "toxic_damage": 3},
        {"_label": "standard", "toxic_damage": 5},
        {"_label": "heavy", "toxic_damage": 7},
    ],
    "health_kit_cap": [
        {"_label": "low", "max_health_kits": 60},
        {"_label": "standard", "max_health_kits": 80},
        {"_label": "high", "max_health_kits": 100},
    ],
    "maze_density": [
        {"_label": "open", "maze_density": 0.45},
        {"_label": "standard", "maze_density": 0.6},
        {"_label": "dense", "maze_density": 0.75},
    ],
    "team_size": [
        {"_label": "2p", "max_players": 2},
        {"_label": "3p", "max_players": 3},
        {"_label": "4p", "max_players": 4},
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
