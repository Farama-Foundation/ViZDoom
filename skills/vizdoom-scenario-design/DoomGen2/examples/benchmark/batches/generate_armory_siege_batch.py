from _batch_utils import generate_batch
from examples.benchmark.armory_siege import ArmorySiegeScenario


SCENARIO_CLS = ArmorySiegeScenario
BASE_NAME = "armory_siege"
OUTPUT_SUBDIR = "batch_armory_siege"
EXPECTED_COUNT = 1215
AXIS_OPTIONS = {
    "room_distance": [
        {"_label": "close", "distance": 500},
        {"_label": "mid", "distance": 700},
        {"_label": "far", "distance": 900},
    ],
    "corridor_width": [
        {"_label": "tight", "corridor_width": 128},
        {"_label": "standard", "corridor_width": 192},
        {"_label": "wide", "corridor_width": 256},
    ],
    "door_timing": [
        {"_label": "brief", "door_timer": 150},
        {"_label": "standard", "door_timer": 300},
        {"_label": "long", "door_timer": 450},
    ],
    "core_health_budget": [
        {"_label": "fragile", "core_health": 600},
        {"_label": "baseline", "core_health": 1000},
        {"_label": "durable", "core_health": 1400},
    ],
    "enemy_pressure": [
        {"_label": "light", "enemy_difficulty": 0.5},
        {"_label": "standard", "enemy_difficulty": 1.0},
        {"_label": "heavy", "enemy_difficulty": 1.5},
    ],
    "team_size": [
        {"_label": "2p", "max_players": 2},
        {"_label": "3p", "max_players": 3},
        {"_label": "4p", "max_players": 4},
        {"_label": "6p", "max_players": 6},
        {"_label": "8p", "max_players": 8},
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
