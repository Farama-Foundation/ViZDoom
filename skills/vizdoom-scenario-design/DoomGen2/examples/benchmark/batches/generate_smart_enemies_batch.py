from _batch_utils import generate_batch
from examples.benchmark.smart_enemies import SmartEnemiesScenario


SCENARIO_CLS = SmartEnemiesScenario
BASE_NAME = "smart_enemies"
OUTPUT_SUBDIR = "batch_smart_enemies"
EXPECTED_COUNT = 729
AXIS_OPTIONS = {
    "arena_size": [
        {"_label": "compact", "arena_radius": 700},
        {"_label": "standard", "arena_radius": 850},
        {"_label": "wide", "arena_radius": 1000},
    ],
    "obstacle_layout_count": [
        {"_label": "sparse_layout_a", "num_obstacles": 18, "seed": 41},
        {"_label": "balanced_layout_b", "num_obstacles": 30, "seed": 42},
        {"_label": "dense_layout_c", "num_obstacles": 42, "seed": 43},
    ],
    "enemy_cap": [
        {"_label": "low", "num_enemies": 3},
        {"_label": "medium", "num_enemies": 5},
        {"_label": "high", "num_enemies": 7},
    ],
    "spawn_interval": [
        {"_label": "fast", "spawn_interval": 2},
        {"_label": "standard", "spawn_interval": 3},
        {"_label": "slow", "spawn_interval": 4},
    ],
    "proximity_radius": [
        {"_label": "tight", "proximity_radius": 384},
        {"_label": "standard", "proximity_radius": 512},
        {"_label": "wide", "proximity_radius": 640},
    ],
    "team_size": [
        {"_label": "2p", "player_count": 2},
        {"_label": "3p", "player_count": 3},
        {"_label": "4p", "player_count": 4},
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
