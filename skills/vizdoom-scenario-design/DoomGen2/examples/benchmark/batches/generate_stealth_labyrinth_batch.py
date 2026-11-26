from _batch_utils import generate_batch
from examples.benchmark.stealth_labyrinth import StealthLabyrinthScenario


SCENARIO_CLS = StealthLabyrinthScenario
BASE_NAME = "stealth_labyrinth"
OUTPUT_SUBDIR = "batch_stealth_labyrinth"
EXPECTED_COUNT = 486
AXIS_OPTIONS = {
    "branch_count": [
        {"_label": "single_branch", "num_branches": 1},
        {"_label": "double_branch", "num_branches": 2},
        {"_label": "triple_branch", "num_branches": 3},
    ],
    "rooms_per_branch": [
        {"_label": "short_branch", "rooms_per_branch": 1},
        {"_label": "standard_branch", "rooms_per_branch": 2},
        {"_label": "deep_branch", "rooms_per_branch": 3},
    ],
    "room_corridor_geometry": [
        {"_label": "compact", "branch_room_width": 384, "branch_room_length": 768, "corridor_width": 5},
        {"_label": "standard", "branch_room_width": 416, "branch_room_length": 896, "corridor_width": 4},
        {"_label": "spacious", "branch_room_width": 512, "branch_room_length": 1024, "corridor_width": 3},
    ],
    "enemy_group_size": [
        {"_label": "solo", "min_enemies_per_room": 1, "max_enemies_per_room": 1},
        {"_label": "mixed", "min_enemies_per_room": 1, "max_enemies_per_room": 2},
        {"_label": "paired", "min_enemies_per_room": 2, "max_enemies_per_room": 2},
    ],
    "turret_strength": [
        {
            "_label": "soft",
            "turret_health": 120,
            "turret_damage": 1,
            "turret_projectile_speed": 12,
            "turret_burst_count": 2,
            "turret_refire_tics": 18,
        },
        {
            "_label": "standard",
            "turret_health": 140,
            "turret_damage": 2,
            "turret_projectile_speed": 14,
            "turret_burst_count": 2,
            "turret_refire_tics": 14,
        },
        {
            "_label": "hard",
            "turret_health": 180,
            "turret_damage": 3,
            "turret_projectile_speed": 16,
            "turret_burst_count": 3,
            "turret_refire_tics": 10,
        },
    ],
    "first_room_bootstrap": [
        {"_label": "strong_bootstrap", "first_room_turret_health": 70, "first_room_turret_wall_margin": 360},
        {"_label": "light_bootstrap", "first_room_turret_health": 110, "first_room_turret_wall_margin": 280},
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
