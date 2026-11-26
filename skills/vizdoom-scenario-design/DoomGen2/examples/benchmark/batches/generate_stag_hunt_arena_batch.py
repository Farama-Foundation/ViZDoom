from _batch_utils import generate_batch
from examples.benchmark.stag_hunt_arena import StagHuntArenaScenario


SCENARIO_CLS = StagHuntArenaScenario
BASE_NAME = "stag_hunt_arena"
OUTPUT_SUBDIR = "batch_stag_hunt_arena"
EXPECTED_COUNT = 486
AXIS_OPTIONS = {
    "arena_size": [
        {"_label": "compact", "arena_size": 1792},
        {"_label": "standard", "arena_size": 2048},
        {"_label": "expansive", "arena_size": 2304},
    ],
    "alcove_layout": [
        {"_label": "few_shallow", "num_alcoves": 8, "alcove_width": 160, "alcove_depth": 96},
        {"_label": "balanced", "num_alcoves": 12, "alcove_width": 192, "alcove_depth": 128},
        {"_label": "many_deep", "num_alcoves": 16, "alcove_width": 224, "alcove_depth": 160},
    ],
    "stag_settings": [
        {
            "_label": "soft_stag",
            "stag_health": 400,
            "stag_regen_rate": 3,
            "stag_spawn_delay": 210,
            "stag_respawn_killed": 420,
            "stag_despawn_time": 2100,
            "stag_respawn_despawn": 160,
        },
        {
            "_label": "baseline_stag",
            "stag_health": 500,
            "stag_regen_rate": 4,
            "stag_spawn_delay": 175,
            "stag_respawn_killed": 500,
            "stag_despawn_time": 2450,
            "stag_respawn_despawn": 200,
        },
        {
            "_label": "hard_stag",
            "stag_health": 650,
            "stag_regen_rate": 5,
            "stag_spawn_delay": 140,
            "stag_respawn_killed": 580,
            "stag_despawn_time": 2800,
            "stag_respawn_despawn": 240,
        },
    ],
    "cooperative_radius": [
        {"_label": "tight", "stag_range": 420},
        {"_label": "standard", "stag_range": 555},
        {"_label": "wide", "stag_range": 700},
    ],
    "rabbit_pressure": [
        {
            "_label": "low_pressure",
            "rabbit_spawn_interval": 28,
            "max_rabbits": 8,
            "rabbit_speed": 20,
            "rabbit_kill_radius": 480,
        },
        {
            "_label": "baseline_pressure",
            "rabbit_spawn_interval": 20,
            "max_rabbits": 12,
            "rabbit_speed": 24,
            "rabbit_kill_radius": 555,
        },
        {
            "_label": "high_pressure",
            "rabbit_spawn_interval": 14,
            "max_rabbits": 16,
            "rabbit_speed": 28,
            "rabbit_kill_radius": 640,
        },
    ],
    "payoff_schedule": [
        {
            "_label": "hare_tilted",
            "rabbit_kill_reward": 4,
            "stag_kill_health_reward": 40,
            "stag_kill_ammo_reward": 16,
        },
        {
            "_label": "stag_tilted",
            "rabbit_kill_reward": 2,
            "stag_kill_health_reward": 60,
            "stag_kill_ammo_reward": 24,
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
