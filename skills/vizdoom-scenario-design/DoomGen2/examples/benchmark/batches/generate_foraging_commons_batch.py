from _batch_utils import generate_batch
from examples.benchmark.foraging_commons import ForagingCommonsScenario


SCENARIO_CLS = ForagingCommonsScenario
BASE_NAME = "foraging_commons"
OUTPUT_SUBDIR = "batch_foraging_commons"
EXPECTED_COUNT = 729
AXIS_OPTIONS = {
    "field_cleanup_geometry": [
        {
            "_label": "compact",
            "field_size": 1400,
            "cleanup_size": 320,
            "corridor_width": 128,
        },
        {
            "_label": "standard",
            "field_size": 1600,
            "cleanup_size": 384,
            "corridor_width": 192,
        },
        {
            "_label": "wide",
            "field_size": 1800,
            "cleanup_size": 448,
            "corridor_width": 256,
        },
    ],
    "cleanup_separation": [
        {"_label": "near", "cleanup_distance": 900},
        {"_label": "mid", "cleanup_distance": 1200},
        {"_label": "far", "cleanup_distance": 1500},
    ],
    "agent_count": [
        {"_label": "duo", "num_agents": 2},
        {"_label": "triad", "num_agents": 3},
        {"_label": "squad", "num_agents": 4},
    ],
    "commons_dynamics": [
        {
            "_label": "generous",
            "initial_spawn_rate": 90,
            "harvest_degrade": 1,
            "natural_decay_rate": 1,
            "cleanup_boost": 18,
            "cleanup_time": 56,
        },
        {
            "_label": "baseline",
            "initial_spawn_rate": 80,
            "harvest_degrade": 2,
            "natural_decay_rate": 1,
            "cleanup_boost": 15,
            "cleanup_time": 72,
        },
        {
            "_label": "fragile",
            "initial_spawn_rate": 70,
            "harvest_degrade": 3,
            "natural_decay_rate": 2,
            "cleanup_boost": 12,
            "cleanup_time": 88,
        },
    ],
    "health_drain": [
        {"_label": "light", "hp_drain": 1},
        {"_label": "standard", "hp_drain": 2},
        {"_label": "heavy", "hp_drain": 3},
    ],
    "spawn_spot_density": [
        {"_label": "sparse", "spots_per_room": 12},
        {"_label": "baseline", "spots_per_room": 16},
        {"_label": "dense", "spots_per_room": 20},
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
