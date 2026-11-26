from _batch_utils import generate_batch
from examples.benchmark.rhythm_sync import RhythmSyncScenario


SCENARIO_CLS = RhythmSyncScenario
BASE_NAME = "rhythm_sync"
OUTPUT_SUBDIR = "batch_rhythm_sync"
EXPECTED_COUNT = 243
AXIS_OPTIONS = {
    "section_count": [
        {"_label": "short", "num_sections": 20},
        {"_label": "medium", "num_sections": 24},
        {"_label": "long", "num_sections": 28},
    ],
    "stage_type_split": [
        {"_label": "regular_heavy", "intro_regular_sections": 6, "jitter_stage_count": 3},
        {"_label": "balanced", "intro_regular_sections": 8, "jitter_stage_count": 5},
        {"_label": "single_observer_heavy", "intro_regular_sections": 10, "jitter_stage_count": 7},
    ],
    "beat_tolerance_schedule": [
        {
            "_label": "forgiving",
            "beat_tics": 84,
            "jitter_tics": 12,
            "delta_t_tics": 20,
            "delta_t_minimum": 10,
            "delta_t_decay": 1,
        },
        {
            "_label": "baseline",
            "beat_tics": 72,
            "jitter_tics": 24,
            "delta_t_tics": 16,
            "delta_t_minimum": 8,
            "delta_t_decay": 1,
        },
        {
            "_label": "strict",
            "beat_tics": 60,
            "jitter_tics": 36,
            "delta_t_tics": 12,
            "delta_t_minimum": 6,
            "delta_t_decay": 2,
        },
    ],
    "room_geometry": [
        {
            "_label": "compact",
            "stage_step": 448,
            "room_width": 448,
            "room_height": 224,
            "top_y": 112,
            "bottom_y": -112,
            "switch_use_range": 56,
        },
        {
            "_label": "standard",
            "stage_step": 512,
            "room_width": 512,
            "room_height": 256,
            "top_y": 128,
            "bottom_y": -128,
            "switch_use_range": 64,
        },
        {
            "_label": "large",
            "stage_step": 576,
            "room_width": 576,
            "room_height": 288,
            "top_y": 144,
            "bottom_y": -144,
            "switch_use_range": 72,
        },
    ],
    "switch_layout_jitter": [
        {
            "_label": "mild",
            "switch_wall_margin": 48,
            "switch_intro_stages": 10,
            "switch_intro_jitter_frac": 0.05,
            "switch_pos_seed_offset": 0,
        },
        {
            "_label": "moderate",
            "switch_wall_margin": 32,
            "switch_intro_stages": 8,
            "switch_intro_jitter_frac": 0.125,
            "switch_pos_seed_offset": 37,
        },
        {
            "_label": "heavy",
            "switch_wall_margin": 20,
            "switch_intro_stages": 6,
            "switch_intro_jitter_frac": 0.25,
            "switch_pos_seed_offset": 73,
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
