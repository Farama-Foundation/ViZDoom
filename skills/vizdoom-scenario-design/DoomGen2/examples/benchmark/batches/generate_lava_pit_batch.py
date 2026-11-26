from _batch_utils import generate_batch
from examples.benchmark.lavapit import LavaPitScenario


SCENARIO_CLS = LavaPitScenario
BASE_NAME = "lava_pit"
OUTPUT_SUBDIR = "batch_lava_pit"
EXPECTED_COUNT = 486
AXIS_OPTIONS = {
    "platform_count": [
        {"_label": "short", "num_platforms": 10},
        {"_label": "standard", "num_platforms": 12},
        {"_label": "long", "num_platforms": 14},
    ],
    "platform_gap_profile": [
        {
            "_label": "wide_safe",
            "platform_width_min": 400,
            "platform_width_max": 520,
            "platform_depth_min": 260,
            "platform_depth_max": 340,
            "gap_min": 220,
            "gap_max": 300,
        },
        {
            "_label": "baseline",
            "platform_width_min": 360,
            "platform_width_max": 460,
            "platform_depth_min": 240,
            "platform_depth_max": 320,
            "gap_min": 260,
            "gap_max": 360,
        },
        {
            "_label": "narrow_risky",
            "platform_width_min": 320,
            "platform_width_max": 420,
            "platform_depth_min": 220,
            "platform_depth_max": 300,
            "gap_min": 300,
            "gap_max": 420,
        },
    ],
    "route_curvature": [
        {"_label": "low", "lane_jitter": 24, "lane_span": 280, "curve_amplitude": 120, "curve_waves": 1},
        {"_label": "medium", "lane_jitter": 48, "lane_span": 360, "curve_amplitude": 220, "curve_waves": 1},
        {"_label": "high", "lane_jitter": 72, "lane_span": 440, "curve_amplitude": 320, "curve_waves": 2},
    ],
    "opening_bootstrap": [
        {
            "_label": "strong_bootstrap",
            "bootstrap_bridges": 4,
            "first_handoff_platform_width": 640,
            "first_handoff_platform_depth": 400,
            "first_handoff_gap": 128,
            "first_handoff_plate_scale": 2.2,
            "bridge0_support_latch_tics": 140,
        },
        {
            "_label": "baseline_bootstrap",
            "bootstrap_bridges": 3,
            "first_handoff_platform_width": 560,
            "first_handoff_platform_depth": 360,
            "first_handoff_gap": 160,
            "first_handoff_plate_scale": 1.9,
            "bridge0_support_latch_tics": 105,
        },
        {
            "_label": "light_bootstrap",
            "bootstrap_bridges": 2,
            "first_handoff_platform_width": 480,
            "first_handoff_platform_depth": 320,
            "first_handoff_gap": 192,
            "first_handoff_plate_scale": 1.6,
            "bridge0_support_latch_tics": 70,
        },
    ],
    "lava_severity": [
        {"_label": "mild", "lava_damage": 60},
        {"_label": "standard", "lava_damage": 100},
        {"_label": "harsh", "lava_damage": 140},
    ],
    "timing_jitter": [
        {"_label": "deterministic", "runtime_jitter_tics": 0},
        {"_label": "jittered", "runtime_jitter_tics": 20},
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
