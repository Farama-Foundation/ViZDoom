import math
import statistics
from typing import Dict, Sequence


SHOOTER_WIN = 1
HIDER_WIN = 2
HIDER_ESCAPE = 3
DRAW = 4


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - margin, center + margin


def hide_and_seek_success_metrics(
    outcomes: Sequence[int], durations_seconds: Sequence[float]
) -> Dict[str, float]:
    """Aggregate terminal outcomes and right-censored episode durations."""
    if len(outcomes) != len(durations_seconds):
        raise ValueError("outcomes and durations_seconds must have the same length")
    if not outcomes:
        return {}
    if any(
        outcome not in (SHOOTER_WIN, HIDER_WIN, HIDER_ESCAPE, DRAW)
        for outcome in outcomes
    ):
        raise ValueError("all hide-and-seek outcomes must be terminal")

    total = len(outcomes)
    outcome_counts = {
        "capture": sum(outcome == SHOOTER_WIN for outcome in outcomes),
        "hider_win": sum(outcome == HIDER_WIN for outcome in outcomes),
        "escape": sum(outcome == HIDER_ESCAPE for outcome in outcomes),
        "draw": sum(outcome == DRAW for outcome in outcomes),
    }
    outcome_counts["hider_success"] = (
        outcome_counts["hider_win"] + outcome_counts["escape"]
    )

    metrics = {"episodes": float(total)}
    for name, count in outcome_counts.items():
        low, high = _wilson_interval(count, total)
        metrics[f"{name}_count"] = float(count)
        metrics[f"{name}_rate"] = count / total
        metrics[f"{name}_rate_ci95_low"] = low
        metrics[f"{name}_rate_ci95_high"] = high

    durations = [float(duration) for duration in durations_seconds]
    metrics["hider_survival_time_seconds_mean"] = statistics.fmean(durations)
    metrics["hider_survival_time_seconds_median"] = statistics.median(durations)
    capture_times = [
        duration
        for outcome, duration in zip(outcomes, durations)
        if outcome == SHOOTER_WIN
    ]
    if capture_times:
        metrics["capture_time_seconds_mean"] = statistics.fmean(capture_times)
        metrics["capture_time_seconds_median"] = statistics.median(capture_times)
    return metrics
