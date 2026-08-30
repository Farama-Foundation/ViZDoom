from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


DEFAULT_BOT_PROFILES: Mapping[str, str] = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}


class PolicyAdapter(Protocol):
    """Required by DoomGame"""

    def reset(self, seed: int | None = None) -> None:
        ...

    def act(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class BotEvalConfig:
    scenario: str = "multi_duel_pistol_big"
    scenario_config: str | None = None
    resolution: str = "320X240"
    skip_frames: int = 1
    # None keeps whatever episode_timeout the scenario .cfg has
    episode_timeout: int | None = None
    num_bots: int = 1
    learner_name: str = "Learner"
    require_deathmatch: bool = True
    ticrate: int = 35
    # Respawn delay in SECONDS
    respawn_delay: int = 0
    screening_valid_episodes: int = 10
    screening_max_attempts: int = 20
    final_valid_episodes: int = 50
    final_max_attempts: int = 100
    interval_episodes: int = 100
    seed: int = 42
    bootstrap_confidence: float = 0.95
    bootstrap_samples: int = 10_000
    video_dir: str = "bot_eval_videos"
    video_fps: int = 35
    output_dir: str | None = None
    tiers: tuple[str, ...] = ("easy", "medium", "hard")


@dataclass
class EpisodeResult:
    seed: int
    tier: str
    valid: bool
    learner_frags: int | None = None
    bot_frags: int | None = None
    learner_deaths: int | None = None
    learner_damage_made: float | None = None
    learner_damage_taken: float | None = None
    duration_seconds: float | None = None
    engine_tics: int | None = None
    policy_steps: int | None = None
    timeout: bool = False
    invalid_reason: str | None = None
    outcome: str | None = None
    bot_profile: str | None = None

    @property
    def frag_diff(self) -> int | None:
        if self.learner_frags is None or self.bot_frags is None:
            return None
        return int(self.learner_frags - self.bot_frags)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frag_diff"] = self.frag_diff
        return payload


@dataclass
class TierSummary:
    tier: str
    valid_episodes: int
    invalid_episodes: int
    frag_diff_mean: float
    frag_diff_std: float
    frag_diff_sem: float
    frag_diff_ci_low: float
    frag_diff_ci_high: float
    win_rate: float
    tie_rate: float
    loss_rate: float
    learner_frags_mean: float
    bot_frags_mean: float
    learner_damage_made_mean: float
    learner_damage_taken_mean: float
    learner_deaths_mean: float
    timeout_rate: float
    bootstrap_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeedSchedule:
    base_seed: int
    tiers: tuple[str, ...]
    screening: Mapping[str, tuple[int, ...]]
    final: Mapping[str, tuple[int, ...]]
    showcase: Mapping[str, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "tiers": list(self.tiers),
            "screening": {k: list(v) for k, v in self.screening.items()},
            "final": {k: list(v) for k, v in self.final.items()},
            "showcase": {k: list(v) for k, v in self.showcase.items()},
        }


def classify_outcome(learner_frags: int, bot_frags: int) -> str:
    if learner_frags > bot_frags:
        return "win"
    if learner_frags < bot_frags:
        return "loss"
    return "tie"


def bootstrap_percentile_ci(
    values: Sequence[float],
    seed: int,
    confidence: float = 0.95,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Return bootstrap CI for sample mean"""
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(int(seed))
    draws = rng.choice(
        values_array, size=(int(samples), values_array.size), replace=True
    )
    means = draws.mean(axis=1)
    alpha = (1.0 - confidence) * 50.0
    low, high = np.percentile(means, [alpha, 100.0 - alpha])
    return float(low), float(high)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_tier(
    tier: str,
    results: Sequence[EpisodeResult],
    bootstrap_seed: int,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
) -> TierSummary:
    valid = [result for result in results if result.valid]
    diffs = [
        float(result.frag_diff) for result in valid if result.frag_diff is not None
    ]
    outcomes = [result.outcome for result in valid if result.outcome is not None]
    wins = [outcome for outcome in outcomes if outcome == "win"]
    ties = [outcome for outcome in outcomes if outcome == "tie"]
    losses = [outcome for outcome in outcomes if outcome == "loss"]
    diff_low, diff_high = bootstrap_percentile_ci(
        diffs,
        seed=bootstrap_seed,
        confidence=confidence,
        samples=bootstrap_samples,
    )

    def numeric(name: str) -> list[float]:
        return [
            float(getattr(result, name))
            for result in valid
            if getattr(result, name) is not None
        ]

    denominator = len(valid)
    return TierSummary(
        tier=tier,
        valid_episodes=denominator,
        invalid_episodes=len(results) - denominator,
        frag_diff_mean=_mean(diffs),
        # ddof=1: these are samples from the episode distribution, not the
        # population. Matters at n=10.
        frag_diff_std=float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        frag_diff_sem=(
            float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            if len(diffs) > 1
            else 0.0
        ),
        frag_diff_ci_low=diff_low,
        frag_diff_ci_high=diff_high,
        win_rate=len(wins) / denominator if denominator else 0.0,
        tie_rate=len(ties) / denominator if denominator else 0.0,
        loss_rate=len(losses) / denominator if denominator else 0.0,
        learner_frags_mean=_mean(numeric("learner_frags")),
        bot_frags_mean=_mean(numeric("bot_frags")),
        learner_damage_made_mean=_mean(numeric("learner_damage_made")),
        learner_damage_taken_mean=_mean(numeric("learner_damage_taken")),
        learner_deaths_mean=_mean(numeric("learner_deaths")),
        timeout_rate=(
            sum(result.timeout for result in valid) / denominator
            if denominator
            else 0.0
        ),
        bootstrap_seed=int(bootstrap_seed),
    )


def _seed(base_seed: int, namespace: str, tier: str, index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{namespace}:{tier}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def build_seed_schedule(
    base_seed: int,
    tiers: Sequence[str] = ("easy", "medium", "hard"),
    screening_attempts: int = 20,
    final_attempts: int = 100,
) -> SeedSchedule:
    normalized_tiers = tuple(str(tier).lower() for tier in tiers)
    namespaces = {
        "screening": int(screening_attempts),
        "final": int(final_attempts),
        "showcase": 1,
    }
    suites: dict[str, dict[str, tuple[int, ...]]] = {}
    for namespace, count in namespaces.items():
        if count < 1:
            raise ValueError(f"{namespace} seed count must be positive")
        suites[namespace] = {
            tier: tuple(_seed(base_seed, namespace, tier, i) for i in range(count))
            for tier in normalized_tiers
        }
    return SeedSchedule(
        base_seed=int(base_seed),
        tiers=normalized_tiers,
        screening=suites["screening"],
        final=suites["final"],
        showcase=suites["showcase"],
    )
