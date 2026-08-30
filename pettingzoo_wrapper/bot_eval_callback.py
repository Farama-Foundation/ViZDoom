from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from benchmarl.experiment.callback import Callback

from .bot_eval_duel import BotDuelEvaluator
from .bot_eval_policy import TorchRLPolicyAdapter, load_bot_eval_experiment
from .bot_eval_types import (
    BotEvalConfig,
    EpisodeResult,
    SeedSchedule,
    TierSummary,
    build_seed_schedule,
    summarize_tier,
)
from .video_recorder import log_wandb_video, write_frames_to_mp4


def _joint_done_count(batch: Any) -> int:
    done = batch.get(("next", "done"))
    if done is None:
        return 0
    done = np.asarray(done.detach().cpu().numpy(), dtype=bool)
    if done.shape[-1] == 1:
        done = np.squeeze(done, axis=-1)
    if done.ndim > 2:
        done = np.any(done, axis=-1)
    return int(done.sum())


def _build_schedule(config: BotEvalConfig) -> SeedSchedule:
    return build_seed_schedule(
        config.seed,
        tiers=config.tiers,
        screening_attempts=config.screening_max_attempts,
        final_attempts=config.final_max_attempts,
    )


def _checkpoint_step(checkpoint: Path) -> int:
    return int(checkpoint.stem.removeprefix("checkpoint_"))


def _output_root(config: BotEvalConfig, experiment: Any) -> Path:
    if config.output_dir:
        return Path(config.output_dir).expanduser().resolve()
    return Path(experiment.folder_name).resolve() / "bot_eval"


def _run_tier_episodes(
    evaluator: Any,
    policy: Any,
    tiers: Sequence[str],
    seeds: Mapping[str, Sequence[int]],
    *,
    valid_target: int,
) -> tuple[dict[str, list[EpisodeResult]], str]:
    """Run episodes per tier until enough valid ones or attempts run out"""
    results: dict[str, list[EpisodeResult]] = {tier: [] for tier in tiers}
    status = "complete"
    for tier in tiers:
        for seed in seeds[tier]:
            if sum(result.valid for result in results[tier]) >= valid_target:
                break
            run = evaluator.run_episode(
                seed=seed,
                tier=tier,
                policy=policy,
            )
            results[tier].append(run.result)
            if not run.result.valid:
                print(
                    f"[BotEval] invalid episode tier={tier} seed={seed}: "
                    f"{run.result.invalid_reason}"
                )
        if sum(result.valid for result in results[tier]) < valid_target:
            status = "ineligible"
    return results, status


def _summarize_tiers(
    results: Mapping[str, Sequence[EpisodeResult]],
    tiers: Sequence[str],
    config: BotEvalConfig,
    bootstrap_seed: int,
) -> dict[str, TierSummary]:
    summaries: dict[str, TierSummary] = {}
    for tier in tiers:
        summaries[tier] = summarize_tier(
            tier,
            results[tier],
            bootstrap_seed=int(bootstrap_seed),
            bootstrap_samples=config.bootstrap_samples,
            confidence=config.bootstrap_confidence,
        )
    return summaries


def _write_manifest(
    path: Path,
    header: Mapping[str, Any],
    results: Mapping[str, Sequence[EpisodeResult]],
    tiers: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8") as manifest:
        manifest.write(json.dumps(header, sort_keys=True) + "\n")
        for tier in tiers:
            for result in results[tier]:
                manifest.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


@dataclass
class EvaluationEvent:
    event_id: str
    threshold_episode: int
    status: str
    summaries: Mapping[str, TierSummary]
    manifest_path: Path


class BotEvaluationRunner:
    """Run synchronously after a training update, seeded"""

    def __init__(
        self,
        config: BotEvalConfig,
        *,
        duel_evaluator: Any | None = None,
        policy_factory: Callable[[Any], Any] | None = None,
        metric_logger: Callable[..., None] | None = None,
    ) -> None:
        self.config = config
        self.duel_evaluator = duel_evaluator or BotDuelEvaluator(config)
        self.policy_factory = policy_factory or self.default_policy_factory
        self.metric_logger = metric_logger
        self.seed_schedule = _build_schedule(config)

    @staticmethod
    def default_policy_factory(experiment: Any):
        group_name = next(iter(experiment.group_map.keys()))
        return TorchRLPolicyAdapter.from_experiment(experiment, group_name=group_name)

    def _log_summaries(
        self, summaries: Mapping[str, TierSummary], threshold_episode: int
    ) -> None:
        if self.metric_logger is None:
            return
        # Every field TierSummary computes
        fields = (
            "frag_diff_mean",
            "frag_diff_std",
            "frag_diff_sem",
            "frag_diff_ci_low",
            "frag_diff_ci_high",
            "win_rate",
            "tie_rate",
            "loss_rate",
            "learner_frags_mean",
            "bot_frags_mean",
            "learner_deaths_mean",
            "learner_damage_made_mean",
            "learner_damage_taken_mean",
            "timeout_rate",
            "valid_episodes",
            "invalid_episodes",
        )
        payload = {
            f"eval/bot/{tier}/{field_name}": float(getattr(summary, field_name))
            for tier, summary in summaries.items()
            for field_name in fields
        }
        self.metric_logger(payload, step=threshold_episode)

    def run_screening(self, experiment: Any, threshold_episode: int) -> EvaluationEvent:
        event_id = f"episode_{int(threshold_episode)}"
        output_root = _output_root(self.config, experiment)
        output_root.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        policy = self.policy_factory(experiment)

        results, status = _run_tier_episodes(
            self.duel_evaluator,
            policy,
            self.seed_schedule.tiers,
            self.seed_schedule.screening,
            valid_target=self.config.screening_valid_episodes,
        )
        summaries = _summarize_tiers(
            results,
            self.seed_schedule.tiers,
            self.config,
            bootstrap_seed=self.config.seed + threshold_episode,
        )
        manifest_path = output_root / f"{event_id}.jsonl"
        header = {
            "event_id": event_id,
            "threshold_episode": int(threshold_episode),
            "status": status,
            "elapsed_seconds": time.monotonic() - started,
            "seed_schedule": self.seed_schedule.to_dict(),
            "summaries": {
                tier: summary.to_dict() for tier, summary in summaries.items()
            },
        }
        _write_manifest(manifest_path, header, results, self.seed_schedule.tiers)
        self._log_summaries(summaries, threshold_episode)
        return EvaluationEvent(
            event_id=event_id,
            threshold_episode=int(threshold_episode),
            status=status,
            summaries=summaries,
            manifest_path=manifest_path,
        )


def run_final_bot_evaluation(
    checkpoint_path: str | Path,
    config: BotEvalConfig,
    *,
    duel_evaluator: Any | None = None,
    policy_factory: Callable[[Any], Any] | None = None,
) -> EvaluationEvent:
    """Evaluate a saved checkpoint on the final episode set"""
    checkpoint = Path(checkpoint_path).resolve()
    experiment = load_bot_eval_experiment(checkpoint)
    evaluator = (
        duel_evaluator if duel_evaluator is not None else BotDuelEvaluator(config)
    )
    policy = (policy_factory or BotEvaluationRunner.default_policy_factory)(experiment)
    schedule = _build_schedule(config)
    started = time.monotonic()
    results, status = _run_tier_episodes(
        evaluator,
        policy,
        schedule.tiers,
        schedule.final,
        valid_target=config.final_valid_episodes,
    )
    summaries = _summarize_tiers(
        results, schedule.tiers, config, bootstrap_seed=config.seed + 10_000
    )
    event_id = f"final_checkpoint_{_checkpoint_step(checkpoint)}"
    output_root = _output_root(config, experiment)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{event_id}.jsonl"
    header = {
        "event_id": event_id,
        "checkpoint_path": str(checkpoint),
        "status": status,
        "elapsed_seconds": time.monotonic() - started,
        "seed_schedule": schedule.to_dict(),
        "summaries": {tier: summary.to_dict() for tier, summary in summaries.items()},
    }
    _write_manifest(manifest_path, header, results, schedule.tiers)
    return EvaluationEvent(
        event_id=event_id,
        threshold_episode=0,
        status=status,
        summaries=summaries,
        manifest_path=manifest_path,
    )


class BotEvaluationCallback(Callback):
    """Callback to count episode completions"""

    def __init__(self, runner: Any, interval_episodes: int = 100) -> None:
        super().__init__()
        if interval_episodes < 1:
            raise ValueError("interval_episodes must be positive")
        self.interval_episodes = int(interval_episodes)
        self.runner = runner
        self.completed_training_episodes = 0
        self.next_threshold = self.interval_episodes
        self.pending_thresholds: list[int] = []

    def on_batch_collected(self, batch: Any) -> None:
        self.completed_training_episodes += _joint_done_count(batch)
        while self.completed_training_episodes >= self.next_threshold:
            self.pending_thresholds.append(self.next_threshold)
            self.next_threshold += self.interval_episodes

    def on_train_end(self, training_td: Any, group: str) -> None:
        if not self.pending_thresholds:
            return
        skipped = len(self.pending_thresholds) - 1
        threshold = self.pending_thresholds[-1]
        self.pending_thresholds.clear()
        if skipped:
            print(
                f"[BotEval] collapsed {skipped} stale threshold(s), "
                f"screening once at episode {threshold}"
            )
        self.runner.run_screening(self.experiment, threshold)

    def on_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict["bot_eval_callback"] = {
            "completed_training_episodes": self.completed_training_episodes,
            "next_threshold": self.next_threshold,
            "pending_thresholds": list(self.pending_thresholds),
        }

    def on_load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved = state_dict.get("bot_eval_callback")
        if saved is None:
            return
        self.completed_training_episodes = int(saved["completed_training_episodes"])
        self.next_threshold = int(saved["next_threshold"])
        self.pending_thresholds = [int(value) for value in saved["pending_thresholds"]]


def run_final_bot_eval_videos(
    checkpoint_path: str | Path,
    config: BotEvalConfig,
    *,
    duel_evaluator: Any | None = None,
    policy_factory: Callable[[Any], Any] | None = None,
) -> list[Path]:
    """Record one video per difficulty for the final checkpoint"""
    evaluator = (
        duel_evaluator if duel_evaluator is not None else BotDuelEvaluator(config)
    )
    policy_factory = policy_factory or BotEvaluationRunner.default_policy_factory
    schedule = _build_schedule(config)
    written: list[Path] = []
    checkpoint = Path(checkpoint_path).resolve()
    experiment = load_bot_eval_experiment(checkpoint)
    policy = policy_factory(experiment)
    training_step = _checkpoint_step(checkpoint)
    for tier in schedule.tiers:
        seed = schedule.showcase[tier][0]
        run = evaluator.run_episode(
            seed=seed,
            tier=tier,
            policy=policy,
            capture_video=True,
        )
        if not run.result.valid or not run.frames:
            continue
        output = (
            Path(config.video_dir)
            / tier
            / f"checkpoint_{training_step}"
            / f"episode_{seed}.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_frames_to_mp4(run.frames, output, config.video_fps)
        written.append(output)
        log_wandb_video(output, f"bot_eval/{tier}/checkpoint_{training_step}")
    return written


def run_post_training_bot_eval(
    experiment_folder: str | Path, config: BotEvalConfig
) -> EvaluationEvent | None:
    """Evaluate the final training checkpoint and record videos"""
    folder = Path(experiment_folder)
    checkpoints = sorted(
        (folder / "checkpoints").glob("checkpoint_*.pt"),
        key=_checkpoint_step,
    )
    if not checkpoints:
        return None
    final_checkpoint = checkpoints[-1]
    final_event = run_final_bot_evaluation(final_checkpoint, config)
    if final_event.status == "complete":
        run_final_bot_eval_videos(final_checkpoint, config)
    else:
        print(
            f"[BotEval] status={final_event.status!r} for {final_event.event_id}, "
            "skipping showcase videos"
        )
    return final_event
