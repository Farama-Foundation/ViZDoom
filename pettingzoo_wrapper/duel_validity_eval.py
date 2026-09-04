"""
Evaluation for self-play. Output:
1. a cross-play table: policy vs no-op, vs uniform random, vs itself, vs the previous checkpoint and vs the built-in bots (easy/medium/hard);
2. a blind ablation (frozen reset frame / frames of the same episode in shuffled order) against the random opponent - a policy that uses vision must collapse;
3. line-of-sight / yaw-error / aim metrics from the ACS globals exported by the scenario (USER54/55 alive, USER57/58 LOS, USER59/60 yaw error), per-button rates and rockets fired;

into `validity_eval/summary.json`, `validity_eval/report.md`, per-episode `validity_eval/episodes.jsonl`.

python -m pettingzoo_wrapper.duel_validity_eval <path_to_checkpoint.pt> --episodes 5 --workers 6 --modes deterministic --wandb
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base_env_common import TRAINING_RESPAWN_DELAY
from .bot_eval_types import BotEvalConfig, bootstrap_percentile_ci


AGENTS = ("agent_0", "agent_1")
OPPONENTS = ("noop", "random", "self", "prev")
CORRUPTIONS = ("none", "frozen", "shuffled")
BOT_TIERS = ("easy", "medium", "hard")
MODES = ("stochastic", "deterministic")
# Extra game variables the evaluation needs on top of the scenario cfg
EXTRA_GAME_VARIABLES = ("POSITION_X", "POSITION_Y", "ANGLE", "DEAD")
DIAGNOSTIC_USER_VARIABLES = tuple(f"USER{i}" for i in (54, 55, 57, 58, 59, 60))
EVAL_BASE_PORT = 47000
EVAL_PORT_STRIDE = 100
# A killed player must stay dead at least this long for frags to be trustworthy
MIN_RESPAWN_DELAY_SECONDS = 1.0
# Blind policy may keep at most this fraction of the sighted frag difference
# improvement over the opponent before the "uses vision" gate fails
VISION_COLLAPSE_RATIO = 0.5
# Mean policy entropy / ln(num_actions) above which the policy is treated as
# uniform random (FAIL) or nearly uniform (weak)
UNIFORM_ENTROPY_RATIO = 0.97
NEAR_UNIFORM_ENTROPY_RATIO = 0.9


def checkpoint_step(checkpoint: Path) -> int:
    return int(checkpoint.stem.removeprefix("checkpoint_"))


def resolve_checkpoints(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        return [root]
    if (root / "checkpoints").is_dir():
        candidates = [root]
    else:
        candidates = sorted(
            child for child in root.iterdir() if (child / "checkpoints").is_dir()
        )
    if not candidates:
        raise FileNotFoundError(f"no experiment folder with checkpoints under {root}")
    if len(candidates) > 1:
        # save_folder with several experiments: take the most recent one
        candidates.sort(key=lambda folder: folder.stat().st_mtime)
        print(
            f"[ValidityEval] {len(candidates)} experiments under {root}, "
            f"evaluating the newest: {candidates[-1].name}"
        )
    folder = candidates[-1]
    checkpoints = sorted(
        (folder / "checkpoints").glob("checkpoint_*.pt"), key=checkpoint_step
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_*.pt in {folder / 'checkpoints'}")
    return checkpoints


def select_checkpoints(
    checkpoints: Sequence[Path], every: int, last_n: int
) -> list[Path]:
    """Keep every `every`-th checkpoint (from the end) and always the last `last_n`."""
    if last_n <= 0:
        last_n = 1
    selected = set(checkpoints[-last_n:])
    if every > 0:
        for index in range(len(checkpoints) - 1, -1, -every):
            selected.add(checkpoints[index])
    return sorted(selected, key=checkpoint_step)


@dataclass
class PolicyBundle:
    checkpoint: Path
    step: int
    scenario: str
    buttons: tuple
    button_names: tuple[str, ...]
    skip_frames: int
    frame_stack: int
    resolution: str
    run_id: str | None
    adapters: dict[int, Any]


def _close_reloaded_experiment(experiment: Any) -> None:
    for name in ("rollout_env", "test_env"):
        env = getattr(experiment, name, None)
        if env is None:
            continue
        try:
            env.close()
        except RuntimeError as exc:
            if "closed environment" not in str(exc):
                raise


def load_policy_bundle(checkpoint: Path, device: str = "cpu") -> PolicyBundle:
    from .bot_eval_policy import TorchRLPolicyAdapter, load_bot_eval_experiment

    experiment = load_bot_eval_experiment(checkpoint)
    group = next(iter(experiment.group_map))
    config = experiment.task.config
    # The eval env stacks frames itself (frame_stack of the task), so the adapter must not stack again
    adapters = {
        index: TorchRLPolicyAdapter(
            experiment.group_policies[group],
            group_name=group,
            agent_index=index,
            agent_count=len(experiment.group_map[group]),
            frame_stack=1,
            device=device,
        )
        for index in range(len(experiment.group_map[group]))
    }
    _close_reloaded_experiment(experiment)
    buttons = tuple(experiment.task.action_buttons)
    return PolicyBundle(
        checkpoint=checkpoint,
        step=checkpoint_step(checkpoint),
        scenario=str(config["scenario"]),
        buttons=buttons,
        button_names=tuple(button.name for button in buttons),
        skip_frames=int(config.get("skip_frames", 1)),
        frame_stack=int(config.get("frame_stack", 1)),
        resolution=str(config.get("resolution", "160X120")),
        run_id=config.get("run_id"),
        adapters=adapters,
    )


@dataclass
class EpisodeStats:
    checkpoint_step: int
    opponent: str
    corruption: str
    mode: str
    seed: int
    episode: int
    steps: int = 0
    frags: float = 0.0
    deaths: float = 0.0
    opp_frags: float = 0.0
    opp_deaths: float = 0.0
    damage_made: float = 0.0
    damage_taken: float = 0.0
    reward: float | None = 0.0
    opp_reward: float | None = 0.0
    rockets_fired: int | None = 0
    # mean entropy (nats) of the policy's action distribution over the episode
    policy_entropy: float | None = None
    policy_entropy_ratio: float | None = None  # entropy / ln(num_actions)
    button_rates: dict[str, float] = field(default_factory=dict)
    opp_button_rates: dict[str, float] = field(default_factory=dict)
    both_alive_steps: int = 0
    los_fraction: float | None = None
    yaw_error_mean_deg: float | None = None
    yaw_error_median_deg: float | None = None
    aim_within_10deg_fraction: float | None = None
    opp_los_fraction: float | None = None
    opp_yaw_error_mean_deg: float | None = None
    respawn_delays_tics: list[int] = field(default_factory=list)
    spawn_points: list[list[int]] = field(default_factory=list)
    same_side_spawns: int = 0
    start_same_side: bool = False
    duration_seconds: float = 0.0
    valid: bool = True
    error: str | None = None

    @property
    def frag_diff(self) -> float:
        return float(self.frags - self.opp_frags)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frag_diff"] = self.frag_diff
        return payload


class _Opponent:
    """agent_1 controller"""

    def __init__(
        self, kind: str, n_buttons: int, adapter: Any | None, rng: random.Random
    ):
        self.kind = kind
        self.n_actions = 1 << n_buttons
        self.adapter = adapter
        self.rng = rng

    def reset(self, seed: int) -> None:
        if self.adapter is not None:
            self.adapter.reset(seed)

    def act(self, observation: np.ndarray, deterministic: bool) -> int:
        if self.kind == "noop":
            return 0
        if self.kind == "random":
            return self.rng.randrange(self.n_actions)
        assert self.adapter is not None
        return int(self.adapter.act(observation, deterministic=deterministic)[0])


def _bits(action: int, n_buttons: int) -> np.ndarray:
    return np.array(
        [(int(action) >> index) & 1 for index in range(n_buttons)], dtype=np.float64
    )


def _side(info: Mapping[str, Any]) -> int:
    return 1 if float(info["POSITION_Y"]) > 0 else -1


def run_duel_episode(
    env,
    policy: Any,
    opponent: _Opponent,
    *,
    corruption: str,
    deterministic: bool,
    button_names: Sequence[str],
    skip_frames: int,
    rng: random.Random,
    stats: EpisodeStats,
) -> EpisodeStats:
    n_buttons = len(button_names)
    started = time.monotonic()
    obs, infos = env.reset()
    policy.reset(stats.seed)
    opponent.reset(stats.seed + 1)

    i0, i1 = infos[AGENTS[0]], infos[AGENTS[1]]
    base = {
        agent: {
            key: float(infos[agent][key])
            for key in ("FRAGCOUNT", "DEATHCOUNT", "DAMAGECOUNT", "DAMAGE_TAKEN")
        }
        for agent in AGENTS
    }
    stats.start_same_side = _side(i0) == _side(i1)
    stats.spawn_points.append(
        [int(round(i0["POSITION_X"])), int(round(i0["POSITION_Y"]))]
    )

    frozen = obs[AGENTS[0]].copy()
    seen_frames: list[np.ndarray] = [frozen]
    counts0 = np.zeros(n_buttons)
    counts1 = np.zeros(n_buttons)
    los: list[float] = []
    yaw: list[float] = []
    opp_los: list[float] = []
    opp_yaw: list[float] = []
    entropies: list[float] = []
    dead_since: int | None = None
    prev = {agent: dict(infos[agent]) for agent in AGENTS}
    step = 0
    done = False
    while not done:
        if corruption == "none":
            o0 = obs[AGENTS[0]]
        elif corruption == "frozen":
            o0 = frozen
        elif corruption == "shuffled":
            # a real frame of this episode, but uncorrelated with the current state
            o0 = seen_frames[rng.randrange(len(seen_frames))]
        else:
            raise ValueError(corruption)
        a0 = int(policy.act(o0, deterministic=deterministic)[0])
        entropy = policy.last_entropy() if hasattr(policy, "last_entropy") else None
        if entropy is not None:
            entropies.append(entropy)
        a1 = opponent.act(obs[AGENTS[1]], deterministic)
        counts0 += _bits(a0, n_buttons)
        counts1 += _bits(a1, n_buttons)
        obs, rewards, terms, truncs, infos = env.step({AGENTS[0]: a0, AGENTS[1]: a1})
        step += 1
        if corruption == "shuffled" and len(seen_frames) < 4096:
            seen_frames.append(obs[AGENTS[0]].copy())
        stats.reward += float(rewards[AGENTS[0]])
        stats.opp_reward += float(rewards[AGENTS[1]])

        i0, i1 = infos[AGENTS[0]], infos[AGENTS[1]]
        p0 = prev[AGENTS[0]]
        # rockets fired: ammo decrements while alive (respawn frame excluded)
        if not p0["DEAD"] and not i0["DEAD"]:
            drop = float(p0["SELECTED_WEAPON_AMMO"]) - float(i0["SELECTED_WEAPON_AMMO"])
            if drop > 0:
                stats.rockets_fired += int(drop)
        if not p0["DEAD"] and i0["DEAD"]:
            dead_since = step
        if p0["DEAD"] and not i0["DEAD"]:
            if dead_since is not None:
                stats.respawn_delays_tics.append((step - dead_since) * skip_frames)
            dead_since = None
            stats.spawn_points.append(
                [int(round(i0["POSITION_X"])), int(round(i0["POSITION_Y"]))]
            )
            if not i1["DEAD"] and _side(i0) == _side(i1):
                stats.same_side_spawns += 1
        if i0.get("USER54", 0) and i0.get("USER55", 0):
            los.append(float(i0["USER57"]))
            yaw.append(float(i0["USER59"]) / 1000.0)
            opp_los.append(float(i0["USER58"]))
            opp_yaw.append(float(i0["USER60"]) / 1000.0)
        prev = {agent: dict(infos[agent]) for agent in AGENTS}
        done = all(terms.values()) or all(truncs.values())

    i0, i1 = infos[AGENTS[0]], infos[AGENTS[1]]
    stats.steps = step
    stats.frags = float(i0["FRAGCOUNT"]) - base[AGENTS[0]]["FRAGCOUNT"]
    stats.deaths = float(i0["DEATHCOUNT"]) - base[AGENTS[0]]["DEATHCOUNT"]
    stats.damage_made = float(i0["DAMAGECOUNT"]) - base[AGENTS[0]]["DAMAGECOUNT"]
    stats.damage_taken = float(i0["DAMAGE_TAKEN"]) - base[AGENTS[0]]["DAMAGE_TAKEN"]
    stats.opp_frags = float(i1["FRAGCOUNT"]) - base[AGENTS[1]]["FRAGCOUNT"]
    stats.opp_deaths = float(i1["DEATHCOUNT"]) - base[AGENTS[1]]["DEATHCOUNT"]
    stats.button_rates = {
        name: float(c / max(step, 1)) for name, c in zip(button_names, counts0)
    }
    stats.opp_button_rates = {
        name: float(c / max(step, 1)) for name, c in zip(button_names, counts1)
    }
    stats.both_alive_steps = len(los)
    if los:
        stats.los_fraction = float(np.mean(los))
        stats.yaw_error_mean_deg = float(np.mean(yaw))
        stats.yaw_error_median_deg = float(np.median(yaw))
        stats.aim_within_10deg_fraction = float(np.mean(np.asarray(yaw) < 10.0))
        stats.opp_los_fraction = float(np.mean(opp_los))
        stats.opp_yaw_error_mean_deg = float(np.mean(opp_yaw))
    if entropies:
        stats.policy_entropy = float(np.mean(entropies))
        stats.policy_entropy_ratio = stats.policy_entropy / math.log(1 << n_buttons)
    stats.duration_seconds = time.monotonic() - started
    return stats


@dataclass(frozen=True)
class Job:
    index: int
    checkpoint: str
    prev_checkpoint: str | None
    opponent: str  # noop | random | self | prev | bots
    corruption: str
    mode: str
    episodes: int
    seed: int
    scenario_config: str | None
    tiers: tuple[str, ...] = BOT_TIERS


_WORKER_LOCK = None
_WORKER_THREADS = 1
_BUNDLE_CACHE: dict[str, PolicyBundle] = {}


def _worker_init(lock, threads: int) -> None:
    global _WORKER_LOCK, _WORKER_THREADS
    _WORKER_LOCK = lock
    _WORKER_THREADS = int(threads)
    import torch

    torch.set_num_threads(max(1, _WORKER_THREADS))


def _bundle(checkpoint: str) -> PolicyBundle:
    bundle = _BUNDLE_CACHE.get(checkpoint)
    if bundle is None:
        # Experiment.reload_from_file spins up real ViZDoom hosts on the task's
        # default ports, so loads must not overlap between workers.
        if _WORKER_LOCK is not None:
            with _WORKER_LOCK:
                bundle = load_policy_bundle(Path(checkpoint))
        else:
            bundle = load_policy_bundle(Path(checkpoint))
        while len(_BUNDLE_CACHE) >= 2:
            _BUNDLE_CACHE.pop(next(iter(_BUNDLE_CACHE)))
        _BUNDLE_CACHE[checkpoint] = bundle
    return bundle


def _scenario_config_path(scenario: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    import vizdoom as vzd

    packaged = Path(vzd.__file__).with_name("scenarios") / f"{scenario}.cfg"
    if packaged.is_file():
        return packaged
    return (
        Path(__file__).resolve().parents[1] / "scenarios" / f"{scenario}.cfg"
    ).resolve()


def write_eval_config(scenario_config: Path, target_dir: Path) -> Path:
    """Copy of the scenario cfg (+ its wad, ViZDoom resolves the wad relative to
    the cfg directory even for absolute paths) with the extra game variables."""
    lines = []
    for raw in scenario_config.read_text(encoding="utf-8").splitlines():
        key = raw.split("#", 1)[0].split("=", 1)[0].strip().lower().replace("_", "")
        if key == "doomscenariopath":
            wad = raw.split("=", 1)[1].split("#", 1)[0].strip()
            wad_path = Path(wad)
            if not wad_path.is_absolute():
                wad_path = (scenario_config.parent / wad_path).resolve()
            shutil.copy(wad_path, target_dir / wad_path.name)
            raw = f"doom_scenario_path = {wad_path.name}"
        lines.append(raw)
    lines.append("")
    lines.append("# added by duel_validity_eval")
    lines.append(
        "available_game_variables += { "
        + " ".join(
            EXTRA_GAME_VARIABLES + ("SELECTED_WEAPON_AMMO",) + DIAGNOSTIC_USER_VARIABLES
        )
        + " }"
    )
    target = target_dir / f"{scenario_config.stem}_validity_eval.cfg"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _run_bot_job(job: Job, bundle: PolicyBundle) -> list[EpisodeStats]:
    from .bot_eval_duel import BotDuelEvaluator

    config = BotEvalConfig(
        scenario=bundle.scenario,
        scenario_config=job.scenario_config,
        resolution=bundle.resolution,
        skip_frames=bundle.skip_frames,
        respawn_delay=TRAINING_RESPAWN_DELAY,
        seed=job.seed,
    )
    evaluator = BotDuelEvaluator(config)
    # bot eval feeds raw frames, so this adapter has to stack itself
    from .bot_eval_policy import TorchRLPolicyAdapter

    adapter = bundle.adapters[0]
    policy = TorchRLPolicyAdapter(
        adapter.policy,
        group_name=adapter.group_name,
        agent_index=0,
        agent_count=adapter.agent_count,
        frame_stack=bundle.frame_stack,
        device=adapter.device,
    )
    results: list[EpisodeStats] = []
    for tier in job.tiers:
        for episode in range(job.episodes):
            seed = job.seed + 1000 * BOT_TIERS.index(tier) + episode
            stats = EpisodeStats(
                checkpoint_step=bundle.step,
                opponent=f"bot_{tier}",
                corruption="none",
                mode=job.mode,
                seed=seed,
                episode=episode,
                # not measured by BotDuelEvaluator
                reward=None,
                opp_reward=None,
                rockets_fired=None,
            )
            run = evaluator.run_episode(
                seed=seed,
                tier=tier,
                policy=policy,
                deterministic=(job.mode == "deterministic"),
            )
            result = run.result
            stats.valid = result.valid
            stats.error = result.invalid_reason
            stats.duration_seconds = float(result.duration_seconds or 0.0)
            if result.valid:
                stats.steps = int(result.policy_steps or 0)
                stats.frags = float(result.learner_frags or 0)
                stats.opp_frags = float(result.bot_frags or 0)
                stats.deaths = float(result.learner_deaths or 0)
                stats.damage_made = float(result.learner_damage_made or 0.0)
                stats.damage_taken = float(result.learner_damage_taken or 0.0)
            results.append(stats)
    return results


def run_job(job: Job) -> tuple[Job, list[EpisodeStats], str | None]:
    """Executed in a worker process."""
    try:
        bundle = _bundle(job.checkpoint)
        if job.opponent == "bots":
            return job, _run_bot_job(job, bundle), None

        from . import make

        rng = random.Random(job.seed)
        opponent_adapter = None
        if job.opponent == "self":
            opponent_adapter = bundle.adapters[1]
        elif job.opponent == "prev":
            if job.prev_checkpoint is None:
                return job, [], None
            opponent_adapter = _bundle(job.prev_checkpoint).adapters[1]
        opponent = _Opponent(job.opponent, len(bundle.buttons), opponent_adapter, rng)

        tmp_dir = Path(tempfile.mkdtemp(prefix="duel_validity_eval_"))
        cfg = write_eval_config(
            _scenario_config_path(bundle.scenario, job.scenario_config), tmp_dir
        )
        port = EVAL_BASE_PORT + (job.index * EVAL_PORT_STRIDE) % (
            65000 - EVAL_BASE_PORT
        )
        env = make(
            scenario=bundle.scenario,
            config_file=str(cfg),
            num_agents=2,
            resolution=bundle.resolution,
            skip_frames=bundle.skip_frames,
            frame_stack=bundle.frame_stack,
            seed=job.seed,
            enable_video=False,
            port=port,
            available_buttons=bundle.buttons,
        )
        results: list[EpisodeStats] = []
        try:
            for episode in range(job.episodes):
                stats = EpisodeStats(
                    checkpoint_step=bundle.step,
                    opponent=job.opponent,
                    corruption=job.corruption,
                    mode=job.mode,
                    seed=job.seed + episode,
                    episode=episode,
                )
                try:
                    run_duel_episode(
                        env,
                        bundle.adapters[0],
                        opponent,
                        corruption=job.corruption,
                        deterministic=(job.mode == "deterministic"),
                        button_names=bundle.button_names,
                        skip_frames=bundle.skip_frames,
                        rng=rng,
                        stats=stats,
                    )
                except Exception as exc:  # keep the other episodes
                    stats.valid = False
                    stats.error = f"{type(exc).__name__}: {exc}"[:300]
                results.append(stats)
        finally:
            try:
                env.close()
            except Exception:
                pass
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return job, results, None
    except Exception:
        return job, [], traceback.format_exc()


def _mean(values: Sequence[float]) -> float | None:
    values = [
        float(v)
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return float(np.mean(values)) if values else None


def _sem(values: Sequence[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def summarize_condition(episodes: Sequence[EpisodeStats], seed: int) -> dict[str, Any]:
    valid = [e for e in episodes if e.valid]
    n = len(valid)
    if n == 0:
        return {"episodes": 0, "invalid": len(episodes)}
    diffs = [e.frag_diff for e in valid]
    ci_low, ci_high = (
        bootstrap_percentile_ci(diffs, seed=seed) if n > 1 else (diffs[0], diffs[0])
    )
    button_names = list(valid[0].button_rates)
    summary: dict[str, Any] = {
        "episodes": n,
        "invalid": len(episodes) - n,
        "frags": _mean([e.frags for e in valid]),
        "deaths": _mean([e.deaths for e in valid]),
        "opp_frags": _mean([e.opp_frags for e in valid]),
        "frag_diff": _mean(diffs),
        "frag_diff_sem": _sem(diffs),
        "frag_diff_ci_low": float(ci_low),
        "frag_diff_ci_high": float(ci_high),
        "win_rate": float(np.mean([d > 0 for d in diffs])),
        "loss_rate": float(np.mean([d < 0 for d in diffs])),
        "damage_made": _mean([e.damage_made for e in valid]),
        "damage_taken": _mean([e.damage_taken for e in valid]),
        "reward": _mean([e.reward for e in valid]),
        "rockets_fired": _mean([e.rockets_fired for e in valid]),
        "policy_entropy": _mean([e.policy_entropy for e in valid]),
        "policy_entropy_ratio": _mean([e.policy_entropy_ratio for e in valid]),
        "los_fraction": _mean([e.los_fraction for e in valid]),
        "yaw_error_mean_deg": _mean([e.yaw_error_mean_deg for e in valid]),
        "yaw_error_median_deg": _mean([e.yaw_error_median_deg for e in valid]),
        "aim_within_10deg_fraction": _mean(
            [e.aim_within_10deg_fraction for e in valid]
        ),
        "opp_los_fraction": _mean([e.opp_los_fraction for e in valid]),
        "opp_yaw_error_mean_deg": _mean([e.opp_yaw_error_mean_deg for e in valid]),
        "button_rates": {
            name: _mean([e.button_rates.get(name) for e in valid])
            for name in button_names
        },
        "respawn_delay_tics_min": (
            min(d for e in valid for d in e.respawn_delays_tics)
            if any(e.respawn_delays_tics for e in valid)
            else None
        ),
        "respawn_delay_tics_mean": _mean(
            [d for e in valid for d in e.respawn_delays_tics]
        ),
        "respawns": int(sum(len(e.respawn_delays_tics) for e in valid)),
        "spawns": int(sum(len(e.spawn_points) for e in valid)),
        "distinct_spawn_points": len({tuple(p) for e in valid for p in e.spawn_points}),
        "same_side_spawns": int(sum(e.same_side_spawns for e in valid)),
        "start_same_side": int(sum(e.start_same_side for e in valid)),
        "seconds_per_episode": _mean([e.duration_seconds for e in valid]),
    }
    return summary


def _key(opponent: str, corruption: str) -> str:
    return opponent if corruption == "none" else f"{opponent}[{corruption}]"


def aggregate(
    episodes: Sequence[EpisodeStats], seed: int
) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    """-> {step: {mode: {condition: summary}}}"""
    grouped: dict[tuple[int, str, str], list[EpisodeStats]] = {}
    for e in episodes:
        grouped.setdefault(
            (e.checkpoint_step, e.mode, _key(e.opponent, e.corruption)), []
        ).append(e)
    table: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for (step, mode, condition), items in grouped.items():
        table.setdefault(step, {}).setdefault(mode, {})[
            condition
        ] = summarize_condition(items, seed=seed + step % 10_000)
    return table


def _status(passed: bool | None, weak: bool = False) -> str:
    if passed is None:
        return "n/a"
    if passed:
        return "pass"
    return "weak" if weak else "FAIL"


def evaluate_gates(
    table: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    mode: str,
    configured_respawn_delay: float,
) -> dict[str, dict[str, Any]]:
    steps = sorted(table)
    final = steps[-1]
    conditions = table[final].get(mode, {})
    gates: dict[str, dict[str, Any]] = {}

    def get(condition: str, key: str):
        summary = conditions.get(condition)
        return None if summary is None else summary.get(key)

    noop = get("noop", "frag_diff")
    gates["beats_noop"] = {
        "status": _status(None if noop is None else noop > 0),
        "frag_diff_vs_noop": noop,
        "note": "a policy that aims at all must frag a standing target",
    }

    rnd = get("random", "frag_diff")
    rnd_low = get("random", "frag_diff_ci_low")
    gates["beats_random"] = {
        "status": _status(
            None if rnd is None else (rnd_low is not None and rnd_low > 0),
            weak=(rnd is not None and rnd > 0),
        ),
        "frag_diff_vs_random": rnd,
        "ci_low": rnd_low,
        "ci_high": get("random", "frag_diff_ci_high"),
        "win_rate": get("random", "win_rate"),
        "note": "pass = bootstrap 95% CI of the frag difference above 0; weak = mean above 0 only",
    }

    for corruption in ("frozen", "shuffled"):
        blind_key = _key("random", corruption)
        blind = get(blind_key, "frag_diff")
        blind_frags = get(blind_key, "frags")
        sighted_frags = get("random", "frags")
        if rnd is None or blind is None:
            status = "n/a"
            retained = None
        else:
            retained = (
                None
                if sighted_frags is None or sighted_frags <= 0
                else max(0.0, float(blind_frags)) / float(sighted_frags)
            )
            collapsed = (
                blind <= rnd - max(1.0, (1.0 - VISION_COLLAPSE_RATIO) * max(rnd, 0.0))
            ) or (retained is not None and retained <= VISION_COLLAPSE_RATIO)
            status = _status(collapsed and rnd > 0, weak=(rnd > 0 and blind < rnd))
        gates[f"uses_vision_{corruption}"] = {
            "status": status,
            "sighted_frag_diff_vs_random": rnd,
            "blind_frag_diff_vs_random": blind,
            "blind_frags_retained_fraction": retained,
            "blind_los_fraction": get(blind_key, "los_fraction"),
            "sighted_los_fraction": get("random", "los_fraction"),
            "note": (
                f"pass = blind policy keeps <= {VISION_COLLAPSE_RATIO:.0%} of the sighted frags "
                "(or loses >= 1 frag-diff / half the margin) against the same random opponent"
            ),
        }

    prev = get("prev", "frag_diff")
    gates["improves_over_previous_checkpoint"] = {
        "status": _status(None if prev is None else prev >= 0, weak=False),
        "frag_diff_vs_previous": prev,
        "win_rate": get("prev", "win_rate"),
        "note": "informational: later checkpoint should not lose to the earlier one",
    }

    bot_diffs = {
        tier: [
            table[s].get(mode, {}).get(f"bot_{tier}", {}).get("frag_diff")
            for s in steps
        ]
        for tier in BOT_TIERS
    }
    bot_changes = any(
        len({round(v, 3) for v in diffs if v is not None}) > 1
        for diffs in bot_diffs.values()
    )
    gates["bot_eval_varies_across_checkpoints"] = {
        "status": _status(None if len(steps) < 2 else bot_changes),
        "final_frag_diff": {tier: diffs[-1] for tier, diffs in bot_diffs.items()},
        "note": "a bot-eval table identical for every checkpoint means the argmax never depended on training",
    }

    delays = [
        summary.get("respawn_delay_tics_min")
        for s in steps
        for m in table[s].values()
        for summary in m.values()
        if summary.get("respawn_delay_tics_min") is not None
    ]
    measured_min = min(delays) if delays else None
    ok = configured_respawn_delay >= MIN_RESPAWN_DELAY_SECONDS and (
        measured_min is None or measured_min >= MIN_RESPAWN_DELAY_SECONDS * 35 - 4
    )
    gates["respawn_delay"] = {
        "status": _status(ok),
        "configured_seconds": configured_respawn_delay,
        "measured_min_tics": measured_min,
        "note": f"TRAINING_RESPAWN_DELAY >= {MIN_RESPAWN_DELAY_SECONDS} s and measured dead time consistent with it",
    }

    spawns = sum(
        summary.get("spawns", 0)
        for m in table[final].values()
        for summary in m.values()
        if "spawns" in summary
    )
    distinct = max(
        (
            summary.get("distinct_spawn_points", 0)
            for m in table[final].values()
            for summary in m.values()
        ),
        default=0,
    )
    same_side = sum(
        summary.get("same_side_spawns", 0) + summary.get("start_same_side", 0)
        for m in table[final].values()
        for summary in m.values()
    )
    gates["randomised_opposite_spawns"] = {
        "status": _status(spawns > 0 and same_side == 0 and distinct >= min(8, spawns)),
        "spawns_observed": spawns,
        "max_distinct_spawn_points_in_one_condition": distinct,
        "same_side_spawns": same_side,
        "note": "every spawn on the player's own side and spawn points not repeating",
    }

    ratio = get("self", "policy_entropy_ratio")
    if ratio is None:
        ratio = get("random", "policy_entropy_ratio")
    gates["policy_not_uniform"] = {
        "status": _status(
            None if ratio is None else ratio <= NEAR_UNIFORM_ENTROPY_RATIO,
            weak=(ratio is not None and ratio <= UNIFORM_ENTROPY_RATIO),
        ),
        "entropy_over_max_entropy": ratio,
        "button_rates": get("self", "button_rates") or get("random", "button_rates"),
        "note": (
            "mean action-distribution entropy / ln(num_actions). "
            f"Above {UNIFORM_ENTROPY_RATIO} = uniform random (FAIL), "
            f"above {NEAR_UNIFORM_ENTROPY_RATIO} = weak. "
            "Balanced button marginals alone are not a failure"
        ),
    }
    return gates


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(
    table: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    gates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    meta: Mapping[str, Any],
) -> str:
    lines = [
        f"# Duel validity evaluation: {meta.get('run_id') or meta.get('experiment')}",
        "",
    ]
    lines.append(
        f"checkpoints: {', '.join(str(s) for s in sorted(table))} | episodes/condition: {meta['episodes']} "
        f"| modes: {', '.join(meta['modes'])} | scenario: {meta.get('scenario')} | "
        f"respawn delay: {meta['respawn_delay_seconds']} s"
    )
    lines.append("")
    for mode, mode_gates in gates.items():
        lines.append(f"## Gates ({mode})")
        lines.append("")
        lines.append("| gate | status | evidence |")
        lines.append("|---|---|---|")
        for name, gate in mode_gates.items():
            evidence = ", ".join(
                f"{k}={_fmt(v)}"
                for k, v in gate.items()
                if k not in ("status", "note") and not isinstance(v, dict)
            )
            lines.append(f"| {name} | **{gate['status']}** | {evidence} |")
        lines.append("")
    for step in sorted(table):
        for mode, conditions in table[step].items():
            lines.append(f"## checkpoint {step} ({mode})")
            lines.append("")
            lines.append(
                "| opponent | n | frags | deaths | frag diff (±sem) | 95% CI | win | dmg made | dmg taken | "
                "reward | fired | LOS | yaw err° | aim<10° | H/Hmax | button rates |"
            )
            lines.append("|---|" + "---|" * 15)
            for condition in sorted(conditions, key=_condition_order):
                s = conditions[condition]
                if not s.get("episodes"):
                    lines.append(
                        f"| {condition} | 0 | invalid: {s.get('invalid')} |" + " |" * 13
                    )
                    continue
                rates = " ".join(
                    f"{k.lower()}={_fmt(v)}"
                    for k, v in (s.get("button_rates") or {}).items()
                )
                lines.append(
                    f"| {condition} | {s['episodes']} | {_fmt(s['frags'], 1)} | {_fmt(s['deaths'], 1)} | "
                    f"{_fmt(s['frag_diff'])} ± {_fmt(s['frag_diff_sem'])} | "
                    f"[{_fmt(s['frag_diff_ci_low'], 1)}, {_fmt(s['frag_diff_ci_high'], 1)}] | "
                    f"{_fmt(s['win_rate'])} | {_fmt(s['damage_made'], 0)} | {_fmt(s['damage_taken'], 0)} | "
                    f"{_fmt(s['reward'])} | {_fmt(s['rockets_fired'], 0)} | {_fmt(s['los_fraction'])} | "
                    f"{_fmt(s['yaw_error_mean_deg'], 1)} | {_fmt(s['aim_within_10deg_fraction'])} | "
                    f"{_fmt(s['policy_entropy_ratio'])} | {rates} |"
                )
            lines.append("")
    return "\n".join(lines)


def _condition_order(condition: str) -> tuple[int, str]:
    order = ["noop", "random", "prev", "self", "bot_easy", "bot_medium", "bot_hard"]
    base = condition.split("[", 1)[0]
    return (order.index(base) if base in order else len(order), condition)


def build_jobs(
    checkpoints: Sequence[Path],
    *,
    episodes: int,
    modes: Sequence[str],
    seed: int,
    scenario_config: str | None,
    blind_every_checkpoint: bool,
    bots: bool,
    opponents: Sequence[str],
    corruptions: Sequence[str],
) -> list[Job]:
    jobs: list[Job] = []
    last = checkpoints[-1]
    for index, checkpoint in enumerate(checkpoints):
        prev = str(checkpoints[index - 1]) if index > 0 else None
        for mode in modes:
            for opponent in opponents:
                if opponent == "prev" and prev is None:
                    continue
                jobs.append(
                    Job(
                        index=len(jobs),
                        checkpoint=str(checkpoint),
                        prev_checkpoint=prev,
                        opponent=opponent,
                        corruption="none",
                        mode=mode,
                        episodes=episodes,
                        seed=seed + 100 * len(jobs),
                        scenario_config=scenario_config,
                    )
                )
            if (checkpoint == last or blind_every_checkpoint) and "random" in opponents:
                for corruption in corruptions:
                    if corruption == "none":
                        continue
                    jobs.append(
                        Job(
                            index=len(jobs),
                            checkpoint=str(checkpoint),
                            prev_checkpoint=prev,
                            opponent="random",
                            corruption=corruption,
                            mode=mode,
                            episodes=episodes,
                            seed=seed + 100 * len(jobs),
                            scenario_config=scenario_config,
                        )
                    )
            if bots:
                jobs.append(
                    Job(
                        index=len(jobs),
                        checkpoint=str(checkpoint),
                        prev_checkpoint=prev,
                        opponent="bots",
                        corruption="none",
                        mode=mode,
                        episodes=episodes,
                        seed=seed + 100 * len(jobs),
                        scenario_config=scenario_config,
                    )
                )
    return jobs


def run_jobs(
    jobs: Sequence[Job], workers: int, errors: list[str] | None = None
) -> list[EpisodeStats]:
    """Run all jobs; tracebacks of jobs that crashed are appended to `errors`."""
    episodes: list[EpisodeStats] = []
    started = time.monotonic()
    if workers <= 1:
        _worker_init(None, max(1, os.cpu_count() or 1))
        for job in jobs:
            _, results, error = run_job(job)
            _report_job(job, results, error, started)
            if error and errors is not None:
                errors.append(error)
            episodes.extend(results)
        return episodes

    context = get_context("spawn")
    lock = context.Lock()
    threads = max(1, (os.cpu_count() or workers) // workers)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(lock, threads),
    ) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            job, results, error = future.result()
            _report_job(job, results, error, started)
            if error and errors is not None:
                errors.append(error)
            episodes.extend(results)
    return episodes


def _report_job(
    job: Job, results: Sequence[EpisodeStats], error: str | None, started: float
) -> None:
    elapsed = time.monotonic() - started
    label = f"ckpt={checkpoint_step(Path(job.checkpoint))} {job.mode} {_key(job.opponent, job.corruption)}"
    if error:
        print(f"[ValidityEval {elapsed:7.0f}s] {label}: FAILED\n{error}", flush=True)
        return
    valid = [r for r in results if r.valid]
    if not valid:
        print(
            f"[ValidityEval {elapsed:7.0f}s] {label}: no valid episodes ({len(results)} attempted)",
            flush=True,
        )
        return
    by_opp = Counter(r.opponent for r in valid)
    detail = " ".join(
        f"{opp}: fd={np.mean([r.frag_diff for r in valid if r.opponent == opp]):+.2f}"
        for opp in by_opp
    )
    print(
        f"[ValidityEval {elapsed:7.0f}s] {label}: {len(valid)} episodes, {detail}",
        flush=True,
    )


def _log_wandb(
    project: str,
    meta: Mapping[str, Any],
    table: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    gates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_dir: Path,
    entity: str | None,
) -> None:
    import wandb

    run_name = f"{meta.get('run_id') or meta.get('experiment')}-validity"
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=meta.get("run_id"),
        job_type="validity_eval",
        config=dict(meta),
        reinit=True,
    )
    try:
        columns = [
            "checkpoint_step",
            "mode",
            "opponent",
            "episodes",
            "frags",
            "deaths",
            "frag_diff",
            "frag_diff_ci_low",
            "frag_diff_ci_high",
            "win_rate",
            "damage_made",
            "damage_taken",
            "reward",
            "rockets_fired",
            "los_fraction",
            "yaw_error_mean_deg",
            "aim_within_10deg_fraction",
            "policy_entropy_ratio",
        ]
        rows = []
        for step in sorted(table):
            for mode, conditions in table[step].items():
                for condition, s in conditions.items():
                    if not s.get("episodes"):
                        continue
                    rows.append(
                        [step, mode, condition] + [s.get(c) for c in columns[3:]]
                    )
                    for key in columns[3:]:
                        if s.get(key) is not None:
                            wandb.log(
                                {
                                    f"validity/{mode}/{condition}/{key}": s[key],
                                    "checkpoint_step": step,
                                }
                            )
                    for name, rate in (s.get("button_rates") or {}).items():
                        wandb.log(
                            {
                                f"validity/{mode}/{condition}/{name.lower()}_rate": rate,
                                "checkpoint_step": step,
                            }
                        )
        wandb.log({"validity/crossplay": wandb.Table(columns=columns, data=rows)})
        for mode, mode_gates in gates.items():
            for name, gate in mode_gates.items():
                run.summary[f"gate/{mode}/{name}"] = gate["status"]
        artifact = wandb.Artifact(f"{run_name}-report", type="evaluation")
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
    finally:
        run.finish()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path", help="checkpoint .pt, experiment folder or BenchMARL save_folder"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="episodes per (checkpoint, opponent, mode)",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2))
    )
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=["stochastic", "deterministic"]
    )
    parser.add_argument(
        "--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS)
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS[1:],
        default=list(CORRUPTIONS[1:]),
    )
    parser.add_argument(
        "--no-bots",
        dest="bots",
        action="store_false",
        help="skip the easy/medium/hard bot duels",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="evaluate every k-th checkpoint counted from the last (0 = only --last)",
    )
    parser.add_argument(
        "--last", type=int, default=1, help="always evaluate the last N checkpoints"
    )
    parser.add_argument(
        "--blind_every_checkpoint",
        action="store_true",
        help="blind ablation for all checkpoints, not only the last",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--scenario_config", default=None, help="override the scenario .cfg"
    )
    parser.add_argument(
        "--output_dir", default=None, help="default: <experiment_folder>/validity_eval"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="log the tables to a <run_id>-validity W&B run",
    )
    parser.add_argument("--wandb_project", default="benchmarl-vizdoom")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args(argv)

    checkpoints = resolve_checkpoints(args.path)
    checkpoints = select_checkpoints(checkpoints, args.every, args.last)
    experiment_folder = checkpoints[-1].parent.parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else experiment_folder / "validity_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[ValidityEval] checkpoints: {[checkpoint_step(c) for c in checkpoints]}",
        flush=True,
    )

    jobs = build_jobs(
        checkpoints,
        episodes=args.episodes,
        modes=args.modes,
        seed=args.seed,
        scenario_config=args.scenario_config,
        blind_every_checkpoint=args.blind_every_checkpoint,
        bots=args.bots,
        opponents=args.opponents,
        corruptions=args.corruptions,
    )
    print(f"[ValidityEval] {len(jobs)} jobs, {args.workers} workers", flush=True)
    started = time.monotonic()
    job_errors: list[str] = []
    episodes = run_jobs(jobs, args.workers, job_errors)
    if not any(e.valid for e in episodes):
        print(
            f"[ValidityEval] no valid episodes, nothing to report "
            f"({len(job_errors)}/{len(jobs)} jobs crashed)",
            file=sys.stderr,
        )
        if job_errors:
            # the per-job tracebacks went to stdout as they happened; repeat the
            # last line of the first one here so the failure is visible in stderr
            first = job_errors[0].strip().splitlines()[-1]
            print(f"[ValidityEval] first job error: {first}", file=sys.stderr)
            if "Can't get attribute" in first and "__main__" in first:
                print(
                    "[ValidityEval] config.pkl references classes of the training "
                    "script as __main__.<Name>; run with PYTHONPATH containing the "
                    "repo root so examples.python.pettingzoo_learning can be "
                    "imported (see bot_eval_policy.register_training_script_classes)",
                    file=sys.stderr,
                )
        return 1
    if job_errors:
        print(
            f"[ValidityEval] warning: {len(job_errors)}/{len(jobs)} jobs crashed, "
            "report is incomplete",
            file=sys.stderr,
        )

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for e in episodes:
            handle.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")

    table = aggregate(episodes, seed=args.seed)
    gates = {
        mode: evaluate_gates(table, mode, TRAINING_RESPAWN_DELAY) for mode in args.modes
    }

    scenario = None
    run_id = None
    config_pkl = experiment_folder / "config.pkl"
    try:
        import pickle

        # BenchMARL pickles the task object itself (VizdoomTask.config holds the CLI task config)
        with config_pkl.open("rb") as handle:
            config = pickle.load(handle)
        task = config.get("task") if isinstance(config, dict) else config
        task_config = getattr(task, "config", None)
        if isinstance(task_config, dict):
            scenario = task_config.get("scenario")
            run_id = task_config.get("run_id")
    except Exception:
        pass
    meta = {
        "experiment": experiment_folder.name,
        "run_id": run_id,
        "scenario": scenario,
        "checkpoints": [checkpoint_step(c) for c in checkpoints],
        "episodes": args.episodes,
        "modes": list(args.modes),
        "seed": args.seed,
        "respawn_delay_seconds": TRAINING_RESPAWN_DELAY,
        "elapsed_seconds": time.monotonic() - started,
    }
    summary = {
        "meta": meta,
        "gates": gates,
        "table": {str(k): v for k, v in table.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = render_report(table, gates, meta)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[ValidityEval] wrote {output_dir}", flush=True)

    if args.wandb:
        try:
            _log_wandb(
                args.wandb_project, meta, table, gates, output_dir, args.wandb_entity
            )
        except Exception as exc:
            print(
                f"[ValidityEval] W&B logging failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    failed = [
        f"{mode}:{name}"
        for mode, mode_gates in gates.items()
        for name, g in mode_gates.items()
        if g["status"] == "FAIL"
    ]
    print(f"[ValidityEval] failed gates: {failed or 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
