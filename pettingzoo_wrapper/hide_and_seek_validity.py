"""
* shooter_vs_noop_hider
* shooter_vs_random_hider against random_shooter_vs_random_hider.
* shooter[frozen|shuffled]_vs_random_hider (blind shooter).
* hider_vs_turret{0,4,8} - learned hider against the ACS turret
* hider[frozen|shuffled]_vs_turret4 (blind hider).
* self (both learned), shooter_vs_prev_hider, prev_shooter_vs_hider.
"""

from __future__ import annotations

import math
import random
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hide_and_seek_metrics import wilson_interval


SHOOTER, HIDER = "agent_0", "agent_1"
ROLES = ("shooter", "hider")
TURRET_SIGMAS_DEG = (0, 4, 8)
TURRET_ABLATION_SIGMA_DEG = 4
TURRET_ENABLE_SCRIPT = 10
TURRET_DISABLE_SCRIPT = 11
OUTCOME_NAMES = {
    0: "ongoing",
    1: "shooter_win",
    2: "hider_win",
    3: "hider_escape",
    4: "draw",
    5: "hider_suicide",
}
# hider reaction window after a launch (rocket flight to the strip is ~0.9-1.4 s)
REACTION_WINDOW_SECONDS = (0.3, 1.3)
SHOOTER_VS_NOOP_MIN_WIN_RATE = 0.9
SHOOTER_VS_NOOP_WEAK_WIN_RATE = 0.5
# a blind role may keep at most this share of the sighted improvement over the
# random controller of the same role
VISION_COLLAPSE_RATIO = 0.5


@dataclass(frozen=True)
class HSJob:
    index: int
    checkpoint: str
    prev_checkpoint: str | None
    condition: str
    shooter: str  # policy | prev | noop | random | turret:<sigma_millideg>
    hider: str  # policy | prev | noop | random
    corruption: str  # none | frozen | shuffled (applies to the learned focus role)
    focus: str  # shooter | hider | both | none  (which learned role is judged)
    mode: str
    episodes: int
    seed: int
    scenario_config: str | None


@dataclass
class HSEpisodeStats:
    checkpoint_step: int
    condition: str
    shooter: str
    hider: str
    corruption: str
    focus: str
    mode: str
    seed: int
    episode: int
    steps: int = 0
    round_seconds: float = 0.0
    outcome: str = "ongoing"
    escape_reason: str | None = None  # budget | timeout
    shooter_won: bool = False
    hider_survived: bool = False
    time_to_kill_seconds: float | None = None
    rockets_fired: int = 0
    shooter_damage: float = 0.0
    hider_damage_taken: float = 0.0
    shooter_reward: float = 0.0
    hider_reward: float = 0.0
    shooter_yaw_error_mean_deg: float | None = None
    shooter_aim_within_10deg_fraction: float | None = None
    hider_yaw_error_mean_deg: float | None = None
    hider_speed_in_window: float | None = None
    hider_speed_outside_window: float | None = None
    hider_reaction_speed_ratio: float | None = None
    hider_direction_changes_in_window: float | None = None
    hider_direction_changes_outside_window: float | None = None
    shooter_entropy_ratio: float | None = None
    hider_entropy_ratio: float | None = None
    shooter_button_rates: dict[str, float] = field(default_factory=dict)
    hider_button_rates: dict[str, float] = field(default_factory=dict)
    start_same_side: bool = False
    shooter_spawn: list[int] = field(default_factory=list)
    hider_spawn: list[int] = field(default_factory=list)
    hider_start_distance: float | None = None
    spawn_retries: int = 0
    duration_seconds: float = 0.0
    valid: bool = True
    error: str | None = None

    # duel_validity_eval's job reporter prints a "frag diff": +-1 round outcome
    # from the shooter's point of view
    @property
    def frag_diff(self) -> float:
        return 1.0 if self.shooter_won else -1.0

    @property
    def opponent(self) -> str:
        return self.condition

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frag_diff"] = self.frag_diff
        return payload


class _Controller:
    def __init__(
        self, kind: str, n_buttons: int, adapter: Any | None, rng: random.Random
    ):
        self.kind = kind
        self.n_actions = 1 << n_buttons
        self.adapter = adapter
        self.rng = rng

    @property
    def learned(self) -> bool:
        return self.adapter is not None

    def reset(self, seed: int) -> None:
        if self.adapter is not None:
            self.adapter.reset(seed)

    def act(self, observation: np.ndarray, deterministic: bool) -> int:
        if self.kind == "random":
            return self.rng.randrange(self.n_actions)
        if self.adapter is None:
            # noop and turret rows: the shooter's body must not act
            return 0
        return int(self.adapter.act(observation, deterministic=deterministic)[0])

    def entropy(self) -> float | None:
        if self.adapter is None or not hasattr(self.adapter, "last_entropy"):
            return None
        return self.adapter.last_entropy()


def turret_sigma_millideg(spec: str) -> int | None:
    if not spec.startswith("turret:"):
        return None
    return int(spec.split(":", 1)[1])


def _bits(action: int, n_buttons: int) -> np.ndarray:
    return np.array(
        [(int(action) >> index) & 1 for index in range(n_buttons)], dtype=np.float64
    )


def _base_env(env):
    node = env
    while not hasattr(node, "send_game_command"):
        node = getattr(node, "env", None)
        if node is None:
            raise AttributeError(
                "no env in the wrapper chain supports send_game_command"
            )
    return node


def reaction_metrics(
    positions: Sequence[tuple[float, float]],
    launch_steps: Sequence[int],
    skip_frames: int,
    ticrate: int = 35,
) -> dict[str, float | None]:
    """Hider lateral speed / direction changes inside vs. outside launch windows.

    positions[k] is the hider position after step k (index 0 = spawn),
    launch_steps the step indices at which a rocket left the launcher.
    """
    if len(positions) < 3:
        return {
            "speed_in": None,
            "speed_out": None,
            "ratio": None,
            "turns_in": None,
            "turns_out": None,
        }
    pos = np.asarray(positions, dtype=np.float64)
    deltas = np.diff(pos, axis=0)  # movement during step k+1
    speed = np.linalg.norm(deltas, axis=1) / skip_frames  # units per tic
    # lateral direction = sign of the x component; a change counts as a turn
    lateral = np.sign(deltas[:, 0])
    turns = np.zeros(len(speed), dtype=bool)
    turns[1:] = (lateral[1:] != lateral[:-1]) & (lateral[1:] != 0) & (lateral[:-1] != 0)
    steps_per_second = ticrate / skip_frames
    lo = int(math.ceil(REACTION_WINDOW_SECONDS[0] * steps_per_second))
    hi = int(math.floor(REACTION_WINDOW_SECONDS[1] * steps_per_second))
    in_window = np.zeros(len(speed), dtype=bool)
    for launch in launch_steps:
        # movement index k corresponds to step k+1
        start = max(0, launch + lo - 1)
        stop = min(len(speed), launch + hi)
        if stop > start:
            in_window[start:stop] = True
    out_window = ~in_window
    speed_in = float(speed[in_window].mean()) if in_window.any() else None
    speed_out = float(speed[out_window].mean()) if out_window.any() else None
    ratio = (
        None
        if speed_in is None or speed_out is None or speed_out <= 1e-6
        else speed_in / speed_out
    )
    turns_in = float(turns[in_window].mean()) if in_window.any() else None
    turns_out = float(turns[out_window].mean()) if out_window.any() else None
    return {
        "speed_in": speed_in,
        "speed_out": speed_out,
        "ratio": ratio,
        "turns_in": turns_in,
        "turns_out": turns_out,
    }


def run_round(
    env,
    shooter: _Controller,
    hider: _Controller,
    *,
    corruption: str,
    focus: str,
    deterministic: bool,
    button_names: Sequence[str],
    skip_frames: int,
    rng: random.Random,
    stats: HSEpisodeStats,
) -> HSEpisodeStats:
    n_buttons = len(button_names)
    started = time.monotonic()
    obs, infos = env.reset()
    shooter.reset(stats.seed)
    hider.reset(stats.seed + 1)

    i0, i1 = infos[SHOOTER], infos[HIDER]
    stats.spawn_retries = int(i0.get("hide_and_seek_spawn_retries", 0) or 0)
    stats.start_same_side = (float(i0["POSITION_Y"]) > 0) == (
        float(i1["POSITION_Y"]) > 0
    )
    stats.shooter_spawn = [int(round(i0["POSITION_X"])), int(round(i0["POSITION_Y"]))]
    stats.hider_spawn = [int(round(i1["POSITION_X"])), int(round(i1["POSITION_Y"]))]
    stats.hider_start_distance = float(
        math.hypot(
            i0["POSITION_X"] - i1["POSITION_X"], i0["POSITION_Y"] - i1["POSITION_Y"]
        )
    )
    base_damage = float(i0.get("DAMAGECOUNT", 0.0))
    base_taken = float(i1.get("DAMAGE_TAKEN", 0.0))

    frozen = {SHOOTER: obs[SHOOTER].copy(), HIDER: obs[HIDER].copy()}
    seen: dict[str, list[np.ndarray]] = {
        SHOOTER: [frozen[SHOOTER]],
        HIDER: [frozen[HIDER]],
    }
    blind_agents = (
        {
            "shooter": (SHOOTER,),
            "hider": (HIDER,),
            "both": (SHOOTER, HIDER),
            "none": (),
        }[focus]
        if corruption != "none"
        else ()
    )

    def observation(agent: str):
        if agent not in blind_agents:
            return obs[agent]
        if corruption == "frozen":
            return frozen[agent]
        if corruption == "shuffled":
            frames = seen[agent]
            return frames[rng.randrange(len(frames))]
        raise ValueError(corruption)

    counts = {SHOOTER: np.zeros(n_buttons), HIDER: np.zeros(n_buttons)}
    entropies: dict[str, list[float]] = {SHOOTER: [], HIDER: []}
    shooter_yaw: list[float] = []
    hider_yaw: list[float] = []
    positions: list[tuple[float, float]] = [
        (float(i1["POSITION_X"]), float(i1["POSITION_Y"]))
    ]
    launch_steps: list[int] = []
    fired_prev = float(i0.get("hide_and_seek_rocket_shots", 0.0) or 0.0)
    step = 0
    done = False
    while not done:
        a0 = shooter.act(observation(SHOOTER), deterministic)
        a1 = hider.act(observation(HIDER), deterministic)
        for agent, controller in ((SHOOTER, shooter), (HIDER, hider)):
            entropy = controller.entropy()
            if entropy is not None:
                entropies[agent].append(entropy)
        counts[SHOOTER] += _bits(a0, n_buttons)
        counts[HIDER] += _bits(a1, n_buttons)
        obs, rewards, terms, truncs, infos = env.step({SHOOTER: a0, HIDER: a1})
        step += 1
        if corruption == "shuffled":
            for agent in blind_agents:
                if len(seen[agent]) < 4096:
                    seen[agent].append(obs[agent].copy())
        stats.shooter_reward += float(rewards.get(SHOOTER, 0.0))
        stats.hider_reward += float(rewards.get(HIDER, 0.0))
        i0, i1 = infos[SHOOTER], infos[HIDER]
        fired = float(i0.get("hide_and_seek_rocket_shots", 0.0) or 0.0)
        if fired > fired_prev:
            launch_steps.append(step)
            fired_prev = fired
        positions.append((float(i1["POSITION_X"]), float(i1["POSITION_Y"])))
        if float(i0.get("USER54", 0) or 0) > 0 and float(i0.get("USER55", 0) or 0) > 0:
            shooter_yaw.append(float(i0["USER59"]) / 1000.0)
            hider_yaw.append(float(i0["USER60"]) / 1000.0)
        done = all(terms.values()) or all(truncs.values())

    code = int(float(i0.get("hide_and_seek_outcome_code", 0.0) or 0.0))
    stats.outcome = OUTCOME_NAMES.get(code, str(code))
    stats.steps = step
    stats.round_seconds = step * skip_frames / 35.0
    stats.shooter_won = stats.outcome in ("shooter_win", "hider_suicide")
    stats.hider_survived = stats.outcome in ("hider_win", "hider_escape")
    if stats.outcome == "hider_escape":
        stats.escape_reason = (
            "budget" if float(i0.get("USER53", 0.0) or 0.0) > 0 else "timeout"
        )
    if stats.outcome == "shooter_win":
        stats.time_to_kill_seconds = step * skip_frames / 35.0
    stats.rockets_fired = int(fired_prev)
    stats.shooter_damage = float(i0.get("DAMAGECOUNT", 0.0)) - base_damage
    stats.hider_damage_taken = float(i1.get("DAMAGE_TAKEN", 0.0)) - base_taken
    stats.shooter_button_rates = {
        name: float(c / max(step, 1)) for name, c in zip(button_names, counts[SHOOTER])
    }
    stats.hider_button_rates = {
        name: float(c / max(step, 1)) for name, c in zip(button_names, counts[HIDER])
    }
    if shooter_yaw:
        stats.shooter_yaw_error_mean_deg = float(np.mean(shooter_yaw))
        stats.shooter_aim_within_10deg_fraction = float(
            np.mean(np.asarray(shooter_yaw) < 10.0)
        )
        stats.hider_yaw_error_mean_deg = float(np.mean(hider_yaw))
    reaction = reaction_metrics(positions, launch_steps, skip_frames)
    stats.hider_speed_in_window = reaction["speed_in"]
    stats.hider_speed_outside_window = reaction["speed_out"]
    stats.hider_reaction_speed_ratio = reaction["ratio"]
    stats.hider_direction_changes_in_window = reaction["turns_in"]
    stats.hider_direction_changes_outside_window = reaction["turns_out"]
    max_entropy = math.log(1 << n_buttons)
    if entropies[SHOOTER]:
        stats.shooter_entropy_ratio = float(np.mean(entropies[SHOOTER])) / max_entropy
    if entropies[HIDER]:
        stats.hider_entropy_ratio = float(np.mean(entropies[HIDER])) / max_entropy
    stats.duration_seconds = time.monotonic() - started
    return stats


def build_jobs(
    checkpoints: Sequence[Path],
    *,
    episodes: int,
    modes: Sequence[str],
    seed: int,
    scenario_config: str | None,
    blind_every_checkpoint: bool,
    corruptions: Sequence[str],
    turret_sigmas_deg: Sequence[int] = TURRET_SIGMAS_DEG,
) -> list[HSJob]:
    jobs: list[HSJob] = []
    last = checkpoints[-1]

    def add(checkpoint, prev, mode, condition, shooter, hider, corruption, focus):
        jobs.append(
            HSJob(
                index=len(jobs),
                checkpoint=str(checkpoint),
                prev_checkpoint=prev,
                condition=condition,
                shooter=shooter,
                hider=hider,
                corruption=corruption,
                focus=focus,
                mode=mode,
                episodes=episodes,
                seed=seed + 100 * len(jobs),
                scenario_config=scenario_config,
            )
        )

    for index, checkpoint in enumerate(checkpoints):
        prev = str(checkpoints[index - 1]) if index > 0 else None
        is_last = checkpoint == last
        for mode in modes:
            # shooter rows
            add(
                checkpoint,
                prev,
                mode,
                "shooter_vs_noop_hider",
                "policy",
                "noop",
                "none",
                "shooter",
            )
            add(
                checkpoint,
                prev,
                mode,
                "shooter_vs_random_hider",
                "policy",
                "random",
                "none",
                "shooter",
            )
            # hider rows (ACS turret with parameterised aim noise)
            for sigma in turret_sigmas_deg:
                spec = f"turret:{int(sigma) * 1000}"
                add(
                    checkpoint,
                    prev,
                    mode,
                    f"hider_vs_turret{sigma}",
                    spec,
                    "policy",
                    "none",
                    "hider",
                )
            # both learned
            add(checkpoint, prev, mode, "self", "policy", "policy", "none", "both")
            if prev is not None:
                add(
                    checkpoint,
                    prev,
                    mode,
                    "shooter_vs_prev_hider",
                    "policy",
                    "prev",
                    "none",
                    "shooter",
                )
                add(
                    checkpoint,
                    prev,
                    mode,
                    "prev_shooter_vs_hider",
                    "prev",
                    "policy",
                    "none",
                    "hider",
                )
            if is_last or blind_every_checkpoint:
                for corruption in corruptions:
                    if corruption == "none":
                        continue
                    add(
                        checkpoint,
                        prev,
                        mode,
                        f"shooter[{corruption}]_vs_random_hider",
                        "policy",
                        "random",
                        corruption,
                        "shooter",
                    )
                    add(
                        checkpoint,
                        prev,
                        mode,
                        f"hider[{corruption}]_vs_turret{TURRET_ABLATION_SIGMA_DEG}",
                        f"turret:{TURRET_ABLATION_SIGMA_DEG * 1000}",
                        "policy",
                        corruption,
                        "hider",
                    )
        if is_last:
            # reference rows without a learned policy (mode-independent, run once)
            mode = modes[0]
            add(
                checkpoint,
                prev,
                mode,
                "random_shooter_vs_random_hider",
                "random",
                "random",
                "none",
                "none",
            )
            add(
                checkpoint,
                prev,
                mode,
                "random_shooter_vs_noop_hider",
                "random",
                "noop",
                "none",
                "none",
            )
            for sigma in turret_sigmas_deg:
                spec = f"turret:{int(sigma) * 1000}"
                add(
                    checkpoint,
                    prev,
                    mode,
                    f"random_hider_vs_turret{sigma}",
                    spec,
                    "random",
                    "none",
                    "none",
                )
                add(
                    checkpoint,
                    prev,
                    mode,
                    f"noop_hider_vs_turret{sigma}",
                    spec,
                    "noop",
                    "none",
                    "none",
                )
    return jobs


def run_job(job: HSJob) -> tuple[HSJob, list[HSEpisodeStats], str | None]:
    """Executed in a duel_validity_eval worker process."""
    from . import make
    from .duel_validity_eval import (
        EVAL_BASE_PORT,
        EVAL_PORT_STRIDE,
        checkpoint_step,
        load_cached_bundle,
        scenario_config_path,
        write_eval_config,
    )

    try:
        bundle = load_cached_bundle(job.checkpoint)
        rng = random.Random(job.seed)
        n_buttons = len(bundle.buttons)

        def controller(spec: str, agent_index: int) -> _Controller:
            if spec == "policy":
                return _Controller(spec, n_buttons, bundle.adapters[agent_index], rng)
            if spec == "prev":
                if job.prev_checkpoint is None:
                    raise RuntimeError("prev controller without a previous checkpoint")
                prev = load_cached_bundle(job.prev_checkpoint)
                return _Controller(spec, n_buttons, prev.adapters[agent_index], rng)
            return _Controller(spec, n_buttons, None, rng)

        shooter = controller(job.shooter, 0)
        hider = controller(job.hider, 1)
        sigma = turret_sigma_millideg(job.shooter)

        tmp_dir = Path(tempfile.mkdtemp(prefix="hs_validity_eval_"))
        cfg = write_eval_config(
            scenario_config_path(bundle.scenario, job.scenario_config), tmp_dir
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
        results: list[HSEpisodeStats] = []
        try:
            base = _base_env(env)
            # The turret request lives in an ACS global, so it survives the per-round map change.
            if sigma is not None:
                base.send_game_command(f"puke {TURRET_ENABLE_SCRIPT} {sigma}")
            else:
                base.send_game_command(f"puke {TURRET_DISABLE_SCRIPT}")
            for episode in range(job.episodes):
                stats = HSEpisodeStats(
                    checkpoint_step=checkpoint_step(Path(job.checkpoint)),
                    condition=job.condition,
                    shooter=job.shooter,
                    hider=job.hider,
                    corruption=job.corruption,
                    focus=job.focus,
                    mode=job.mode,
                    seed=job.seed + episode,
                    episode=episode,
                )
                try:
                    run_round(
                        env,
                        shooter,
                        hider,
                        corruption=job.corruption,
                        focus=job.focus,
                        deterministic=(job.mode == "deterministic"),
                        button_names=bundle.button_names,
                        skip_frames=bundle.skip_frames,
                        rng=rng,
                        stats=stats,
                    )
                except Exception as exc:  # keep the other rounds
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


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _rate(flags: Sequence[bool]) -> tuple[float, float, float]:
    n = len(flags)
    k = int(sum(bool(f) for f in flags))
    low, high = wilson_interval(k, n) if n else (0.0, 0.0)
    return (k / n if n else 0.0), low, high


def summarize_condition(
    episodes: Sequence[HSEpisodeStats], seed: int
) -> dict[str, Any]:
    valid = [e for e in episodes if e.valid]
    n = len(valid)
    if n == 0:
        return {"episodes": 0, "invalid": len(episodes)}
    win, win_low, win_high = _rate([e.shooter_won for e in valid])
    surv, surv_low, surv_high = _rate([e.hider_survived for e in valid])
    outcomes = {name: 0 for name in OUTCOME_NAMES.values()}
    for e in valid:
        outcomes[e.outcome] = outcomes.get(e.outcome, 0) + 1
    escapes = [e.escape_reason for e in valid if e.escape_reason]
    button_names = list(valid[0].shooter_button_rates)
    return {
        "episodes": n,
        "invalid": len(episodes) - n,
        "shooter": valid[0].shooter,
        "hider": valid[0].hider,
        "focus": valid[0].focus,
        "shooter_win_rate": win,
        "shooter_win_rate_ci_low": win_low,
        "shooter_win_rate_ci_high": win_high,
        "hider_survival_rate": surv,
        "hider_survival_rate_ci_low": surv_low,
        "hider_survival_rate_ci_high": surv_high,
        "outcome_fractions": {k: v / n for k, v in outcomes.items() if v},
        "budget_exhausted_fraction": sum(r == "budget" for r in escapes) / n,
        "timeout_fraction": sum(r == "timeout" for r in escapes) / n,
        "time_to_kill_seconds": _mean([e.time_to_kill_seconds for e in valid]),
        "round_seconds": _mean([e.round_seconds for e in valid]),
        "rockets_fired": _mean([e.rockets_fired for e in valid]),
        "shooter_damage": _mean([e.shooter_damage for e in valid]),
        "shooter_reward": _mean([e.shooter_reward for e in valid]),
        "hider_reward": _mean([e.hider_reward for e in valid]),
        "shooter_yaw_error_mean_deg": _mean(
            [e.shooter_yaw_error_mean_deg for e in valid]
        ),
        "shooter_aim_within_10deg_fraction": _mean(
            [e.shooter_aim_within_10deg_fraction for e in valid]
        ),
        "hider_yaw_error_mean_deg": _mean([e.hider_yaw_error_mean_deg for e in valid]),
        "hider_speed_in_window": _mean([e.hider_speed_in_window for e in valid]),
        "hider_speed_outside_window": _mean(
            [e.hider_speed_outside_window for e in valid]
        ),
        "hider_reaction_speed_ratio": _mean(
            [e.hider_reaction_speed_ratio for e in valid]
        ),
        "hider_direction_changes_in_window": _mean(
            [e.hider_direction_changes_in_window for e in valid]
        ),
        "hider_direction_changes_outside_window": _mean(
            [e.hider_direction_changes_outside_window for e in valid]
        ),
        "shooter_entropy_ratio": _mean([e.shooter_entropy_ratio for e in valid]),
        "hider_entropy_ratio": _mean([e.hider_entropy_ratio for e in valid]),
        "shooter_button_rates": {
            name: _mean([e.shooter_button_rates.get(name) for e in valid])
            for name in button_names
        },
        "hider_button_rates": {
            name: _mean([e.hider_button_rates.get(name) for e in valid])
            for name in button_names
        },
        "start_same_side": int(sum(e.start_same_side for e in valid)),
        "distinct_shooter_spawns": len({tuple(e.shooter_spawn) for e in valid}),
        "distinct_hider_spawns": len({tuple(e.hider_spawn) for e in valid}),
        "hider_start_distance_mean": _mean([e.hider_start_distance for e in valid]),
        "spawn_retries": int(sum(e.spawn_retries for e in valid)),
        "seconds_per_round": _mean([e.duration_seconds for e in valid]),
        "_shooter_wins": [bool(e.shooter_won) for e in valid],
        "_hider_survivals": [bool(e.hider_survived) for e in valid],
    }


def aggregate(
    episodes: Sequence[HSEpisodeStats], seed: int
) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    grouped: dict[tuple[int, str, str], list[HSEpisodeStats]] = {}
    for e in episodes:
        grouped.setdefault((e.checkpoint_step, e.mode, e.condition), []).append(e)
    table: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for (step, mode, condition), items in grouped.items():
        table.setdefault(step, {}).setdefault(mode, {})[
            condition
        ] = summarize_condition(items, seed=seed + step % 10_000)
    return table


def rate_difference_ci(
    a: Sequence[bool], b: Sequence[bool], seed: int, samples: int = 10_000
) -> tuple[float, float]:
    """95% CI of mean(a) - mean(b) for two independent Bernoulli samples"""
    if not a or not b:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    da = rng.choice(a_arr, size=(samples, a_arr.size), replace=True).mean(axis=1)
    db = rng.choice(b_arr, size=(samples, b_arr.size), replace=True).mean(axis=1)
    low, high = np.percentile(da - db, [2.5, 97.5])
    return float(low), float(high)


def _status(passed: bool | None, weak: bool = False) -> str:
    if passed is None:
        return "n/a"
    if passed:
        return "pass"
    return "weak" if weak else "FAIL"


def evaluate_gates(
    table: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    mode: str,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    steps = sorted(table)
    final = steps[-1]
    conditions = table[final].get(mode, {})
    # the reference rows were run in the first mode only
    references: dict[str, Mapping[str, Any]] = {}
    for m in table[final].values():
        for name, summary in m.items():
            if summary.get("focus") == "none":
                references.setdefault(name, summary)
    gates: dict[str, dict[str, Any]] = {}

    def get(condition: str, key: str):
        summary = conditions.get(condition) or references.get(condition)
        return None if summary is None else summary.get(key)

    # 1. absolute aim sanity check
    win_noop = get("shooter_vs_noop_hider", "shooter_win_rate")
    gates["shooter_hits_static_hider"] = {
        "status": _status(
            None if win_noop is None else win_noop >= SHOOTER_VS_NOOP_MIN_WIN_RATE,
            weak=(win_noop is not None and win_noop >= SHOOTER_VS_NOOP_WEAK_WIN_RATE),
        ),
        "win_rate_vs_noop_hider": win_noop,
        "random_shooter_win_rate_vs_noop_hider": get(
            "random_shooter_vs_noop_hider", "shooter_win_rate"
        ),
        "time_to_kill_seconds": get("shooter_vs_noop_hider", "time_to_kill_seconds"),
        "note": f"a shooter that cannot hit a standing target invalidates every hider metric (pass >= {SHOOTER_VS_NOOP_MIN_WIN_RATE:.0%})",
    }

    # 2. shooter vs random hider must beat a random shooter
    wins = get("shooter_vs_random_hider", "_shooter_wins") or []
    ref_wins = get("random_shooter_vs_random_hider", "_shooter_wins") or []
    if wins and ref_wins:
        low, high = rate_difference_ci(wins, ref_wins, seed=seed + 1)
        diff = float(np.mean(wins) - np.mean(ref_wins))
        status = _status(low > 0, weak=diff > 0)
    else:
        low = high = diff = None
        status = "n/a"
    gates["shooter_beats_random_shooter"] = {
        "status": status,
        "win_rate_vs_random_hider": get("shooter_vs_random_hider", "shooter_win_rate"),
        "random_shooter_win_rate": get(
            "random_shooter_vs_random_hider", "shooter_win_rate"
        ),
        "win_rate_difference": diff,
        "ci_low": low,
        "ci_high": high,
        "note": "pass = bootstrap 95% CI of the win-rate difference above 0; weak = mean above 0 only",
    }

    # 3. hider vs turret must survive more often than a random hider
    sigma = TURRET_ABLATION_SIGMA_DEG
    per_sigma = {}
    pooled_h: list[bool] = []
    pooled_r: list[bool] = []
    for s in TURRET_SIGMAS_DEG:
        h = get(f"hider_vs_turret{s}", "_hider_survivals") or []
        r = get(f"random_hider_vs_turret{s}", "_hider_survivals") or []
        per_sigma[f"sigma{s}"] = {
            "hider": _mean(h) if h else None,
            "random_hider": _mean(r) if r else None,
            "noop_hider": get(f"noop_hider_vs_turret{s}", "hider_survival_rate"),
        }
        pooled_h += list(h)
        pooled_r += list(r)
    if pooled_h and pooled_r:
        low, high = rate_difference_ci(pooled_h, pooled_r, seed=seed + 2)
        diff = float(np.mean(pooled_h) - np.mean(pooled_r))
        status = _status(low > 0, weak=diff > 0)
    else:
        low = high = diff = None
        status = "n/a"
    gates["hider_beats_random_hider"] = {
        "status": status,
        "survival_rate_pooled": _mean(pooled_h) if pooled_h else None,
        "random_hider_survival_rate_pooled": _mean(pooled_r) if pooled_r else None,
        "survival_difference": diff,
        "ci_low": low,
        "ci_high": high,
        "per_sigma": per_sigma,
        "note": "survival vs the ACS turret pooled over sigma in {0,4,8} deg; pass = 95% CI of the difference to a random hider above 0",
    }

    # 4./5. blind ablations per role
    def vision_gate(
        role: str, sighted_key: str, blind_key: str, random_key: str, metric: str
    ):
        sighted = get(sighted_key, metric)
        blind = get(blind_key, metric)
        reference = get(random_key, metric)
        if sighted is None or blind is None or reference is None:
            return {
                "status": "n/a",
                "sighted": sighted,
                "blind": blind,
                "random": reference,
            }
        margin = sighted - reference
        retained = None if margin <= 1e-9 else (blind - reference) / margin
        collapsed = margin > 0 and (
            retained is not None and retained <= VISION_COLLAPSE_RATIO
        )
        return {
            "status": _status(collapsed, weak=(margin > 0 and blind < sighted)),
            f"sighted_{metric}": sighted,
            f"blind_{metric}": blind,
            f"random_{role}_{metric}": reference,
            "blind_margin_retained_fraction": retained,
            "note": f"pass = blind {role} keeps <= {VISION_COLLAPSE_RATIO:.0%} of the sighted margin over the random {role}",
        }

    for corruption in ("frozen", "shuffled"):
        gates[f"shooter_uses_vision_{corruption}"] = vision_gate(
            "shooter",
            "shooter_vs_random_hider",
            f"shooter[{corruption}]_vs_random_hider",
            "random_shooter_vs_random_hider",
            "shooter_win_rate",
        )
        gates[f"hider_uses_vision_{corruption}"] = vision_gate(
            "hider",
            f"hider_vs_turret{sigma}",
            f"hider[{corruption}]_vs_turret{sigma}",
            f"random_hider_vs_turret{sigma}",
            "hider_survival_rate",
        )

    # 6. per-role previous-checkpoint cross-play (informational)
    prev_step = steps[-2] if len(steps) > 1 else None
    prev_self = (
        table[prev_step].get(mode, {}).get("self") if prev_step is not None else None
    )
    new_s_prev_h = get("shooter_vs_prev_hider", "shooter_win_rate")
    prev_s_new_h = get("prev_shooter_vs_hider", "hider_survival_rate")
    gates["shooter_improves_over_previous"] = {
        "status": _status(
            None
            if new_s_prev_h is None or prev_self is None
            else new_s_prev_h >= prev_self.get("shooter_win_rate", 0.0)
        ),
        "new_shooter_vs_prev_hider_win_rate": new_s_prev_h,
        "prev_shooter_vs_prev_hider_win_rate": (
            None if prev_self is None else prev_self.get("shooter_win_rate")
        ),
        "note": "informational: the newer shooter should do at least as well against the previous hider as the previous shooter did",
    }
    gates["hider_improves_over_previous"] = {
        "status": _status(
            None
            if prev_s_new_h is None or prev_self is None
            else prev_s_new_h >= prev_self.get("hider_survival_rate", 0.0)
        ),
        "new_hider_vs_prev_shooter_survival_rate": prev_s_new_h,
        "prev_hider_vs_prev_shooter_survival_rate": (
            None if prev_self is None else prev_self.get("hider_survival_rate")
        ),
        "note": "informational: the newer hider should survive the previous shooter at least as often as the previous hider did",
    }

    # 7. diagnostics
    self_row = conditions.get("self", {})
    gates["round_diagnostics"] = {
        "status": "n/a",
        "self_shooter_win_rate": self_row.get("shooter_win_rate"),
        "self_outcomes": self_row.get("outcome_fractions"),
        "self_budget_exhausted_fraction": self_row.get("budget_exhausted_fraction"),
        "self_timeout_fraction": self_row.get("timeout_fraction"),
        "self_time_to_kill_seconds": self_row.get("time_to_kill_seconds"),
        "self_rockets_fired": self_row.get("rockets_fired"),
        "self_shooter_yaw_error_deg": self_row.get("shooter_yaw_error_mean_deg"),
        "self_shooter_aim_within_10deg": self_row.get(
            "shooter_aim_within_10deg_fraction"
        ),
        "note": "informational: training-equilibrium round statistics",
    }
    ratio = get(f"hider_vs_turret{sigma}", "hider_reaction_speed_ratio")
    random_ratio = get(f"random_hider_vs_turret{sigma}", "hider_reaction_speed_ratio")
    gates["hider_reacts_to_launches"] = {
        "status": _status(
            None if ratio is None else ratio > 1.1,
            weak=(ratio is not None and ratio > 1.0),
        ),
        "speed_ratio_window_vs_outside": ratio,
        "random_hider_speed_ratio": random_ratio,
        "direction_changes_in_window": get(
            f"hider_vs_turret{sigma}", "hider_direction_changes_in_window"
        ),
        "direction_changes_outside_window": get(
            f"hider_vs_turret{sigma}", "hider_direction_changes_outside_window"
        ),
        "note": f"hider lateral speed {REACTION_WINDOW_SECONDS[0]}-{REACTION_WINDOW_SECONDS[1]} s after a launch relative to the rest of the round (a schedule-based dodger shows ~1.0)",
    }

    # spawn layout (all rows)
    same_side = sum(
        s.get("start_same_side", 0) for m in table[final].values() for s in m.values()
    )
    rounds = sum(
        s.get("episodes", 0) for m in table[final].values() for s in m.values()
    )
    per_condition = max(
        (s.get("episodes", 0) for m in table[final].values() for s in m.values()),
        default=0,
    )
    distinct = max(
        (
            s.get("distinct_hider_spawns", 0)
            for m in table[final].values()
            for s in m.values()
        ),
        default=0,
    )
    retries = sum(
        s.get("spawn_retries", 0) for m in table[final].values() for s in m.values()
    )
    gates["randomised_opposite_spawns"] = {
        "status": _status(
            rounds > 0 and same_side == 0 and distinct >= min(4, per_condition)
        ),
        "rounds": rounds,
        "same_side_starts": same_side,
        "max_distinct_hider_spawns_in_one_condition": distinct,
        "spawn_retries_after_telefrag": retries,
        "note": "shooter and hider always start on opposite strips, positions do not repeat",
    }

    for role, key in (
        ("shooter", "shooter_entropy_ratio"),
        ("hider", "hider_entropy_ratio"),
    ):
        ratio = self_row.get(key)
        gates[f"{role}_policy_not_uniform"] = {
            "status": _status(
                None if ratio is None else ratio <= 0.9,
                weak=(ratio is not None and ratio <= 0.97),
            ),
            "entropy_over_max_entropy": ratio,
            "button_rates": self_row.get(f"{role}_button_rates"),
            "note": "mean action-distribution entropy / ln(num_actions) in the self row",
        }
    return gates


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _condition_order(condition: str) -> tuple[int, str]:
    order = [
        "shooter_vs_noop_hider",
        "shooter_vs_random_hider",
        "random_shooter_vs_noop_hider",
        "random_shooter_vs_random_hider",
        "self",
        "shooter_vs_prev_hider",
        "prev_shooter_vs_hider",
    ]
    base = condition.split("[", 1)[0]
    if condition in order:
        return (order.index(condition), condition)
    if base in order:
        return (order.index(base), condition)
    return (len(order), condition)


def render_report(
    table: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    gates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    meta: Mapping[str, Any],
) -> str:
    lines = [
        f"# Hide-and-seek validity evaluation: {meta.get('run_id') or meta.get('experiment')}",
        "",
        f"checkpoints: {', '.join(str(s) for s in sorted(table))} | rounds/condition: {meta['episodes']} "
        f"| modes: {', '.join(meta['modes'])} | scenario: {meta.get('scenario')} | "
        "roles: agent_0 shooter, agent_1 hider",
        "",
    ]
    for mode, mode_gates in gates.items():
        lines += [
            f"## Gates ({mode})",
            "",
            "| gate | status | evidence |",
            "|---|---|---|",
        ]
        for name, gate in mode_gates.items():
            evidence = ", ".join(
                f"{k}={_fmt(v)}"
                for k, v in gate.items()
                if k not in ("status", "note") and not isinstance(v, (dict, list))
            )
            lines.append(f"| {name} | **{gate['status']}** | {evidence} |")
        lines.append("")
    for step in sorted(table):
        for mode, conditions in table[step].items():
            lines += [
                f"## checkpoint {step} ({mode})",
                "",
                "| condition | n | shooter win | 95% CI | hider survival | outcomes | TTK s | fired | "
                "yaw err° | aim<10° | react ratio | H/Hmax S/H | shooter buttons | hider buttons |",
                "|---|" + "---|" * 13,
            ]
            for condition in sorted(conditions, key=_condition_order):
                s = conditions[condition]
                if not s.get("episodes"):
                    lines.append(
                        f"| {condition} | 0 | invalid: {s.get('invalid')} |" + " |" * 11
                    )
                    continue
                outcomes = " ".join(
                    f"{k}={v:.2f}"
                    for k, v in (s.get("outcome_fractions") or {}).items()
                )
                s_rates = " ".join(
                    f"{k.lower()}={_fmt(v)}"
                    for k, v in (s.get("shooter_button_rates") or {}).items()
                )
                h_rates = " ".join(
                    f"{k.lower()}={_fmt(v)}"
                    for k, v in (s.get("hider_button_rates") or {}).items()
                )
                lines.append(
                    f"| {condition} | {s['episodes']} | {_fmt(s['shooter_win_rate'])} | "
                    f"[{_fmt(s['shooter_win_rate_ci_low'])}, {_fmt(s['shooter_win_rate_ci_high'])}] | "
                    f"{_fmt(s['hider_survival_rate'])} | {outcomes} | {_fmt(s['time_to_kill_seconds'], 1)} | "
                    f"{_fmt(s['rockets_fired'], 1)} | {_fmt(s['shooter_yaw_error_mean_deg'], 1)} | "
                    f"{_fmt(s['shooter_aim_within_10deg_fraction'])} | {_fmt(s['hider_reaction_speed_ratio'])} | "
                    f"{_fmt(s['shooter_entropy_ratio'])}/{_fmt(s['hider_entropy_ratio'])} | {s_rates} | {h_rates} |"
                )
            lines.append("")
    return "\n".join(lines)


def public_table(
    table: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> dict[str, Any]:
    """summary.json payload without the per-round bootstrap samples."""
    return {
        str(step): {
            mode: {
                condition: {k: v for k, v in summary.items() if not k.startswith("_")}
                for condition, summary in conditions.items()
            }
            for mode, conditions in modes.items()
        }
        for step, modes in table.items()
    }


WANDB_COLUMNS = (
    "shooter_win_rate",
    "shooter_win_rate_ci_low",
    "shooter_win_rate_ci_high",
    "hider_survival_rate",
    "budget_exhausted_fraction",
    "timeout_fraction",
    "time_to_kill_seconds",
    "rockets_fired",
    "shooter_damage",
    "shooter_yaw_error_mean_deg",
    "shooter_aim_within_10deg_fraction",
    "hider_reaction_speed_ratio",
    "shooter_entropy_ratio",
    "hider_entropy_ratio",
)
