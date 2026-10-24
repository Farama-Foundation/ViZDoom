"""
Evaluate the rule-based oracle on health_gathering_multi_agent and record a video.

The oracle plays using the engine's ground truth data (medkit and player positions,
exact health) rather than pixels, so its score is a rough upper reference point for
the learned policies.

Two modes are available and are worth running together (`--mode both`):

  collaborator - an agent only obtains a kit once its health is below --consume-health
            (50 by default), so the full heal is utilized. Until then, it just waits
            beside a kit and walks around the others. The agent lowest on health calls
            "dibs" for the closest kit. The dibs is not respected if it is not possible
            for said dying agent to reach the kit.
  defector     - every agent grabs its own nearest kit on sight, at any health.

By default, measured over 10 episodes x 2625 steps (a full 5 minute episode, 10500 tics
at a frame skip of 4), 2 agents. With the individual rewards setting, the collaborator
ought to score about twice as much as the defector.

The collaborator takes fewer kits and still scores twice as much, because a death costs
-10. Note the scenario pays +1 per step in which health rose rather than per health point,
so restraint is not otherwise rewarded. A pickup at 90 health pays the same as one at 40.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pettingzoo_wrapper import make  # noqa: E402
from pettingzoo_wrapper.oracle_base import summarize_decisions  # noqa: E402
from pettingzoo_wrapper.oracle_health_gathering import (  # noqa: E402
    HealthGatheringOracle,
    HealthGatheringOracleConfig,
)


DEFAULT_SCENARIO = "health_gathering_multi_agent"

# Health below which an agent counts as needing to obtain a kit
DEFAULT_URGENT_HEALTH = 50.0


# --------------------------------------------------------------------- stats


@dataclass
class AgentStats:
    """Per-agent bookkeeping for one episode."""

    ret: float = 0.0
    pickups: int = 0
    early_pickups: int = 0
    pickup_health: List[float] = field(default_factory=list)
    health_gained: float = 0.0
    deaths: int = 0
    defections: int = 0
    health_samples: List[float] = field(default_factory=list)
    steps_urgent: int = 0

    def health_wasted(self, kit_value: float) -> float:
        return max(0.0, self.pickups * kit_value - self.health_gained)


@dataclass
class EpisodeStats:
    steps: int = 0
    agents: Dict[str, AgentStats] = field(default_factory=dict)

    def agent(self, name: str) -> AgentStats:
        return self.agents.setdefault(name, AgentStats())


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


# --------------------------------------------------------------------- video


class TiledVideoRecorder:
    """Tiles the agents' screens side by side and draws a health/state HUD."""

    _STATE_COLORS = {
        "go": (80, 220, 120),
        "defer": (80, 180, 255),
        "hold": (250, 210, 70),
        "escape": (255, 140, 60),
        "dead": (230, 60, 60),
    }

    def __init__(self, agents: List[str], fps: float, max_frames: int = 4000) -> None:
        self.agents = list(agents)
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.frames: List[np.ndarray] = []

    def capture(self, obs: Dict[str, np.ndarray], decisions: Dict[str, Any]) -> None:
        if len(self.frames) >= self.max_frames:
            return
        panels = []
        for agent in self.agents:
            frame = obs.get(agent)
            if frame is None:
                continue
            panels.append(self._decorate(np.ascontiguousarray(frame), decisions.get(agent)))
        if panels:
            self.frames.append(np.concatenate(panels, axis=1))

    def _decorate(self, frame: np.ndarray, decision: Any) -> np.ndarray:
        frame = frame.astype(np.uint8, copy=True)
        if decision is None:
            return frame
        height, width = frame.shape[:2]
        bar_height = max(4, height // 24)

        health = max(0.0, min(100.0, float(decision.health)))
        filled = int(width * health / 100.0)
        frame[:bar_height, :filled] = (60, 200, 90) if health > 40 else (230, 90, 60)
        frame[:bar_height, filled:] = (40, 40, 40)

        if decision.dead:
            state = "dead"
        elif decision.escaping:
            state = "escape"
        elif decision.holding:
            state = "hold"
        elif decision.deferred:
            state = "defer"
        else:
            state = "go"
        marker = max(6, height // 16)
        frame[bar_height : bar_height + marker, :marker] = self._STATE_COLORS[state]
        return frame

    def write(self, path: str) -> Optional[str]:
        if not self.frames:
            return None
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[video] imageio not installed, skipping video", flush=True)
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        writer = imageio.get_writer(path, fps=self.fps, quality=8, macro_block_size=1)
        try:
            for frame in self.frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return path


# ------------------------------------------------------------------ episodes


def run_episode(
    env,
    oracle: HealthGatheringOracle,
    *,
    max_steps: int,
    recorder: Optional[TiledVideoRecorder],
    config: HealthGatheringOracleConfig,
    seed: Optional[int],
    log_every: int,
    urgent_health: float,
    debug_pickups: bool = False,
) -> EpisodeStats:
    obs, infos = env.reset(seed=seed)
    oracle.reset()
    stats = EpisodeStats()
    previous_health = {
        agent: float(info.get("HEALTH", config.max_health))
        for agent, info in infos.items()
    }

    for step in range(1, max_steps + 1):
        actions = oracle.act(infos)
        decisions = oracle.last_decisions
        if recorder is not None:
            recorder.capture(obs, decisions)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        stats.steps = step

        health_now = {
            agent: float(info.get("HEALTH", 0.0)) for agent, info in infos.items()
        }
        urgent_agents = {
            agent for agent, health in health_now.items() if health < urgent_health
        }

        for agent, info in infos.items():
            agent_stats = stats.agent(agent)
            agent_stats.ret += float(rewards.get(agent, 0.0))
            health = health_now[agent]
            agent_stats.health_samples.append(health)
            if health < urgent_health:
                agent_stats.steps_urgent += 1

            died = bool(info.get("just_died", False))
            if died:
                agent_stats.deaths += 1

            was = previous_health.get(agent, health)
            # A respawn also raises health, so only count a real pickup.
            if not died and was > 0.0 and health > was:
                agent_stats.pickups += 1
                agent_stats.health_gained += health - was
                agent_stats.pickup_health.append(was)
                if debug_pickups:
                    decision = decisions.get(agent)
                    state = (
                        "none"
                        if decision is None
                        else (
                            "escape" if decision.escaping
                            else "hold" if decision.holding
                            else "defer" if decision.deferred
                            else "go"
                        )
                    )
                    distance = float("nan") if decision is None else decision.distance
                    print(
                        f"    pickup step {step:4d} {agent} hp {was:5.1f} -> {health:5.1f}"
                        f"  state={state:6s} target_dist={distance:6.0f}",
                        flush=True,
                    )
                # The pickup happens somewhere inside the frame skip, so the
                # health at the previous step boundary overstates it by up to
                # one step of drain.
                if was - config.drain_per_step > config.consume_health:
                    agent_stats.early_pickups += 1
                partner_urgent = bool(urgent_agents - {agent})
                if was > config.consume_health and partner_urgent:
                    agent_stats.defections += 1
            previous_health[agent] = health

        if log_every and step % log_every == 0:
            print(
                f"  step {step:4d}  {summarize_decisions(decisions)}",
                flush=True,
            )

        if any(terminations.values()) or any(truncations.values()):
            break

    if recorder is not None:
        recorder.capture(obs, oracle.last_decisions)
    return stats


def run_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    config: HealthGatheringOracleConfig,
) -> List[EpisodeStats]:
    env = make(
        scenario=args.scenario,
        num_agents=args.num_agents,
        resolution=args.resolution,
        skip_frames=args.skip_frames,
        async_mode=False,
        render_mode=None,
        port=args.port,
        netmode=0,
        ticrate=args.ticrate,
        seed=args.seed,
        enable_video=False,
        verbose=args.verbose,
        daemon=True,
        vector_obs=False,
        privileged_info=True,
        reward_mode=args.reward_mode,
        shared_reward_agg=args.shared_reward_agg,
    )
    oracle = HealthGatheringOracle(env, mode=mode, config=config)
    episodes: List[EpisodeStats] = []
    try:
        for episode in range(args.episodes):
            record = (not args.no_video) and episode == 0
            recorder = (
                TiledVideoRecorder(
                    agents=list(env.possible_agents),
                    fps=args.ticrate / max(1, args.skip_frames),
                )
                if record
                else None
            )
            print(f"[{mode}] episode {episode + 1}/{args.episodes}", flush=True)
            stats = run_episode(
                env,
                oracle,
                max_steps=args.max_steps,
                recorder=recorder,
                config=config,
                seed=None if episode else args.seed,
                log_every=args.log_every,
                urgent_health=args.urgent_health,
                debug_pickups=args.debug_pickups,
            )
            episodes.append(stats)
            if recorder is not None:
                path = args.video.replace(".mp4", f"_{mode}.mp4")
                written = recorder.write(path)
                if written:
                    print(f"[{mode}] wrote {written} ({len(recorder.frames)} frames)")
    finally:
        env.close()
    return episodes


# ------------------------------------------------------------------- report


def report(mode: str, episodes: List[EpisodeStats], config: HealthGatheringOracleConfig) -> Dict[str, float]:
    agents = sorted({agent for episode in episodes for agent in episode.agents})
    print(f"\n=== oracle [{mode}] over {len(episodes)} episode(s) ===")
    header = (
        f"{'agent':<10}{'return':>10}{'kits':>7}{'early':>7}{'pickHP':>8}{'gained':>9}"
        f"{'wasted':>9}{'deaths':>8}{'defect':>8}{'meanHP':>8}{'minHP':>7}{'urgent%':>9}"
    )
    print(header)
    print("-" * len(header))

    totals = {"return": 0.0, "pickups": 0.0, "deaths": 0.0, "wasted": 0.0, "defections": 0.0}
    per_agent_pickups: List[float] = []
    for agent in agents:
        rows = [episode.agents[agent] for episode in episodes if agent in episode.agents]
        returns = [row.ret for row in rows]
        pickups = [float(row.pickups) for row in rows]
        early = [float(row.early_pickups) for row in rows]
        pickup_hp = [_mean(row.pickup_health) for row in rows if row.pickup_health]
        gained = [row.health_gained for row in rows]
        wasted = [row.health_wasted(config.kit_value) for row in rows]
        deaths = [float(row.deaths) for row in rows]
        defections = [float(row.defections) for row in rows]
        mean_hp = [_mean(row.health_samples) for row in rows]
        min_hp = [min(row.health_samples) if row.health_samples else 0.0 for row in rows]
        urgent = [
            100.0 * row.steps_urgent / max(1, len(row.health_samples)) for row in rows
        ]
        print(
            f"{agent:<10}{_mean(returns):>10.1f}{_mean(pickups):>7.1f}"
            f"{_mean(early):>7.1f}{_mean(pickup_hp):>8.0f}{_mean(gained):>9.0f}"
            f"{_mean(wasted):>9.0f}"
            f"{_mean(deaths):>8.1f}"
            f"{_mean(defections):>8.1f}{_mean(mean_hp):>8.1f}{_mean(min_hp):>7.0f}"
            f"{_mean(urgent):>9.1f}"
        )
        totals["return"] += _mean(returns)
        totals["pickups"] += _mean(pickups)
        totals["deaths"] += _mean(deaths)
        totals["wasted"] += _mean(wasted)
        totals["defections"] += _mean(defections)
        per_agent_pickups.append(_mean(pickups))

    team_returns = [
        sum(row.ret for row in episode.agents.values()) for episode in episodes
    ]
    fairness = (
        min(per_agent_pickups) / max(per_agent_pickups)
        if per_agent_pickups and max(per_agent_pickups) > 0
        else float("nan")
    )
    steps = [float(episode.steps) for episode in episodes]
    print("-" * len(header))
    print(
        f"team return {_mean(team_returns):.1f} +/- {_stdev(team_returns):.1f}   "
        f"kits {totals['pickups']:.1f}   health wasted {totals['wasted']:.0f}   "
        f"deaths {totals['deaths']:.1f}   defections {totals['defections']:.1f}   "
        f"kit share min/max {fairness:.2f}   steps {_mean(steps):.0f}"
    )
    # Use a per-episode series to report the results
    per_episode = {
        "team_return": team_returns,
        "per_agent_return": [total / max(1, len(agents)) for total in team_returns],
        "deaths_total": [
            sum(row.deaths for row in episode.agents.values()) for episode in episodes
        ],
        "reward_gap": [
            max(row.ret for row in episode.agents.values())
            - min(row.ret for row in episode.agents.values())
            for episode in episodes
            if len(episode.agents) > 1
        ],
    }
    summary = {
        "episodes": len(episodes),
        "steps_per_episode": _mean(steps),
        "kits": totals["pickups"],
        "wasted": totals["wasted"],
        "defections": totals["defections"],
        "fairness": fairness,
        "metrics": {
            name: {"mean": _mean(values), "std": _stdev(values), "n": len(values)}
            for name, values in per_episode.items()
            if values
        },
    }
    # Add some averaged results as well
    summary.update(
        {
            "team_return": _mean(team_returns),
            "pickups": totals["pickups"],
            "deaths": totals["deaths"],
        }
    )
    return summary


# --------------------------------------------------------------------- main


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--mode", choices=("collaborator", "defector", "both"),
                        default="both")
    parser.add_argument("--resolution", default="320X240")
    parser.add_argument("--skip-frames", type=int, default=4)
    parser.add_argument("--ticrate", type=int, default=35)
    parser.add_argument("--max-steps", type=int, default=2625,
                        help="2625 = the scenario's 10500 tic (5 minute) timeout at skip 4")
    parser.add_argument("--reward-mode", choices=("individual", "shared"), default="individual")
    parser.add_argument("--shared-reward-agg", choices=("sum", "mean"), default="sum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=5029)
    parser.add_argument("--video", default="oracle.mp4",
                        help="output path; the mode name is appended per run")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--kit-value", type=float, default=50.0,
                        help="health a medkit restores in the scenario wad; sets the "
                             "restraint threshold and the wasted-health metric")
    parser.add_argument("--urgent-health", type=float, default=DEFAULT_URGENT_HEALTH,
                        help="reporting threshold: health below which an agent counts "
                             "as needing a kit (urgent%% and defections columns)")
    parser.add_argument("--consume-health", type=float, default=None,
                        help="the collaborator only picks a kit up below this health "
                             "(default: max_health - kit value, the waste-free point)")
    parser.add_argument("--no-restraint", action="store_true",
                        help="ablation: the collaborator keeps its claim priority "
                             "but takes kits at any health")
    parser.add_argument("--log-every", type=int, default=0,
                        help="print the oracle's per-agent decisions every N steps")
    parser.add_argument("--json-out", default=None,
                        help="write the results as a JSON file for later plotting convenience")
    parser.add_argument("--debug-pickups", action="store_true",
                        help="print one line per medkit pickup with the oracle's state")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = HealthGatheringOracleConfig(
        skip_frames=args.skip_frames,
        kit_value=args.kit_value,
        consume_health=args.consume_health,
    )
    if args.no_restraint:
        # Holding back from grabbing health kits would be "wait while health > consume_health",
        # so "no restraint" would mean to set the threshold at the full health value
        config.consume_health = config.max_health
    modes = ("collaborator", "defector") if args.mode == "both" else (args.mode,)

    summaries: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        episodes = run_mode(mode=mode, args=args, config=config)
        summaries[mode] = report(mode, episodes, config)

    if args.json_out:
        payload = {
            "scenario": args.scenario,
            "num_agents": args.num_agents,
            "reward_mode": args.reward_mode,
            "max_steps": args.max_steps,
            "skip_frames": args.skip_frames,
            "modes": {
                mode: summary["metrics"] for mode, summary in summaries.items()
            },
        }
        path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote baselines to {path}")

    if len(summaries) > 1:
        collaborator = summaries["collaborator"]
        defector = summaries["defector"]
        print("\n=== collaborator - defector ===")
        for key in ("team_return", "pickups", "wasted", "deaths", "defections"):
            delta = collaborator[key] - defector[key]
            print(
                f"{key:>12}: {collaborator[key]:>8.1f} vs {defector[key]:>8.1f}"
                f"  ({delta:+.1f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
