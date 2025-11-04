from typing import Any, Dict, Optional
import numpy as np
from pettingzoo import ParallelEnv

class HealthGatheringRewardWrapper(ParallelEnv):
    """
    Dense shaping on HEALTH. Optional sparse goal on reaching max health.
    """
    def __init__(
            self,
            env: ParallelEnv,
            *,
            death_penalty: float = -10.0,
            medkit_reward: float = 1.0,
            health_key: str = "HEALTH",
            dead_key: str = "DEAD",
    ):
        self.env = env
        self.death_penalty = death_penalty
        self.medkit_reward = medkit_reward
        self.health_key = health_key
        self.dead_key = dead_key
        self._prev_health: Dict[str, Optional[float]] = {}
        self.possible_agents = env.possible_agents
        self.agents = env.agents

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._prev_health = {a: infos[a][self.health_key] for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        for a in self.agents:
            r = rewards[a]
            h_cur = infos[a].get(self.health_key, 0.0)
            h_prev = self._prev_health[a]

            if h_cur > h_prev:
                r += self.medkit_reward

            if infos[a]["just_died"]:
                r += self.death_penalty

            rewards[a] = r
            self._prev_health[a] = h_cur

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class PitfallRewardWrapper(ParallelEnv):
    """
    Dense shaping on POSITION_X (forward-only). Optional sparse goal on crossing goal_x.
    """
    def __init__(
            self,
            env: ParallelEnv,
            *,
            scaler: float = 0.01,
            death_penalty: float = -1.0,
            keep_lb: bool = True,
            goal_x: Optional[float] = None,
            goal_reward: float = 10.0,
            pos_key: str = "POSITION_X",
            dead_key: str = "DEAD",
            x_start: float = 32.0,
    ):
        self.env = env
        self.scaler = float(scaler)
        self.death_penalty = float(death_penalty)
        self.keep_lb = bool(keep_lb)
        self.goal_x = goal_x
        self.goal_reward = float(goal_reward)
        self.pos_key = pos_key
        self.dead_key = dead_key
        self.x_start = float(x_start)

        self._prev_x: Dict[str, Optional[float]] = {}
        self._best_x: Dict[str, float] = {}
        self._goal_given: Dict[str, bool] = {}

        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = env.possible_agents
        self.agents = env.agents

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._prev_x = {a: infos[a].get(self.pos_key, None) for a in self.agents}
        self._best_x = {a: (self._prev_x[a] if self._prev_x[a] is not None else -np.inf) for a in self.agents}
        self._goal_given = {a: False for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, base_rewards, terminations, truncations, infos = self.env.step(actions)

        shaped: Dict[str, float] = {}
        for a in self.agents:
            r = float(base_rewards.get(a, 0.0))
            x_cur = infos[a].get(self.pos_key, None)
            x_prev = self._prev_x.get(a, None)

            if x_prev is not None and x_cur is not None:
                if self.keep_lb:
                    inc = max(0.0, x_cur - max(self._best_x[a], x_prev))
                    if inc > 0:
                        r += self.scaler * inc
                    self._best_x[a] = max(self._best_x[a], x_cur)
                else:
                    dx = max(0.0, x_cur - x_prev)
                    if dx > 0:
                        r += self.scaler * dx

            info_a = infos.get(a, {})
            just_died = bool(info_a.get("just_died", False))
            dead_now = bool(info_a.get(self.dead_key, 0))
            if just_died or (dead_now and not just_died):
                r += self.death_penalty
                self._best_x[a] = self.x_start  # reset best_x on death

            if self.goal_x is not None and not self._goal_given.get(a, False):
                if x_cur is not None and x_cur > float(self.goal_x):
                    r += self.goal_reward
                    self._goal_given[a] = True

            shaped[a] = r
            self._prev_x[a] = x_cur

        return obs, shaped, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
