# pettingzoo_wrapper/reward_wrappers.py
from typing import Any, Dict, Optional
import numpy as np
from pettingzoo import ParallelEnv

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
    ):
        self.env = env
        self.scaler = float(scaler)
        self.death_penalty = float(death_penalty)
        self.keep_lb = bool(keep_lb)
        self.goal_x = goal_x
        self.goal_reward = float(goal_reward)
        self.pos_key = pos_key
        self.dead_key = dead_key

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
        self._prev_x = {a: self._extract_x(infos, a) for a in self.agents}
        self._best_x = {a: (self._prev_x[a] if self._prev_x[a] is not None else -np.inf) for a in self.agents}
        self._goal_given = {a: False for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, base_rewards, terminations, truncations, infos = self.env.step(actions)

        shaped: Dict[str, float] = {}
        for a in self.agents:
            r = float(base_rewards.get(a, 0.0))
            x_cur = self._extract_x(infos, a)
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

            info_a = infos.get(a, {}) or {}
            gv = (info_a.get("game_variables", {}) or {})
            just_died = bool(info_a.get("just_died", False))
            dead_now = bool(gv.get(self.dead_key, 0))
            if just_died or (dead_now and not just_died):
                r += self.death_penalty

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

    def _extract_x(self, infos: Dict[str, Dict[str, Any]], agent: str) -> Optional[float]:
        gv = (infos.get(agent, {}).get("game_variables", {}) or {})
        x = gv.get(self.pos_key, None)
        return float(x) if x is not None else None
