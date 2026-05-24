from typing import Any, Dict, Optional

import cv2
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class ObservationWrapper(ParallelEnv):
    def __init__(
        self,
        env: ParallelEnv,
        *,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_AREA,
    ):
        self.env = env
        self.width = int(width)
        self.height = int(height)
        self.interpolation = interpolation
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.metadata = getattr(env, "metadata", {})

        sample_space = env.observation_space(self.possible_agents[0])
        channels = 1 if len(sample_space.shape) < 3 else sample_space.shape[-1]
        self._observation_space = spaces.Box(
            low=sample_space.low.flat[0],
            high=sample_space.high.flat[0],
            shape=(self.height, self.width, channels),
            dtype=sample_space.dtype,
        )

    def _resize_obs(self, obs: np.ndarray) -> np.ndarray:
        resized = cv2.resize(obs, (self.width, self.height), interpolation=self.interpolation)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        return resized

    def _resize_obs_dict(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {agent: self._resize_obs(obs) for agent, obs in obs_dict.items()}

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self._observation_space

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        return self._resize_obs_dict(obs), infos

    def step(self, actions: Dict[str, Any]):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents[:]
        return self._resize_obs_dict(obs), rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
