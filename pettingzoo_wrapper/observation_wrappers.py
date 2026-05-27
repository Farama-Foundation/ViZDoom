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
        if not isinstance(sample_space, spaces.Box):
            raise TypeError("ObservationWrapper requires Box observation spaces")
        sample_shape = sample_space.shape
        if sample_shape is None:
            raise ValueError("Observation space shape must be defined")
        channels = 1 if len(sample_shape) < 3 else sample_shape[-1]
        self._observation_space = spaces.Box(
            low=sample_space.low.flat[0],
            high=sample_space.high.flat[0],
            shape=(self.height, self.width, channels),
            dtype=sample_space.dtype,
        )

    def _resize_obs(self, obs: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            obs, (self.width, self.height), interpolation=self.interpolation
        )
        if resized.ndim == 2:
            resized = resized[:, :, None]
        return resized

    def _resize_obs_dict(
        self, obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return {agent: self._resize_obs(obs) for agent, obs in obs_dict.items()}

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self._observation_space

    @property
    def state_space(self):
        return spaces.Box(
            low=self._observation_space.low.flat[0],
            high=self._observation_space.high.flat[0],
            shape=(
                self.height,
                self.width,
                self._observation_space.shape[-1] * self.num_agents,
            ),
            dtype=self._observation_space.dtype,
        )

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def state(self):
        return np.concatenate(
            [self.env_state_observation(agent) for agent in self.possible_agents],
            axis=-1,
        )

    def env_state_observation(self, agent: str):
        if hasattr(self.env, "state_observation"):
            obs = self.env.state_observation(agent)
        else:
            obs = getattr(self.env, "_last_frames", {}).get(agent)
        if obs is None:
            observation_space = self.env.observation_space(agent)
            if not isinstance(observation_space, spaces.Box):
                raise TypeError("ObservationWrapper requires Box observation spaces")
            observation_shape = observation_space.shape
            if observation_shape is None:
                raise ValueError("Observation space shape must be defined")
            obs = np.zeros(observation_shape, dtype=observation_space.dtype)
        return self._resize_obs(obs)

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
