"""
Shared base class and DoomGame configuration helper for ViZDoom PettingZoo envs.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv

import vizdoom as vzd
from vizdoom.pettingzoo_wrapper.utils import (
    discover_buttons,
    get_screen_resolution,
    parse_hw,
)


# Policy-facing limit for raw delta actions.  The value is expressed in the
# native ViZDoom units (degrees for view-angle deltas); the value itself is
# passed through to ViZDoom without quantisation.
RAW_DELTA_ACTION_LIMIT = 180.0

# Respawn delay in seconds (viz_respawn_delay is scaled by ticrate)
# Export so bot eval can use the same (in BotEvalConfig.respawn_delay)
TRAINING_RESPAWN_DELAY = 0


def encode_env_action(
    agent_action: Any, available_buttons: tuple[vzd.Button, ...] | list[vzd.Button]
) -> list[float]:
    """Discretize actions for those with buttons, otherwise the action distribution becomes weird"""
    raw = np.asarray(agent_action, dtype=np.float32).reshape(-1)
    if not any(vzd.is_delta_button(button) for button in available_buttons):
        if raw.size != 1:
            raise ValueError(f"expected one discrete action, got {raw.size} values")
        action = int(raw.item())
        action_count = 1 << len(available_buttons)
        if action != raw.item() or not 0 <= action < action_count:
            raise ValueError(f"discrete action must be in [0, {action_count})")
        return [float((action >> index) & 1) for index in range(len(available_buttons))]
    if raw.size != len(available_buttons):
        raise ValueError(
            f"expected {len(available_buttons)} raw button values, got {raw.size}"
        )
    encoded = raw.copy()
    binary_indices = [
        index
        for index, button in enumerate(available_buttons)
        if not vzd.is_delta_button(button)
    ]
    if binary_indices:
        encoded[binary_indices] = (raw[binary_indices] >= 0.5).astype(np.float32)
    return encoded.tolist()


def configure_doom_game(
    *,
    config_path: str,
    resolution: str,
    ticrate: int,
    async_mode: bool,
    timeout: int | None,
    seed: int | None,
    is_host: bool,
    num_agents: int,
    host_address: str,
    port: int,
    netmode: int,
    agent_idx: int,
) -> vzd.DoomGame:
    """
    Create and configure a DoomGame instance without calling game.init().
    Each worker is responsible for calling game.init() so it can add its own
    error handling and IPC-specific logic around the blocking network call.
    """
    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_window_visible(False)
    game.set_screen_resolution(get_screen_resolution(resolution))
    game.set_ticrate(ticrate)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if async_mode else vzd.Mode.PLAYER)
    if timeout is not None:
        game.set_episode_timeout(timeout)
    if seed is not None:
        game.set_seed(int(seed))
    if is_host:
        game.add_game_args(
            f"-host {num_agents} -port {port} -netmode {netmode} "
            "+timelimit 0 +sv_noautoaim 1 +sv_nocrouch 1 +sv_nofreelook 1 "
            f"+sv_forcerespawn 1 +viz_respawn_delay {TRAINING_RESPAWN_DELAY} "
            "+viz_connect_timeout 60"
        )
    else:
        game.add_game_args(
            f"-join {host_address}:{port} -netmode {netmode} +viz_connect_timeout 60"
        )
    game.add_game_args(f"+name Player{agent_idx} +colorset {agent_idx}")
    return game


class VizdoomParallelEnvBase(ParallelEnv):
    """
    Abstract base for multi-agent ViZDoom PettingZoo environments.

    Handles everything that does not depend on the IPC mechanism:
    spaces, action encoding/decoding, and rendering. Subclasses implement
    reset(), step(), and close() with their chosen transport.
    """

    def __init__(
        self,
        *,
        config_file: str,
        num_agents: int = 2,
        resolution: str = "160X120",
        timeout: int | None = None,
        skip_frames: int | None = 1,
        frame_stack: int = 1,
        async_mode: bool = False,
        host_address: str = "127.0.0.1",
        port: int = 5029,
        netmode: int = 0,
        ticrate: int = vzd.DEFAULT_TICRATE,
        render_mode: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
        daemon: bool = True,
    ) -> None:
        assert num_agents >= 1
        self.config_file = config_file
        self._num_agents = num_agents
        self.host_address = host_address
        self.port = int(port)
        self.resolution = resolution
        self.netmode = int(netmode)
        self.async_mode = bool(async_mode)
        self.ticrate = int(ticrate)
        self._timeout = int(timeout) if timeout is not None else None
        self._skip_frames = skip_frames
        self.frame_stack = int(frame_stack)
        self.render_mode = render_mode
        self._ext_seed = seed
        self.verbose = verbose
        self.daemon = daemon

        self.possible_agents: list[str] = [
            f"agent_{i}" for i in range(self._num_agents)
        ]
        self.agents: list[str] = self.possible_agents[:]

        self.available_buttons = discover_buttons(config_file)
        self._delta_indices = [
            i
            for i, button in enumerate(self.available_buttons)
            if vzd.is_delta_button(button)
        ]
        self._binary_indices = [
            i
            for i, button in enumerate(self.available_buttons)
            if not vzd.is_delta_button(button)
        ]
        self._delta_count = len(self._delta_indices)
        self._binary_count = len(self._binary_indices)
        self._act_len = self._delta_count + self._binary_count
        self._action_space = (
            self._continuous_space()
            if self._delta_count
            else spaces.Discrete(1 << self._binary_count)
        )

        w, h = parse_hw(resolution)
        self._raw_obs_shape = (h, w, 3)
        self._obs_shape = (h, w, 3 * self.frame_stack)
        self._observation_space = spaces.Box(
            0, 255, shape=self._obs_shape, dtype=np.uint8
        )

        self._last_frames: dict[str, np.ndarray] = {}
        self._frame_history: dict[str, list[np.ndarray]] = {}
        self._screen: pygame.Surface | None = None

    # ------------- space helpers -------------

    def _continuous_space(self) -> spaces.Space:
        low = np.zeros((self._act_len,), dtype=np.float32)
        high = np.ones((self._act_len,), dtype=np.float32)
        for button_index in self._delta_indices:
            low[button_index] = -RAW_DELTA_ACTION_LIMIT
            high[button_index] = RAW_DELTA_ACTION_LIMIT
        return spaces.Box(low, high, dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    @property
    def state_space(self) -> spaces.Space:
        return spaces.Box(
            0,
            255,
            shape=(*self._obs_shape[:2], self._obs_shape[2] * self.num_agents),
            dtype=np.uint8,
        )

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def state_observation(self, agent: str) -> np.ndarray:
        obs = self._last_frames.get(agent)
        if obs is None:
            return np.zeros(self._obs_shape, dtype=np.uint8)
        return obs

    def state(self) -> np.ndarray:
        return np.concatenate(
            [self.state_observation(agent) for agent in self.possible_agents],
            axis=-1,
        )

    # ------------- action encoding -------------

    def _noop_action(self):
        """Build a valid 'do nothing' action matching self._action_space."""
        if isinstance(self._action_space, spaces.Discrete):
            return 0
        return np.zeros((self._act_len,), dtype=np.float32)

    def _encode_env_action(self, agent_action: Any) -> list[float]:
        return encode_env_action(agent_action, self.available_buttons)

    def _stack_observations(
        self, observations: dict[str, np.ndarray], *, reset: bool
    ) -> dict[str, np.ndarray]:
        stacked = {}
        for agent, frame in observations.items():
            if reset or agent not in self._frame_history:
                self._frame_history[agent] = [frame] * self.frame_stack
            else:
                self._frame_history[agent] = [
                    *self._frame_history[agent][1:],
                    frame,
                ]
            stacked[agent] = np.concatenate(self._frame_history[agent], axis=-1)
        self._last_frames = dict(stacked)
        return stacked

    # ------------------- rendering -------------------

    def render(self) -> np.ndarray | None:
        if self.render_mode is None or not self._last_frames:
            return None
        frames = [self._last_frames[a] for a in self.agents if a in self._last_frames]
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        cols = min(self._num_agents, max(1, 1920 // w))
        rows = math.ceil(self._num_agents / cols)
        total_w, total_h = cols * w, rows * h

        if self.render_mode == "human":
            if self._screen is None or self._screen.get_size() != (total_w, total_h):
                pygame.init()
                self._screen = pygame.display.set_mode((total_w, total_h))
                pygame.display.set_caption("ViZDoom Multi-Agent")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            for i, frame in enumerate(frames):
                col, row = i % cols, i // cols
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self._screen.blit(surf, (col * w, row * h))
            pygame.display.flip()
            time.sleep(0.01)
            return None
        else:  # rgb_array
            ch = frames[0].shape[2]
            canvas = np.zeros((total_h, total_w, ch), dtype=frames[0].dtype)
            for i, frame in enumerate(frames):
                col, row = i % cols, i // cols
                x, y = col * w, row * h
                canvas[y : y + h, x : x + w] = frame
            return canvas
