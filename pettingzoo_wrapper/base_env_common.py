"""
Shared base class and DoomGame configuration helper for ViZDoom PettingZoo envs.
"""
from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv

import vizdoom as vzd
from pettingzoo_wrapper.utils import discover_buttons, get_screen_resolution, parse_hw
from vizdoom import Mode


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
    game.set_screen_resolution(get_screen_resolution(resolution))
    game.set_ticrate(ticrate)
    game.set_mode(Mode.ASYNC_PLAYER if async_mode else Mode.PLAYER)
    if timeout is not None:
        game.set_episode_timeout(timeout)
    if seed is not None:
        game.set_seed(int(seed))
    if is_host:
        game.add_game_args(
            f"-host {num_agents} -port {port} -netmode {netmode} "
            "+timelimit 0 +sv_noautoaim 1 +sv_nocrouch 1 +sv_nofreelook 1 "
            "+sv_spawnfarthest 1 +sv_forcerespawn 1 +viz_respawn_delay 0 "
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
        async_mode: bool = False,
        host_address: str = "127.0.0.1",
        port: int = 5029,
        netmode: int = 0,
        ticrate: int = vzd.DEFAULT_TICRATE,
        render_mode: str | None = None,
        use_multi_binary_action_space: bool = False,
        simple_discrete: bool = True,
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
        self.render_mode = render_mode
        self.use_multi_binary_action_space = bool(use_multi_binary_action_space)
        self.simple_discrete = bool(simple_discrete)
        self._ext_seed = seed
        self.verbose = verbose
        self.daemon = daemon

        self.possible_agents: list[str] = [
            f"agent_{i}" for i in range(self._num_agents)
        ]
        self.agents: list[str] = self.possible_agents[:]

        self._delta_count, self._binary_count = discover_buttons(config_file)
        self._simple_n = (3**self._delta_count) * (2**self._binary_count)
        self._act_len = self._delta_count + self._binary_count
        self._action_space = self._build_action_space()

        w, h = parse_hw(resolution)
        self._obs_shape = (h, w, 3)
        self._observation_space = spaces.Box(
            0, 255, shape=self._obs_shape, dtype=np.uint8
        )

        self._last_frames: dict[str, np.ndarray] = {}
        self._screen: pygame.Surface | None = None

    # ------------- space helpers -------------

    def _build_action_space(self) -> spaces.Space:
        if self.simple_discrete:
            return spaces.Discrete(max(1, self._simple_n))
        if self._delta_count == 0:
            return self._binary_space()
        if self._binary_count == 0:
            return self._continuous_space()
        return spaces.Dict(
            {
                "binary": self._binary_space(),
                "continuous": self._continuous_space(),
            }
        )

    def _binary_space(self) -> spaces.Space:
        if self.use_multi_binary_action_space:
            return spaces.MultiBinary(self._binary_count)
        return spaces.MultiDiscrete([2] * self._binary_count)

    def _continuous_space(self) -> spaces.Space:
        low, high = np.finfo(np.float32).min, np.finfo(np.float32).max
        return spaces.Box(low, high, (self._delta_count,), dtype=np.float32)

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
        if isinstance(self._action_space, spaces.Dict):
            out = {}
            if self._delta_count:
                out["continuous"] = np.zeros((self._delta_count,), dtype=np.float32)
            if self._binary_count:
                dtype = np.int8 if self.use_multi_binary_action_space else np.int64
                out["binary"] = np.zeros((self._binary_count,), dtype=dtype)
            return out
        if isinstance(self._action_space, spaces.Discrete):
            return 0
        if isinstance(self._action_space, spaces.MultiBinary):
            return np.zeros((self._binary_count,), dtype=np.int8)
        if isinstance(self._action_space, spaces.MultiDiscrete):
            return np.zeros((self._binary_count,), dtype=np.int64)
        if isinstance(self._action_space, spaces.Box):
            return np.zeros((self._delta_count,), dtype=np.float32)
        raise NotImplementedError(type(self._action_space))

    def _decode_simple_discrete(self, idx: int) -> list[float]:
        """Decode a Discrete index -> flat [delta..., binary...] list for ViZDoom.

        Deltas are radix-3 mapped {0,1,2} -> {-1,0,+1}; binaries are radix-2.
        """
        D, B = self._delta_count, self._binary_count
        out = np.zeros((self._act_len,), dtype=np.float32)
        x = int(idx)
        for i in range(B):
            out[D + i] = float(x & 1)
            x >>= 1
        for i in range(D):
            out[i] = float([-1, 0, +1][x % 3])
            x //= 3
        return out.tolist()

    def _encode_env_action(self, agent_action: Any) -> list[float]:
        if self.simple_discrete:
            return self._decode_simple_discrete(int(agent_action))
        out = np.zeros((self._act_len,), dtype=np.float32)
        if isinstance(self._action_space, spaces.Dict):
            if self._delta_count:
                cont = (
                    agent_action["continuous"]
                    if isinstance(agent_action, dict)
                    else agent_action[0]
                )
                out[: self._delta_count] = np.asarray(cont, dtype=np.float32)
            if self._binary_count:
                bin_act = (
                    agent_action["binary"]
                    if isinstance(agent_action, dict)
                    else agent_action[1]
                )
                out[self._delta_count :] = np.asarray(
                    bin_act, dtype=np.float32
                ).reshape(-1)
        else:
            if self._delta_count:
                out[: self._delta_count] = np.asarray(agent_action, dtype=np.float32)
            else:
                out[self._delta_count :] = np.asarray(agent_action, dtype=np.float32)
        return out.tolist()

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
        max_w, max_h = 1920, 1080
        scale = min(
            max_w / total_w if total_w > max_w else 1.0,
            max_h / total_h if total_h > max_h else 1.0,
        )
        sw, sh = int(w * scale), int(h * scale)
        disp_w = min(cols * sw, max_w)
        disp_h = min(rows * sh, max_h)

        if self.render_mode == "human":
            if self._screen is None or self._screen.get_size() != (disp_w, disp_h):
                pygame.init()
                self._screen = pygame.display.set_mode((disp_w, disp_h))
                pygame.display.set_caption("ViZDoom Multi-Agent")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            for i, frame in enumerate(frames):
                if scale < 1.0:
                    frame = cv2.resize(frame, (sw, sh))
                col, row = i % cols, i // cols
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self._screen.blit(surf, (col * sw, row * sh))
            pygame.display.flip()
            time.sleep(0.01)
            return None
        else:  # rgb_array
            ch = frames[0].shape[2]
            canvas = np.zeros((disp_h, disp_w, ch), dtype=frames[0].dtype)
            for i, frame in enumerate(frames):
                if scale < 1.0:
                    frame = cv2.resize(frame, (sw, sh))
                col, row = i % cols, i // cols
                x, y = col * sw, row * sh
                canvas[y : y + sh, x : x + sw] = frame[:sh, :sw]
            return canvas
