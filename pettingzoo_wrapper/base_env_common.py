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


# Button pairs that cancel out when pressed together
# Player has 16 unit radius, so at the duel 632 unit spawn means 2*atan(16/632) = 2.9 deg
# Binary TURN_LEFT/TURN_RIGHT turns a fixed 7.03 degrees per action at skip_frames=4,
# which is more than 2 times wider than the target
CONFLICT_BUTTONS = (
    (vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD),
    (vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT),
    (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT),
    (vzd.Button.MOVE_UP, vzd.Button.MOVE_DOWN),
    (vzd.Button.LOOK_UP, vzd.Button.LOOK_DOWN),
)

ANGLE_DELTA_BUTTONS = (
    vzd.Button.TURN_LEFT_RIGHT_DELTA,
    vzd.Button.LOOK_UP_DOWN_DELTA,
)

_DELTA_SUPERSEDES = {
    vzd.Button.TURN_LEFT_RIGHT_DELTA: (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT),
    vzd.Button.LOOK_UP_DOWN_DELTA: (vzd.Button.LOOK_UP, vzd.Button.LOOK_DOWN),
}
DEFAULT_ANGLE_DEGREES_PER_ACTION = (2.0, 6.0, 18.0, 50.0)


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
            "+sv_forcerespawn 1 +viz_respawn_delay 0 "
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
        factored_actions: bool = True,
        angle_degrees_per_action: tuple = DEFAULT_ANGLE_DEGREES_PER_ACTION,
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
        self.factored_actions = bool(factored_actions)
        self._angle_degrees_per_action = tuple(angle_degrees_per_action)
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
        self._simple_n = (3**self._delta_count) * (2**self._binary_count)
        self._act_len = self._delta_count + self._binary_count
        self._factors = self._build_factors() if self.factored_actions else []
        self._action_space = self._build_action_space()

        w, h = parse_hw(resolution)
        self._obs_shape = (h, w, 3)
        self._observation_space = spaces.Box(
            0, 255, shape=self._obs_shape, dtype=np.uint8
        )

        self._last_frames: dict[str, np.ndarray] = {}
        self._screen: pygame.Surface | None = None

    # ------------- space helpers -------------

    def _build_factors(self) -> list[list[dict[int, float]]]:
        """Group buttons into independent choices, one per physical axis.
        Conflicting combo like MOVE_LEFT+MOVE_RIGHT can't be pressed at once.
        - Single buttons become 2-way
        - Angle delta buttons get magnitude incr. in degrees
        - Other delta buttons become 3-way over {0, -1, +1}
        """
        remaining = set(self._binary_indices)
        by_button = {button: i for i, button in enumerate(self.available_buttons)}
        factors: list[list[dict[int, float]]] = []

        # Avoid the both delta and discrete declared together
        for delta_button, superseded in _DELTA_SUPERSEDES.items():
            if delta_button in by_button:
                clashing = [b.name for b in superseded if b in by_button]
                if clashing:
                    raise ValueError(
                        f"{self.config_file} declares {delta_button.name} together with "
                        f"{', '.join(clashing)}, both drive the same axis and would be "
                        "summed. Remove the binary buttons and keep the delta or vice versa."
                    )

        for first, second in CONFLICT_BUTTONS:
            i, j = by_button.get(first), by_button.get(second)
            if i in remaining and j in remaining:
                remaining -= {i, j}
                factors.append([{}, {i: 1.0}, {j: 1.0}])

        # Keep button order
        for i in sorted(remaining):
            factors.append([{}, {i: 1.0}])

        # delta button applies its value once per tic
        tics = max(1, int(self._skip_frames or 1))
        for i in self._delta_indices:
            if self.available_buttons[i] in ANGLE_DELTA_BUTTONS:
                magnitudes = [float(d) for d in self._angle_degrees_per_action if d > 0]
                if not magnitudes:
                    raise ValueError(
                        "angle_degrees_per_action must contain a positive value"
                    )
                magnitudes.sort()
                # Lower option index always means "further clockwise"
                degrees = [-d for d in reversed(magnitudes)] + [0.0] + magnitudes
                # Positive delta value decrease ANGLE (turns right)
                # so we need to negate to make label match the rotation
                factors.append([{i: -d / tics} for d in degrees])
            else:
                factors.append([{i: 0.0}, {i: -1.0}, {i: +1.0}])

        if not factors:
            raise ValueError(
                f"factored_actions requires at least one available button,"
                f"got {self.config_file}"
            )
        return factors

    def factor_description(self) -> list[str]:
        """Label per factor, for logs. Same order as factor_sizes."""
        labels = []
        for options in self._factors:
            touched = sorted({i for option in options for i in option})
            names = "+".join(self.available_buttons[i].name for i in touched) or "noop"
            if (
                len(touched) == 1
                and self.available_buttons[touched[0]] in ANGLE_DELTA_BUTTONS
            ):
                tics = max(1, int(self._skip_frames or 1))
                # Negate to match _build_factors
                degrees = [
                    round(-next(iter(option.values())) * tics, 2) for option in options
                ]
                names = f"{names} yaw_deg/action={degrees}"
            labels.append(f"{names} (n={len(options)})")
        return labels

    @property
    def neutral_factor_indices(self) -> list[int]:
        """
        Option index that presses nothing/rotates by zero per factor.
        """
        indices = []
        for factor_index, options in enumerate(self._factors):
            neutral = next(
                (
                    option_index
                    for option_index, option in enumerate(options)
                    if all(value == 0.0 for value in option.values())
                ),
                None,
            )
            if neutral is None:
                raise ValueError(
                    f"factor {factor_index} has no neutral option, cannot build a noop"
                )
            indices.append(neutral)
        return indices

    @property
    def factor_sizes(self) -> list[int]:
        """nvec of the factored action space. Empty unless factored_actions."""
        return [len(options) for options in self._factors]

    def _decode_factored(self, agent_action) -> list[float]:
        out = np.zeros((self._act_len,), dtype=np.float32)
        indices = np.asarray(agent_action, dtype=np.int64).reshape(-1)
        if indices.size != len(self._factors):
            raise ValueError(
                f"expected {len(self._factors)} factor indices, got {indices.size}"
            )
        for factor, choice in zip(self._factors, indices):
            if not 0 <= int(choice) < len(factor):
                raise ValueError(
                    f"factor index {int(choice)} out of range for size {len(factor)}"
                )
            for button_index, value in factor[int(choice)].items():
                out[button_index] = value
        return out.tolist()

    def _build_action_space(self) -> spaces.Space:
        if self.factored_actions:
            return spaces.MultiDiscrete(self.factor_sizes)
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
        if self.factored_actions:
            return np.asarray(self.neutral_factor_indices, dtype=np.int64)
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
        """Decode a Discrete index in the button order configured by ViZDoom.

        Deltas are radix-3 mapped {0,1,2} -> {0,-1,+1}; binaries are radix-2.
        """
        out = np.zeros((self._act_len,), dtype=np.float32)
        x = int(idx)
        for button_index in self._binary_indices:
            out[button_index] = float(x & 1)
            x >>= 1
        for button_index in self._delta_indices:
            out[button_index] = float([0, -1, +1][x % 3])
            x //= 3
        return out.tolist()

    def _encode_env_action(self, agent_action: Any) -> list[float]:
        if self.factored_actions:
            return self._decode_factored(agent_action)
        if self.simple_discrete:
            raw_action = np.asarray(agent_action)
            if raw_action.ndim == 1 and raw_action.size == self._act_len:
                return raw_action.astype(np.float32).tolist()
            return self._decode_simple_discrete(int(agent_action))
        out = np.zeros((self._act_len,), dtype=np.float32)
        if isinstance(self._action_space, spaces.Dict):
            if self._delta_count:
                cont = (
                    agent_action["continuous"]
                    if isinstance(agent_action, dict)
                    else agent_action[0]
                )
                out[self._delta_indices] = np.asarray(cont, dtype=np.float32)
            if self._binary_count:
                bin_act = (
                    agent_action["binary"]
                    if isinstance(agent_action, dict)
                    else agent_action[1]
                )
                out[self._binary_indices] = np.asarray(
                    bin_act, dtype=np.float32
                ).reshape(-1)
        else:
            if self._delta_count:
                out[self._delta_indices] = np.asarray(agent_action, dtype=np.float32)
            else:
                out[self._binary_indices] = np.asarray(agent_action, dtype=np.float32)
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
