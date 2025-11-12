"""
PettingZoo Parallel wrapper for multi-agent ViZDoom.

- Spawns one process per agent (host + peers). Communication via duplex Pipes.
- Parent sends simple commands ("reset", "step", "close") to each child and
  receives observations, rewards, terminations, truncations, and infos.

Usage:
    env = VizdoomParallelEnv(
        scenario="health_gathering",
        num_agents=2,
        resolution="160X120",
        skip_frames=4,
        async_mode=True,
        host_address="127.0.0.1",
        port=5029,
        netmode=0,  # 0: p2p, 1: client-server
        seed=0,
    )

    obs, infos = env.reset()
    done = False
    while not any(env._terminations.values()) and not any(env._truncations.values()):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, infos = env.step(actions)
    env.close()
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
from multiprocessing import Process, Event, shared_memory, Array, Value

ctx = mp.get_context("spawn")
import time

from pettingzoo import ParallelEnv
from pettingzoo_wrapper.utils import get_screen_resolution, parse_hw, get_flat_game_vars, read_frame, discover_buttons, \
    sync_agent_init
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces

import vizdoom as vzd
from vizdoom import Mode, GameVariable
import pygame
import cv2


# ------------------------- child process worker ---------------------------

def agent_process(
        *,
        shared_command,
        step_event,
        all_done_event,
        num_completed,
        shm_name,
        obs_shape,
        agent_id,
        config_path: str,
        resolution: str,
        timeout: int,
        skip_frames: Optional[int],
        num_agents: int,
        is_host: bool,
        host_address: str,
        port: int,
        async_mode: bool,
        netmode: int,
        ticrate: int,
        seed: Optional[int],
        verbose: bool,
) -> None:
    game = vzd.DoomGame()
    game.load_config(config_path)

    # headless
    game.set_screen_resolution(get_screen_resolution(resolution))
    game.set_ticrate(ticrate)
    game.set_mode(Mode.ASYNC_PLAYER if async_mode else Mode.PLAYER)

    if timeout is not None:
        game.set_episode_timeout(timeout)
    if seed is not None:
        game.set_seed(int(seed))

    if is_host:
        game.add_game_args(
            f"-host {num_agents} -port {port} -netmode {netmode} +sv_spawnfarthest 1"
        )
        agent = "host"
    else:
        game.add_game_args(f"-join {host_address} -port {port} -netmode {netmode}")
        agent = f"peer{agent_id}"

    # cosmetics / identity
    game.add_game_args(f"+name Player{agent_id} +colorset {agent_id}")
    game.add_game_args(f"+playernumber {agent_id}")

    # synchronize initialization
    game.init()
    game.send_game_command("viz_respawn_delay 0")

    # Connect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    observations = np.ndarray(obs_shape, dtype=np.uint8, buffer=existing_shm.buf)

    # Get available game variables for mapping indices to names
    available_game_vars = game.get_available_game_variables()

    episodes = 0
    steps = 0
    frames_per_step = skip_frames if skip_frames else 1

    try:
        while True:
            # Wait for the step event
            step_event.wait()
            # Do not clear the step_event here, the main process will clear it if necessary

            cmd_bytes = shared_command['cmd'].value
            cmd = cmd_bytes.decode().strip()
            data = list(shared_command['data'][:])

            if cmd == "reset":
                game.new_episode()
                game.respawn_player()

                state = game.get_state()
                frame = read_frame(state, resolution)

                info = {
                    "num_frames": frames_per_step,
                    "player_dead": False,
                    "just_died": False,
                    "step": steps
                }
                info.update(get_flat_game_vars(state, available_game_vars))

                # Write observation into shared memory
                observations[agent_id] = frame

                # Write info JSON into the fixed-size buffer
                try:
                    write_info_to_mem(info, shared_command)
                except Exception as e:
                    info = {"error": f"reset_info_serialize_failed:{str(e)[:50]}", "agent_id": agent_id}
                    write_info_to_mem(info, shared_command)

                # Reset reward/terminated scalars
                shared_command['reward'].value = 0.0
                shared_command['terminated'].value = False

                # Barrier book-keeping
                with num_completed.get_lock():
                    num_completed.value += 1
                    if num_completed.value == num_agents:
                        # Last agent to finish reset
                        all_done_event.set()

                episodes += 1
                steps = 0

            elif cmd == "step":
                action = data

                is_dead = game.is_player_dead()
                if is_dead:
                    if verbose:
                        print(f"Player {agent} respawning at step {game.get_episode_time()}...")
                    game.respawn_player()
                    if verbose:
                        print(f"Player {agent} respawned at step {game.get_episode_time()}")
                    reward = 0.0
                else:
                    reward = game.make_action(action, skip_frames)

                # Check if player died during this step
                was_dead_before = is_dead
                just_died = not was_dead_before and is_dead
                terminated = game.is_episode_finished()

                if verbose and terminated:
                    print(f"Player {agent} terminated at step {game.get_episode_time()}")

                state = game.get_state()
                frame = read_frame(state, resolution)

                info = {
                    "num_frames": frames_per_step,
                    "player_dead": is_dead,
                    "just_died": just_died,
                    "step": steps
                }
                info.update(get_flat_game_vars(state, available_game_vars))

                # Write observation into shared memory
                observations[agent_id] = frame

                # Write info JSON into the fixed-size buffer
                try:
                    write_info_to_mem(info, shared_command)
                except Exception as e:
                    info = {"error": f"step_info_serialize_failed:{str(e)[:50]}", "agent_id": agent_id}
                    write_info_to_mem(info, shared_command)

                # Write results to shared_command
                shared_command['reward'].value = reward
                shared_command['terminated'].value = terminated

                # Increment the shared counter
                with num_completed.get_lock():
                    num_completed.value += 1
                    if num_completed.value == num_agents:
                        # Last agent to finish steps
                        all_done_event.set()

                steps += frames_per_step

            elif cmd == "close":
                game.close()
                existing_shm.close()
                break

            else:
                print(f"Unknown command: {cmd}")
    finally:
        try:
            game.close()
        except Exception:
            pass


def write_info_to_mem(info, shared_command):
    info_json = json.dumps(info)
    info_bytes = info_json.encode()
    info_bytes += b'\x00' * (1024 - len(info_bytes))
    shared_command['info'][:] = info_bytes


# -------------------------- main PettingZoo env ---------------------------

class VizdoomParallelEnv(ParallelEnv):

    def __init__(
            self,
            *,
            config_file: str,
            num_agents: int = 2,
            resolution: str = "160X120",
            timeout: Optional[int] = None,
            skip_frames: Optional[int] = 1,
            async_mode: bool = True,
            host_address: str = "127.0.0.1",
            port: int = 5029,
            netmode: int = 0,
            ticrate: int = vzd.DEFAULT_TICRATE,
            render_mode: Optional[str] = None,
            use_multi_binary_action_space: bool = False,
            simple_discrete: bool = True,
            seed: Optional[int] = None,
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
        self.render_mode = render_mode
        self.use_multi_binary_action_space = bool(use_multi_binary_action_space)
        self.simple_discrete = bool(simple_discrete)
        self.ext_seed = seed

        # names
        self.possible_agents: List[str] = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents: List[str] = self.possible_agents[:]

        # Discover spaces (no net init needed)
        self.delta_count, self._binary_count = discover_buttons(config_file)
        self.simple_n = (3 ** self.delta_count) * (2 ** self._binary_count)
        self.act_len = self.delta_count + self._binary_count
        self._action_space = self._build_action_space()

        # Observation space
        width, height = parse_hw(resolution)
        self.channels = 3  # update on first reset if GRAY8
        self.obs_shape = (height, width, self.channels)
        self._observation_space = spaces.Box(0, 255, shape=self.obs_shape, dtype=np.uint8)

        # Create shared memory for observations
        multi_obs_shape = (self.num_agents, *self.obs_shape)
        obs_size = np.prod(multi_obs_shape) * np.dtype(np.uint8).itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=obs_size)
        self.observations = np.ndarray(multi_obs_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Create shared commands and events for synchronization
        self.shared_commands = []
        self.processes = []

        # Shared synchronization primitives
        self.step_event = Event()
        self.all_done_event = Event()
        self.num_completed = Value('i', 0)

        for agent_id in range(self._num_agents):
            # Shared command dictionary using multiprocessing.Array and Value
            shared_command = {
                'cmd': Array('c', 10),  # Command string, max length 10
                'data': Array('i', self.act_len),  # Array to hold each discrete action
                'reward': Value('d', 0.0),
                'terminated': Value('b', False),
                'info': Array('c', 1024)  # Increased buffer size for JSON info
            }
            self.shared_commands.append(shared_command)

            # Create and start the game process, passing shared memory details
            process = Process(
                target=agent_process,
                kwargs=dict(
                    shared_command=shared_command,
                    step_event=self.step_event,
                    all_done_event=self.all_done_event,
                    num_completed=self.num_completed,
                    shm_name=self.shm.name,
                    obs_shape=multi_obs_shape,
                    agent_id=agent_id,
                    config_path=self.config_file,
                    resolution=self.resolution,
                    timeout=timeout,
                    skip_frames=skip_frames,
                    num_agents=self.num_agents,
                    is_host=(agent_id == 0),
                    host_address=self.host_address,
                    port=self.port,
                    async_mode=self.async_mode,
                    netmode=self.netmode,
                    ticrate=self.ticrate,
                    seed=(None if seed is None else int(seed) + agent_id),
                    verbose=verbose,
                ),
                daemon=daemon, # terminate child if parent dies
            )
            process.start()
            self.processes.append(process)

        # timeout / PZ bookkeeping
        self.frames_advanced = 0
        self.timeout = int(timeout) if timeout is not None else None

        # last frames for rendering
        self.last_frames: Dict[str, np.ndarray] = {}

        # Rendering surface
        self.screen: Optional[pygame.Surface] = None

    def _barrier(self):
        with self.num_completed.get_lock():
            self.num_completed.value = 0
        self.step_event.set()
        self.all_done_event.wait()
        self.all_done_event.clear()
        self.step_event.clear()

    # ------------- space helpers -------------

    def _build_action_space(self) -> spaces.Space:
        if self.simple_discrete:
            return spaces.Discrete(max(1, self.simple_n))
        if self.delta_count == 0:
            return self._binary_space()
        if self._binary_count == 0:
            return self._continuous_space()
        return spaces.Dict({
            "binary": self._binary_space(),
            "continuous": self._continuous_space(),
        })

    def _binary_space(self) -> spaces.Space:
        if self.use_multi_binary_action_space:
            return spaces.MultiBinary(self._binary_count)
        return spaces.MultiDiscrete([2] * self._binary_count)

    def _continuous_space(self) -> spaces.Space:
        low, high = np.finfo(np.float32).min, np.finfo(np.float32).max
        return spaces.Box(low, high, (self.delta_count,), dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    @property
    def num_agents(self) -> int:
        return self._num_agents

    # ------------- PZ API -------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.ext_seed = int(seed)
        self.frames_advanced = 0

        for i in range(self.num_agents):
            self.shared_commands[i]['cmd'].value = b'reset'
        self._barrier()

        observations = {a: self.observations[i].copy() for i, a in enumerate(self.agents)}
        self.last_frames = observations

        # Parse infos from the per-agent buffers
        infos: Dict[str, Dict] = {}
        for i, a in enumerate(self.agents):
            info_bytes = self.shared_commands[i]['info'][:]
            info_str = bytes(info_bytes).decode().strip('\x00')
            info = json.loads(info_str)
            infos[a] = info

        return observations, infos

    def step(self, actions: Dict[str, Any]):
        # 1) Handle dead agents from previous step and encode actions for alive agents
        flat_actions: List[List[int]] = []
        for agent in self.agents:
            a = actions.get(agent, self._noop_action())
            env_action = self._encode_env_action(a)
            if len(env_action) != self.act_len:
                raise ValueError(f"Encoded action length {len(env_action)} != expected {self.act_len}")
            flat_actions.append(env_action)

        # Send step commands
        for i, action in enumerate(flat_actions):
            sc = self.shared_commands[i]
            sc['cmd'].value = b'step'
            if len(action) != len(sc['data']):
                raise ValueError(f"Action tuple length {len(action)} does not match expected {len(sc['data'])}.")
            sc['data'][:] = action

        # 2) Barrier (set -> wait -> clear)
        self._barrier()

        # 3) Gather results
        observations: Dict[str, np.ndarray] = {a: self.observations[i].copy() for i, a in enumerate(self.agents)}
        self.last_frames = observations
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        infos: Dict[str, Dict] = {}

        for i, a in enumerate(self.agents):
            sc = self.shared_commands[i]
            rewards[a] = float(sc['reward'].value)
            terminations[a] = bool(sc['terminated'].value)

            info_bytes = sc['info'][:]
            info_str = bytes(info_bytes).decode().strip('\x00')
            info = json.loads(info_str)
            infos[a] = info

        # If any agent finishes, finish the episode for ALL agents this step.
        if any(terminations.values()):
            for a in self.agents:
                terminations[a] = True
        truncations: Dict[str, bool] = terminations.copy()

        return observations, rewards, terminations, truncations, infos

    def close(self):
        # Set 'close' command for all agents
        for shared_command in self.shared_commands:
            shared_command['cmd'].value = b'close'
        # Signal all agents to proceed
        self.step_event.set()
        # Wait for processes to finish
        for process in self.processes:
            process.join()

        # Clean up pygame resources
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
            self.screen = None

        # Clean up shared memory
        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            # Shared memory already cleaned up, ignore
            pass

    # ------------- helpers -------------
    def _noop_action(self):
        """Build a valid 'do nothing' action matching self._action_space."""
        if isinstance(self._action_space, spaces.Dict):
            out = {}
            if self.delta_count:
                out["continuous"] = np.zeros((self.delta_count,), dtype=np.float32)
            if self._binary_count:
                if self.use_multi_binary_action_space:
                    out["binary"] = np.zeros((self._binary_count,), dtype=np.int8)
                else:
                    out["binary"] = np.zeros((self._binary_count,), dtype=np.int64)
            return out
        else:
            if isinstance(self._action_space, spaces.Discrete):
                return 0
            if isinstance(self._action_space, spaces.MultiBinary):
                return np.zeros((self._binary_count,), dtype=np.int8)
            if isinstance(self._action_space, spaces.MultiDiscrete):
                return np.zeros((self._binary_count,), dtype=np.int64)
            if isinstance(self._action_space, spaces.Box):
                return np.zeros((self.delta_count,), dtype=np.float32)
            raise NotImplementedError(type(self._action_space))

    def _decode_simple_discrete(self, idx: int) -> List[float]:
        """Decode Discrete index -> flat [delta..., binary...] for ViZDoom.

        delta in {-1, 0, +1} (radix-3), then binary in {0,1} (radix-2).
        Order is [delta_0, ..., delta_{D-1}, bin_0, ..., bin_{B-1}]
        """
        D, B = self.delta_count, self._binary_count
        out = np.zeros((self.act_len,), dtype=np.int8)
        x = int(idx)

        # Binary tail (radix-2)
        for i in range(B):
            out[D + i] = int(x & 1)
            x >>= 1

        # Delta head (radix-3) mapped {0,1,2} -> {-1,0,+1}
        for i in range(D):
            digit = x % 3
            out[i] = int([-1, 0, +1][digit])
            x //= 3

        return out.tolist()

    def _encode_env_action(self, agent_action: Any) -> List[float]:
        if self.simple_discrete:
            # agent_action is an integer index from spaces.Discrete
            return self._decode_simple_discrete(int(agent_action))

        # Map user action (matching self._action_space) -> flat vector [delta..., binary...]
        out = np.zeros((self.act_len,), dtype=np.float32)
        if isinstance(self._action_space, spaces.Dict):
            # Dict with continuous and binary
            if self.delta_count:
                cont = agent_action["continuous"] if isinstance(agent_action, dict) else agent_action[0]
                out[: self.delta_count] = np.asarray(cont, dtype=np.float32)
            if self._binary_count:
                bin_act = agent_action["binary"] if isinstance(agent_action, dict) else agent_action[1]
                bin_arr = np.asarray(bin_act, dtype=np.float32)
                out[self.delta_count:] = bin_arr.reshape(-1)
        else:
            if self.delta_count:
                out[: self.delta_count] = np.asarray(agent_action, dtype=np.float32)
            else:
                out[self.delta_count:] = np.asarray(agent_action, dtype=np.float32)
        return out.tolist()

    # ------------------- rendering -------------------
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None or not self.last_frames:
            return None
        frames = [self.last_frames[a] for a in self.agents if a in self.last_frames]
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        cols = min(self._num_agents, max(1, 1920 // w))
        rows = math.ceil(self._num_agents / cols)
        total_w = cols * w
        total_h = rows * h
        max_w, max_h = 1920, 1080
        scale = min(max_w / total_w if total_w > max_w else 1.0, max_h / total_h if total_h > max_h else 1.0)
        sw, sh = int(w * scale), int(h * scale)
        disp_w, disp_h = min(cols * sw, max_w), min(rows * sh, max_h)

        if self.render_mode == "human":
            if self.screen is None or self.screen.get_size() != (disp_w, disp_h):
                pygame.init()
                self.screen = pygame.display.set_mode((disp_w, disp_h))
                pygame.display.set_caption("ViZDoom Multi-Agent")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            for i, frame in enumerate(frames):
                if scale < 1.0:
                    frame = cv2.resize(frame, (sw, sh))
                col = i % cols
                row = i // cols
                x, y = col * sw, row * sh
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.screen.blit(surf, (x, y))
            pygame.display.flip()
            time.sleep(0.01)
            return None
        else:  # rgb_array
            ch = frames[0].shape[2]
            canvas = np.zeros((disp_h, disp_w, ch), dtype=frames[0].dtype)
            for i, frame in enumerate(frames):
                if scale < 1.0:
                    frame = cv2.resize(frame, (sw, sh))
                col = i % cols
                row = i // cols
                x, y = col * sw, row * sh
                canvas[y: y + sh, x: x + sw] = frame[: sh, : sw]
            return canvas


