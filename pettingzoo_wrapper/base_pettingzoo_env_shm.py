"""
PettingZoo Parallel wrapper for multi-agent ViZDoom — shared-memory IPC.

One process per agent; observations are exchanged via shared memory,
commands/rewards/info via multiprocessing Arrays/Values with a barrier.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from multiprocessing import Process, Event, shared_memory, Array, Value
from typing import Dict, List, Optional

import numpy as np

from pettingzoo_wrapper.base_env_common import VizdoomParallelEnvBase, configure_doom_game
from pettingzoo_wrapper.utils import get_flat_game_vars, read_frame

ctx = mp.get_context("spawn")

_BARRIER_TIMEOUT = 30.0  # seconds to wait for all workers at each barrier
_INIT_TIMEOUT = 90.0  # seconds to wait for each worker to finish game.init()


# ------------------------- helpers ---------------------------

def _write_info_to_mem(info: dict, shared_command: dict) -> None:
    info_bytes = json.dumps(info).encode()
    info_bytes += b'\x00' * (1024 - len(info_bytes))
    shared_command['info'][:] = info_bytes


# ------------------------- child process worker ---------------------------

def agent_process(
        *,
        shared_command,
        step_event,
        all_done_event,
        num_completed,
        shm_name: str,
        obs_shape,
        agent_id: int,
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
    agent = "host" if is_host else f"peer{agent_id}"
    agent_workdir = tempfile.mkdtemp(prefix=f"vizdoom_pz_agent{agent_id}_")
    original_cwd = os.getcwd()
    os.chdir(agent_workdir)

    game = configure_doom_game(
        config_path=config_path,
        resolution=resolution,
        ticrate=ticrate,
        async_mode=async_mode,
        timeout=timeout,
        seed=seed,
        is_host=is_host,
        num_agents=num_agents,
        host_address=host_address,
        port=port,
        netmode=netmode,
        agent_idx=agent_id,
    )

    try:
        game.init()
        game.send_game_command("viz_respawn_delay 0")
    except Exception as e:
        shared_command['init_error'].value = True
        raise

    # Signal to the parent that this worker is ready
    shared_command['ready'].value = True

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    observations = np.ndarray(obs_shape, dtype=np.uint8, buffer=existing_shm.buf)
    available_game_vars = game.get_available_game_variables()

    episodes = 0
    steps = 0
    frames_per_step = skip_frames if skip_frames else 1

    try:
        while True:
            step_event.wait()

            cmd = shared_command['cmd'].value.decode().strip()
            data = list(shared_command['data'][:])

            if cmd == "reset":
                game.new_episode()
                game.respawn_player()
                state = game.get_state()
                observations[agent_id] = read_frame(state, resolution)
                info = {
                    "num_frames": frames_per_step,
                    "player_dead": False,
                    "just_died": False,
                    "step": steps,
                }
                info.update(get_flat_game_vars(state, available_game_vars))
                try:
                    _write_info_to_mem(info, shared_command)
                except Exception as e:
                    _write_info_to_mem({"error": f"reset_info_serialize_failed:{str(e)[:50]}", "agent_id": agent_id},
                                       shared_command)
                shared_command['reward'].value = 0.0
                shared_command['terminated'].value = False
                episodes += 1
                steps = 0

            elif cmd == "step":
                action = data
                is_dead = game.is_player_dead()
                if is_dead:
                    if verbose:
                        print(f"Player {agent} respawning at step {game.get_episode_time()}...")
                    game.respawn_player()
                    reward = 0.0
                else:
                    reward = game.make_action(action, skip_frames)

                was_dead_before = is_dead
                just_died = not was_dead_before and is_dead
                terminated = game.is_episode_finished()
                if verbose and terminated:
                    print(f"Player {agent} terminated at step {game.get_episode_time()}")

                state = game.get_state()
                observations[agent_id] = read_frame(state, resolution)
                info = {
                    "num_frames": frames_per_step,
                    "player_dead": is_dead,
                    "just_died": just_died,
                    "step": steps,
                }
                info.update(get_flat_game_vars(state, available_game_vars))
                try:
                    _write_info_to_mem(info, shared_command)
                except Exception as e:
                    _write_info_to_mem({"error": f"step_info_serialize_failed:{str(e)[:50]}", "agent_id": agent_id},
                                       shared_command)
                shared_command['reward'].value = reward
                shared_command['terminated'].value = terminated
                steps += frames_per_step

            elif cmd == "close":
                game.close()
                existing_shm.close()
                break

            else:
                print(f"Unknown command: {cmd}")

            with num_completed.get_lock():
                num_completed.value += 1
                if num_completed.value == num_agents:
                    all_done_event.set()

    finally:
        try:
            game.close()
        except Exception:
            pass
        try:
            existing_shm.close()
        except Exception:
            pass
        os.chdir(original_cwd)
        shutil.rmtree(agent_workdir, ignore_errors=True)


# -------------------------- main PettingZoo env ---------------------------

class VizdoomParallelEnv(VizdoomParallelEnvBase):

    def __init__(self, **kwargs) -> None:
        timeout = kwargs.get("timeout")
        skip_frames = kwargs.get("skip_frames", 1)
        seed = kwargs.get("seed")
        verbose = kwargs.get("verbose", False)
        daemon = kwargs.get("daemon", True)

        super().__init__(**kwargs)

        multi_obs_shape = (self._num_agents, *self._obs_shape)
        obs_size = int(np.prod(multi_obs_shape)) * np.dtype(np.uint8).itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=obs_size)
        self._shm_observations = np.ndarray(multi_obs_shape, dtype=np.uint8, buffer=self.shm.buf)

        self.step_event = Event()
        self.all_done_event = Event()
        self.num_completed = Value('i', 0)

        self.shared_commands: List[dict] = []
        self.processes: List[Process] = []

        for agent_id in range(self._num_agents):
            shared_command = {
                'cmd': Array('c', 10),
                'data': Array('d', self._act_len),
                'reward': Value('d', 0.0),
                'terminated': Value('b', False),
                'info': Array('c', 1024),
                'ready': Value('b', False),  # set by worker after game.init()
                'init_error': Value('b', False),  # set by worker on init failure
            }
            self.shared_commands.append(shared_command)

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
                    num_agents=self._num_agents,
                    is_host=(agent_id == 0),
                    host_address=self.host_address,
                    port=self.port,
                    async_mode=self.async_mode,
                    netmode=self.netmode,
                    ticrate=self.ticrate,
                    seed=(None if seed is None else int(seed) + agent_id),
                    verbose=verbose,
                ),
                daemon=daemon,
            )
            process.start()
            self.processes.append(process)

        self._wait_for_workers_ready()

        self.frames_advanced = 0

    # ------------- init sync -------------

    def _wait_for_workers_ready(self) -> None:
        """Poll shared ready flags until all workers confirm game.init() completed."""
        deadline = time.monotonic() + _INIT_TIMEOUT
        for i, (proc, sc) in enumerate(zip(self.processes, self.shared_commands)):
            role = "host" if i == 0 else f"peer {i}"
            print(f"Waiting for agent {i} ({role}) to init...")
            while not sc['ready'].value:
                if time.monotonic() > deadline:
                    self._terminate_all()
                    raise TimeoutError(
                        f"Agent {i} init timeout after {_INIT_TIMEOUT}s"
                    )
                if not proc.is_alive():
                    self._terminate_all()
                    raise RuntimeError(f"Agent {i} process died during init")
                if sc['init_error'].value:
                    self._terminate_all()
                    raise RuntimeError(f"Agent {i} reported an init error")
                time.sleep(0.1)
            print(f"Agent {i} ({role}) ready")

    # ------------- barrier -------------

    def _barrier(self) -> None:
        with self.num_completed.get_lock():
            self.num_completed.value = 0
        self.step_event.set()
        completed = self.all_done_event.wait(timeout=_BARRIER_TIMEOUT)
        self.all_done_event.clear()
        self.step_event.clear()
        if not completed:
            dead = [i for i, p in enumerate(self.processes) if not p.is_alive()]
            raise TimeoutError(
                f"Barrier timeout after {_BARRIER_TIMEOUT}s. Dead agents: {dead}"
            )

    # ------------- PZ API -------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._ext_seed = int(seed)
        self.frames_advanced = 0

        for i in range(self._num_agents):
            self.shared_commands[i]['cmd'].value = b'reset'
        self._barrier()

        obs = {a: self._shm_observations[i].copy() for i, a in enumerate(self.agents)}
        self._last_frames = dict(obs)

        infos: Dict[str, Dict] = {}
        for i, a in enumerate(self.agents):
            info_bytes = bytes(self.shared_commands[i]['info'][:])
            infos[a] = json.loads(info_bytes.decode().strip('\x00'))

        return obs, infos

    def step(self, actions):
        flat_actions: List[List[int]] = []
        for agent in self.agents:
            a = actions.get(agent, self._noop_action())
            env_action = self._encode_env_action(a)
            if len(env_action) != self._act_len:
                raise ValueError(f"Encoded action length {len(env_action)} != expected {self._act_len}")
            flat_actions.append(env_action)

        for i, action in enumerate(flat_actions):
            sc = self.shared_commands[i]
            sc['cmd'].value = b'step'
            sc['data'][:] = action
        self._barrier()

        obs = {a: self._shm_observations[i].copy() for i, a in enumerate(self.agents)}
        self._last_frames = dict(obs)

        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        infos: Dict[str, Dict] = {}

        for i, a in enumerate(self.agents):
            sc = self.shared_commands[i]
            rewards[a] = float(sc['reward'].value)
            terminations[a] = bool(sc['terminated'].value)
            info_bytes = bytes(sc['info'][:])
            infos[a] = json.loads(info_bytes.decode().strip('\x00'))

        if any(terminations.values()):
            for a in self.agents:
                terminations[a] = True
        truncations = terminations.copy()

        return obs, rewards, terminations, truncations, infos

    # ------------- cleanup -------------

    def _terminate_all(self) -> None:
        """Best-effort termination of all worker processes."""
        for p in self.processes:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in self.processes:
            try:
                if p.is_alive():
                    p.kill()
            except Exception:
                pass

    def close(self) -> None:
        for sc in self.shared_commands:
            sc['cmd'].value = b'close'
        self.step_event.set()

        time.sleep(0.5)

        for i, p in enumerate(self.processes):
            try:
                p.join(timeout=2.0)
                if p.is_alive():
                    print(f"Agent {i} didn't exit, terminating.")
                    p.terminate()
                    p.join(timeout=1.0)
                    if p.is_alive():
                        p.kill()
            except Exception as e:
                print(f"Join agent {i} failed: {e}")

        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._screen = None

        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            pass
