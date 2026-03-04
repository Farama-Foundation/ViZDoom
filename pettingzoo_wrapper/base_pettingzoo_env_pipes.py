"""
PettingZoo Parallel wrapper for multi-agent ViZDoom — pipe-based IPC.

One process per agent; parent communicates via duplex Pipes.
Commands: "reset", "step", "respawn", "close".
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, List, Optional

import numpy as np
from vizdoom import GameVariable

from pettingzoo_wrapper.base_env_common import VizdoomParallelEnvBase, configure_doom_game
from pettingzoo_wrapper.utils import get_flat_game_vars, parse_hw, read_frame, sync_agent_init

ctx = mp.get_context("spawn")


# ------------------------- child process worker ---------------------------

def _agent_process(
        *,
        pipe_end,
        config_path: str,
        resolution: str,
        timeout: int,
        skip_frames: Optional[int],
        num_agents: int,
        agent_idx: int,
        is_host: bool,
        host_address: str,
        port: int,
        async_mode: bool,
        netmode: int,
        ticrate: int,
        seed: Optional[int],
        verbose: bool,
) -> None:
    agent = "host" if is_host else f"peer{agent_idx}"
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
        agent_idx=agent_idx,
    )

    pipe_end.send({"status": "initializing", "agent": agent})

    try:
        if not is_host:
            import random
            time.sleep(0.5 + random.uniform(0.5, 1.0))
            game.add_game_args("+viz_connect_timeout 45")
        game.init()
        game.send_game_command("viz_respawn_delay 0")
    except Exception as e:
        try:
            pipe_end.send({"status": "init_failed", "error": str(e), "agent": agent})
        except (BrokenPipeError, EOFError):
            pass
        raise

    try:
        pipe_end.send({"status": "ready", "agent": agent})
    except (BrokenPipeError, EOFError):
        return

    max_players = int(game.get_game_variable(GameVariable.USER1))
    if num_agents > max_players:
        raise ValueError(f"Scenario supports {max_players} players, but {num_agents} were requested.")

    available_game_vars = game.get_available_game_variables()
    frames_per_step = skip_frames if skip_frames else 1
    steps = 0

    try:
        while True:
            try:
                cmd, payload = pipe_end.recv()
            except (EOFError, BrokenPipeError):
                break

            if cmd == "reset":
                game.new_episode()
                game.respawn_player()
                state = game.get_state()
                info = {
                    "num_frames": frames_per_step,
                    "player_dead": False,
                    "just_died": False,
                    "step": steps,
                }
                info.update(get_flat_game_vars(state, available_game_vars))
                pipe_end.send({
                    "obs": read_frame(state, resolution),
                    "reward": 0.0,
                    "terminated": False,
                    "info": info,
                })
                steps = 0

            elif cmd == "step":
                action = payload

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
                info = {
                    "num_frames": frames_per_step,
                    "player_dead": is_dead,
                    "just_died": just_died,
                    "step": steps,
                }
                info.update(get_flat_game_vars(state, available_game_vars))
                pipe_end.send({
                    "obs": read_frame(state, resolution),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": terminated,
                    "info": info,
                })
                steps += frames_per_step

            elif cmd == "close":
                break

            else:
                h, w = parse_hw(resolution)
                pipe_end.send(
                    {"obs": np.zeros((h, w, 3), dtype=np.uint8), "reward": 0.0, "terminated": False, "info": {}})
    finally:
        try:
            game.close()
        except Exception:
            pass
        try:
            pipe_end.close()
        except Exception:
            pass


# -------------------------- main PettingZoo env ---------------------------

class VizdoomParallelEnv(VizdoomParallelEnvBase):

    def __init__(self, **kwargs) -> None:
        timeout = kwargs.get("timeout")
        skip_frames = kwargs.get("skip_frames", 1)
        seed = kwargs.get("seed")
        verbose = kwargs.get("verbose", False)
        daemon = kwargs.get("daemon", True)

        super().__init__(**kwargs)

        self._pipes_parent = []
        self._procs: List[ctx.Process] = []

        for i in range(self._num_agents):
            parent_end, child_end = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_agent_process,
                kwargs=dict(
                    pipe_end=child_end,
                    config_path=self.config_file,
                    resolution=self.resolution,
                    timeout=timeout,
                    skip_frames=skip_frames,
                    num_agents=self._num_agents,
                    agent_idx=i,
                    is_host=(i == 0),
                    host_address=self.host_address,
                    port=self.port,
                    async_mode=self.async_mode,
                    netmode=self.netmode,
                    ticrate=self.ticrate,
                    seed=(None if seed is None else int(seed) + i),
                    verbose=verbose,
                ),
                daemon=daemon,
            )
            p.start()
            self._pipes_parent.append(parent_end)
            self._procs.append(p)

        sync_agent_init(self._pipes_parent, self._procs)

        self._frames_advanced = 0
        self._terminations: Dict[str, bool] = {a: False for a in self.agents}
        self._truncations: Dict[str, bool] = {a: False for a in self.agents}

    # ------------- PZ API -------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._ext_seed = int(seed)
        self._frames_advanced = 0
        self._terminations = {a: False for a in self.agents}
        self._truncations = {a: False for a in self.agents}

        for pipe in self._pipes_parent:
            pipe.send(("reset", None))

        results = []
        for i, pipe in enumerate(self._pipes_parent):
            if pipe.poll(timeout=30.0):
                try:
                    results.append(pipe.recv())
                except Exception as e:
                    raise RuntimeError(f"No reset result from agent {i}: {e}")
            else:
                raise TimeoutError(f"Agent {i} reset timeout: 30s")

        obs: Dict[str, np.ndarray] = {}
        infos: Dict[str, Dict[str, Any]] = {}
        for i, agent in enumerate(self.agents):
            frame = results[i]["obs"]
            c = frame.shape[2]
            if c != self._obs_shape[2]:
                self._obs_shape = (self._obs_shape[0], self._obs_shape[1], c)
                from gymnasium import spaces
                self._observation_space = spaces.Box(0, 255, shape=self._obs_shape, dtype=np.uint8)
            obs[agent] = frame
            infos[agent] = results[i].get("info", {})
            self._last_frames[agent] = frame

        return obs, infos

    def step(self, actions: Dict[str, Any]):
        # Encode all actions upfront
        flat_actions: List[List[float]] = []
        for agent in self.agents:
            a = actions.get(agent, self._noop_action())
            env_action = self._encode_env_action(a)
            if len(env_action) != self._act_len:
                raise ValueError(f"Encoded action length {len(env_action)} != expected {self._act_len}")
            flat_actions.append(env_action)

        for i in range(len(self.agents)):
            self._pipes_parent[i].send(("step", flat_actions[i]))

        results = []
        for i, pipe in enumerate(self._pipes_parent):
            if pipe.poll(timeout=30.0):
                try:
                    results.append(pipe.recv())
                except Exception as e:
                    raise RuntimeError(f"No step result from agent {i}: {e}")
            else:
                raise TimeoutError(f"Agent {i} step timeout: 30s")

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for i, agent in enumerate(self.agents):
            r = results[i]
            observations[agent] = r["obs"]
            rewards[agent] = float(r.get("reward", 0.0))
            self._terminations[agent] = bool(r.get("terminated", False))
            infos[agent] = r.get("info", {})
            self._last_frames[agent] = r["obs"]

        frames_advanced = next(iter(infos.values())).get("num_frames", 1) if infos else 1
        self._frames_advanced += int(frames_advanced)
        if self._timeout is not None and self._frames_advanced >= self._timeout:
            for a in self.agents:
                self._truncations[a] = True
                infos[a]["TimeLimit.truncated"] = True

        any_term = any(bool(r.get("terminated", False)) for r in results)
        any_trunc = any(bool(r.get("truncated", False)) for r in results)
        if any_term:
            for a in self.agents:
                self._terminations[a] = True
        if any_trunc:
            for a in self.agents:
                self._truncations[a] = True

        return observations, rewards, self._terminations.copy(), self._truncations.copy(), infos

    def close(self):
        for i, (pipe, proc) in enumerate(zip(self._pipes_parent, self._procs)):
            if pipe is None or pipe.closed:
                continue
            if proc is not None and not proc.is_alive():
                try:
                    pipe.close()
                except Exception:
                    pass
                self._pipes_parent[i] = None
                continue
            try:
                while pipe.poll(timeout=0.05):
                    try:
                        pipe.recv()
                    except (EOFError, BrokenPipeError, OSError):
                        break
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                try:
                    pipe.close()
                except Exception:
                    pass
                self._pipes_parent[i] = None
            except Exception as e:
                print(f"Send close to agent {i} failed: {e}")

        time.sleep(0.5)

        for i, p in enumerate(self._procs):
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

        for i, pipe in enumerate(self._pipes_parent):
            if pipe is None:
                continue
            try:
                pipe.close()
            except Exception:
                pass
            self._pipes_parent[i] = None

        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._screen = None
