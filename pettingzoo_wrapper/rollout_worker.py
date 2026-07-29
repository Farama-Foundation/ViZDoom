from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.utils import ExplorationType, set_exploration_type


_ENV_INSTANCE_SEED_STRIDE = 1_000
_RESTART_SEED_STRIDE = 100_000
_MAX_ENV_INSTANCE_RECREATE_ATTEMPTS = 1
_MAX_FAILED_ROUNDS = 3


class _EnvInstance:
    def __init__(
        self,
        env_builder: Callable[[int, int], Any],
        seed: int,
        env_instance_index: int,
    ):
        self._env_builder = env_builder
        self._base_seed = int(seed)
        self.env_instance_index = int(env_instance_index)
        self.total_restarts = 0
        self.consecutive_failures = 0
        self.last_seed = self._base_seed
        self.last_error = "none"
        self.env: Any | None = None
        self.agent_names: list[str] = []
        self.n_agents = 0
        self._obs: dict[str, np.ndarray] = {}
        self._episode_reward: np.ndarray | None = None
        self._recreate_env()

    def _format_error(self, exc: BaseException | str) -> str:
        if isinstance(exc, BaseException):
            return f"{type(exc).__name__}: {exc}"
        return str(exc)

    def _recreate_env(self) -> None:
        self.last_seed = self._base_seed + self.total_restarts * _RESTART_SEED_STRIDE
        env = self._env_builder(self.last_seed, self.env_instance_index)
        self.env = env
        self.agent_names = list(env.possible_agents)
        self.n_agents = len(self.agent_names)
        self._reset()

    def _reset(self) -> None:
        if self.env is None:
            raise RuntimeError(
                "RolloutWorker env instance reset called before env creation"
            )
        self._obs, _ = self.env.reset()
        self._episode_reward = np.zeros((self.n_agents, 1), dtype=np.float32)

    def ensure_ready(self) -> None:
        if self.env is None or self._episode_reward is None or not self._obs:
            self.restart("env instance was not ready for collection")

    def current_observation(self) -> np.ndarray:
        self.ensure_ready()
        return np.stack([self._obs[agent] for agent in self.agent_names], axis=0)

    def current_state(self) -> np.ndarray:
        self.ensure_ready()
        if self.env is not None and hasattr(self.env, "state"):
            return np.asarray(self.env.state())
        return self.current_observation()

    def step(self, actions):
        self.ensure_ready()
        if self.env is None or self._episode_reward is None:
            raise RuntimeError(
                "RolloutWorker env instance step called before initialization"
            )

        action_dict = {
            agent: actions[index] for index, agent in enumerate(self.agent_names)
        }
        next_obs, rewards, terminations, truncations, _ = self.env.step(action_dict)

        reward = np.asarray(
            [float(rewards[agent]) for agent in self.agent_names], dtype=np.float32
        ).reshape(self.n_agents, 1)
        self._episode_reward += reward
        episode_reward = self._episode_reward.copy()
        terminated = np.asarray(
            [bool(terminations[agent]) for agent in self.agent_names], dtype=np.bool_
        ).reshape(self.n_agents, 1)
        truncated = np.asarray(
            [bool(truncations[agent]) for agent in self.agent_names], dtype=np.bool_
        ).reshape(self.n_agents, 1)
        next_observation = np.stack(
            [next_obs[agent] for agent in self.agent_names], axis=0
        )
        if self.env is not None and hasattr(self.env, "state"):
            next_state = np.asarray(self.env.state())
        else:
            next_state = next_observation
        done = np.logical_or(terminated, truncated)

        if bool(done.any()):
            self._reset()
        else:
            self._obs = next_obs

        self.consecutive_failures = 0
        self.last_error = "none"
        return (
            next_observation,
            next_state,
            reward,
            episode_reward,
            done,
            terminated,
            truncated,
        )

    def restart(self, reason: BaseException | str) -> None:
        self.consecutive_failures += 1
        self.last_error = self._format_error(reason)
        last_exc: BaseException | None = None
        delay_sec = min(1.0, 0.25 * self.consecutive_failures)
        self.close()
        self.env = None
        self._obs = {}
        self._episode_reward = None

        for _ in range(_MAX_ENV_INSTANCE_RECREATE_ATTEMPTS):
            self.total_restarts += 1
            try:
                if delay_sec > 0:
                    time.sleep(delay_sec)
                self._recreate_env()
                return
            except Exception as exc:
                last_exc = exc
                self.last_error = self._format_error(exc)
                self.close()
                self.env = None
                self._obs = {}
                self._episode_reward = None

        raise RuntimeError(
            f"RolloutWorker env instance {self.env_instance_index} failed to recover after "
            f"{_MAX_ENV_INSTANCE_RECREATE_ATTEMPTS} recreate attempts ({self.last_error})"
        ) from last_exc

    def debug_status(self) -> dict[str, Any]:
        if self.env is None:
            return {
                "env_instance_index": self.env_instance_index,
                "seed": self.last_seed,
                "env": "missing",
            }
        debug_status = getattr(self.env, "debug_status", None)
        if callable(debug_status):
            try:
                return debug_status()
            except Exception as exc:
                return {
                    "env_instance_index": self.env_instance_index,
                    "seed": self.last_seed,
                    "debug_status_error": self._format_error(exc),
                }
        return {
            "env_instance_index": self.env_instance_index,
            "seed": self.last_seed,
            "env_type": type(self.env).__name__,
        }

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


class _RolloutStorage:
    """
    Preallocated rollout buffers, in place, at [env_index, t]
    """

    def __init__(self, num_envs: int, steps_per_env: int) -> None:
        self.num_envs = int(num_envs)
        self.steps_per_env = int(steps_per_env)
        self.fill = [0] * self.num_envs
        self._buffers: dict[str, torch.Tensor] = {}

    def _buffer_for(self, name: str, value: torch.Tensor) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if buffer is None:
            buffer = torch.empty(
                (self.num_envs, self.steps_per_env, *value.shape),
                dtype=value.dtype,
            )
            self._buffers[name] = buffer
        return buffer

    def write(self, env_index: int, fields: dict[str, torch.Tensor]) -> None:
        t = self.fill[env_index]
        if t >= self.steps_per_env:
            raise RuntimeError(f"RolloutStorage overflow for env {env_index} (t={t})")
        for name, value in fields.items():
            self._buffer_for(name, value)[env_index, t].copy_(value)
        self.fill[env_index] = t + 1

    def is_full(self, env_index: int) -> bool:
        return self.fill[env_index] >= self.steps_per_env

    def all_full(self) -> bool:
        return all(fill >= self.steps_per_env for fill in self.fill)

    def total_collected(self) -> int:
        return sum(self.fill)

    def get(self, name: str) -> torch.Tensor | None:
        return self._buffers.get(name)


@dataclass
class _GroupState:
    env_indices: list[int]
    observation: torch.Tensor  # uint8 [k, n_agents, H, W, C]
    state: torch.Tensor | None
    actions: torch.Tensor
    log_prob: torch.Tensor | None
    futures: list[Future] | None
    results: list[tuple[int, Any]] | None


class RolloutWorker:
    def __init__(
        self,
        *,
        env_builder: Callable[[int, int], Any],
        policy: TensorDictModuleBase,
        action_spec,
        group_name: str,
        n_agents: int,
        frames_per_batch: int,
        num_envs: int,
        sampling_device: str | torch.device,
        seed: int,
        parallel_collection: bool = True,
        double_buffer: bool = True,
        collect_state: bool = False,
    ) -> None:
        self.policy = policy
        self.action_spec = action_spec
        self.group_name = group_name
        self.n_agents = int(n_agents)
        self.frames_per_batch = int(frames_per_batch)
        self.num_envs = int(num_envs)
        self.sampling_device = torch.device(sampling_device)
        self.parallel_collection = bool(parallel_collection) and self.num_envs > 1
        self.double_buffer = bool(double_buffer) and self.parallel_collection
        self.collect_state = bool(collect_state)
        try:
            self.policy_device = next(policy.parameters()).device
        except StopIteration:
            self.policy_device = self.sampling_device
        self.last_profile: dict[str, float] = {}
        self._policy_time = 0.0
        self._env_wait_time = 0.0
        self._pinned_buffers: dict[int, torch.Tensor] = {}

        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if self.frames_per_batch < self.num_envs:
            raise ValueError("frames_per_batch must be at least num_envs")
        if self.frames_per_batch % self.num_envs != 0:
            raise ValueError("frames_per_batch must be divisible by num_envs")

        self._step_executor = (
            ThreadPoolExecutor(
                max_workers=self.num_envs,
                thread_name_prefix="vizdoom-rollout-worker",
            )
            if self.parallel_collection
            else None
        )

        num_groups = 2 if self.double_buffer and self.num_envs >= 2 else 1
        self._env_groups: list[list[int]] = [
            list(range(self.num_envs))[group_index::num_groups]
            for group_index in range(num_groups)
        ]

        self._env_instances = self._build_env_instances(
            env_builder=env_builder, seed=seed
        )

    def _build_env_instances(self, *, env_builder, seed):
        if self.num_envs == 1:
            return [
                _EnvInstance(
                    env_builder=env_builder,
                    seed=seed,
                    env_instance_index=0,
                )
            ]

        env_instances: list[_EnvInstance | None] = [None] * self.num_envs
        with ThreadPoolExecutor(
            max_workers=self.num_envs, thread_name_prefix="vizdoom-rollout-worker-init"
        ) as executor:
            futures = {
                executor.submit(
                    _EnvInstance,
                    env_builder=env_builder,
                    seed=seed + env_instance_index * _ENV_INSTANCE_SEED_STRIDE,
                    env_instance_index=env_instance_index,
                ): env_instance_index
                for env_instance_index in range(self.num_envs)
            }
            for future in as_completed(futures):
                env_instance_index = futures[future]
                env_instances[env_instance_index] = future.result()

        return [
            env_instance for env_instance in env_instances if env_instance is not None
        ]

    def __iter__(self):
        return self

    def __next__(self):
        collection_started = time.perf_counter()
        self._policy_time = 0.0
        self._env_wait_time = 0.0
        storage = _RolloutStorage(
            num_envs=self.num_envs,
            steps_per_env=self.frames_per_batch // self.num_envs,
        )

        num_groups = len(self._env_groups)
        states: list[_GroupState | None] = [None] * num_groups
        stalled_iterations = 0
        max_stalled_iterations = _MAX_FAILED_ROUNDS * num_groups
        group_index = 0

        # Launch (inference + submit env steps) for one group, then collect other group env steps in parallel
        while not storage.all_full():
            collected_before = storage.total_collected()
            if states[group_index] is None:
                states[group_index] = self._launch_group(group_index, storage)
            group_index = (group_index + 1) % num_groups
            if states[group_index] is not None:
                self._collect_group(states[group_index], storage)
                states[group_index] = None
            # Only actual transitions written to smh counts
            # Prevent crashed env restart being counted
            if storage.total_collected() > collected_before:
                stalled_iterations = 0
            else:
                stalled_iterations += 1
                if stalled_iterations >= max_stalled_iterations:
                    raise RuntimeError(stalled_iterations)

        batch_assembly_started = time.perf_counter()
        batch = self._finalize_batch(storage)
        batch_assembly_time = time.perf_counter() - batch_assembly_started
        collection_time = time.perf_counter() - collection_started
        self.last_profile = {
            "timers/collector_policy_inference": self._policy_time,
            "timers/collector_env_step": self._env_wait_time,
            "timers/collector_batch_assembly": batch_assembly_time,
            "timers/collector_other": max(
                0.0,
                collection_time
                - self._policy_time
                - self._env_wait_time
                - batch_assembly_time,
            ),
        }
        return batch

    def _launch_group(self, group_index, storage):
        pending = [
            env_index
            for env_index in self._env_groups[group_index]
            if not storage.is_full(env_index)
        ]
        if not pending:
            return None

        ready_env_indices = []
        obs_list = []
        state_list = []
        for env_index in pending:
            env_instance = self._env_instances[env_index]
            try:
                obs = env_instance.current_observation()
                state_obs = env_instance.current_state() if self.collect_state else None
                obs_list.append(obs)
                if state_obs is not None:
                    state_list.append(state_obs)
                ready_env_indices.append(env_index)
            except Exception as exc:
                try:
                    env_instance.restart(exc)
                except Exception:
                    pass
        if not ready_env_indices:
            return None

        observation = torch.from_numpy(np.stack(obs_list, axis=0))
        state = torch.from_numpy(np.stack(state_list, axis=0)) if state_list else None

        policy_started = time.perf_counter()
        policy_obs = self._to_policy_obs(group_index, observation)
        policy_td = TensorDict(
            {
                self.group_name: TensorDict(
                    {"observation": policy_obs},
                    batch_size=[len(ready_env_indices), self.n_agents],
                )
            },
            batch_size=[len(ready_env_indices)],
            device=self.policy_device,
        )
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            policy_td = self.policy(policy_td)
        actions = policy_td.get((self.group_name, "action")).detach().cpu()
        log_prob = policy_td.get((self.group_name, "log_prob"), None)
        if log_prob is not None:
            log_prob = log_prob.detach().cpu()
        self._policy_time += time.perf_counter() - policy_started

        env_actions = self._decode_policy_actions(actions)
        futures: list[Future] | None = None
        results: list[tuple[int, Any]] | None = None
        # go through the executor so double buffer overlap not lost
        if self._step_executor is not None:
            futures = [
                self._step_executor.submit(
                    self._step_env_instance, env_index, env_action
                )
                for env_index, env_action in zip(ready_env_indices, env_actions)
            ]
        else:
            step_started = time.perf_counter()
            results = [
                self._step_env_instance(env_index, env_action)
                for env_index, env_action in zip(ready_env_indices, env_actions)
            ]
            self._env_wait_time += time.perf_counter() - step_started

        return _GroupState(
            env_indices=ready_env_indices,
            observation=observation,
            state=state,
            actions=actions,
            log_prob=log_prob,
            futures=futures,
            results=results,
        )

    def _collect_group(self, state: _GroupState, storage: _RolloutStorage) -> bool:
        if state.futures is not None:
            wait_started = time.perf_counter()
            results = [future.result() for future in state.futures]
            self._env_wait_time += time.perf_counter() - wait_started
        else:
            results = state.results or []

        wrote_any = False
        for offset, (env_index, step_result) in enumerate(results):
            if step_result is None:
                continue
            storage.write(env_index, self._env_fields(state, offset, step_result))
            wrote_any = True
        return wrote_any

    def _env_fields(self, state: _GroupState, offset: int, step_result) -> dict:
        (
            next_obs_np,
            next_state_np,
            reward_np,
            episode_reward_np,
            done_np,
            terminated_np,
            truncated_np,
        ) = step_result
        done = torch.from_numpy(done_np)
        terminated = torch.from_numpy(terminated_np)
        truncated = torch.from_numpy(truncated_np)
        fields: dict[str, torch.Tensor] = {
            "observation": state.observation[offset],
            "action": state.actions[offset],
            "reward": torch.from_numpy(reward_np),
            "episode_reward": torch.from_numpy(episode_reward_np),
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "next_observation": torch.from_numpy(next_obs_np),
        }
        if state.log_prob is not None:
            fields["log_prob"] = state.log_prob[offset]
        if self.collect_state and state.state is not None:
            fields["state"] = state.state[offset]
            fields["next_state"] = torch.from_numpy(next_state_np)
        return fields

    def _to_policy_obs(self, group_index: int, observation) -> torch.Tensor:
        if self.policy_device.type == "cuda":
            capacity = len(self._env_groups[group_index])
            pinned = self._pinned_buffers.get(group_index)
            if (
                pinned is None
                or pinned.shape[0] < capacity
                or pinned.shape[1:] != observation.shape[1:]
            ):
                pinned = torch.empty(
                    (capacity, *observation.shape[1:]),
                    dtype=observation.dtype,
                    pin_memory=True,
                )
                self._pinned_buffers[group_index] = pinned
            k = observation.shape[0]
            pinned[:k].copy_(observation)
            return pinned[:k].to(self.policy_device, non_blocking=True)
        if self.policy_device != observation.device:
            observation = observation.to(self.policy_device)
        return observation

    def _finalize_batch(self, storage: _RolloutStorage) -> TensorDict:
        observation_u8 = storage.get("observation")
        next_observation_u8 = storage.get("next_observation")
        if observation_u8 is None or next_observation_u8 is None:
            raise ValueError("Cannot assemble an empty transition batch")

        # obs is unint8 before going into replay buffer
        observation = observation_u8
        next_observation = next_observation_u8

        action = storage.get("action")
        reward = storage.get("reward")
        episode_reward = storage.get("episode_reward")
        done = storage.get("done")
        terminated = storage.get("terminated")
        truncated = storage.get("truncated")

        def _any_over_agents(flags: torch.Tensor | None) -> torch.Tensor | None:
            if flags is None:
                return None
            return flags.flatten(2).any(2, keepdim=True)

        next_done = _any_over_agents(done)
        next_terminated = _any_over_agents(terminated)
        next_truncated = _any_over_agents(truncated)
        log_prob = storage.get("log_prob")
        state_u8 = storage.get("state")
        next_state_u8 = storage.get("next_state")

        batch_size = [storage.num_envs, storage.steps_per_env]

        group_data = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "episode_reward": episode_reward,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
        if log_prob is not None:
            group_data["log_prob"] = log_prob
        group_td = TensorDict(group_data, batch_size=[*batch_size, self.n_agents])
        next_group_td = TensorDict(
            {
                "observation": next_observation,
                "reward": reward,
                "episode_reward": episode_reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=[*batch_size, self.n_agents],
        )
        next_td = TensorDict(
            {
                self.group_name: next_group_td,
                "done": next_done,
                "terminated": next_terminated,
                "truncated": next_truncated,
            },
            batch_size=batch_size,
        )
        # To satisfy pyright
        assert next_done is not None
        assert next_terminated is not None
        assert next_truncated is not None
        root_data = {
            self.group_name: group_td,
            "done": next_done.clone(),
            "terminated": next_terminated.clone(),
            "truncated": next_truncated.clone(),
            "next": next_td,
        }
        if state_u8 is not None and next_state_u8 is not None:
            root_data["state"] = state_u8.to(torch.float32).div_(255.0)
            next_td["state"] = next_state_u8.to(torch.float32).div_(255.0)
        return TensorDict(root_data, batch_size=batch_size)

    # Other methods
    def update_policy_weights_(self):
        return None

    def state_dict(self) -> dict[str, int]:
        return {"env_instance_cursor": 0}

    def load_state_dict(self, state_dict: dict[str, int]):
        return None

    def shutdown(self):
        for env_instance in self._env_instances:
            env_instance.close()
        if self._step_executor is not None:
            self._step_executor.shutdown(wait=True)

    def _decode_policy_actions(self, actions: torch.Tensor) -> list[list[Any]]:
        action_np = self.action_spec.to_numpy(actions)
        if isinstance(action_np, np.ndarray):
            if action_np.ndim == 3 and action_np.shape[-1] == 1:
                action_np = action_np.squeeze(-1)
            if action_np.ndim == 1:
                action_np = action_np.reshape(1, -1)
            return [
                [
                    int(action_np[env_instance_index, agent_index])
                    for agent_index in range(self.n_agents)
                ]
                for env_instance_index in range(action_np.shape[0])
            ]
        raise TypeError(f"RolloutWorker got {type(action_np)!r}")

    def _step_env_instance(self, env_instance_index: int, env_action: Sequence[Any]):
        env_instance = self._env_instances[env_instance_index]
        try:
            return env_instance_index, env_instance.step(env_action)
        except Exception as exc:
            try:
                env_instance.restart(exc)
            except Exception:
                pass
            return env_instance_index, None
