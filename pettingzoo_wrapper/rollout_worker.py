from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@dataclass
class _Transition:
    observation: torch.Tensor
    state: torch.Tensor | None
    action: torch.Tensor
    log_prob: torch.Tensor | None
    reward: torch.Tensor
    episode_reward: torch.Tensor
    done: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_observation: torch.Tensor
    next_state: torch.Tensor | None
    next_done: torch.Tensor
    next_terminated: torch.Tensor
    next_truncated: torch.Tensor


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
        self.collect_state = bool(collect_state)
        try:
            self.policy_device = next(policy.parameters()).device
        except StopIteration:
            self.policy_device = self.sampling_device
        self._env_instance_cursor = 0
        self._consecutive_failed_rounds = 0
        self.last_profile: dict[str, float] = {}
        self._step_executor = (
            ThreadPoolExecutor(
                max_workers=self.num_envs,
                thread_name_prefix="vizdoom-rollout-worker",
            )
            if self.parallel_collection
            else None
        )

        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if self.frames_per_batch < self.num_envs:
            raise ValueError("frames_per_batch must be at least num_envs")
        if self.frames_per_batch % self.num_envs != 0:
            raise ValueError("frames_per_batch must be divisible by num_envs")

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
        policy_time = 0.0
        env_step_time = 0.0
        frames_per_env = self.frames_per_batch // self.num_envs
        transitions: list[list[_Transition]] = [[] for _ in range(self.num_envs)]
        transitions_collected = 0

        while transitions_collected < self.frames_per_batch:
            transitions_before_round = transitions_collected
            env_instance_indices = [
                env_instance_index
                for env_instance_index in self._next_env_instance_indices(self.num_envs)
                if len(transitions[env_instance_index]) < frames_per_env
            ]

            ready_env_instance_indices = []
            current_obs_np = []
            current_state_np = []
            for env_instance_index in env_instance_indices:
                env_instance = self._env_instances[env_instance_index]
                try:
                    current_obs_np.append(env_instance.current_observation())
                    if self.collect_state:
                        current_state_np.append(env_instance.current_state())
                    ready_env_instance_indices.append(env_instance_index)
                except Exception as exc:
                    try:
                        env_instance.restart(exc)
                    except Exception:
                        pass

            if not ready_env_instance_indices:
                self._register_failed_round()
                continue

            current_obs = (
                torch.from_numpy(np.stack(current_obs_np, axis=0))
                .to(torch.float32)
                .div_(255.0)
            )
            current_state = None
            if self.collect_state:
                current_state = (
                    torch.from_numpy(np.stack(current_state_np, axis=0))
                    .to(torch.float32)
                    .div_(255.0)
                )
            policy_started = time.perf_counter()
            policy_obs = current_obs
            if self.policy_device.type == "cuda":
                policy_obs = policy_obs.pin_memory().to(
                    self.policy_device, non_blocking=True
                )
            elif self.policy_device != current_obs.device:
                policy_obs = policy_obs.to(self.policy_device)

            policy_td = TensorDict(
                {
                    self.group_name: TensorDict(
                        {"observation": policy_obs},
                        batch_size=[len(ready_env_instance_indices), self.n_agents],
                    )
                },
                batch_size=[len(ready_env_instance_indices)],
                device=self.policy_device,
            )

            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                policy_td = self.policy(policy_td)

            actions = policy_td.get((self.group_name, "action")).detach().cpu()
            log_prob = policy_td.get((self.group_name, "log_prob"), None)
            if log_prob is not None:
                log_prob = log_prob.detach().cpu()
            policy_time += time.perf_counter() - policy_started
            current_obs = current_obs.cpu()
            env_actions = self._decode_policy_actions(actions)
            ready_env_instance_offsets = {
                env_instance_index: offset
                for offset, env_instance_index in enumerate(ready_env_instance_indices)
            }

            env_step_started = time.perf_counter()
            step_results = self._step_ready_env_instances(
                ready_env_instance_indices=ready_env_instance_indices,
                env_actions=env_actions,
            )
            env_step_time += time.perf_counter() - env_step_started
            for env_instance_index, step_result in step_results:
                if step_result is None:
                    continue
                offset = ready_env_instance_offsets[env_instance_index]

                (
                    next_obs_np,
                    next_state_np,
                    reward_np,
                    episode_reward_np,
                    done_np,
                    terminated_np,
                    truncated_np,
                ) = step_result
                next_obs = torch.from_numpy(next_obs_np).to(torch.float32).div_(255.0)
                next_state = None
                if self.collect_state:
                    next_state = (
                        torch.from_numpy(next_state_np).to(torch.float32).div_(255.0)
                    )
                done = torch.from_numpy(done_np)
                terminated = torch.from_numpy(terminated_np)
                truncated = torch.from_numpy(truncated_np)
                next_done = done.any().reshape(1)
                next_terminated = terminated.any().reshape(1)
                next_truncated = truncated.any().reshape(1)
                transitions[env_instance_index].append(
                    _Transition(
                        observation=current_obs[offset],
                        state=None if current_state is None else current_state[offset],
                        action=actions[offset],
                        log_prob=None if log_prob is None else log_prob[offset],
                        reward=torch.from_numpy(reward_np),
                        episode_reward=torch.from_numpy(episode_reward_np),
                        done=done,
                        terminated=terminated,
                        truncated=truncated,
                        next_observation=next_obs,
                        next_state=next_state,
                        next_done=next_done,
                        next_terminated=next_terminated,
                        next_truncated=next_truncated,
                    )
                )
                transitions_collected += 1

            if transitions_collected == transitions_before_round:
                self._register_failed_round()
            else:
                self._consecutive_failed_rounds = 0

        batch_assembly_started = time.perf_counter()
        batch = self._stack_transitions(transitions)
        batch_assembly_time = time.perf_counter() - batch_assembly_started
        collection_time = time.perf_counter() - collection_started
        self.last_profile = {
            "timers/collector_policy_inference": policy_time,
            "timers/collector_env_step": env_step_time,
            "timers/collector_batch_assembly": batch_assembly_time,
            "timers/collector_other": max(
                0.0,
                collection_time - policy_time - env_step_time - batch_assembly_time,
            ),
        }
        return batch

    def update_policy_weights_(self):
        return None

    def state_dict(self) -> dict[str, int]:
        return {"env_instance_cursor": self._env_instance_cursor}

    def load_state_dict(self, state_dict: dict[str, int]):
        legacy_cursor = state_dict.get("slot_cursor", 0)
        self._env_instance_cursor = (
            int(state_dict.get("env_instance_cursor", legacy_cursor)) % self.num_envs
        )

    def shutdown(self):
        for env_instance in self._env_instances:
            env_instance.close()
        if self._step_executor is not None:
            self._step_executor.shutdown(wait=True)

    def _register_failed_round(self):
        self._consecutive_failed_rounds += 1
        if self._consecutive_failed_rounds >= _MAX_FAILED_ROUNDS:
            raise RuntimeError(self._format_failure_summary())

    def _format_failure_summary(self) -> str:
        details = []
        for env_instance in self._env_instances:
            details.append(
                "env_instance="
                f"{env_instance.env_instance_index}:restarts={env_instance.total_restarts},"
                f"consecutive_failures={env_instance.consecutive_failures},"
                f"last_seed={env_instance.last_seed},"
                f"last_error={env_instance.last_error},"
                f"debug_status={env_instance.debug_status()}"
            )
        return (
            f"RolloutWorker could not collect any transitions after {self._consecutive_failed_rounds} consecutive recovery rounds. "
            + "; ".join(details)
        )

    def _next_env_instance_indices(self, count: int) -> list[int]:
        indices: list[int] = []
        if count <= 0:
            return indices
        cursor = self._env_instance_cursor
        while len(indices) < count:
            indices.append(cursor)
            cursor = (cursor + 1) % self.num_envs
        self._env_instance_cursor = cursor
        return indices

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
        raise TypeError(
            f"RolloutWorker only supports ndarray discrete actions, got {type(action_np)!r}"
        )

    def _stack_transitions(self, transitions: Sequence[Sequence[_Transition]]) -> TensorDict:
        if not transitions or not transitions[0]:
            raise ValueError("Cannot stack an empty transition batch")
        time_steps = len(transitions[0])
        if any(len(env_transitions) != time_steps for env_transitions in transitions):
            raise ValueError("All environments must contribute the same number of steps")

        batch_size = [len(transitions), time_steps]

        def stack_field(field: str) -> torch.Tensor:
            return torch.stack(
                [
                    torch.stack(
                        [getattr(transition, field) for transition in env_transitions],
                        dim=0,
                    )
                    for env_transitions in transitions
                ],
                dim=0,
            )

        observation = stack_field("observation")
        state = None
        if transitions[0][0].state is not None:
            state = stack_field("state")
        action = stack_field("action")
        reward = stack_field("reward")
        episode_reward = stack_field("episode_reward")
        done = stack_field("done")
        terminated = stack_field("terminated")
        truncated = stack_field("truncated")
        next_observation = stack_field("next_observation")
        next_state = None
        if transitions[0][0].next_state is not None:
            next_state = stack_field("next_state")
        next_done = stack_field("next_done")
        next_terminated = stack_field("next_terminated")
        next_truncated = stack_field("next_truncated")

        group_data = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "episode_reward": episode_reward,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
        if transitions[0][0].log_prob is not None:
            group_data["log_prob"] = stack_field("log_prob")
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
        root_data = {
            self.group_name: group_td,
            "done": next_done.clone(),
            "terminated": next_terminated.clone(),
            "truncated": next_truncated.clone(),
            "next": next_td,
        }
        if state is not None and next_state is not None:
            root_data["state"] = state
            next_td["state"] = next_state
        return TensorDict(root_data, batch_size=batch_size)

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

    def _step_ready_env_instances(
        self,
        *,
        ready_env_instance_indices: Sequence[int],
        env_actions: Sequence[Sequence[Any]],
    ):
        env_instance_actions = list(zip(ready_env_instance_indices, env_actions))
        if self._step_executor is None or len(env_instance_actions) < 2:
            return [
                self._step_env_instance(env_instance_index, env_action)
                for env_instance_index, env_action in env_instance_actions
            ]

        results = {}
        future_to_env_instance = {
            self._step_executor.submit(
                self._step_env_instance, env_instance_index, env_action
            ): env_instance_index
            for env_instance_index, env_action in env_instance_actions
        }
        for future in as_completed(future_to_env_instance):
            env_instance_index, step_result = future.result()
            results[env_instance_index] = step_result

        return [
            (env_instance_index, results.get(env_instance_index))
            for env_instance_index, _ in env_instance_actions
        ]
