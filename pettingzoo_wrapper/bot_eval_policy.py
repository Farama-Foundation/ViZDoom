from __future__ import annotations

import importlib
import inspect
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


TRAINING_SCRIPT_MODULE = "examples.python.pettingzoo_learning"


def _policy_device(policy: Any) -> str:
    try:
        return str(next(policy.parameters()).device)
    except (AttributeError, StopIteration):
        return "cpu"


class TorchRLPolicyAdapter:
    """Adapt a BenchMARL/TorchRL group policy to one raw DoomGame player."""

    def __init__(
        self,
        policy: Any,
        *,
        group_name: str,
        agent_index: int = 0,
        agent_count: int = 1,
        frame_stack: int = 1,
        device: str = "cpu",
    ) -> None:
        if agent_index < 0 or agent_index >= agent_count:
            raise ValueError("agent_index must be within agent_count")
        self.policy = policy
        self.group_name = group_name
        self.agent_index = int(agent_index)
        self.agent_count = int(agent_count)
        self.frame_stack = int(frame_stack)
        self.device = device
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        # Logits of the last act() call (categorical policies only), for entropy diagnostics
        self.last_logits: np.ndarray | None = None

    @classmethod
    def from_experiment(
        cls,
        experiment: Any,
        group_name: str,
        agent_index: int = 0,
        device: str | None = None,
    ) -> TorchRLPolicyAdapter:
        policy = experiment.group_policies[group_name]
        agent_count = len(experiment.group_map[group_name])
        if device is None:
            device = _policy_device(policy)
        return cls(
            policy,
            group_name=group_name,
            agent_index=agent_index,
            agent_count=agent_count,
            frame_stack=int(experiment.task.config.get("frame_stack", 1)),
            device=device,
        )

    def reset(self, seed: int | None = None) -> None:
        self._frames.clear()
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(seed)

    def act(self, observation: np.ndarray, deterministic: bool = True):
        import torch
        from tensordict import TensorDict
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        obs = np.asarray(observation)
        if obs.ndim != 3:
            raise ValueError(f"expected HWC observation, got {tuple(obs.shape)}")
        if not self._frames:
            self._frames.extend([obs] * self.frame_stack)
        else:
            self._frames.append(obs)
        obs = np.concatenate(self._frames, axis=-1)
        obs_tensor = (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
        )
        group_obs = torch.zeros(
            (1, self.agent_count, *obs.shape), dtype=torch.float32, device=self.device
        )
        group_obs[:, self.agent_index] = obs_tensor
        td = TensorDict({}, batch_size=[1], device=self.device)
        td.set((self.group_name, "observation"), group_obs)

        exploration = (
            ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
        )
        with torch.no_grad(), set_exploration_type(exploration):
            output = self.policy(td)
        logits = output.get((self.group_name, "logits"), None)
        if logits is not None and logits.ndim >= 2:
            self.last_logits = logits[0, self.agent_index].detach().cpu().numpy()
        action = output.get((self.group_name, "action"))
        if action is None:
            raise RuntimeError(
                f"policy did not produce {(self.group_name, 'action')!r}"
            )
        if action.ndim >= 3:
            action = action[0, self.agent_index]
        elif action.ndim >= 2:
            action = action[0, self.agent_index]
        return action.detach().cpu().numpy().reshape(-1).tolist()

    def last_entropy(self) -> float | None:
        """Entropy (nats) of the categorical distribution behind the last act() call."""
        if self.last_logits is None:
            return None
        logits = self.last_logits.astype(np.float64)
        logits = logits - logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def register_training_script_classes(
    module_name: str = TRAINING_SCRIPT_MODULE,
) -> list[str]:
    """
    Returns the names that were added. A no-op when ``__main__`` already is the training script
    """
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return []
    spec = getattr(main_module, "__spec__", None)
    main_file = getattr(main_module, "__file__", None) or ""
    if (spec is not None and spec.name == module_name) or main_file.endswith(
        module_name.replace(".", "/") + ".py"
    ):
        # __main__ is the training script itself: nothing to alias
        return []
    try:
        training = importlib.import_module(module_name)
    except ImportError as exc:
        print(
            f"[bot_eval] cannot import {module_name} ({exc}) - __main__ pickles "
            "of the training script will not resolve",
            file=sys.stderr,
        )
        return []
    added: list[str] = []
    for name, value in vars(training).items():
        if not inspect.isclass(value) or value.__module__ != training.__name__:
            continue
        if hasattr(main_module, name):
            continue
        setattr(main_module, name, value)
        added.append(name)
    return added


def load_bot_eval_experiment(checkpoint_path: str | Path):
    from benchmarl.experiment import Experiment

    register_training_script_classes()
    checkpoint = Path(checkpoint_path).resolve()
    event_dir = checkpoint.parent.parent
    return Experiment.reload_from_file(
        str(checkpoint),
        experiment_patch={
            "save_folder": str(event_dir.parent),
            "collect_with_grad": True,
            "on_policy_n_envs_per_worker": 1,
            "off_policy_n_envs_per_worker": 1,
            "sampling_device": "cpu",
            "train_device": "cpu",
            "buffer_device": "cpu",
            "restore_map_location": "cpu",
            "loggers": [],
            "evaluation": False,
            "render": False,
        },
    )
