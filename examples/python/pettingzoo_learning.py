"""
Example usage:

pip install vizdoom==1.3.0 \
            benchmarl==1.5.2 \
            torchrl==0.11.1 \
            tensordict==0.11.0 \
            pettingzoo==1.24.3 \
            wandb==0.22.1 \
            gymnasium==0.29.1 \
            pygame-ce==2.5.7 \
            imageio==2.37.3 \
            imageio-ffmpeg==0.6.0 \
            opencv-python-headless

python -m examples.python.pettingzoo_learning
    --algo mappo \
    --scenario health_gathering_multi_agent \
    --total_steps 3000000 \
    --num_agents 2 \
    --num_envs 8 \
    --train_device cuda \
    --sampling_device cpu \
    --buffer_device cpu \
    --record_every 10 \
    --parallel_collection \
"""

import time
from argparse import ArgumentParser, BooleanOptionalAction
from collections import deque
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from benchmarl.algorithms import IppoConfig, MasacConfig, QmixConfig
from benchmarl.algorithms.mappo import MappoConfig
from benchmarl.environments import TaskClass
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.logger import Logger
from benchmarl.models import CnnConfig
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import Composite
from torchrl.data.tensor_specs import UnboundedContinuous
from torchrl.envs import Compose, EnvBase, RemoveEmptySpecs, TransformedEnv
from torchrl.envs.libs.pettingzoo import MarlGroupMapType, PettingZooWrapper
from torchrl.envs.transforms import ObservationTransform, SelectTransform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.record.loggers.wandb import WandbLogger

import vizdoom as vzd
from pettingzoo_wrapper import make
from pettingzoo_wrapper.rollout_worker import RolloutWorker


DEFAULT_BASE_UDP_PORT = 40300
SLOT_PORT_STRIDE = 100
ENV_INSTANCE_INDEX_BASE = 100


class WandbLoggingWrapper(Logger):
    env_step_metric = "_env_steps"

    def __init__(
        self,
        *args,
        skip_frames: int = 1,
        num_actions: int | None = None,
        factor_sizes: list | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._skip_frames = max(1, int(skip_frames))
        self._num_actions = num_actions
        self._factor_sizes = factor_sizes
        self._env_steps = 0
        self._fps_samples = deque(maxlen=12)
        self._has_wandb_logger = any(
            isinstance(logger, WandbLogger) for logger in self.loggers
        )
        self._pending_metrics = None
        if self._has_wandb_logger:
            import wandb

            wandb.define_metric(self.env_step_metric, hidden=True)
            wandb.define_metric("*", step_metric=self.env_step_metric)

    def _prepare_iteration_metrics(self, total_frames: int):
        self._env_steps = int(total_frames) * self._skip_frames
        self._pending_metrics = None
        now = time.perf_counter()
        self._fps_samples.append((now, self._env_steps))
        if len(self._fps_samples) > 1:
            t0, env_steps0 = self._fps_samples[0]
            delta_time = now - t0
            delta_env_steps = self._env_steps - env_steps0
            if delta_time > 0 and delta_env_steps > 0:
                self._pending_metrics = {
                    "counters/wallclock_fps": float(delta_env_steps / delta_time)
                }

    def _add_fps(self, dict_to_log):
        payload = dict(dict_to_log)
        if self._pending_metrics is not None:
            payload.update(self._pending_metrics)
        collection_profile = getattr(self, "_collection_profile", None)
        if callable(collection_profile):
            payload.update(collection_profile())

        collection_time = payload.get("timers/collection_time")
        current_frames = payload.get("counters/current_frames")
        if (
            collection_time is not None
            and current_frames is not None
            and collection_time > 0
        ):
            payload["counters/fps"] = (
                float(current_frames) * self._skip_frames / float(collection_time)
            )
        return payload

    def log_collection(
        self, batch: TensorDictBase, task: TaskClass, total_frames: int, step: int
    ):
        self._prepare_iteration_metrics(total_frames)
        return super().log_collection(
            batch=batch,
            task=task,
            total_frames=total_frames,
            step=step,
        )

    def _log_individual_and_group_rewards(
        self,
        group: str,
        batch: TensorDictBase,
        global_done: torch.Tensor,
        any_episode_ended: bool,
        to_log: Dict[str, torch.Tensor],
        prefix: str = "collection",
        log_individual_agents: bool = True,
    ) -> torch.Tensor:
        return super()._log_individual_and_group_rewards(
            group=group,
            batch=batch,
            global_done=global_done,
            any_episode_ended=any_episode_ended,
            to_log=to_log,
            prefix=prefix,
            log_individual_agents=True,
        )

    def _log_global_episode_reward(
        self,
        episode_rewards: list[torch.Tensor],
        to_log: Dict[str, torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:
        if prefix == "collection" and len(episode_rewards) == 1:
            return episode_rewards[0]
        return super()._log_global_episode_reward(
            episode_rewards=episode_rewards,
            to_log=to_log,
            prefix=prefix,
        )

    def _evaluation_metrics(self, rollouts):
        metrics = {}
        if not rollouts:
            return metrics
        combat_info = {
            "DAMAGECOUNT": "damage",
            "FRAGCOUNT": "frags",
            "DEATHCOUNT": "deaths",
        }
        for group, agents in self.group_map.items():
            returns = torch.stack(
                [
                    self._get_reward(group, rollout).sum(0).squeeze(-1)
                    for rollout in rollouts
                ]
            )
            for agent_index, agent in enumerate(agents):
                self._log_min_mean_max(
                    metrics,
                    f"eval/{group}/reward/{agent}/episode_reward",
                    returns[:, agent_index],
                )

                for info_key, metric_name in combat_info.items():
                    values = []
                    for rollout in rollouts:
                        info = rollout.get(("next", group, "info"), None)
                        value = None if info is None else info.get(info_key, None)
                        if value is None:
                            break
                        deltas = value[1:, agent_index] - value[:-1, agent_index]
                        reset = torch.zeros_like(deltas, dtype=torch.bool)
                        for reset_key in ("DAMAGECOUNT", "DEATHCOUNT"):
                            counter = info.get(reset_key, None)
                            if counter is not None:
                                reset |= (
                                    counter[1:, agent_index] < counter[:-1, agent_index]
                                )
                        values.append(
                            deltas.masked_fill(reset, 0).sum(dim=0).float().mean()
                        )
                    if len(values) == len(rollouts):
                        self._log_min_mean_max(
                            metrics,
                            f"eval/{group}/combat/{agent}/{metric_name}",
                            torch.stack(values),
                        )

                if self._factor_sizes is not None:
                    # Actions are (time, n_agents, n_factors)
                    per_agent = torch.cat(
                        [
                            rollout.get((group, "action"))[:, agent_index]
                            .reshape(-1, len(self._factor_sizes))
                            .long()
                            for rollout in rollouts
                        ]
                    )
                    for factor_index, size in enumerate(self._factor_sizes):
                        column = per_agent[:, factor_index]
                        for option in range(size):
                            metrics[
                                f"eval/{group}/actions/{agent}/factor{factor_index}_option{option}_frequency"
                            ] = ((column == option).float().mean().item())
                elif self._num_actions is not None:
                    actions = torch.cat(
                        [
                            rollout.get((group, "action"))[:, agent_index]
                            .reshape(-1)
                            .long()
                            for rollout in rollouts
                        ]
                    )
                    for action in range(self._num_actions):
                        metrics[
                            f"eval/{group}/actions/{agent}/action_{action}_frequency"
                        ] = ((actions == action).float().mean().item())
        return metrics

    def log_evaluation(self, rollouts, total_frames: int, step: int, video_frames=None):
        self._env_steps = int(total_frames) * self._skip_frames
        result = super().log_evaluation(
            rollouts=rollouts,
            total_frames=total_frames,
            step=step,
            video_frames=video_frames,
        )
        self.log(self._evaluation_metrics(rollouts), step=step)
        return result

    def log(self, dict_to_log: Dict, step: int = None):
        payload = self._add_fps(dict_to_log)
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_payload = {
                    **payload,
                    self.env_step_metric: float(self._env_steps),
                }
                logger.experiment.log(wandb_payload, commit=False)
            else:
                for key, value in payload.items():
                    logger.log_scalar(key.replace("/", "_"), value, step=step)
        self._pending_metrics = None


class FactoredCategorical(torch.distributions.Distribution):
    """
    Split logit dimension to 1 chunk per factor, size nvec, then sums log-probs and entropies across factors
    """

    arg_constraints: Dict[str, Any] = {}
    has_rsample = False

    def __init__(self, logits: torch.Tensor, nvec, validate_args=None):
        self.nvec = [int(n) for n in nvec]
        expected = sum(self.nvec)
        if logits.shape[-1] != expected:
            raise ValueError(
                f"FactoredCategorical expected {expected} logits for nvec {self.nvec}, "
                f"got {logits.shape[-1]}"
            )
        self._dists = [
            torch.distributions.Categorical(logits=chunk)
            for chunk in logits.split(self.nvec, dim=-1)
        ]
        super().__init__(
            batch_shape=logits.shape[:-1],
            event_shape=torch.Size([len(self.nvec)]),
            validate_args=False,
        )

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self._dists], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [d.log_prob(value[..., i]) for i, d in enumerate(self._dists)], dim=-1
        ).sum(-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([d.entropy() for d in self._dists], dim=-1).sum(-1)

    @property
    def mode(self) -> torch.Tensor:
        return torch.stack([d.logits.argmax(-1) for d in self._dists], dim=-1)

    @property
    def deterministic_sample(self) -> torch.Tensor:
        # TorchRL ExplorationType.DETERMINISTIC looks for this first
        return self.mode


def _factored_nvec(spec) -> Optional[list]:
    """nvec of a MultiDiscrete action spec. None if isn't one."""
    nvec = getattr(spec, "nvec", None)
    if nvec is None:
        return None
    nvec = torch.as_tensor(nvec)
    while nvec.ndim > 1:
        nvec = nvec[0]
    return [int(n) for n in nvec.tolist()]


def _patch_benchmarl_for_factored_actions() -> None:
    """
    BenchMARL decides discrete vs cont. with `not isinstance(action_space, (Categorical, OneHot))`,
    so MultiCategorical spec is classified as continuous. This patches that.
    """
    from benchmarl.algorithms.ippo import Ippo
    from benchmarl.algorithms.mappo import Mappo
    from torchrl.data import Composite as _Composite
    from torchrl.data import Unbounded as _Unbounded
    from torchrl.modules import ProbabilisticActor

    for cls in (Ippo, Mappo):
        if getattr(cls, "_factored_patched", False):
            continue
        original = cls._get_policy_for_loss

        def _get_policy_for_loss(self, group, model_config, continuous, _orig=original):
            action_spec = self.action_spec[group, "action"]
            nvec = _factored_nvec(action_spec)
            if nvec is None:
                return _orig(self, group, model_config, continuous)

            n_agents = len(self.group_map[group])
            actor_module = model_config.get_model(
                input_spec=_Composite(
                    {group: self.observation_spec[group].clone().to(self.device)}
                ),
                output_spec=_Composite(
                    {
                        group: _Composite(
                            {"logits": _Unbounded(shape=[n_agents, sum(nvec)])},
                            shape=(n_agents,),
                        )
                    }
                ),
                agent_group=group,
                input_has_agent_dim=True,
                n_agents=n_agents,
                centralised=False,
                share_params=self.experiment_config.share_policy_params,
                device=self.device,
                action_spec=self.action_spec,
            )
            print(
                f"[factored-actions] {group}: MultiDiscrete nvec={nvec} "
                f"({sum(nvec)} logits, {int(torch.tensor(nvec).prod())} combinations)",
                flush=True,
            )
            return ProbabilisticActor(
                module=actor_module,
                spec=action_spec,
                in_keys=[(group, "logits")],
                out_keys=[(group, "action")],
                distribution_class=FactoredCategorical,
                distribution_kwargs={"nvec": nvec},
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        cls._get_policy_for_loss = _get_policy_for_loss
        cls._factored_patched = True


# BenchMARL builds (group, "advantage") as (*group_shape, 1), i.e.
# (batch, time, n_agents, 1), so the agent dimension is -2. Asserted at runtime in
# _enable_per_agent_advantage_normalization rather than trusted.
_ADVANTAGE_AGENT_DIM = -2


def _enable_per_agent_advantage_normalization(
    loss_module, group: str, n_agents: int
) -> None:
    """
    BenchMARL hardcodes `normalize_advantage=False` and doesn't expose it on the config,
    which deviates loss_obj from entropy bonus.

    `normalize_advantage_exclude_dims` (in TorchRL) is what we want, but it's not implemented in BenchMARL.
    The flag will be added in this PR: https://github.com/facebookresearch/BenchMARL/pull/256.
    """
    for attribute in ("normalize_advantage", "normalize_advantage_exclude_dims"):
        if not hasattr(loss_module, attribute):
            raise RuntimeError(
                f"{type(loss_module).__name__} has no {attribute!r}. Check TorchRL version"
            )

    loss_module.normalize_advantage = True
    loss_module.normalize_advantage_exclude_dims = (_ADVANTAGE_AGENT_DIM,)

    # check -2 is agent dimension
    advantage_key = loss_module.tensor_keys.advantage
    inner_forward = loss_module.forward
    state = {"verified": False}

    def forward(tensordict, *args, **kwargs):
        if not state["verified"]:
            advantage = tensordict.get(advantage_key, None)
            if advantage is None:
                raise RuntimeError(
                    f"[{group}] {advantage_key} absent before the loss forward"
                )
            shape = tuple(advantage.shape)
            if advantage.ndim < 3 or shape[-1] != 1 or shape[-2] != n_agents:
                raise RuntimeError(
                    f"[{group}] expected advantage of shape (..., {n_agents}, 1) so "
                    f"that dim {_ADVANTAGE_AGENT_DIM} holds agents, got {shape}."
                )
            print(
                f"[adv-norm] {group}: (advantage {shape}, agent dim {_ADVANTAGE_AGENT_DIM}, {n_agents} agents)",
                flush=True,
            )
            state["verified"] = True
        return inner_forward(tensordict, *args, **kwargs)

    loss_module.forward = forward


class VizdoomExperiment(Experiment):
    """
    Experiment subclass that injects structured W&B metadata:

    - job_type  : algorithm name
    - group     : "<environment>/<task>"
    - id / name : "<algo>_<task>_<N>agents_seed<S>_<timestamp>"

    and turns on per-agent advantage normalization.
    """

    def _setup_algorithm(self):
        super()._setup_algorithm()
        for group, loss_module in self.losses.items():
            if hasattr(loss_module, "normalize_advantage"):
                _enable_per_agent_advantage_normalization(
                    loss_module, group, len(self.group_map[group])
                )

    def _setup_logger(self):
        num_agents = sum(len(v) for v in self.group_map.values())
        run_id = self.task.config.get("run_id")
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = (
                f"{self.algorithm_name}_{self.task_name}"
                f"_{num_agents}agents_seed{self.seed}_{timestamp}"
            )

        extra = {
            "job_type": self.algorithm_name,
            "group": self.task_name,
            "id": run_id,
            "name": run_id,
        }

        action_spec = self.test_env.input_spec[
            "full_action_spec", next(iter(self.group_map)), "action"
        ]
        original = self.config.wandb_extra_kwargs
        self.config.wandb_extra_kwargs = {**original, **extra}
        try:
            hparams_kwargs = {
                "task_name": self.task_name,
                "algorithm_name": self.algorithm_name,
                "model_name": self.model_name,
                "critic_model_name": self.critic_model_name,
                "experiment_config": self.config.__dict__,
                "algorithm_config": self.algorithm_config.__dict__,
                "model_config": self.model_config.__dict__,
                "critic_model_config": self.critic_model_config.__dict__,
                "task_config": self.task.config,
                "continuous_actions": self.continuous_actions,
                "on_policy": self.on_policy,
                "environment_name": self.environment_name,
                "seed": self.seed,
            }
            self.logger = WandbLoggingWrapper(
                experiment_name=self.name,
                folder_name=str(self.folder_name),
                experiment_config=self.config,
                algorithm_name=self.algorithm_name,
                model_name=self.model_name,
                environment_name=self.environment_name,
                task_name=self.task_name,
                group_map=self.group_map,
                seed=self.seed,
                project_name=self.config.project_name,
                wandb_extra_kwargs={
                    **self.config.wandb_extra_kwargs,
                    "config": hparams_kwargs,
                },
                skip_frames=self.task.config.get("skip_frames", 1),
                num_actions=getattr(action_spec, "n", None),
                factor_sizes=_factored_nvec(action_spec),
            )
            self.logger._collection_profile = lambda: self.collector.last_profile
            self.logger.log_hparams(**hparams_kwargs)
        finally:
            self.config.wandb_extra_kwargs = original

    def _setup_collector(self):
        self.policy = self.algorithm.get_policy_for_collection()
        self.group_policies = {}
        for group in self.group_map.keys():
            group_policy = self.policy.select_subsequence(out_keys=[(group, "action")])
            assert len(group_policy) == 1
            self.group_policies[group] = group_policy[0]

        if len(self.group_map) != 1:
            raise ValueError("The custom RolloutWorker supports only one agent group")
        group_name = next(iter(self.group_map.keys()))
        n_agents = len(self.group_map[group_name])
        rollout_worker_kwargs = dict(
            policy=self.policy,
            action_spec=self.test_env.input_spec[
                "full_action_spec", group_name, "action"
            ],
            group_name=group_name,
            n_agents=n_agents,
            frames_per_batch=self.config.collected_frames_per_batch(self.on_policy),
            num_envs=int(self.config.n_envs_per_worker(self.on_policy)),
            sampling_device=self.config.sampling_device,
            seed=self.seed,
            parallel_collection=bool(getattr(self.config, "parallel_collection", True)),
            double_buffer=bool(self.task.config.get("double_buffer", True)),
            collect_state=False,
        )

        def env_builder(seed, env_instance_index):
            return self.task.build_parallel_env(
                seed=seed,
                env_instance_index=self.task.env_instance_index(env_instance_index),
                enable_video=False,
            )

        rollout_worker = RolloutWorker(
            env_builder=env_builder,
            **rollout_worker_kwargs,
        )
        self.collector = rollout_worker


_OBS_SCALE = 1.0 / 255.0


def _normalize_byte_observations(model):
    obs_keys = [
        key
        for key in model.in_keys
        if (key[-1] if isinstance(key, tuple) else key) in ("observation", "state")
    ]
    if not obs_keys:
        return model

    inner_forward = model._forward

    def _forward(tensordict, *args, **kwargs):
        for key in obs_keys:
            obs = tensordict.get(key, None)
            if obs is not None and obs.dtype == torch.uint8:
                tensordict.set(key, obs.mul(_OBS_SCALE))
        return inner_forward(tensordict, *args, **kwargs)

    model._forward = _forward
    return model


class ByteObsCnnConfig(CnnConfig):
    def get_model(self, *args, **kwargs):
        return _normalize_byte_observations(super().get_model(*args, **kwargs))


class AHWCToTensor(ObservationTransform):
    """
    Keep AHWC layout, convert to float tensor
    """

    def __init__(
        self,
        key=("agent", "observation"),
        dtype: torch.dtype | None = None,
    ):
        super().__init__(in_keys=[key], out_keys=[key])
        self.key = key
        self.dtype = dtype if dtype is not None else torch.float32

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)
        obs = obs.div(255).to(self.dtype)
        if obs.ndim not in (3, 4):
            raise ValueError(f"{self.key} must be 3D/4D AHWC, got {tuple(obs.shape)}")
        return obs

    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        leaf = obs_spec[self.key]
        obs_spec[self.key] = UnboundedContinuous(
            shape=leaf.shape,
            device=leaf.device,
            dtype=self.dtype,
        )
        return obs_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class VizdoomTask(TaskClass):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name=config["scenario"], config=config)
        self._next_env_instance_index = 0

    @staticmethod
    def env_name() -> str:
        return "vizdoom"

    def _env_instance_base_port(self, env_instance_index: int) -> int:
        base_port = int(self.config.get("base_port", DEFAULT_BASE_UDP_PORT))
        span = max(SLOT_PORT_STRIDE, 65535 - base_port - SLOT_PORT_STRIDE)
        return base_port + (int(env_instance_index) * SLOT_PORT_STRIDE) % span

    def _allocate_env_instance_index(self) -> int:
        env_instance_index = self._next_env_instance_index
        self._next_env_instance_index += 1
        return env_instance_index

    def env_instance_index(self, env_instance_offset: int) -> int:
        return ENV_INSTANCE_INDEX_BASE + int(env_instance_offset)

    def build_parallel_env(
        self,
        seed: int,
        env_instance_index: int = 0,
        enable_video: Optional[bool] = None,
        async_mode: Optional[bool] = None,
    ):
        cfg = self.config
        base_port = self._env_instance_base_port(env_instance_index)
        return make(
            scenario=cfg["scenario"],
            num_agents=cfg["num_agents"],
            resolution=cfg["resolution"],
            skip_frames=cfg["skip_frames"],
            async_mode=cfg["async_mode"],
            render_mode=cfg["render_mode"],
            host_address=cfg.get("host_address", "127.0.0.1"),
            port=base_port,
            slot_index=env_instance_index,
            netmode=cfg["netmode"],
            ticrate=cfg["ticrate"],
            use_multi_binary_action_space=False,
            factored_actions=cfg.get("factored_actions", True),
            seed=seed,
            enable_video=cfg["enable_video"] if enable_video is None else enable_video,
            record_every=cfg["record_every"],
            video_fps=cfg["video_fps"],
            verbose=cfg.get("verbose", False),
            daemon=cfg["daemon"],
        )

    def _build_training_env(self, seed: int, env_instance_index: int):
        cfg = self.config
        env = PettingZooWrapper(
            env=self.build_parallel_env(
                seed=seed, env_instance_index=env_instance_index
            ),
            group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
            return_state=False,
        )
        group_name = next(iter(env.group_map.keys()))
        selected_keys = [(group_name, "observation"), (group_name, "info")]
        transforms = [
            SelectTransform(*selected_keys),
            AHWCToTensor(key=(group_name, "observation")),
            RemoveEmptySpecs(),
        ]
        env = TransformedEnv(env, Compose(*transforms))
        env = env.to(cfg.get("sampling_device", "cpu"))
        return env

    def get_env_fun(
        self, num_envs: int, continuous_actions: bool, seed: int | None, device=None
    ):
        def _make():
            env_instance_index = self._allocate_env_instance_index()
            return self._build_training_env(
                seed=seed, env_instance_index=env_instance_index
            )

        return _make

    def action_spec(self, env: EnvBase) -> Composite:
        return env.action_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return getattr(env, "action_mask_spec", None)

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return getattr(env, "info_spec", None)

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def group_map(self, env=None):
        """
        Return a dict group_map (group_name -> [agent_names]).
        BenchMARL calls this with the env; use its computed map if available.
        Fallback to ALL_IN_ONE_GROUP as we only have 1 group. Use ONE_GROUP_PER_AGENT otherwise.
        """
        # Try to use the env's own group_map (already a dict on PettingZooWrapper)
        if env is not None:
            gm = getattr(env, "group_map", None)
            if isinstance(gm, dict) and gm:
                return gm
            # Fallback: build one-group-per-agent from possible_agents if present
            agents = getattr(env, "possible_agents", None)
            if agents:
                return MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(list(agents))

        # Last-resort fallback using config’s num_agents
        n = int(self.config.get("num_agents", 2))
        agents = [f"agent_{i}" for i in range(n)]
        return MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agents)

    def has_render(self, env: EnvBase) -> bool:
        return hasattr(env, "render")

    def max_steps(self, env: EnvBase) -> int:
        return int(self.config.get("timeout", 1000))

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True


# ----------------- Script entry -----------------
ALGOS: Dict[str, Any] = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "qmix": QmixConfig,
    "masac": MasacConfig,
}


def override_config(config, overrides) -> Any:
    valid_fields = {field.name for field in fields(type(config))}
    for key, value in overrides.items():
        if value is not None and key in valid_fields:
            setattr(config, key, value)
    return config


def override_algo_config(args):
    algo_cfg = ALGOS[args.algo].get_from_yaml()

    ppo_overrides = {
        "share_param_critic": True,
        "clip_epsilon": args.clip_eps,
        "entropy_coef": args.entropy_coef,
        "critic_coef": args.vf_coef,
        "loss_critic_type": "l2",
        "lmbda": args.gae_lambda,
        "scale_mapping": "biased_softplus_1.0",
        "use_tanh_normal": True,
        "minibatch_advantage": False,
    }

    algo_overrides = {
        "mappo": ppo_overrides,
        "ippo": ppo_overrides,
        "qmix": {},
        "masac": {
            "share_param_critic": True,
            "scale_mapping": "biased_softplus_1.0",
            "use_tanh_normal": True,
        },
    }

    return override_config(algo_cfg, algo_overrides[args.algo])


def override_experiment_config(args, on_policy: bool, on_policy_minibatch_size: int):
    overrides = {
        "sampling_device": args.sampling_device,
        "train_device": args.train_device,
        "buffer_device": args.buffer_device,
        "share_policy_params": True,
        "parallel_collection": args.parallel_collection,
        "max_n_frames": int(args.total_steps),
        "gamma": args.gamma,
        "lr": args.lr,
        # eval / logging / ckpts
        "evaluation": True,  # Must be enabled for video logging
        "render": False,
        "evaluation_interval": args.rollout_steps * 25,
        "evaluation_episodes": 2,
        "loggers": ["wandb"],
        "project_name": "benchmarl-vizdoom",
        "checkpoint_interval": args.rollout_steps * 100,
        "checkpoint_at_end": True,
        "keep_checkpoints_num": args.keep_checkpoints_num,
        "exclude_buffer_from_checkpoint": not args.save_replay_buffer,
    }

    if on_policy:
        overrides.update(
            {
                "on_policy_collected_frames_per_batch": args.rollout_steps,
                "on_policy_n_envs_per_worker": args.num_envs,
                "on_policy_n_minibatch_iters": args.num_epochs,
                "on_policy_minibatch_size": on_policy_minibatch_size,
            }
        )
    else:
        overrides.update(
            {
                "off_policy_collected_frames_per_batch": args.rollout_steps,
                "off_policy_n_envs_per_worker": args.num_envs,
                "off_policy_n_optimizer_steps": args.optimizer_steps,
                "off_policy_train_batch_size": args.batch_size,
                "off_policy_memory_size": args.off_policy_memory_size,
            }
        )

    return overrides


def main():
    ap = ArgumentParser()
    # Env args
    ap.add_argument("--scenario", type=str, default="pitfall_multi_agent")
    ap.add_argument("--num_agents", type=int, default=2)
    ap.add_argument("--resolution", type=str, default="160X120")
    ap.add_argument("--skip_frames", type=int, default=4)
    ap.add_argument("--factored_actions", action=BooleanOptionalAction, default=True)
    ap.add_argument("--async-mode", action=BooleanOptionalAction, default=False)
    ap.add_argument("--host_address", type=str, default="127.0.0.1")
    ap.add_argument("--base_port", type=int, default=DEFAULT_BASE_UDP_PORT)
    ap.add_argument("--netmode", type=int, default=0)
    ap.add_argument("--ticrate", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", default=False)
    ap.add_argument(
        "--daemon", dest="daemon", action=BooleanOptionalAction, default=True
    )

    # Train args
    ap.add_argument("--algo", type=str, default="mappo", choices=list(ALGOS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total_steps", type=float, default=1e6)
    ap.add_argument(
        "--train_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--sampling_device", type=str, default="cpu")
    ap.add_argument("--buffer_device", type=str, default="cpu")
    ap.add_argument("--rollout_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument(
        "--off_policy_memory_size", type=int, default=16384
    )  # 2048 batch * 8 optimizer steps
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.1)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=1.0)
    ap.add_argument("--num_minibatches", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=8)
    ap.add_argument("--optimizer_steps", type=int, default=8)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--parallel_collection", action=BooleanOptionalAction, default=True)
    ap.add_argument(
        "--double_buffer",
        action=BooleanOptionalAction,
        default=True,
        help=("policy inference in 1/2 of envs overlaps, env stepping the other 1/2."),
    )
    ap.add_argument(
        "--keep_checkpoints_num",
        type=int,
        default=1,
        help="How many checkpoints to keep",
    )
    ap.add_argument(
        "--save_replay_buffer",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Include replay buffers in checkpoints. This lets exact off-policy resume but can create really heavy checkpoint files for image observations."
        ),
    )

    # Video recording
    ap.add_argument("--enable_video", action=BooleanOptionalAction, default=True)
    ap.add_argument("--record_every", type=int, default=100)
    ap.add_argument("--video_fps", type=int, default=35)
    ap.add_argument(
        "--render_mode", type=str, default="rgb_array", choices=["rgb_array", "human"]
    )

    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if not args.async_mode and args.ticrate is not None:
        raise ValueError("--ticrate can only be set when --async-mode is enabled")
    args.ticrate = (
        int(args.ticrate) if args.ticrate is not None else vzd.DEFAULT_TICRATE
    )
    root_path = Path(__file__).parent.parent.parent
    checkpoints_path = root_path / "checkpoints"
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{args.algo}_{args.scenario}"
        f"_{args.num_agents}agents_{args.num_envs}envs"
        f"_seed{args.seed}_{run_timestamp}"
    )

    if args.algo not in ALGOS:
        raise NotImplementedError(
            f"{args.algo} is not currently implemented in this script"
        )

    algo_cfg = override_algo_config(args)

    # Nature-style front end + 512 MLP head
    cnn_num_cells = [32, 64, 64]
    cnn_kernel_sizes = [8, 4, 3]
    cnn_strides = [4, 2, 1]
    cnn_paddings = [0, 0, 0]
    cnn_activation_class = nn.ReLU
    mlp_num_cells = [512]
    mlp_layer_class = nn.Linear
    mlp_activation_class = nn.ReLU

    model_cfg = ByteObsCnnConfig(
        cnn_num_cells=cnn_num_cells,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
        cnn_paddings=cnn_paddings,
        cnn_activation_class=cnn_activation_class,
        mlp_num_cells=mlp_num_cells,
        mlp_layer_class=mlp_layer_class,
        mlp_activation_class=mlp_activation_class,
    )

    critic_cfg = ByteObsCnnConfig(
        cnn_num_cells=cnn_num_cells,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
        cnn_paddings=cnn_paddings,
        cnn_activation_class=cnn_activation_class,
        mlp_num_cells=mlp_num_cells,
        mlp_layer_class=mlp_layer_class,
        mlp_activation_class=mlp_activation_class,
    )

    exp_cfg = ExperimentConfig.get_from_yaml()
    is_on_policy = algo_cfg.on_policy()

    # compute any derived values first
    on_policy_minibatch_size = max(1, args.batch_size // max(1, args.num_minibatches))

    overrides = override_experiment_config(
        args,
        on_policy=is_on_policy,
        on_policy_minibatch_size=on_policy_minibatch_size,
    )
    overrides["save_folder"] = str(checkpoints_path)
    override_config(exp_cfg, overrides)

    # keep eval interval aligned with horizon (collector-friendly)
    h = exp_cfg.collected_frames_per_batch(is_on_policy)
    if h and exp_cfg.evaluation_interval % h != 0:
        exp_cfg.evaluation_interval = ((exp_cfg.evaluation_interval + h - 1) // h) * h

    task_cfg = {
        "scenario": args.scenario,
        "num_agents": args.num_agents,
        "resolution": args.resolution,
        "skip_frames": args.skip_frames,
        "factored_actions": args.factored_actions,
        "async_mode": args.async_mode,
        "render_mode": args.render_mode,
        "host_address": args.host_address,
        "base_port": args.base_port,
        "netmode": args.netmode,
        "ticrate": args.ticrate,
        "enable_video": args.enable_video,
        "record_every": args.record_every,
        "video_fps": args.video_fps,
        "sampling_device": args.sampling_device,
        "daemon": args.daemon,
        "verbose": args.verbose,
        "run_id": run_id,
        "double_buffer": args.double_buffer,
    }
    task = VizdoomTask(task_cfg)

    if args.factored_actions:
        _patch_benchmarl_for_factored_actions()

    experiment = VizdoomExperiment(
        task=task,
        algorithm_config=algo_cfg,
        model_config=model_cfg,
        critic_model_config=critic_cfg,
        seed=args.seed,
        config=exp_cfg,
    )

    Path(str(exp_cfg.save_folder)).mkdir(parents=True, exist_ok=True)
    experiment.run()
    experiment.close()


if __name__ == "__main__":
    main()
