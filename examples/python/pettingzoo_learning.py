import time
from argparse import ArgumentParser, BooleanOptionalAction
from collections import deque
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import vizdoom as vzd
from benchmarl.algorithms import QmixConfig, MasacConfig
from benchmarl.algorithms.mappo import MappoConfig
from benchmarl.environments import TaskClass
from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.experiment.logger import Logger
from benchmarl.models import CnnConfig
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import Composite
from torchrl.data.tensor_specs import UnboundedContinuous
from torchrl.envs import EnvBase, RemoveEmptySpecs
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.libs.pettingzoo import MarlGroupMapType, PettingZooWrapper
from torchrl.envs.transforms import ObservationTransform
from torchrl.envs.transforms import SelectTransform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.record.loggers.wandb import WandbLogger

from pettingzoo_wrapper import make
from pettingzoo_wrapper.collector import Collector

DEFAULT_BASE_UDP_PORT = 40300
SLOT_PORT_STRIDE = 100
ENV_INSTANCE_INDEX_BASE = 100


class WandbLoggingWrapper(Logger):
    env_step_metric = "_env_steps"

    def __init__(self, *args, skip_frames: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_frames = max(1, int(skip_frames))
        self._env_steps = 0
        self._fps_samples = deque(maxlen=12)
        self._has_wandb_logger = any(
            isinstance(logger, WandbLogger) for logger in self.loggers
        )
        self._pending_wandb_metrics = None
        if self._has_wandb_logger:
            import wandb

            wandb.define_metric(self.env_step_metric, hidden=True)
            wandb.define_metric("*", step_metric=self.env_step_metric)

    def _prepare_iteration_metrics(self, total_frames: int):
        self._env_steps = int(total_frames) * self._skip_frames
        self._pending_wandb_metrics = None
        now = time.perf_counter()
        self._fps_samples.append((now, self._env_steps))
        if len(self._fps_samples) > 1:
            t0, env_steps0 = self._fps_samples[0]
            delta_time = now - t0
            delta_env_steps = self._env_steps - env_steps0
            if delta_time > 0 and delta_env_steps > 0:
                self._pending_wandb_metrics = {
                    "counters/fps": float(delta_env_steps / delta_time)
                }

    def log_collection(self, batch: TensorDictBase, task: TaskClass, total_frames: int, step: int):
        self._prepare_iteration_metrics(total_frames)
        return super().log_collection(
            batch=batch,
            task=task,
            total_frames=total_frames,
            step=step,
        )

    def log_evaluation(self, rollouts, total_frames: int, step: int, video_frames=None):
        self._env_steps = int(total_frames) * self._skip_frames
        return super().log_evaluation(
            rollouts=rollouts,
            total_frames=total_frames,
            step=step,
            video_frames=video_frames,
        )

    def log(self, dict_to_log: Dict, step: int = None):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_payload = {
                    **dict_to_log,
                    self.env_step_metric: float(self._env_steps),
                }
                if self._pending_wandb_metrics is not None:
                    wandb_payload.update(self._pending_wandb_metrics)
                logger.experiment.log(wandb_payload, commit=False)
            else:
                for key, value in dict_to_log.items():
                    logger.log_scalar(key.replace("/", "_"), value, step=step)
        self._pending_wandb_metrics = None


class VizdoomExperiment(Experiment):
    """
    Experiment subclass that injects structured W&B metadata:

    - job_type  : algorithm name
    - group     : "<environment>/<task>"
    - id / name : "<algo>_<task>_<N>agents_seed<S>_<timestamp>"
    """

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
            )
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

        group_name = next(iter(self.group_map.keys()))
        n_agents = len(self.group_map[group_name])
        collector_kwargs = dict(
            policy=self.policy,
            action_spec=self.test_env.input_spec["full_action_spec", group_name, "action"],
            group_name=group_name,
            n_agents=n_agents,
            frames_per_batch=self.config.on_policy_collected_frames_per_batch,
            num_envs=int(self.config.on_policy_n_envs_per_worker),
            sampling_device=self.config.sampling_device,
            seed=self.seed,
            parallel_collection=bool(getattr(self.config, "parallel_collection", True)),
        )
        def env_builder(seed, env_instance_index):
            return self.task.build_parallel_env(
                seed=seed,
                env_instance_index=self.task.env_instance_index(env_instance_index),
                enable_video=False,
            )
        collector = Collector(
            env_builder=env_builder,
            **collector_kwargs,
        )
        self.collector = collector


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
        if obs.ndim != 4:
            raise ValueError(f"{self.key} must be 4D AHWC, got {tuple(obs.shape)}")
        return obs

    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        leaf = obs_spec[self.key]
        obs_spec[self.key] = UnboundedContinuous(
            shape=leaf.shape,
            device=leaf.device,
            dtype=self.dtype,
        )
        return obs_spec

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
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
        return base_port + int(env_instance_index) * SLOT_PORT_STRIDE

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
            seed=seed,
            enable_video=cfg["enable_video"] if enable_video is None else enable_video,
            record_every=cfg["record_every"],
            video_fps=cfg["video_fps"],
            verbose=cfg.get("verbose", False),
            daemon=cfg["daemon"],
            resize_width=128,
            resize_height=72,
        )

    def _build_training_env(self, seed: int, env_instance_index: int):
        cfg = self.config
        env = PettingZooWrapper(
            env=self.build_parallel_env(seed=seed, env_instance_index=env_instance_index)
        )
        env = TransformedEnv(env, Compose(
            SelectTransform(("agent", "observation"), ("agent", "info")),
            AHWCToTensor(key=("agent", "observation")),
            RemoveEmptySpecs(),
        ))
        env = env.to(cfg.get("sampling_device", "cpu"))
        return env

    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: int | None, device=None):
        def _make():
            env_instance_index = self._allocate_env_instance_index()
            return self._build_training_env(seed=seed, env_instance_index=env_instance_index)

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
        Fallback to ONE_GROUP_PER_AGENT using known agent names.
        """
        # Try to use the env's own group_map (already a dict on PettingZooWrapper)
        if env is not None:
            gm = getattr(env, "group_map", None)
            if isinstance(gm, dict) and gm:
                return gm
            # Fallback: build one-group-per-agent from possible_agents if present
            agents = getattr(env, "possible_agents", None)
            if agents:
                return MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(list(agents))

        # Last-resort fallback using config’s num_agents
        n = int(self.config.get("num_agents", 2))
        agents = [f"agent_{i}" for i in range(n)]
        return MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(agents)

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
    "qmix": QmixConfig,
    "masac": MasacConfig,
}


def main():
    ap = ArgumentParser()
    # Env args
    ap.add_argument("--scenario", type=str, default="pitfall_multi_agent")
    ap.add_argument("--num_agents", type=int, default=2)
    ap.add_argument("--resolution", type=str, default="160X120")
    ap.add_argument("--skip_frames", type=int, default=4)
    ap.add_argument("--async-mode", action=BooleanOptionalAction, default=False)
    ap.add_argument("--host_address", type=str, default="127.0.0.1")
    ap.add_argument("--base_port", type=int, default=DEFAULT_BASE_UDP_PORT)
    ap.add_argument("--netmode", type=int, default=0)
    ap.add_argument("--ticrate", type=int, default=None)
    ap.add_argument("--verbose", action='store_true', default=False)
    ap.add_argument("--daemon", dest="daemon", action=BooleanOptionalAction, default=True)

    # Train args
    ap.add_argument("--algo", type=str, default="mappo", choices=list(ALGOS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total_steps", type=float, default=1e6)
    ap.add_argument("--train_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sampling_device", type=str, default="cpu")
    ap.add_argument("--buffer_device", type=str, default="cpu")
    ap.add_argument("--rollout_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.1)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=1.0)
    ap.add_argument("--num_minibatches", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=8)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--parallel_collection", action='store_true', default=True)

    # Video recording
    ap.add_argument("--enable_video", action=BooleanOptionalAction, default=True)
    ap.add_argument("--record_every", type=int, default=100)
    ap.add_argument("--video_fps", type=int, default=35)
    ap.add_argument("--render_mode", type=str, default="rgb_array", choices=["rgb_array", "human"])

    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if not args.async_mode and args.ticrate is not None:
        raise ValueError("--ticrate can only be set when --async-mode is enabled")
    args.ticrate = int(args.ticrate) if args.ticrate is not None else vzd.DEFAULT_TICRATE
    root_path = Path(__file__).parent.parent.parent
    checkpoints_path = root_path / "checkpoints"
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{args.algo}_{args.scenario}"
        f"_{args.num_agents}agents_seed{args.seed}_{run_timestamp}"
    )

    if args.algo not in ALGOS:
        raise NotImplementedError(f"{args.algo} is not currently implemented in this script")

    algo_cfg = ALGOS[args.algo](
        share_param_critic=True,  # share critic across agents
        clip_epsilon=args.clip_eps,  # PPO clip
        entropy_coef=args.entropy_coef,  # entropy bonus
        critic_coef=args.vf_coef,  # value loss coef
        loss_critic_type="l2",  # or "smooth_l1" (Huber)
        lmbda=args.gae_lambda,  # GAE lambda
        scale_mapping="biased_softplus_1.0",  # softplus
        use_tanh_normal=True,  # use tanh Gaussian here
        minibatch_advantage=False,  # compute adv per minibatch
    )

    # Nature-style front end + 512 MLP head
    cnn_num_cells = [32, 64, 64]
    cnn_kernel_sizes = [8, 4, 3]
    cnn_strides = [4, 2, 1]
    cnn_paddings = [0, 0, 0]
    cnn_activation_class = nn.ReLU
    mlp_num_cells = [512]
    mlp_layer_class = nn.Linear
    mlp_activation_class = nn.ReLU

    model_cfg = CnnConfig(
        cnn_num_cells=cnn_num_cells,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
        cnn_paddings=cnn_paddings,
        cnn_activation_class=cnn_activation_class,
        mlp_num_cells=mlp_num_cells,
        mlp_layer_class=mlp_layer_class,
        mlp_activation_class=mlp_activation_class,
    )

    critic_cfg = CnnConfig(
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

    # compute any derived values first
    on_policy_minibatch_size = max(1, args.batch_size // max(1, args.num_minibatches))

    # only the fields you want to control from CLI
    overrides = {
        "sampling_device": args.sampling_device,
        "train_device": args.train_device,
        "buffer_device": args.buffer_device,
        "share_policy_params": True,
        "parallel_collection": args.parallel_collection,
        "max_n_frames": int(args.total_steps),
        "gamma": args.gamma,
        "lr": args.lr,

        # on-policy collection
        "on_policy_collected_frames_per_batch": args.rollout_steps,
        "on_policy_n_envs_per_worker": args.num_envs,
        "on_policy_n_minibatch_iters": args.num_epochs,
        "on_policy_minibatch_size": on_policy_minibatch_size,

        # eval / logging / ckpts
        "evaluation": False, # Disable as I think this is not necessary + it causes sudden drop in throughput
        "render": False,
        "evaluation_interval": args.rollout_steps * 25,
        "evaluation_episodes": 5,
        "loggers": ["wandb"],
        "project_name": "benchmarl-vizdoom",
        "save_folder": str(checkpoints_path),
        "checkpoint_interval": args.rollout_steps * 100,
        "checkpoint_at_end": True,
    }

    # apply safely (only set known fields; skip Nones)
    valid = {f.name for f in fields(ExperimentConfig)}
    for k, v in overrides.items():
        if v is not None and k in valid:
            setattr(exp_cfg, k, v)

    # keep eval interval aligned with horizon (collector-friendly)
    h = exp_cfg.on_policy_collected_frames_per_batch
    if h and exp_cfg.evaluation_interval % h != 0:
        exp_cfg.evaluation_interval = ((exp_cfg.evaluation_interval + h - 1) // h) * h

    task_cfg = {
        "scenario": args.scenario,
        "num_agents": args.num_agents,
        "resolution": args.resolution,
        "skip_frames": args.skip_frames,
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
    }
    task = VizdoomTask(task_cfg)

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
