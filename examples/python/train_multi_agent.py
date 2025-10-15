import argparse
import multiprocessing as mp
import os
import socket
from dataclasses import fields
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from benchmarl.algorithms import QmixConfig, MasacConfig
from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.environments import TaskClass
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import CnnConfig
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import Composite
from torchrl.data.tensor_specs import UnboundedContinuous
from torchrl.envs import EnvCreator, EnvBase, RemoveEmptySpecs, ParallelEnv
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.libs.pettingzoo import MarlGroupMapType, PettingZooWrapper
from torchrl.envs.transforms import ObservationTransform
from torchrl.envs.transforms import SelectTransform
from torchrl.envs.transforms.utils import _set_missing_tolerance

from pettingzoo_wrapper import make


def available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return mp.cpu_count() or 1


class MappoOnDevice(Mappo):
    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(nested_done_key, batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)))
        if nested_terminated_key not in keys:
            batch.set(nested_terminated_key, batch.get(("next", "terminated")).unsqueeze(-1).expand((*group_shape, 1)))
        if nested_reward_key not in keys:
            batch.set(nested_reward_key, batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)))

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(-self.experiment.config.train_minibatch_size(self.on_policy) // batch.shape[1])
        else:
            increment = batch.batch_size[0] + 1
        last_start_index = 0
        start_index = increment
        minibatches = []
        while last_start_index < batch.shape[0]:
            minibatch = batch[last_start_index:start_index]
            minibatch = minibatch.to(self.device, non_blocking=True)  # Move the minibatch to device
            minibatches.append(minibatch)
            with torch.no_grad():
                loss.value_estimator(
                    minibatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        return torch.cat(minibatches, dim=0)


class MappoOnDeviceConfig(MappoConfig):
    @staticmethod
    def associated_class():
        return MappoOnDevice


class AHWCToTensorResize(ObservationTransform):
    """
    Keep AHWC layout, convert to float tensor, optional /255, then resize (H,W).
    In/out: (A,H,W,C) -> (A,h,w,C)
    """

    def __init__(
            self,
            key=("agent", "observation"),
            h: int = 72,
            w: int = 96,
            from_int: bool | None = None,  # True: /255, False: no, None: auto if not float
            dtype: torch.dtype | None = None,
            mode: str = "bilinear",
            antialias: bool = True,
    ):
        super().__init__(in_keys=[key], out_keys=[key])
        self.key = key
        self.h, self.w = int(h), int(w)
        self.from_int = from_int
        self.dtype = dtype if dtype is not None else torch.float32
        self.mode = mode
        self.antialias = antialias

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is the leaf tensor for self.key; we expect (A,H,W,C)
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        # normalize to [0, 1]
        obs = obs.div(255).to(self.dtype)

        if obs.ndim != 4:
            raise ValueError(f"{self.key} must be 4D AHWC, got {tuple(obs.shape)}")

        # Resize through NCHW path for interpolate, then back to AHWC
        x = obs.permute(0, 3, 1, 2)  # A,C,H,W
        align = dict(align_corners=False) if self.mode in ("bilinear", "bicubic") else {}
        x = F.interpolate(x, size=(self.h, self.w), mode=self.mode, antialias=self.antialias, **align)
        x = x.permute(0, 2, 3, 1).contiguous()  # A,h,w,C
        return x

    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        leaf = obs_spec[self.key]
        A, H, W, C = leaf.shape
        obs_spec[self.key] = UnboundedContinuous(
            shape=torch.Size([A, self.h, self.w, C]),
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
        super().__init__("doom", config)
        # Build a prototype env ONCE to fetch specs
        proto_env = self.env_creator(seed=config.get("seed", 0))()
        try:
            self._action_spec = proto_env.action_spec
            self._observation_spec = proto_env.observation_spec
            self._action_mask_spec = getattr(proto_env, "action_mask_spec", None)
            self._info_spec = getattr(proto_env, "info_spec", None)
            self._state_spec = getattr(proto_env, "state_spec", None)
            self._has_render = True
        finally:
            proto_env.close()

    @staticmethod
    def env_name() -> str:
        return "vizdoom"

    def env_creator(self, seed: int):
        # returns an EnvCreator (TorchRL) that builds the env when called
        def _make():
            cfg = self.config
            host_address = cfg.get("host_address", "127.0.0.1")

            def _pick_free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host_address, 0))  # 0 = ask OS for a free port
                    return s.getsockname()[1]

            # ensure each env instance gets its own port
            port = _pick_free_port()

            pz_env = make(
                scenario=cfg["scenario"],
                num_agents=int(cfg.get("num_agents", 2)),
                resolution=str(cfg.get("resolution", "160x120")),
                skip_frames=cfg.get("skip_frames", 4),
                async_mode=bool(cfg.get("async_mode", True)),
                render_mode=str(cfg.get("render_mode", "rgb_array")),
                host_address=str(host_address),
                port=port,
                netmode=int(cfg.get("netmode", 1)),
                ticrate=int(cfg.get("ticrate", 35)),
                use_multi_binary_action_space=False,
                seed=seed,
                enable_video=bool(cfg.get("enable_video", True)),
                record_every=int(cfg.get("record_every", 50)),
                video_fps=int(cfg.get("video_fps", 35)),
            )
            env = PettingZooWrapper(
                env=pz_env,
            )
            env = TransformedEnv(env, Compose(
                SelectTransform(("agent", "observation"), ("agent", "info")),
                AHWCToTensorResize(key=("agent", "observation"), h=72, w=96, mode="bilinear"),
                RemoveEmptySpecs(),
            ))
            env = env.to(cfg.get("sampling_device", "cpu"))
            return env

        return EnvCreator(_make)

    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: int | None, device=None):
        # make_single = self.env_creator(seed)
        # return make_single if num_envs == 1 else EnvCreator(lambda: ParallelEnv(available_cpu_count(), make_single))
        
        # Return non-vec env, avoid vizdoom processes + env vectorization double parallelising
        return self.env_creator(seed if seed is not None else 0)

    def action_spec(self, env: EnvBase) -> Composite:
        return self._action_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        return self._observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return self._action_mask_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return self._info_spec

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
        return self._has_render

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
    ap = argparse.ArgumentParser()
    # Env args
    ap.add_argument("--scenario", type=str, default="pitfall")
    ap.add_argument("--num_agents", type=int, default=2)
    ap.add_argument("--resolution", type=str, default="160x120")
    ap.add_argument("--skip_frames", type=int, default=4)
    ap.add_argument("--async_mode", type=int, default=1)
    ap.add_argument("--host_address", type=str, default="127.0.0.1")
    ap.add_argument("--netmode", type=int, default=1)
    ap.add_argument("--ticrate", type=int, default=35)
    ap.add_argument("--verbose", action='store_true', default=False)

    # Train args
    ap.add_argument("--algo", type=str, default="mappo", choices=list(ALGOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_steps", type=float, default=1e6)
    ap.add_argument("--train_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sampling_device", type=str, default="cpu")
    ap.add_argument("--buffer_device", type=str, default="cpu")
    ap.add_argument("--rollout_steps", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.9)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=1.0)
    ap.add_argument("--num_minibatches", type=int, default=15)
    ap.add_argument("--num_epochs", type=int, default=45)
    ap.add_argument("--num_envs", type=int, default=1)

    # Video recording
    ap.add_argument("--enable_video", type=bool, default=True)
    ap.add_argument("--record_every", type=int, default=50)
    ap.add_argument("--video_fps", type=int, default=35)

    args = ap.parse_args()

    root_path = Path(__file__).parent.parent.parent
    checkpoints_path = root_path / "checkpoints"
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)

    if args.algo == "mappo":
        # Required ctor args for your MAPPO version
        algo_cfg = MappoOnDeviceConfig(
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
    else:
        raise NotImplementedError(
            f"Only --algo mappo is wired without YAML in this script right now. Got: {args.algo}"
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
        "parallel_collection": True,
        "max_n_frames": int(args.total_steps),
        "lr": args.lr,

        # on-policy collection
        "on_policy_collected_frames_per_batch": args.rollout_steps,
        "on_policy_n_envs_per_worker": args.num_envs,
        "on_policy_n_minibatch_iters": args.num_epochs,
        "on_policy_minibatch_size": on_policy_minibatch_size,

        # eval / logging / ckpts
        "evaluation": True,
        "render": False,
        "evaluation_interval": args.rollout_steps * 25,
        "evaluation_episodes": 1,
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
        "async_mode": bool(args.async_mode),
        "host_address": args.host_address,
        "netmode": args.netmode,
        "ticrate": args.ticrate,
        "enable_video": args.enable_video,
        "record_every": args.record_every,
        "video_fps": args.video_fps,
        "sampling_device": args.sampling_device,
    }
    task = VizdoomTask(task_cfg)

    experiment = Experiment(
        task=task,
        algorithm_config=algo_cfg,
        model_config=model_cfg,
        critic_model_config=critic_cfg,
        seed=args.seed,
        config=exp_cfg,
    )

    Path(exp_cfg.save_folder).mkdir(parents=True, exist_ok=True)
    experiment.run()
    experiment.close()


if __name__ == "__main__":
    main()
