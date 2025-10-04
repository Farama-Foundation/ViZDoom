import argparse
import multiprocessing as mp
import os
import socket
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from benchmarl.algorithms import MappoConfig, QmixConfig, MasacConfig
from benchmarl.environments import TaskClass
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import CnnConfig
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import Composite
from torchrl.data.tensor_specs import UnboundedContinuous
from torchrl.envs import EnvCreator, EnvBase, RemoveEmptySpecs
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
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.mode = mode
        self.antialias = antialias

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is the leaf tensor for self.key; we expect (A,H,W,C)
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        # normalize?
        needs_div255 = self.from_int if self.from_int is not None else (not torch.is_floating_point(obs))
        if needs_div255:
            obs = obs.div(255)
        obs = obs.to(self.dtype)

        if obs.ndim != 4:
            raise ValueError(f"{self.key} must be 4D AHWC, got {tuple(obs.shape)}")
        A, H, W, C = obs.shape

        # Resize through NCHW path for interpolate, then back to AHWC
        x = obs.permute(0, 3, 1, 2)  # A,C,H,W
        align = dict(align_corners=False) if self.mode in ("bilinear", "bicubic") else {}
        x = F.interpolate(x, size=(self.h, self.w), mode=self.mode, antialias=self.antialias, **align)
        x = x.permute(0, 2, 3, 1).contiguous()  # A,h,w,C
        return x

    # @_apply_to_composite
    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        leaf = obs_spec[self.key]
        A, H, W, C = leaf.shape
        obs_spec[self.key] = UnboundedContinuous(
            shape=torch.Size([A, self.h, self.w, C]),
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

            def _pick_free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))  # 0 = ask OS for a free port
                    return s.getsockname()[1]

            # ensure each env instance gets its own port
            port = _pick_free_port()

            pz_env = make(
                scenario=cfg["scenario"],
                num_agents=int(cfg.get("num_agents", 2)),
                resolution=str(cfg.get("resolution", "160x120")),
                skip_frames=cfg.get("skip_frames", 4),
                async_mode=bool(cfg.get("async_mode", True)),
                host_address=str(cfg.get("host_address", "127.0.0.1")),
                port=port,
                netmode=int(cfg.get("netmode", 1)),
                ticrate=int(cfg.get("ticrate", 35)),
                use_multi_binary_action_space=False,
                seed=seed,
            )
            env = PettingZooWrapper(
                env=pz_env,
            )
            env = TransformedEnv(env, Compose(
                SelectTransform(("agent", "observation")),
                AHWCToTensorResize(key=("agent", "observation"), h=72, w=96, mode="bilinear"),
                RemoveEmptySpecs(),
            ))
            env = env.to(cfg.get("device", "cpu"))
            return env

        return EnvCreator(_make)

    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: int | None, device=None):
        make_single = self.env_creator(seed)
        # return make_single if num_envs == 1 else EnvCreator(lambda: ParallelEnv(available_cpu_count(), make_single))
        return make_single

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
    ap.add_argument("--port", type=int, default=5029)
    ap.add_argument("--netmode", type=int, default=1)
    ap.add_argument("--ticrate", type=int, default=35)

    # Train args
    ap.add_argument("--algo", type=str, default="mappo", choices=list(ALGOS))
    ap.add_argument("--model", type=str, default="cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_steps", type=int, default=1_000_000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rollout_steps", type=int, default=256)  # PPO horizon
    ap.add_argument("--batch_size", type=int, default=16384)  # PPO batch
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--num_minibatches", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=2)
    args = ap.parse_args()

    root_path = Path(__file__).parent.parent.parent
    checkpoints_path = root_path / "checkpoints"
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)

    if args.algo == "mappo":
        # Required ctor args for your MAPPO version
        algo_cfg = MappoConfig(
            share_param_critic=True,  # share critic across agents
            clip_epsilon=args.clip_eps,  # PPO clip
            entropy_coef=args.entropy_coef,  # entropy bonus
            critic_coef=args.vf_coef,  # value loss coef
            loss_critic_type="l2",  # or "smooth_l1" (Huber)
            lmbda=args.gae_lambda,  # GAE lambda
            scale_mapping="identity",  # no scaling of actions
            use_tanh_normal=False,  # not using tanh Gaussian here
            minibatch_advantage=True,  # compute adv per minibatch
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

    exp_cfg = ExperimentConfig(
        # devices / execution
        sampling_device=args.device,
        train_device=args.device,  # "cuda" or "cpu"
        buffer_device=args.device,
        share_policy_params=True,
        prefer_continuous_actions=False,
        collect_with_grad=False,
        parallel_collection=False,

        # RL basics
        gamma=args.gamma,
        lr=args.lr,
        adam_eps=1e-5,
        clip_grad_norm=0.5,
        clip_grad_val=None,

        # target updates (off-policy; harmless defaults for PPO/MAPPO)
        soft_target_update=False,
        polyak_tau=0.995,
        hard_target_update_frequency=None,

        # ε-greedy (ignored by PPO/MAPPO; keep sane defaults)
        exploration_eps_init=0.2,
        exploration_eps_end=0.01,
        exploration_anneal_frames=100_000,

        # training budget
        max_n_iters=None,
        max_n_frames=args.total_steps,

        # on-policy collection (PPO/MAPPO)
        on_policy_collected_frames_per_batch=args.rollout_steps,
        on_policy_n_envs_per_worker=1,
        on_policy_n_minibatch_iters=args.num_epochs,
        on_policy_minibatch_size=max(1, args.batch_size // max(1, args.num_minibatches)),

        # off-policy knobs (unused by MAPPO)
        off_policy_collected_frames_per_batch=None,
        off_policy_n_envs_per_worker=None,
        off_policy_n_optimizer_steps=None,
        off_policy_train_batch_size=None,
        off_policy_memory_size=None,
        off_policy_init_random_frames=None,
        off_policy_use_prioritized_replay_buffer=False,
        off_policy_prb_alpha=None,
        off_policy_prb_beta=None,

        # eval / logging / ckpts
        evaluation=True,
        render=False,
        evaluation_interval=args.rollout_steps * 30,
        evaluation_episodes=5,
        evaluation_deterministic_actions=True,
        evaluation_static=False,
        loggers=["wandb"],
        project_name="benchmarl-vizdoom",
        create_json=False,
        save_folder=checkpoints_path,
        restore_file=None,
        restore_map_location="cpu",
        checkpoint_interval=args.rollout_steps * 100,
        checkpoint_at_end=True,
        keep_checkpoints_num=3,
    )

    task_cfg = {
        "scenario": args.scenario,
        "num_agents": args.num_agents,
        "resolution": args.resolution,
        "skip_frames": args.skip_frames,
        "async_mode": bool(args.async_mode),
        "host_address": args.host_address,
        "port": args.port,
        "netmode": args.netmode,
        "ticrate": args.ticrate,
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
