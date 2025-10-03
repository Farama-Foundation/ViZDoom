import argparse
import os
import socket
from copy import deepcopy

import numpy as np
import ray
from gymnasium.spaces import Box
from ray import tune, air
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from pettingzoo_wrapper import make

try:
    import cv2

    _USE_CV2 = True
except Exception:
    from PIL import Image

    _USE_CV2 = False

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn


class VizdoomCNN(TorchModelV2, nn.Module):
    """
    CNN head that infers (H, W, C) from obs_space.shape (channel-last).
    Optional overrides via custom_model_config:
        - in_channels: int
        - in_size: (H, W) tuple
        - hidden_size: int (default 256)
        - conv_spec: list like [[32,[8,8],4],[64,[4,4],2],[64,[3,3],1]]
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Infer from obs_space (H, W, C). Allow overrides.
        H, W, C = obs_space.shape
        cfg = model_config.get("custom_model_config", {})
        in_channels = int(cfg.get("in_channels", C))
        in_size = tuple(cfg.get("in_size", (H, W)))
        hidden = int(cfg.get("hidden_size", 256))
        conv_spec = cfg.get("conv_spec", [
            [32, [8, 8], 4],  # out: floor((H-8)/4 + 1)
            [64, [4, 4], 2],
            [64, [3, 3], 1],
        ])

        # Build conv stack from spec
        layers = []
        c = in_channels
        for out_c, k, s in conv_spec:
            kH, kW = k
            layers += [nn.Conv2d(c, out_c, kernel_size=(kH, kW), stride=(s, s)), nn.ReLU()]
            c = out_c
        self.conv = nn.Sequential(*layers)

        # Compute flatten dim from inferred size (no hardcoding)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_size[0], in_size[1])  # CPU is fine
            n_flat = int(self.conv(dummy).view(1, -1).shape[1])

        self.fc = nn.Sequential(nn.Linear(n_flat, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, num_outputs)
        self.v = nn.Linear(hidden, 1)
        self._last_features = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H, W]
        x = self.conv(x).view(x.size(0), -1)
        feat = self.fc(x)
        self._last_features = feat
        logits = self.pi(feat)
        return logits, state

    def value_function(self):
        return self.v(self._last_features).squeeze(-1)


class ResizeNormalizePZ:
    """PettingZoo Parallel wrapper: resize to 84x84 and normalize to [0,1] float32."""

    def __init__(self, env, size=(84, 84), normalize=True):
        self.env = env
        self.size = tuple(size)  # (H, W)
        self.normalize = normalize

        low, high = (np.float32(0.0), np.float32(1.0)) if normalize else (0, 255)
        dtype = np.float32 if normalize else np.uint8
        self._obs_space = Box(low=low, high=high, shape=(self.size[0], self.size[1], 3), dtype=dtype)

    # ---- PZ parallel API passthroughs ----
    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return getattr(self.env, "possible_agents", None)

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), infos

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        return self._process_obs(obs), rews, terms, truncs, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    # ---- helpers ----
    def _resize_img(self, img):
        if _USE_CV2:
            # img: HxWxC (uint8)
            return cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        else:
            # PIL fallback
            return np.asarray(Image.fromarray(img).resize((self.size[1], self.size[0]), Image.BILINEAR))

    def _process_one(self, x):
        # assume HxWxC RGB uint8 coming from VizDoom/PZ
        x = self._resize_img(x)
        if self.normalize:
            x = x.astype(np.float32)
            x *= np.float32(1.0 / 255.0)
        return x

    def _process_obs(self, obs_dict):
        # obs_dict: {agent_id: np.ndarray}
        return {aid: self._process_one(ob) for aid, ob in obs_dict.items()}


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def make_pz_env(env_config):
    cfg = deepcopy(env_config)
    cfg["port"] = _free_port()
    base = make(**cfg)
    wrapped = ResizeNormalizePZ(base, size=(84, 84), normalize=True)
    return ParallelPettingZooEnv(wrapped)


def main():
    parser = argparse.ArgumentParser()
    # --- env ---
    parser.add_argument("--scenario", type=str, default="pitfall")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--resolution", type=str, default="160x120")
    parser.add_argument("--skip_frames", type=int, default=4)
    parser.add_argument("--async_mode", type=int, default=1)
    parser.add_argument("--port", type=int, default=5029)
    parser.add_argument("--netmode", type=int, default=1)
    parser.add_argument("--ticrate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # --- training / resources ---
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--num_gpus", type=float, default=1)
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--rollout_fragment_length", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)

    # --- wandb / run meta ---
    parser.add_argument("--exp_name", type=str, default="vizdoom-ippo")
    parser.add_argument("--wandb_project", type=str, default="vizdoom-marl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="ippo")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    ModelCatalog.register_custom_model("vizdoom_cnn", VizdoomCNN)

    # 1) Register env with RLlib
    env_name = "vizdoom_pz_env"
    register_env(env_name, make_pz_env)

    # 2) Build a tiny dummy env to grab per-agent spaces (needed for IPPO = one policy per agent)
    dummy_env = make(
        scenario=args.scenario,
        num_agents=args.num_agents,
        resolution=args.resolution,
        skip_frames=args.skip_frames,
        async_mode=bool(args.async_mode),
        host_address="127.0.0.1",
        port=_free_port(),
        netmode=args.netmode,
        ticrate=args.ticrate,
        use_multi_binary_action_space=False,
        seed=args.seed,
    )
    dummy_env = ResizeNormalizePZ(dummy_env, size=(84, 84), normalize=True)
    try:
        # PettingZoo wants reset before spaces are fully known sometimes
        dummy_env.reset()
        obs_space = dummy_env.observation_space(dummy_env.agents[0])
        act_space = dummy_env.action_space(dummy_env.agents[0])
    finally:
        dummy_env.close()

    # 3) Create one independent PPO policy *per* agent (IPPO)
    policies = {
        f"pi_{i}": (None, obs_space, act_space, {}) for i in range(args.num_agents)
    }

    def policy_mapping_fn(agent_id, *_, **__):
        # agent ids are "agent_0", "agent_1", ...
        idx = int(str(agent_id).split("_")[-1])
        return f"pi_{idx}"

    # 4) RLlib environment config
    env_config = dict(
        scenario=args.scenario,
        num_agents=args.num_agents,
        resolution=args.resolution,
        skip_frames=args.skip_frames,
        async_mode=bool(args.async_mode),
        host_address="127.0.0.1",
        port=args.port,
        netmode=args.netmode,
        ticrate=args.ticrate,
        use_multi_binary_action_space=False,
        seed=args.seed,
    )

    # 5) RLlib environment config (vision model for RGB frames)
    ppo_cfg = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)  # <- old stable stack
        .framework("torch")
        .environment(env=env_name, env_config=env_config)
        .resources(num_gpus=args.num_gpus)
        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length='auto',
            batch_mode="truncate_episodes",
        )
        .training(
            gamma=args.gamma,
            lr=args.lr,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            vf_clip_param=args.vf_clip_param,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            train_batch_size=args.train_batch_size,
            model={
                "_disable_preprocessor_api": True,
                "custom_model": "vizdoom_cnn",
                "vf_share_layers": True,
                "always_check_shapes": True,
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
        .debugging(log_level="INFO")
    )

    # 5) WandB callback — set WANDB_API_KEY in your env, or W&B will prompt you
    wandb_cb = WandbLoggerCallback(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.run_name or args.exp_name,
        log_config=True,
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            name=args.exp_name,
            stop={"timesteps_total": args.total_timesteps},
            callbacks=[wandb_cb],
            storage_path=os.path.expanduser("~/ray_results"),
            verbose=1,
        ),
        param_space=ppo_cfg.to_dict(),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
