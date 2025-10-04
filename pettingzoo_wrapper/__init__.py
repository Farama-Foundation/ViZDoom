# public factory
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .base_pettingzoo_env import VizdoomParallelEnv
from .reward_wrappers import PitfallRewardWrapper
from .video_recorder import VideoLoggerParallelWrapper

# where your .cfg files live (package data or repo path)
_SCENARIO_DIR = os.path.join(Path(__file__).parent.parent, "scenarios")

# scenario -> { cfg, wrapper, wrapper_defaults }
_SCENARIOS = {
    "pitfall": {
        "cfg": "pitfall.cfg",
        "wrapper": PitfallRewardWrapper,
        "defaults": dict(scaler=0.1, death_penalty=-1.0, keep_lb=True, goal_x=None, goal_reward=1.0),
    },
    # add others here
}


def _resolve_cfg(scenario: str) -> str:
    key = scenario.lower()
    if key not in _SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Known: {list(_SCENARIOS)}")
    p = os.path.join(_SCENARIO_DIR, _SCENARIOS[key]["cfg"])
    if not os.path.exists(p):
        raise FileNotFoundError(f"Config not found: {p}")
    return p


def make(
        *,
        scenario: Optional[str] = None,
        config_file: Optional[str] = None,
        # env kwargs
        num_agents: int = 2,
        resolution: str = "160x120",
        timeout: Optional[int] = None,
        skip_frames: Optional[int] = 1,
        async_mode: bool = True,
        host_address: str = "127.0.0.1",
        port: int = 5029,
        netmode: int = 0,
        ticrate: int = 35,
        render_mode: Optional[str] = None,
        use_multi_binary_action_space: bool = True,
        seed: Optional[int] = None,
        # reward
        reward: str = "auto",  # "auto" | "none" | scenario name
        reward_params: Optional[Dict[str, Any]] = None,
        # video logging
        enable_video: bool = True,
        record_every: int = 50,  # every N episodes
        video_fps: int = 35,
):
    cfg = config_file if config_file is not None else _resolve_cfg(scenario)

    env = VizdoomParallelEnv(
        config_file=cfg,
        num_agents=num_agents,
        resolution=resolution,
        timeout=timeout,
        skip_frames=skip_frames,
        async_mode=async_mode,
        host_address=host_address,
        port=port,
        netmode=netmode,
        ticrate=ticrate,
        render_mode=render_mode,
        use_multi_binary_action_space=use_multi_binary_action_space,
        seed=seed,
    )

    if reward == "none":
        return env

    # pick wrapper key
    wrapper_key = None
    if reward == "auto":
        # use scenario (from cfg basename) if known
        scen_key = os.path.splitext(os.path.basename(cfg))[0].lower()
        if scen_key in _SCENARIOS:
            wrapper_key = scen_key
    else:
        wrapper_key = reward.lower()

    if enable_video:
        env = VideoLoggerParallelWrapper(
            env,
            every_n_episodes=record_every,
            fps=video_fps,
        )

    if wrapper_key and wrapper_key in _SCENARIOS and _SCENARIOS[wrapper_key]["wrapper"]:
        params = {**_SCENARIOS[wrapper_key].get("defaults", {}), **(reward_params or {})}
        return _SCENARIOS[wrapper_key]["wrapper"](env, **params)

    return env
