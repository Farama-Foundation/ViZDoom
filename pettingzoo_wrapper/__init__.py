# public factory
import os
from pathlib import Path
from typing import Optional

from .base_pettingzoo_env import VizdoomParallelEnv
from .reward_wrappers import PitfallRewardWrapper
from .video_recorder import VideoLoggerParallelWrapper

# where the scenario .cfg files live
_SCENARIO_DIR = os.path.join(Path(__file__).parent.parent, "scenarios")

# scenario-specific wrappers
_WRAPPERS = {
    "pitfall_multi_agent": PitfallRewardWrapper,
}


def make(
        *,
        scenario: Optional[str] = None,
        config_file: Optional[str] = None,
        # env kwargs
        num_agents: int = 2,
        resolution: str = "160X120",
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
        # video logging
        enable_video: bool = True,
        record_every: int = 100,  # every N episodes
        video_fps: int = 35,
        daemon: bool = True,
):
    scenario = scenario.lower()
    cfg = config_file if config_file is not None else f"{_SCENARIO_DIR}/{scenario}.cfg"

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
        daemon=daemon,
    )

    if enable_video:
        env = VideoLoggerParallelWrapper(
            env,
            every_n_episodes=record_every,
            fps=video_fps,
        )

    if scenario in _WRAPPERS:
        return _WRAPPERS[scenario](env)

    return env
