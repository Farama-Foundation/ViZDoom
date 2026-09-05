# public factory
import os
from pathlib import Path
from typing import Optional, Sequence

import vizdoom as vzd
from vizdoom.pettingzoo_wrapper.base_pettingzoo_env import VizdoomParallelEnv
from vizdoom.pettingzoo_wrapper.info_wrappers import InternalInfoFilter
from vizdoom.pettingzoo_wrapper.reward_wrappers import (
    DeathmatchRewardWrapper,
    HealthGatheringRewardWrapper,
    HideAndSeekRewardWrapper,
    PitfallRewardWrapper,
    RemedyRushRewardWrapper,
    SimpleTagRewardWrapper,
)
from vizdoom.pettingzoo_wrapper.video_recorder import VideoLoggerParallelWrapper


# where the scenario .cfg files live
_SCENARIO_DIR = os.path.join(Path(__file__).parent.parent, "scenarios")

# scenario-specific wrappers
_WRAPPERS = {
    "simple_tag": SimpleTagRewardWrapper,
    "simple_tag_audio": SimpleTagRewardWrapper,
    "multi_duel": DeathmatchRewardWrapper,
    "multi_duel_hide_and_seek": HideAndSeekRewardWrapper,
    "multi_duel_pistol": DeathmatchRewardWrapper,
    "multi_duel_pistol_big": DeathmatchRewardWrapper,
    "pitfall_multi_agent": PitfallRewardWrapper,
    "remedy_rush_multi_agent": RemedyRushRewardWrapper,
    "health_gathering_multi_agent": HealthGatheringRewardWrapper,
}


def make(
    *,
    scenario: str = None,
    config_file: Optional[str] = None,
    # env kwargs
    num_agents: int = 2,
    resolution: str = "160X120",
    timeout: Optional[int] = None,
    barrier_timeout: Optional[float] = None,
    skip_frames: Optional[int] = 1,
    frame_stack: int = 1,
    async_mode: bool = False,
    host_address: str = "127.0.0.1",
    port: int = 5029,
    slot_index: int = 0,
    netmode: int = 0,
    ticrate: int = 35,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    # video logging
    enable_video: bool = True,
    record_every: int = 100,  # every N episodes
    video_fps: int = 35,
    verbose: bool = False,
    daemon: bool = True,
    available_buttons: Optional[Sequence[vzd.Button]] = None,
):
    scenario = scenario.lower() if scenario is not None else None
    cfg = config_file if config_file is not None else f"{_SCENARIO_DIR}/{scenario}.cfg"

    env = VizdoomParallelEnv(
        config_file=cfg,
        num_agents=num_agents,
        resolution=resolution,
        timeout=timeout,
        barrier_timeout=barrier_timeout,
        skip_frames=skip_frames,
        frame_stack=frame_stack,
        async_mode=async_mode,
        host_address=host_address,
        port=port,
        slot_index=slot_index,
        netmode=netmode,
        ticrate=ticrate,
        render_mode=render_mode,
        seed=seed,
        verbose=verbose,
        daemon=daemon,
        available_buttons=available_buttons,
    )

    if enable_video:
        env = VideoLoggerParallelWrapper(
            env,
            every_n_episodes=record_every,
            fps=video_fps,
        )

    if scenario in _WRAPPERS:
        env = _WRAPPERS[scenario](env)

    # filter info key that is not in its spec
    return InternalInfoFilter(env)
