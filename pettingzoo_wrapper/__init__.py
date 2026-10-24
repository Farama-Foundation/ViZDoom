# public factory
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import vizdoom as vzd
from vizdoom.pettingzoo_wrapper.base_pettingzoo_env import VizdoomParallelEnv
from vizdoom.pettingzoo_wrapper.communication import CommunicationWrapper
from vizdoom.pettingzoo_wrapper.info_wrappers import InternalInfoFilter
from vizdoom.pettingzoo_wrapper.reward_wrappers import (
    CommunicationRewardWrapper,
    DeathmatchRewardWrapper,
    HealthGatheringRewardWrapper,
    HideAndSeekRewardWrapper,
    PitfallRewardWrapper,
    RemedyRushRewardWrapper,
    SharedRewardWrapper,
    SimpleTagRewardWrapper,
)
from vizdoom.pettingzoo_wrapper.vector_obs import (
    SCENARIO_VECTOR_OBS,
    VectorStateObservationWrapper,
)


# from vizdoom.pettingzoo_wrapper.video_recorder import VideoLoggerParallelWrapper


# where the scenario .cfg files live
_SCENARIO_DIR = os.path.join(Path(__file__).parent.parent, "scenarios")

# Only one agent moved and chat enabled
COMMUNICATION_SCENARIOS = ("simple_communication", "simple_no_communication")
# Both agents move and chat channel disabled
NO_CHAT_SCENARIOS = ("simple_no_communication",)

# scenario-specific wrappers
# COMMUNICATION_SCENARIOS are built separately in make(), so they are configured separately below
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
    enable_video: bool = True,
    record_every: int = 100,  # every N episodes
    video_fps: int = 35,
    verbose: bool = False,
    daemon: bool = True,
    available_buttons: Optional[Sequence[vzd.Button]] = None,
    communication: bool = True,
    reward_kwargs: Optional[dict] = None,
    chat_tokens: Optional[Sequence[str]] = None,
    chat_cooldown_tics: int = 35,
    delivery: str = "natural",
    delivery_seed: Optional[int] = None,
    observer_info: bool = False,
    vector_obs: bool = False,
    privileged_info: bool = False,
    reward_mode: str = "individual",
    shared_reward_agg: str = "sum",
    continuous_actions: bool = False,
):
    scenario = scenario.lower() if scenario is not None else None
    cfg = config_file if config_file is not None else f"{_SCENARIO_DIR}/{scenario}.cfg"

    if scenario is None and config_file is not None:
        if Path(config_file).stem.lower() in COMMUNICATION_SCENARIOS:
            scenario = Path(config_file).stem.lower()
    if scenario in COMMUNICATION_SCENARIOS:
        if num_agents != 2:
            raise ValueError(f"{scenario} requires exactly two agents")
        if async_mode:
            raise ValueError(f"{scenario} requires synchronous play")
    if scenario in NO_CHAT_SCENARIOS:
        communication = False

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
        privileged_info=privileged_info,
        continuous_actions=continuous_actions,
    )

    # if enable_video:
    #     env = VideoLoggerParallelWrapper(
    #         env,
    #         every_n_episodes=record_every,
    #         fps=video_fps,
    #     )

    if scenario in COMMUNICATION_SCENARIOS:
        try:
            env = CommunicationWrapper(
                env,
                communication=communication,
                chat_tokens=chat_tokens,
                chat_cooldown_tics=chat_cooldown_tics,
                delivery=delivery,
                delivery_seed=delivery_seed,
                observer_info=observer_info,
            )
            env = CommunicationRewardWrapper(env, **(reward_kwargs or {}))
        except Exception:
            env.close()
            raise
    elif scenario in _WRAPPERS:
        env = _WRAPPERS[scenario](env)

    if reward_mode == "shared":
        env = SharedRewardWrapper(env, agg=shared_reward_agg)
    elif reward_mode != "individual":
        raise ValueError(
            f"reward_mode must be 'individual' or 'shared', got {reward_mode!r}"
        )

    if vector_obs:
        spec = SCENARIO_VECTOR_OBS.get(scenario)
        if spec is None:
            raise ValueError(
                f"vector_obs=True but '{scenario!r}' has no entry in SCENARIO_VECTOR_OBS."
            )
        env = VectorStateObservationWrapper(env, spec)

    # filter info key that is not in its spec
    return InternalInfoFilter(env)
