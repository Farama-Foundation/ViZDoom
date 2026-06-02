#!/usr/bin/env python3

#####################################################################
# Example for running a vizdoom scenario as a Gymnasium env
#####################################################################

import gymnasium

from vizdoom import gymnasium_wrapper  # noqa
from vizdoom.gymnasium_wrapper.telemetry_obs_wrapper import TelemetryWrapper
from stable_baselines3 import PPO
#from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    env = gymnasium.make(
        "VizdoomHealthGatheringSupreme-v1", render_mode="rgb_array", frame_skip=4
    )
    env = TelemetryWrapper(env)

    model = PPO("MultiInputPolicy", env=env, verbose=1)
    model.learn(10000)
    