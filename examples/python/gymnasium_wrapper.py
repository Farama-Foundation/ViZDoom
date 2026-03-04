#!/usr/bin/env python3

#####################################################################
# Example for running a vizdoom scenario as a Gymnasium env
#
# To see the list of available environments go to
# https://vizdoom.farama.org/main/environments/default/
#####################################################################

import time
from argparse import ArgumentParser

import gymnasium
import numpy as np

import vizdoom as vzd

# Importing the wrapper registers the ViZDoom environments in Gymnasium, so it should be imported before creating the environment
from vizdoom import gymnasium_wrapper  # noqa


DEFAULT_ENV = "VizdoomBasic-v1"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]  # type: ignore


def _print_obs_space(obs, obs_space, indent=0):
    """
    Help function to print the observation space and some stats about the observation
    """
    prefix = " " * indent
    if isinstance(obs_space, gymnasium.spaces.Dict):
        print(f"{prefix}Dict:")
        for key, space in obs_space.spaces.items():
            print(f"{prefix} Key: {key}")
            _print_obs_space(obs[key], space, indent + 4)
    else:
        print(f"{prefix}Space: {obs_space}")
        if isinstance(obs, np.ndarray):
            print(
                f"{prefix} Shape: {obs.shape}, Dtype: {obs.dtype}, Min value: {obs.min()}, Max value: {obs.max()}, Mean value: {obs.mean()}"
            )


if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use Gymnasium wrapper.")
    parser.add_argument(
        dest="env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )

    # Create the Gymnasium environment
    env = gymnasium.make(
        # Env ID, render_mode and frame_skip can be changed as needed
        parser.parse_args().env,
        render_mode="human",
        frame_skip=4,
        # Additional parameters can be passed to override the default environment config, however they should not be used for evaluation
        # Any kwargs that are supported by vizdoom.DoomGame.set_config can be passed here
        screen_resolution=vzd.ScreenResolution.RES_640X480,
    )

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs, info = env.reset(seed=42)
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            print("Observation:")
            _print_obs_space(obs, env.observation_space)
            print(f"Reward: {rew}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print(f"Info: {info}")
            print("=====================")

            time.sleep(
                1.0 / vzd.DEFAULT_TICRATE * env.unwrapped.frame_skip
            )  # Make it run at real-time speed

    env.close()
