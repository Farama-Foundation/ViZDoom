#!/usr/bin/env python3

#####################################################################
# Example script of training agents with stable-baselines3
# on ViZDoom using the Gym API
#
# Note: ViZDoom must be installed with optional gym dependencies:
#         pip install vizdoom[gym]
#       You also need stable-baselines3:
#         pip install stable-baselines3
#
# See more stable-baselines3 documentation here:
#   https://stable-baselines3.readthedocs.io/en/master/index.html
#####################################################################

from argparse import ArgumentParser

import cv2
import numpy as np
import gym
import vizdoom.gym_wrapper

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

DEFAULT_ENV = "VizdoomBasic-v0"
AVAILABLE_ENVS = [env for env in [env_spec.id for env_spec in gym.envs.registry.all()] if "Vizdoom" in env]
# Height and width of the resized image
IMAGE_SHAPE = (60, 80)

# Training parameters
TRAINING_TIMESTEPS = int(1e6)
N_STEPS = 128
N_ENVS = 8
FRAME_SKIP = 4


class ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """
    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        num_channels = env.observation_space["rgb"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation["rgb"], self.image_shape_reverse)
        return observation


def main(args):
    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations (takes only the image and resizes it)
    #  2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, and we scale the rewards by 1/100
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    envs = make_vec_env(
        args.env,
        n_envs=N_ENVS,
        wrapper_class=wrap_env
    )

    agent = PPO("CnnPolicy", envs, n_steps=N_STEPS, verbose=1)

    # Do the actual learning
    # This will print out the results in the console.
    # If agent gets better, "ep_rew_mean" should increase steadily
    agent.learn(total_timesteps=TRAINING_TIMESTEPS)


if __name__ == "__main__":
    parser = ArgumentParser("Train stable-baselines3 PPO agents on ViZDoom.")
    parser.add_argument("--env",
                        default=DEFAULT_ENV,
                        choices=AVAILABLE_ENVS,
                        help="Name of the environment to play")
    args = parser.parse_args()
    main(args)
