#!/usr/bin/env python3
#####################################################################
# Example script of training agents with stable-baselines3
# on ViZDoom using the Gymnasium API
#
# Note: For this example to work, you need to install stable-baselines3 and opencv:
#       pip install stable-baselines3 opencv-python
#
# See more stable-baselines3 documentation here:
#   https://stable-baselines3.readthedocs.io/en/master/index.html
#####################################################################

from argparse import ArgumentParser

import cv2
import gymnasium
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import vizdoom.gymnasium_wrapper  # noqa


DEFAULT_ENV = "VizdoomBasicNotifications-v1"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]  # type: ignore

# Height and width of the resized image
IMAGE_SHAPE = (60, 80)

# Training parameters
TRAINING_TIMESTEPS = int(1e6)
N_STEPS = 512
N_ENVS = 8
FRAME_SKIP = 4


class ObservationWrapper(gymnasium.ObservationWrapper):
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
        self.n_features = 256
        self.vectorizer = HashingVectorizer(n_features=self.n_features)

        # Create new observation space with the new shape
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)

        # Get audio observation space if available
        if "audio" in env.observation_space.spaces:
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "screen": gymnasium.spaces.Box(
                        0, 255, shape=new_shape, dtype=np.uint8
                    ),
                    "audio": env.observation_space["audio"],
                }
            )
        elif "notifications" in env.observation_space.spaces:
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "screen": gymnasium.spaces.Box(
                        0, 255, shape=new_shape, dtype=np.uint8
                    ),
                    "notifications": gymnasium.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.n_features,),
                        dtype=np.float32,
                    )
                    # "notifications": env.observation_space["notifications"]
                }
            )
        else:
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "screen": gymnasium.spaces.Box(
                        0, 255, shape=new_shape, dtype=np.uint8
                    )
                }
            )

    def observation(self, observation):
        if "notifications" in self.observation_space.spaces:
            notif = observation["notifications"]
            if isinstance(notif, str):
                notif_vector = (
                    self.vectorizer.fit_transform([notif])
                    .toarray()
                    .astype(np.float32)[0]
                )
            else:
                notif_vector = notif
            observation = {
                "screen": cv2.resize(observation["screen"], self.image_shape_reverse),
                "notifications": notif_vector,
            }
        else:
            observation = {
                "screen": cv2.resize(observation["screen"], self.image_shape_reverse)
            }
        return observation


def main(args):
    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations (takes only the image and resizes it)
    #  2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, and we scale the rewards by 1/100
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    envs = make_vec_env(
        args.env,
        n_envs=N_ENVS,
        wrapper_class=wrap_env,
        env_kwargs=dict(frame_skip=FRAME_SKIP),
    )

    agent = PPO(
        "MultiInputPolicy",
        envs,
        n_steps=N_STEPS,
        verbose=2,
        tensorboard_log=f"{args.dir}",
    )

    # Do the actual learning
    # This will print out the results in the console.
    # If agent gets better, "ep_rew_mean" should increase steadily

    try:
        agent.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            progress_bar=True,
        )
    except ImportError:
        agent.learn(total_timesteps=TRAINING_TIMESTEPS)


if __name__ == "__main__":
    parser = ArgumentParser("Train stable-baselines3 PPO agents on ViZDoom.")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )
    parser.add_argument("--dir", default="./logs/")
    parser.add_argument("--exp", default="Notif")
    args = parser.parse_args()
    main(args)
