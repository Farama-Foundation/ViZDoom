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

import hashlib
from argparse import ArgumentParser

import cv2
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import vizdoom.gymnasium_wrapper  # noqa


DEFAULT_ENV = "VizdoomBasic-v1"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]  # type: ignore

# Height and width of the resized image
IMAGE_SHAPE = (60, 80)  # We use a very small image for faster training
NOTIFICATIONS_FEATURES = 16
IMAGE_BUFFER_KEYS = ("screen", "depth", "labels", "automap")
REWARD_SCALE = 0.01
LEARNING_RATE = 0.001

# Training parameters
MAX_TRAINING_TIMESTEPS = 1000000
N_STEPS = 128
N_ENVS = 16
FRAME_SKIP = 4


class ObservationWrapper(gymnasium.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.

    This wrapper keeps all available buffers, resizes every image-like buffer
    (screen/depth/labels/automap) to a smaller size, normalizes audio buffer,
    and converts notifications text into a fixed-size numeric vector via hashing.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(
        self, env, shape=IMAGE_SHAPE, notifications_features=NOTIFICATIONS_FEATURES
    ):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.notifications_features = notifications_features
        self.image_buffer_keys = [
            key for key in env.observation_space.spaces if key in IMAGE_BUFFER_KEYS
        ]
        self.current_instruction = ""

        spaces = {}
        for key, space in env.observation_space.spaces.items():
            if key in IMAGE_BUFFER_KEYS:
                num_channels = space.shape[-1] if len(space.shape) >= 3 else 1
                new_shape = (shape[0], shape[1], num_channels)
                spaces[key] = gymnasium.spaces.Box(
                    0, 255, shape=new_shape, dtype=np.uint8
                )
            elif key == "notifications":
                spaces[key] = gymnasium.spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.notifications_features,),
                    dtype=np.float32,
                )
            elif key == "audio":
                spaces[key] = gymnasium.spaces.Box(
                    -1.0,
                    1.0,
                    shape=space.shape,
                    dtype=np.float32,
                )
            else:
                spaces[key] = space

        self.observation_space = gymnasium.spaces.Dict(spaces)

    def _resize_buffer(self, buffer):
        resized = cv2.resize(buffer, self.image_shape_reverse)
        if resized.ndim == 2:
            resized = resized[..., None]
        return resized

    def _hash_notifications(self, text):
        vector = np.zeros(self.notifications_features, dtype=np.float32)

        for token in text.split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "little") % self.notifications_features
            vector[idx] += 1.0

        total = vector.sum()
        if total > 0:
            vector /= total
        return vector

    def reset(self, *, seed=None, options=None):
        self.current_instruction = ""
        return super().reset(seed=seed, options=options)

    def observation(self, observation):
        processed = {}
        for key, value in observation.items():
            if key in self.image_buffer_keys:
                processed[key] = self._resize_buffer(value)
            elif key == "notifications":
                if isinstance(value, str) and value:
                    self.current_instruction = value
                processed[key] = self._hash_notifications(self.current_instruction)
            elif key == "audio":
                audio = value.astype(np.float32)
                processed[key] = audio / 32768.0  # Normalize audio to [-1, 1]
            elif key == "gamevariables":
                processed[key] = np.asarray(value, dtype=np.float32)
            else:
                processed[key] = value
        return processed


def main(args):
    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations ( resizes images )
    #  2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, and we scale the rewards by 1/100
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * REWARD_SCALE)
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
        learning_rate=LEARNING_RATE,
    )

    # Do the actual learning
    # This will print out the results in the console.
    # If agent gets better, "ep_rew_mean" should increase steadily

    try:
        agent.learn(
            total_timesteps=MAX_TRAINING_TIMESTEPS,
            progress_bar=True,
        )
    except ImportError:
        agent.learn(total_timesteps=MAX_TRAINING_TIMESTEPS)


if __name__ == "__main__":
    parser = ArgumentParser("Train stable-baselines3 PPO agents on ViZDoom.")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )
    args = parser.parse_args()
    main(args)
