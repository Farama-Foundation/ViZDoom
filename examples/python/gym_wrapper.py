#!/usr/bin/env python3

#####################################################################
# Example for running a vizdoom scenario as a gym env
#####################################################################

import gym

from vizdoom import gym_wrapper  # noqa


if __name__ == "__main__":
    env = gym.make("VizdoomHealthGatheringSupreme-v0", render_mode="human")

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
