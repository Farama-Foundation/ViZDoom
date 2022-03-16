#!/usr/bin/env python3

#####################################################################
# Example for running a vizdoom scenario as a gym env
#####################################################################

import gym
from vizdoom import gym_wrapper

if __name__  == '__main__':
    env = gym.make("VizdoomHealthGatheringSupreme-v0")

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()
