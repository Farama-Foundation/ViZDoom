#!/usr/bin/env python3

# This test can be run as Python script or via PyTest

import gym
import numpy as np
from gym.utils.env_checker import check_env
from vizdoom import gym_wrapper


vizdoom_envs = [env for env in [env_spec.id for env_spec in gym.envs.registry.all()] if "Vizdoom" in env]

# Testing with different non-default kwargs (since each has a different obs space)
def test_gym_wrapper():
    print("Testing Gym wrapper compatibility with gym API")
    for env_name in vizdoom_envs:
        for frame_skip in [1, 4]:
            env = gym.make(
                env_name,
                frame_skip=frame_skip
            )

            # Test if env adheres to Gym API
            check_env(env, warn=True, skip_render_check=True)

            ob_space = env.observation_space
            act_space = env.action_space
            ob = env.reset()
            assert ob_space.contains(ob), f"Reset observation: {ob!r} not in space"

            a = act_space.sample()
            observation, reward, done, _info = env.step(a)
            assert ob_space.contains(
                observation
            ), f"Step observation: {observation!r} not in space"
            assert np.isscalar(reward), f"{reward} is not a scalar for {env}"
            assert isinstance(done, bool), f"Expected {done} to be a boolean"

            env.close()


# Check obs on terminal state (terminal state is handled differently)
def test_gym_wrapper_terminal_state():
    print("Testing Gym rollout (checking terminal state)")
    for env_name in vizdoom_envs:
        for frame_skip in [1, 4]:
            env = gym.make(
                env_name,
                frame_skip=frame_skip
            )

            agent = lambda ob: env.action_space.sample()
            ob = env.reset()
            done = False
            while not done:
                a = agent(ob)
                (ob, _reward, done, _info) = env.step(a)
                if done:
                    break
                env.close()
            assert env.observation_space.contains(ob)


if __name__ == "__main__":
    test_gym_wrapper()
    test_gym_wrapper_terminal_state()
