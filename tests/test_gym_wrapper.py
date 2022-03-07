import gym
import numpy as np
from gym.spaces.box import Box
from gym.utils.env_checker import check_env
from vizdoom import gym_wrapper
from itertools import product

env_conditions = list(product([True, False], [True, False], [True, False], [True, False]))

# Testing with different non-default kwargs (since each has a different obs space)
def test_gym_wrapper():
    print("Testing Gym wrapper compatiblility with gym API")

    for depth, labels, position, health in env_conditions:
        env = gym.make(
            "VizdoomTakeCover-v0",
            depth=depth,
            labels=labels,
            position=position,
            health=health,
        )

        # Test if env adheres to Gym API
        check_env(env, warn=True, skip_render_check=True)

        ob_space = env.observation_space
        act_space = env.action_space
        ob = env.reset()
        assert ob_space.contains(ob), f"Reset observation: {ob!r} not in space"
        if isinstance(ob_space, Box):
            # Only checking dtypes for Box spaces to avoid iterating through tuple entries
            assert (
                ob.dtype == ob_space.dtype
            ), f"Reset observation dtype: {ob.dtype}, expected: {ob_space.dtype}"

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
    for depth, labels, position, health in env_conditions:
        env = gym.make(
            "VizdoomHealthGatheringSupreme-v0",
            depth=depth,
            labels=labels,
            position=position,
            health=health,
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
