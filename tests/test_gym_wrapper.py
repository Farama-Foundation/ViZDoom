#!/usr/bin/env python3

# This test can be run as Python script or via PyTest
import os

import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from gym.utils.env_checker import check_env
from vizdoom import gym_wrapper  # noqa
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv


vizdoom_envs = [
    env
    for env in [env_spec.id for env_spec in gym.envs.registry.values()]
    if "Vizdoom" in env
]
test_env_configs = f"{os.path.dirname(os.path.abspath(__file__))}/env_configs"


# Testing with different non-default kwargs (since each has a different obs space)
# should give warning forcing RGB24 screen type
def test_gym_wrapper():
    print("Testing Gym wrapper compatibility with gym API")
    for env_name in vizdoom_envs:
        for frame_skip in [1, 4]:
            env = gym.make(env_name, frame_skip=frame_skip, max_buttons_pressed=0)

            # Test if env adheres to Gym API
            check_env(env.unwrapped, skip_render_check=True)

            ob_space = env.observation_space
            act_space = env.action_space
            ob, _ = env.reset()
            assert ob_space.contains(ob), f"Reset observation: {ob!r} not in space"

            a = act_space.sample()
            observation, reward, terminated, truncated, _info = env.step(a)
            assert ob_space.contains(
                observation
            ), f"Step observation: {observation!r} not in space"
            assert np.isscalar(reward), f"{reward} is not a scalar for {env}"
            assert isinstance(
                terminated, bool
            ), f"Expected {terminated} to be a boolean"
            assert isinstance(truncated, bool), f"Expected {truncated} to be a boolean"

            env.close()


# Testing obs on terminal state (terminal state is handled differently)
# should give warning forcing RGB24 screen type
def test_gym_wrapper_terminal_state():
    print("Testing Gym rollout (checking terminal state)")
    for env_name in vizdoom_envs:
        for frame_skip in [1, 4]:
            env = gym.make(env_name, frame_skip=frame_skip, max_buttons_pressed=0)

            def agent(ob):
                return env.action_space.sample()

            ob = env.reset()
            terminated = False
            truncated = False
            done = terminated or truncated
            while not done:
                a = agent(ob)
                (ob, _reward, terminated, truncated, _info) = env.step(a)
                done = terminated or truncated
                if done:
                    break
                env.close()
            assert env.observation_space.contains(ob)


# Testing various observation spaces
# Using both screen types `(GRAY8, RGB24)` for various combinations of buffers `(screen|depth|labels|automap)`
def test_gym_wrapper_obs_space():
    print("Testing Gym wrapper observation spaces")
    env_configs = [
        "basic_rgb_i_1_3",
        "basic_g8_i_1_0",
        "basic_g8_idla_4_2",
        "basic_g8_idl_3_1",
        "basic_rgb_id_2_0",
        "basic_rgb_idla_0_1",
    ]
    tri_channel_screen_obs_space = Box(0, 255, (240, 320, 3), dtype=np.uint8)
    single_channel_screen_obs_space = Box(0, 255, (240, 320, 1), dtype=np.uint8)
    observation_spaces = [
        Dict({"screen": tri_channel_screen_obs_space}),
        Dict({"screen": single_channel_screen_obs_space}),
        Dict(
            {
                "screen": single_channel_screen_obs_space,
                "depth": single_channel_screen_obs_space,
                "labels": single_channel_screen_obs_space,
                "automap": single_channel_screen_obs_space,
            }
        ),
        Dict(
            {
                "screen": single_channel_screen_obs_space,
                "depth": single_channel_screen_obs_space,
                "labels": single_channel_screen_obs_space,
            }
        ),
        Dict(
            {
                "screen": tri_channel_screen_obs_space,
                "depth": single_channel_screen_obs_space,
            }
        ),
        Dict(
            {
                "screen": tri_channel_screen_obs_space,
                "depth": single_channel_screen_obs_space,
                "labels": single_channel_screen_obs_space,
                "automap": tri_channel_screen_obs_space,
            }
        ),
    ]

    for i in range(len(env_configs)):
        env = VizdoomEnv(
            level=os.path.join(test_env_configs, env_configs[i] + ".cfg"),
            frame_skip=1,
            max_buttons_pressed=0,
        )
        assert env.observation_space == observation_spaces[i], (
            f"Incorrect observation space: {env.observation_space!r}, "
            f"should be: {observation_spaces[i]!r}"
        )
        obs, _ = env.reset()
        assert env.observation_space.contains(
            obs
        ), f"Step observation: {obs!r} not in space"


# Testing all possible action space combinations
def test_gym_wrapper_action_space():
    print("Testing Gym wrapper action spaces")
    env_configs = [
        "basic_rgb_i_1_3",
        "basic_g8_i_1_0",
        "basic_g8_idla_4_2",
        "basic_g8_idl_3_1",
        "basic_rgb_id_2_0",
        "basic_rgb_idla_0_1",
    ]
    action_spaces = [
        # max_button_pressed = 0, binary action space is MultiDiscrete
        [
            Dict(
                {
                    "binary": MultiDiscrete([2]),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (3,),
                        dtype=np.float32,
                    ),
                }
            ),
            MultiDiscrete([2]),
            Dict(
                {
                    "binary": MultiDiscrete([2, 2, 2, 2]),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (2,),
                        dtype=np.float32,
                    ),
                }
            ),
            Dict(
                {
                    "binary": MultiDiscrete([2, 2, 2]),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (1,),
                        dtype=np.float32,
                    ),
                }
            ),
            MultiDiscrete([2, 2]),
            Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (1,),
                dtype=np.float32,
            ),
        ],
        # max_button_pressed = 1, binary action space is Discrete(num_binary_buttons + 1)
        [
            Dict(
                {
                    "binary": Discrete(2),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (3,),
                        dtype=np.float32,
                    ),
                }
            ),
            Discrete(2),
            Dict(
                {
                    "binary": Discrete(5),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (2,),
                        dtype=np.float32,
                    ),
                }
            ),
            Dict(
                {
                    "binary": Discrete(4),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (1,),
                        dtype=np.float32,
                    ),
                }
            ),
            Discrete(3),
            Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (1,),
                dtype=np.float32,
            ),
        ],
        # max_button_pressed = 2, binary action space is Discrete(m) m=all combinations
        # indices=[0,1] should give warning clipping max_buttons_pressed to 1
        [
            Dict(
                {
                    "binary": Discrete(2),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (3,),
                        dtype=np.float32,
                    ),
                }
            ),
            Discrete(2),
            Dict(
                {
                    "binary": Discrete(11),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (2,),
                        dtype=np.float32,
                    ),
                }
            ),
            Dict(
                {
                    "binary": Discrete(7),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (1,),
                        dtype=np.float32,
                    ),
                }
            ),
            Discrete(4),
            Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (1,),
                dtype=np.float32,
            ),
        ],
        # max_button_pressed = 3, binary action space is Discrete(m) m=all combinations
        # indices=[0,1, 4] should give warning clipping max_buttons_pressed to 1 or 2
        [
            Dict(
                {
                    "binary": Discrete(2),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (3,),
                        dtype=np.float32,
                    ),
                }
            ),
            Discrete(2),
            Dict(
                {
                    "binary": Discrete(15),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (2,),
                        dtype=np.float32,
                    ),
                }
            ),
            Dict(
                {
                    "binary": Discrete(8),
                    "continuous": Box(
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).max,
                        (1,),
                        dtype=np.float32,
                    ),
                }
            ),
            Discrete(4),
            Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (1,),
                dtype=np.float32,
            ),
        ],
    ]
    for max_button_pressed in range(0, 4):
        for i in range(len(env_configs)):
            env = VizdoomEnv(
                level=os.path.join(test_env_configs, env_configs[i] + ".cfg"),
                frame_skip=1,
                max_buttons_pressed=max_button_pressed,
            )
            assert env.action_space == action_spaces[max_button_pressed][i], (
                f"Incorrect action space: {env.action_space!r}, "
                f"should be: {action_spaces[max_button_pressed][i]!r}"
            )
            env.reset()
            # check successful call to step using action_space.sample()
            sample_action = env.action_space.sample()
            env.step(sample_action)


if __name__ == "__main__":
    test_gym_wrapper()
    test_gym_wrapper_terminal_state()
    test_gym_wrapper_action_space()
    test_gym_wrapper_obs_space()
