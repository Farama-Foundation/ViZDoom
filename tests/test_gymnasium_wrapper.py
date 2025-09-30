#!/usr/bin/env python3

# Tests for Gymnasium wrapper.
# This test can be run as Python script or via PyTest

import os
import pickle

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gymnasium.utils.env_checker import check_env, data_equivalence

from vizdoom import gymnasium_wrapper  # noqa
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv


vizdoom_envs = [
    env
    for env in [env_spec.id for env_spec in gymnasium.envs.registry.values()]  # type: ignore
    if "Vizdoom" in env
]
test_env_configs = f"{os.path.dirname(os.path.abspath(__file__))}/env_configs"
envs_with_animated_textures = [
    "VizdoomHealthGathering",
    "VizdoomHealthGatheringSupreme",
    "VizdoomDeathmatch",
]
buffers = [
    "screen",
    "depth",
    "labels",
    "automap",
    "audio",
]


# Testing with different non-default kwargs (since each has a different obs space)
# should give warning forcing RGB24 screen type
def test_gymnasium_wrapper():
    print("Testing Gymnasium wrapper compatibility with gymnasium API")
    for env_name in vizdoom_envs:
        print(f"  Env: {env_name}")

        # Skip environments with animated textures,
        # as they might render different states for the same seeds
        if env_name.split("-")[0] in envs_with_animated_textures:
            continue

        for frame_skip in [1, 4]:
            env = gymnasium.make(env_name, frame_skip=frame_skip)

            # Test if env adheres to Gymnasium API
            check_env(env.unwrapped, skip_render_check=True)

            ob_space = env.observation_space
            act_space = env.action_space
            obs, _ = env.reset()
            assert ob_space.contains(obs), f"Reset observation: {obs!r} not in space"

            a = act_space.sample()
            observation, reward, terminated, truncated, _info = env.step(a)
            assert ob_space.contains(
                observation
            ), f"Step observation: {observation!r} not in space"
            assert np.isscalar(reward), f"{reward} is not a scalar for {env}"
            assert isinstance(
                terminated, bool
            ), f"Expected {terminated} to be a boolean"
            assert isinstance(
                terminated, bool
            ), f"Expected {terminated} to be a boolean"
            assert isinstance(truncated, bool), f"Expected {truncated} to be a boolean"

            env.close()


# Testing obs on terminal state (terminal state is handled differently)
# should give warning forcing RGB24 screen type
def test_gymnasium_wrapper_terminal_state():
    print("Testing Gymnasium rollout (checking terminal state)")
    for env_name in vizdoom_envs:
        print(f"  Env: {env_name}")

        for frame_skip in [1, 4]:
            env = gymnasium.make(env_name, frame_skip=frame_skip, max_buttons_pressed=0)
            obs = env.reset()
            terminated = False
            truncated = False
            done = terminated or truncated
            while not done:
                a = env.action_space.sample()
                (obs, _reward, terminated, truncated, _info) = env.step(a)
                done = terminated or truncated
                if done:
                    break

            assert env.observation_space.contains(obs)
            env.close()


def test_gymnasium_wrapper_truncated_state():
    print("Testing Gymnasium wrapper truncated state")
    env = VizdoomEnv(
        config_file=os.path.join(
            test_env_configs, "basic_rgb_idla_0_1.cfg"
        ),  # For this config it is impossible to get other terminal state than timeout
        frame_skip=10,  # Using frame_skip=10 to speed up the test
        max_buttons_pressed=0,
        treat_episode_timeout_as_truncation=True,
    )
    obs = env.reset()
    terminated = False
    truncated = False
    done = terminated or truncated
    while not done:
        a = env.action_space.sample()
        obs, _reward, terminated, truncated, _info = env.step(a)
        done = terminated or truncated
        if done:
            assert truncated
            env.close()


# Testing various observation spaces
# Using both screen types `(GRAY8, RGB24)` for various combinations of buffers `(screen|depth|labels|automap)`
def test_gymnasium_wrapper_obs_space():
    print("Testing Gymnasium wrapper observation spaces")
    env_configs = [
        "basic_rgb_i_1_3",
        "basic_g8_i_1_0",
        "basic_g8_i_1_0_wAudio",
        "basic_g8_idla_4_2",
        "basic_g8_idl_3_1",
        "basic_rgb_id_2_0",
        "basic_rgb_idla_0_1",
    ]
    tri_channel_screen_obs_space = Box(0, 255, (240, 320, 3), dtype=np.uint8)
    single_channel_screen_obs_space = Box(0, 255, (240, 320, 1), dtype=np.uint8)
    audio_obs_space = Box(
        -32768, 32767, (int(44100 * 1 / 35 * 1), 2), dtype=np.int16
    )  # sampling rate = 44100, frame_skip = 1
    observation_spaces = [
        Dict({"screen": tri_channel_screen_obs_space}),
        Dict({"screen": single_channel_screen_obs_space}),
        Dict({"screen": single_channel_screen_obs_space, "audio": audio_obs_space}),
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
            config_file=os.path.join(test_env_configs, env_configs[i] + ".cfg"),
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


def _compare_action_spaces(env, expected_action_space):
    """
    Helper function to compare the action space of the environment with the expected action space.
    """
    assert env.action_space == expected_action_space, (
        f"Incorrect action space: {env.action_space!r}, "
        f"should be: {expected_action_space!r}"
    )
    env.reset()
    # check successful call to step using action_space.sample()
    sample_action = env.action_space.sample()
    env.step(sample_action)


# Testing all possible action space combinations
def test_gymnasium_wrapper_action_space():
    print("Testing Gymnasium wrapper action spaces")
    env_configs = [
        "basic_rgb_i_1_3",
        "basic_g8_i_1_0",
        "basic_g8_idla_4_2",
        "basic_g8_idl_3_1",
        "basic_rgb_id_2_0",
        "basic_rgb_idla_0_1",
    ]
    # max_button_pressed = 0, binary action space is MultiBinary or MultiDiscrete
    multi_binary_action_spaces = [
        Dict(
            {
                "binary": MultiBinary(1),
                "continuous": Box(
                    np.finfo(np.float32).min,
                    np.finfo(np.float32).max,
                    (3,),
                    dtype=np.float32,
                ),
            }
        ),
        MultiBinary(1),
        Dict(
            {
                "binary": MultiBinary(4),
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
                "binary": MultiBinary(3),
                "continuous": Box(
                    np.finfo(np.float32).min,
                    np.finfo(np.float32).max,
                    (1,),
                    dtype=np.float32,
                ),
            }
        ),
        MultiBinary(2),
        Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            (1,),
            dtype=np.float32,
        ),
    ]

    # max_button_pressed = 0, binary action space is MultiBinary or MultiDiscrete
    multi_discrete_action_spaces = [
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
    ]

    # max_button_pressed = 1, binary action space is Discrete(num_binary_buttons + 1)
    discrete_action_spaces = [
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
        # indices=[0,1,4] should give warning clipping max_buttons_pressed to 1 or 2
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

    for i in range(len(env_configs)):
        env = VizdoomEnv(
            config_file=os.path.join(test_env_configs, env_configs[i] + ".cfg"),
            frame_skip=1,
            max_buttons_pressed=0,
            use_multi_binary_action_space=True,
        )
        _compare_action_spaces(env, multi_binary_action_spaces[i])

        env = VizdoomEnv(
            config_file=os.path.join(test_env_configs, env_configs[i] + ".cfg"),
            frame_skip=1,
            max_buttons_pressed=0,
            use_multi_binary_action_space=False,
        )
        _compare_action_spaces(env, multi_discrete_action_spaces[i])

    for max_button_pressed in range(1, 4):
        for i in range(len(env_configs)):
            env = VizdoomEnv(
                config_file=os.path.join(test_env_configs, env_configs[i] + ".cfg"),
                frame_skip=1,
                max_buttons_pressed=max_button_pressed,
            )
            _compare_action_spaces(
                env, discrete_action_spaces[max_button_pressed - 1][i]
            )


def _compare_envs(
    env1, env2, env1_name="First", env2_name="Second", seed=1993, compare_buffers=True
):
    """
    Helper function to compare two environments.
    It checks if the initial observations, actions, and subsequent observations,
    rewards, termination, truncation, and info are equivalent.
    """
    # Seed environments
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)

    # Seed action space sampler
    env1.action_space.seed(seed)
    env2.action_space.seed(seed)

    # Compare initial states
    if not compare_buffers:
        if "screen" in obs1:
            obs1["screen"] = np.zeros_like(obs1["screen"])
            obs2["screen"] = np.zeros_like(obs2["screen"])

    assert data_equivalence(
        obs1, obs2
    ), f"Initial observations incorrect. {env1_name} environment: {obs1}. {env2_name} environment: {obs2}"

    # Compare sequance of random actions and states
    done = False
    while not done:
        a1 = env1.action_space.sample()
        a2 = env2.action_space.sample()
        assert data_equivalence(
            a1, a2
        ), f"Actions incorrect. First environment: {a1}. Second environment: {a2}"

        obs1, rew1, term1, trunc1, info1 = env1.step(a1)
        obs2, rew2, term2, trunc2, info2 = env2.step(a2)

        if not compare_buffers:
            for buffer in buffers:
                if buffer in obs1:
                    obs1[buffer] = np.zeros_like(obs1[buffer])
                if buffer in obs2:
                    obs2[buffer] = np.zeros_like(obs2[buffer])

        assert data_equivalence(
            obs1, obs2
        ), f"Incorrect observations: {env1_name} environment: {obs1}. {env2_name} environment: {obs2}"
        assert data_equivalence(
            rew1, rew2
        ), f"Incorrect rewards: {env1_name} environment: {rew1}. {env2_name} environment:{rew2}"
        assert data_equivalence(
            term1, term2
        ), f"Incorrect terms: {env1_name} environment: {term1}. {env2_name} environment: {term2}"
        assert data_equivalence(
            trunc1, trunc2
        ), f"Incorrect truncs: {env1_name} environment: {trunc1}. {env2_name} environment: {trunc2}"
        assert data_equivalence(
            info1, info2
        ), f"Incorrect info: {env1_name} environment: {info1}. {env2_name} environment: {info2}"

        done = term1 or trunc1 or term2 or trunc2
    env1.close()
    env2.close()


def test_gymnasium_wrapper_pickle():
    print("Testing Gymnasium wrapper pickling (EzPickle).")
    for env_name in vizdoom_envs:
        print(f"  Env: {env_name}")

        env1 = gymnasium.make(env_name, frame_skip=1, max_buttons_pressed=0)
        env2 = pickle.loads(pickle.dumps(env1))

        _compare_envs(
            env1,
            env2,
            env1_name="Original",
            env2_name="Pickled",
            seed=1993,
            compare_buffers=(env_name.split("-")[0] not in envs_with_animated_textures),
        )


def test_gymnasium_wrapper_seed():
    print("Testing gymnasium wrapper seeding.")
    for env_name in vizdoom_envs:
        print(f"  Env: {env_name}")

        env1 = gymnasium.make(env_name, frame_skip=1, max_buttons_pressed=0)
        env2 = gymnasium.make(env_name, frame_skip=1, max_buttons_pressed=0)

        _compare_envs(
            env1,
            env2,
            env1_name="First",
            env2_name="Second",
            seed=1993,
            compare_buffers=(env_name.split("-")[0] not in envs_with_animated_textures),
        )


if __name__ == "__main__":
    test_gymnasium_wrapper()
    test_gymnasium_wrapper_terminal_state()
    test_gymnasium_wrapper_truncated_state()
    test_gymnasium_wrapper_action_space()
    test_gymnasium_wrapper_obs_space()
    test_gymnasium_wrapper_pickle()
    test_gymnasium_wrapper_seed()
