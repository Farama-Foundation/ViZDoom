#!/usr/bin/env python3

# Tests for Gymnasium wrapper.
# This test can be run as Python script or via PyTest

import os
import pickle

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Text
from gymnasium.utils.env_checker import check_env, data_equivalence

from vizdoom import gymnasium_wrapper, scenarios_path  # noqa
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv


# Ensure pytest.mark.parametrize decorator works without pytest
try:
    import pytest
except ImportError:

    class MockMark:
        def parametrize(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class MockSkip(Exception):
        """Raised to mimic pytest.skip behavior."""

    def _mock_skip(reason: str = "") -> None:
        raise MockSkip(reason)

    _mock_skip.Exception = MockSkip

    class MockPytest:
        mark = MockMark()
        skip = staticmethod(_mock_skip)

    pytest = MockPytest()
    del MockMark, MockPytest, _mock_skip


VIZDOOM_ENVS = [
    env
    for env in [env_spec.id for env_spec in gymnasium.envs.registry.values()]  # type: ignore
    if "Vizdoom" in env
]
# Skip environments with higher skills levels for testing purposes as they only differ with a single parameter
for skill in ["-S2-", "-S3-", "-S4-", "-S5-"]:
    VIZDOOM_ENVS = [env for env in VIZDOOM_ENVS if skill not in env]

TEST_ENV_CONFIGS = f"{os.path.dirname(os.path.abspath(__file__))}/env_configs"
BUFFERS = ["screen", "depth", "labels", "automap", "audio", "notifications"]

FP32_ACT_SPACE = dict(
    low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, dtype=np.float32
)
TRI_CHANNEL_SCREEN_OBS_SPACE = Box(0, 255, (240, 320, 3), dtype=np.uint8)
SINGLE_CHANNEL_SCREEN_OBS_SPACE = Box(0, 255, (240, 320, 1), dtype=np.uint8)
AUDIO_OBS_SPACE = Box(
    -32768, 32767, (int(44100 * 1 / 35 * 1), 2), dtype=np.int16
)  # sampling rate = 44100, frame_skip = 1
NOTIFICATIONS_OBS_SPACE = Text(min_length=0, max_length=32768)

COLOR_SCREEN: dict[str, gymnasium.Space] = {"screen": TRI_CHANNEL_SCREEN_OBS_SPACE}
GREY_SCREEN: dict[str, gymnasium.Space] = {"screen": SINGLE_CHANNEL_SCREEN_OBS_SPACE}
DEPTH_BUFFER: dict[str, gymnasium.Space] = {"depth": SINGLE_CHANNEL_SCREEN_OBS_SPACE}
LABELS_BUFFER: dict[str, gymnasium.Space] = {"labels": SINGLE_CHANNEL_SCREEN_OBS_SPACE}
COLOR_AUTOMAP: dict[str, gymnasium.Space] = {"automap": TRI_CHANNEL_SCREEN_OBS_SPACE}
GRAY_AUTOMAP: dict[str, gymnasium.Space] = {"automap": SINGLE_CHANNEL_SCREEN_OBS_SPACE}
NOTIFICATIONS: dict[str, gymnasium.Space] = {"notifications": NOTIFICATIONS_OBS_SPACE}
AUDIO_BUFFER: dict[str, gymnasium.Space] = {"audio": AUDIO_OBS_SPACE}


def _check_if_main_wad_available(env_name: str, env: gymnasium.Env) -> None:
    """
    Helper function to check if specified main WAD file is available for the given environment.
    """
    main_wad_path = env.unwrapped.game.get_doom_game_path()
    if (
        main_wad_path is not None
        and main_wad_path != ""
        and not os.path.exists(main_wad_path)
        and not os.path.exists(os.path.join(scenarios_path, main_wad_path))
    ):
        pytest.skip(
            f"Main WAD file {main_wad_path} not available for {env_name}, skipping test."
        )


def _run_with_pytest_skip(func, *args, **kwargs):
    skip_exception = getattr(getattr(pytest, "skip", None), "Exception", None)
    try:
        func(*args, **kwargs)
    except BaseException as exc:
        if skip_exception is not None and isinstance(exc, skip_exception):
            print(f"Skipped: {exc}")
            return
        raise


# Testing with different non-default kwargs (since each has a different obs space)
# should give warning forcing RGB24 screen type
@pytest.mark.parametrize("env_name", VIZDOOM_ENVS)
def test_gymnasium_wrapper(env_name: str):
    print(f"Testing Gymnasium wrapper - {env_name}")

    for frame_skip in [1, 4]:
        env = gymnasium.make(env_name, frame_skip=frame_skip)
        _check_if_main_wad_available(env_name, env)

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
        assert isinstance(terminated, bool), f"Expected {terminated} to be a boolean"
        assert isinstance(terminated, bool), f"Expected {terminated} to be a boolean"
        assert isinstance(truncated, bool), f"Expected {truncated} to be a boolean"

        env.close()


# Testing obs on terminal state (terminal state is handled differently)
# should give warning forcing RGB24 screen type
@pytest.mark.parametrize("env_name", VIZDOOM_ENVS)
def test_gymnasium_wrapper_terminal_state(env_name: str):
    print(f"Testing Gymnasium terminal state - {env_name}")

    for frame_skip in [1, 4]:
        env = gymnasium.make(env_name, frame_skip=frame_skip, max_buttons_pressed=0)
        _check_if_main_wad_available(env_name, env)

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
            TEST_ENV_CONFIGS, "basic_rgb_idla_0_1.cfg"
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

OBS_SPACE_PARAMS = [
    ("basic_rgb_i_1_3", Dict(COLOR_SCREEN)),
    ("basic_g8_i_1_0", Dict(GREY_SCREEN)),
    ("basic_g8_i_1_0_notifications", Dict(GREY_SCREEN | NOTIFICATIONS)),
    ("basic_g8_i_1_0_audio", Dict(GREY_SCREEN | AUDIO_BUFFER)),
    (
        "basic_g8_idla_4_2",
        Dict(GREY_SCREEN | DEPTH_BUFFER | LABELS_BUFFER | GRAY_AUTOMAP),
    ),
    ("basic_g8_idl_3_1", Dict(GREY_SCREEN | DEPTH_BUFFER | LABELS_BUFFER)),
    ("basic_rgb_id_2_0", Dict(COLOR_SCREEN | DEPTH_BUFFER)),
    (
        "basic_rgb_idla_0_1",
        Dict(COLOR_SCREEN | DEPTH_BUFFER | LABELS_BUFFER | COLOR_AUTOMAP),
    ),
]


@pytest.mark.parametrize("env_config,obs_space", OBS_SPACE_PARAMS)
def test_gymnasium_wrapper_obs_space(env_config: str, obs_space: Dict):
    env = VizdoomEnv(
        config_file=os.path.join(TEST_ENV_CONFIGS, env_config + ".cfg"),
        frame_skip=1,
        max_buttons_pressed=0,
    )
    _check_if_main_wad_available(env_config, env)
    assert env.observation_space == obs_space, (
        f"Incorrect observation space: {env.observation_space!r}, "
        f"should be: {obs_space!r}"
    )
    obs, _ = env.reset()
    assert env.observation_space.contains(
        obs
    ), f"Step observation: {obs!r} not in space"


def _compare_action_spaces(env: gymnasium.Env, expected_action_space: gymnasium.Space):
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
CONTINUOUS_1 = Box(shape=(1,), **FP32_ACT_SPACE)  # type: ignore
CONTINUOUS_2 = Box(shape=(2,), **FP32_ACT_SPACE)  # type: ignore
CONTINUOUS_3 = Box(shape=(3,), **FP32_ACT_SPACE)  # type: ignore

ACTION_SPACE_PARAMS = [
    (
        "basic_rgb_i_1_3",
        Dict({"binary": MultiBinary(1), "continuous": CONTINUOUS_3}),
        Dict({"binary": MultiDiscrete([2]), "continuous": CONTINUOUS_3}),
        [
            Dict({"binary": Discrete(2), "continuous": CONTINUOUS_3}),
            Dict({"binary": Discrete(2), "continuous": CONTINUOUS_3}),
            Dict({"binary": Discrete(2), "continuous": CONTINUOUS_3}),
        ],
    ),
    (
        "basic_g8_i_1_0",
        MultiBinary(1),
        MultiDiscrete([2]),
        [Discrete(2), Discrete(2), Discrete(2)],
    ),
    (
        "basic_g8_idla_4_2",
        Dict({"binary": MultiBinary(4), "continuous": CONTINUOUS_2}),
        Dict({"binary": MultiDiscrete([2, 2, 2, 2]), "continuous": CONTINUOUS_2}),
        [
            Dict({"binary": Discrete(5), "continuous": CONTINUOUS_2}),
            Dict({"binary": Discrete(11), "continuous": CONTINUOUS_2}),
            Dict({"binary": Discrete(15), "continuous": CONTINUOUS_2}),
        ],
    ),
    (
        "basic_g8_idl_3_1",
        Dict({"binary": MultiBinary(3), "continuous": CONTINUOUS_1}),
        Dict({"binary": MultiDiscrete([2, 2, 2]), "continuous": CONTINUOUS_1}),
        [
            Dict({"binary": Discrete(4), "continuous": CONTINUOUS_1}),
            Dict({"binary": Discrete(7), "continuous": CONTINUOUS_1}),
            Dict({"binary": Discrete(8), "continuous": CONTINUOUS_1}),
        ],
    ),
    (
        "basic_rgb_id_2_0",
        MultiBinary(2),
        MultiDiscrete([2, 2]),
        [Discrete(3), Discrete(4), Discrete(4)],
    ),
    (
        "basic_rgb_idla_0_1",
        CONTINUOUS_1,
        CONTINUOUS_1,
        [CONTINUOUS_1, CONTINUOUS_1, CONTINUOUS_1],
    ),
]


@pytest.mark.parametrize(
    "env_config,multi_binary_action_space,multi_discrete_action_space,discrete_action_spaces",
    ACTION_SPACE_PARAMS,
)
def test_gymnasium_wrapper_action_space(
    env_config: str,
    multi_binary_action_space: gymnasium.Space,
    multi_discrete_action_space: gymnasium.Space,
    discrete_action_spaces: list[gymnasium.Space],
):
    env = VizdoomEnv(
        config_file=os.path.join(TEST_ENV_CONFIGS, env_config + ".cfg"),
        frame_skip=1,
        max_buttons_pressed=0,
        use_multi_binary_action_space=True,
    )
    _compare_action_spaces(env, multi_binary_action_space)

    env = VizdoomEnv(
        config_file=os.path.join(TEST_ENV_CONFIGS, env_config + ".cfg"),
        frame_skip=1,
        max_buttons_pressed=0,
        use_multi_binary_action_space=False,
    )
    _compare_action_spaces(env, multi_discrete_action_space)

    for max_button_pressed, action_space in enumerate(discrete_action_spaces, start=1):
        env = VizdoomEnv(
            config_file=os.path.join(TEST_ENV_CONFIGS, env_config + ".cfg"),
            frame_skip=1,
            max_buttons_pressed=max_button_pressed,
        )
        _compare_action_spaces(env, action_space)


def _compare_envs(
    env1, env2, env1_name="First", env2_name="Second", max_steps=10, seed=1993
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

    assert data_equivalence(
        obs1, obs2
    ), f"Initial observations incorrect. {env1_name} environment: {obs1}. {env2_name} environment: {obs2}"

    # Compare sequance of random actions and states
    done = False
    steps = 0
    while not done and steps < max_steps:
        a1 = env1.action_space.sample()
        a2 = env2.action_space.sample()
        assert data_equivalence(
            a1, a2
        ), f"Actions incorrect. First environment: {a1}. Second environment: {a2}"

        obs1, rew1, term1, trunc1, info1 = env1.step(a1)
        obs2, rew2, term2, trunc2, info2 = env2.step(a2)

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
        steps += 1

    env1.close()
    env2.close()


@pytest.mark.parametrize("env_name", VIZDOOM_ENVS)
def test_gymnasium_wrapper_pickle(env_name: str):
    print(f"Testing Gymnasium wrapper pickle - {env_name}")

    env1 = gymnasium.make(env_name)
    _check_if_main_wad_available(env_name, env1)
    env2 = pickle.loads(pickle.dumps(env1))

    _compare_envs(
        env1,
        env2,
        env1_name="Original",
        env2_name="Pickled",
        seed=1993,
    )


@pytest.mark.parametrize("env_name", VIZDOOM_ENVS)
def test_gymnasium_wrapper_seed(env_name: str):
    print(f"Testing Gymnasium wrapper seed - {env_name}")

    env1 = gymnasium.make(env_name)
    _check_if_main_wad_available(env_name, env1)
    env2 = gymnasium.make(env_name)

    _compare_envs(
        env1,
        env2,
        env1_name="First",
        env2_name="Second",
        seed=1993,
    )


if __name__ == "__main__":
    print("Testing Gymnasium wrapper compatibility with gymnasium API")
    for env_name in VIZDOOM_ENVS:
        _run_with_pytest_skip(test_gymnasium_wrapper, env_name)

    print("Testing Gymnasium rollout (checking terminal state)")
    for env_name in VIZDOOM_ENVS:
        _run_with_pytest_skip(test_gymnasium_wrapper_terminal_state, env_name)

    test_gymnasium_wrapper_truncated_state()

    print("Testing Gymnasium wrapper action spaces")
    for (
        env_config,
        multi_binary_action_space,
        multi_discrete_action_space,
        discrete_action_spaces,
    ) in ACTION_SPACE_PARAMS:
        _run_with_pytest_skip(
            test_gymnasium_wrapper_action_space,
            env_config,
            multi_binary_action_space,
            multi_discrete_action_space,
            discrete_action_spaces,
        )

    print("Testing Gymnasium wrapper observation spaces")
    for env_config, obs_space in OBS_SPACE_PARAMS:
        _run_with_pytest_skip(test_gymnasium_wrapper_obs_space, env_config, obs_space)

    print("Testing Gymnasium wrapper pickling (EzPickle).")
    for env_name in VIZDOOM_ENVS:
        _run_with_pytest_skip(test_gymnasium_wrapper_pickle, env_name)

    print("Testing gymnasium wrapper seeding.")
    for env_name in VIZDOOM_ENVS:
        _run_with_pytest_skip(test_gymnasium_wrapper_seed, env_name)
