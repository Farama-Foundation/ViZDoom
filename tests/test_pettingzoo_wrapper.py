# Tests for PettingZoo wrapper.
# This test can be run as Python script or via PyTest.

import inspect
import itertools
from typing import Optional

import numpy as np
import pytest


pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402
from pettingzoo.test.seed_test import (  # noqa: E402
    check_environment_deterministic_parallel,
)

from vizdoom import Button  # noqa: E402
from vizdoom import pettingzoo_wrapper  # noqa: E402
from vizdoom.pettingzoo_wrapper import base_env_common  # noqa: E402
from vizdoom.pettingzoo_wrapper.base_pettingzoo_env import (  # noqa: E402
    _settle_game_after_reset,
)
from vizdoom.pettingzoo_wrapper.bot_eval_types import (  # noqa: E402
    EpisodeResult,
    build_seed_schedule,
    summarize_tier,
)


_PORTS = itertools.count(50290, 1000)


def test_bot_eval_schedule_and_summary_are_reproducible():
    schedule = build_seed_schedule(7, tiers=("easy",), screening_attempts=2)
    assert schedule == build_seed_schedule(7, tiers=("easy",), screening_attempts=2)
    metrics = {
        "learner_deaths": 0,
        "learner_damage_made": 10,
        "learner_damage_taken": 10,
    }
    results = [
        EpisodeResult(
            seed,
            "easy",
            True,
            learner_frags=learner_frags,
            bot_frags=bot_frags,
            outcome=outcome,
            **metrics,
        )
        for seed, learner_frags, bot_frags, outcome in (
            (1, 3, 1, "win"),
            (2, 1, 1, "tie"),
        )
    ]

    summary = summarize_tier("easy", results, bootstrap_seed=7, bootstrap_samples=20)

    assert summary.frag_diff_mean == 1.0
    assert summary.win_rate == summary.tie_rate == 0.5


def _make_test_env(seed: Optional[int] = None):
    return pettingzoo_wrapper.make(
        scenario="health_gathering_multi_agent",
        num_agents=2,
        resolution="160X120",
        timeout=300,
        barrier_timeout=10.0,
        skip_frames=1,
        async_mode=False,
        ticrate=35,
        seed=seed,
        port=next(_PORTS),
        enable_video=False,
        verbose=False,
    )


def test_pettingzoo_parallel_api():
    env = _make_test_env(seed=123)
    try:
        parallel_api_test(env, num_cycles=25)
    finally:
        env.close()


@pytest.mark.flaky(reruns=3)
def test_pettingzoo_parallel_seed():
    env1 = _make_test_env()
    env2 = _make_test_env()
    try:
        check_environment_deterministic_parallel(env1, env2, num_cycles=10)
    finally:
        env1.close()
        env2.close()


def test_base_env_exposes_direct_continuous_button_vector(monkeypatch):
    monkeypatch.setattr(
        base_env_common,
        "discover_buttons",
        lambda _path: [
            Button.TURN_LEFT_RIGHT_DELTA,
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.ATTACK,
        ],
    )
    env = base_env_common.VizdoomParallelEnvBase(
        config_file="unused.cfg",
        num_agents=2,
        skip_frames=4,
    )

    action_space = env.action_space("agent_0")
    assert isinstance(action_space, base_env_common.spaces.Box)
    assert action_space.shape == (4,)
    assert env._encode_env_action(np.asarray([1.25, 0.2, 0.8, 0.49])) == [
        1.25,
        0.0,
        1.0,
        0.0,
    ]


def test_base_env_exposes_joint_categorical_binary_actions(monkeypatch):
    monkeypatch.setattr(
        base_env_common,
        "discover_buttons",
        lambda _path: [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.ATTACK],
    )
    env = base_env_common.VizdoomParallelEnvBase(
        config_file="unused.cfg",
        num_agents=2,
        skip_frames=4,
    )

    action_space = env.action_space("agent_0")
    assert isinstance(action_space, base_env_common.spaces.Discrete)
    assert action_space.n == 8
    assert env._encode_env_action(0) == [0.0, 0.0, 0.0]
    assert env._encode_env_action(5) == [1.0, 0.0, 1.0]
    assert env._encode_env_action(7) == [1.0, 1.0, 1.0]


def test_base_env_stacks_frames(monkeypatch):
    monkeypatch.setattr(
        base_env_common, "discover_buttons", lambda _path: [Button.ATTACK]
    )
    env = base_env_common.VizdoomParallelEnvBase(
        config_file="unused.cfg",
        num_agents=1,
        resolution="4X3",
        frame_stack=4,
    )
    first = np.full((3, 4, 3), 1, dtype=np.uint8)
    second = np.full((3, 4, 3), 2, dtype=np.uint8)

    reset_obs = env._stack_observations({"agent_0": first}, reset=True)["agent_0"]
    step_obs = env._stack_observations({"agent_0": second}, reset=False)["agent_0"]

    assert env.observation_space("agent_0").shape == (3, 4, 12)
    assert np.array_equal(reset_obs, np.concatenate([first] * 4, axis=-1))
    assert np.array_equal(
        step_obs, np.concatenate([first, first, first, second], axis=-1)
    )


def test_pettingzoo_action_api_selects_space_from_scenario_buttons():
    parameters = inspect.signature(pettingzoo_wrapper.make).parameters
    assert "simple_discrete" not in parameters
    assert "use_multi_binary_action_space" not in parameters


def test_reset_settling_is_mandatory_and_synchronized():
    class FakeGame:
        def __init__(self):
            self.actions = []
            self.advances = []

        def get_available_buttons_size(self):
            return 2

        def set_action(self, action):
            self.actions.append(action)

        def advance_action(self, tics, update_state):
            self.advances.append((tics, update_state))

    class FakeBarrier:
        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            assert timeout > 0
            self.waits += 1

    game = FakeGame()
    barrier = FakeBarrier()

    _settle_game_after_reset(game, barrier)

    assert game.actions == [[0.0, 0.0]]
    assert len(game.advances) == 35
    assert game.advances[:34] == [(1, False)] * 34
    assert game.advances[-1] == (1, True)
    assert barrier.waits == 35
