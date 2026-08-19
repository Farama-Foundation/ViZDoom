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


_PORTS = itertools.count(50290, 1000)


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


def test_pettingzoo_action_api_is_continuous_only():
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
