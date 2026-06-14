#!/usr/bin/env python3

# Tests for PettingZoo wrapper.
# This test can be run as Python script or via PyTest.

import itertools
from typing import Optional

import pytest


pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402
from pettingzoo.test.seed_test import (  # noqa: E402
    check_environment_deterministic_parallel,
)
from vizdoom import pettingzoo_wrapper  # noqa: E402


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
