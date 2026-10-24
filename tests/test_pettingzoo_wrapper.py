# Tests for PettingZoo wrapper.
# This test can be run as Python script or via PyTest.

import inspect
import itertools
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

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
from vizdoom.pettingzoo_wrapper import base_pettingzoo_env as backend  # noqa: E402
from vizdoom.pettingzoo_wrapper.base_pettingzoo_env import (  # noqa: E402
    _settle_game_after_reset,
)
from vizdoom.pettingzoo_wrapper.bot_eval_types import (  # noqa: E402
    BotEvalConfig,
    EpisodeResult,
    build_seed_schedule,
    summarize_tier,
)
from vizdoom.pettingzoo_wrapper.human_policy_duel import (  # noqa: E402
    HUMAN_BUTTONS,
    HUMAN_CONTROL_ARGS,
    HUMAN_CONTROL_COMMANDS,
    _configure_human_controls,
    player_frags,
    summary_payload,
)
from vizdoom.pettingzoo_wrapper.oracle_base import BaseOracle  # noqa: E402
from vizdoom.pettingzoo_wrapper.oracle_health_gathering import (  # noqa: E402
    HealthGatheringOracle,
)
from vizdoom.pettingzoo_wrapper.reward_wrappers import (  # noqa: E402
    DeathmatchRewardWrapper,
    HideAndSeekRewardWrapper,
    SharedRewardWrapper,
)
from vizdoom.pettingzoo_wrapper.vector_obs import (  # noqa: E402
    SCENARIO_VECTOR_OBS,
    VectorStateObservationWrapper,
)


_PORTS = itertools.count(50290, 1000)


class _ScriptedDuelEnv:
    """Two-agent env that replays scripted info dicts (engine counters)."""

    def __init__(self, reset_infos, step_infos):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self.metadata = {}
        self._reset_infos = reset_infos
        self._steps = iter(step_infos)

    def reset(self, seed=None, options=None):
        return {a: None for a in self.agents}, self._reset_infos

    def step(self, actions):
        infos = next(self._steps)
        done = {a: False for a in self.agents}
        return (
            {a: None for a in self.agents},
            {a: 0.0 for a in self.agents},
            done,
            done,
            infos,
        )


def _duel_info(frags=0, deaths=0, damage=0, taken=0, ammo=50):
    return {
        "FRAGCOUNT": float(frags),
        "DEATHCOUNT": float(deaths),
        "DAMAGECOUNT": float(damage),
        "DAMAGE_TAKEN": float(taken),
        "SELECTED_WEAPON_AMMO": float(ammo),
    }


def test_deathmatch_reward_is_zero_sum_for_a_frag_and_charges_ammo():
    idle = _duel_info()
    env = _ScriptedDuelEnv(
        {"agent_0": idle, "agent_1": idle},
        [
            # agent_0 fires one rocket (ammo 50 -> 49) and hits for 40 hp
            {
                "agent_0": _duel_info(damage=40, ammo=49),
                "agent_1": _duel_info(taken=40),
            },
            # second rocket kills: +1 frag / +1 death
            {
                "agent_0": _duel_info(frags=1, damage=140, ammo=48),
                "agent_1": _duel_info(deaths=1, taken=140),
            },
            # agent_1 respawns with refilled ammo: refills are free, and
            # nothing else changed -> no reward for either agent
            {
                "agent_0": _duel_info(frags=1, damage=140, ammo=48),
                "agent_1": _duel_info(deaths=1, taken=140, ammo=50),
            },
        ],
    )
    wrapped = DeathmatchRewardWrapper(env)
    wrapped.reset()

    _, r1, *_ = wrapped.step({})
    assert r1["agent_0"] == pytest.approx(40 * 0.0025 - 0.005)
    assert r1["agent_1"] == pytest.approx(-40 * 0.0025)

    _, r2, *_ = wrapped.step({})
    assert r2["agent_0"] == pytest.approx(1.0 + 100 * 0.0025 - 0.005)
    assert r2["agent_1"] == pytest.approx(-1.0 - 100 * 0.0025)
    # a full kill is worth at most 1.25 frag-equivalents, not 2 (old 0.01/hp)
    assert r2["agent_0"] < 1.3

    _, r3, *_ = wrapped.step({})
    assert r3 == {"agent_0": 0.0, "agent_1": 0.0}


def test_deathmatch_reward_penalizes_suicides_and_ignores_counter_resets():
    idle = _duel_info()
    env = _ScriptedDuelEnv(
        {"agent_0": idle, "agent_1": idle},
        [
            # agent_0 blows itself up: ZDoom decrements FRAGCOUNT, DEATHCOUNT +1
            {"agent_0": _duel_info(frags=-1, deaths=1, taken=100), "agent_1": idle},
            # per-episode engine counters drop back to zero on a hidden reset:
            # monotonic counters must not produce a (negative) reward
            {"agent_0": _duel_info(frags=-1), "agent_1": idle},
        ],
    )
    wrapped = DeathmatchRewardWrapper(env)
    wrapped.reset()

    _, r1, *_ = wrapped.step({})
    assert r1["agent_0"] == pytest.approx(-1.0 - 1.0 - 100 * 0.0025)
    assert r1["agent_1"] == 0.0

    _, r2, *_ = wrapped.step({})
    assert r2 == {"agent_0": 0.0, "agent_1": 0.0}


def test_deathmatch_reward_does_not_bill_the_respawn_frame_as_ammo_spent():
    idle = _duel_info(ammo=47)
    env = _ScriptedDuelEnv(
        {"agent_0": idle, "agent_1": idle},
        [
            # agent_0 dies (already rewarded by DEATHCOUNT), ammo still 47
            {"agent_0": _duel_info(deaths=1, taken=100, ammo=47), "agent_1": idle},
            # respawn frame: no weapon selected yet -> SELECTED_WEAPON_AMMO = -1
            {"agent_0": _duel_info(deaths=1, taken=100, ammo=-1), "agent_1": idle},
            # weapon back, ammo carried over
            {"agent_0": _duel_info(deaths=1, taken=100, ammo=47), "agent_1": idle},
            # a real shot afterwards is still charged
            {"agent_0": _duel_info(deaths=1, taken=100, ammo=46), "agent_1": idle},
        ],
    )
    wrapped = DeathmatchRewardWrapper(env)
    wrapped.reset()
    _, r1, *_ = wrapped.step({})
    assert r1["agent_0"] == pytest.approx(-1.0 - 100 * 0.0025)
    _, r2, *_ = wrapped.step({})
    assert r2["agent_0"] == 0.0
    _, r3, *_ = wrapped.step({})
    assert r3["agent_0"] == 0.0
    _, r4, *_ = wrapped.step({})
    assert r4["agent_0"] == pytest.approx(-0.005)


def test_deathmatch_reward_works_without_ammo_variable():
    idle = {k: v for k, v in _duel_info().items() if k != "SELECTED_WEAPON_AMMO"}
    hit = dict(idle, DAMAGECOUNT=10.0)
    env = _ScriptedDuelEnv(
        {"agent_0": idle, "agent_1": idle}, [{"agent_0": hit, "agent_1": idle}]
    )
    wrapped = DeathmatchRewardWrapper(env)
    wrapped.reset()
    _, rewards, *_ = wrapped.step({})
    assert rewards["agent_0"] == pytest.approx(10 * 0.0025)


class _ScriptedRoundEnv:
    """Two-agent env replaying reset infos (one per reset) and step infos."""

    def __init__(self, reset_infos, step_infos, truncate_at=None):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self.metadata = {}
        self._resets = iter(reset_infos)
        self._steps = iter(step_infos)
        self._truncate_at = truncate_at
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, seed=None, options=None):
        self.reset_calls += 1
        return {a: None for a in self.agents}, next(self._resets)

    def step(self, actions):
        self.step_calls += 1
        infos = next(self._steps)
        truncated = (
            self._truncate_at is not None and self.step_calls >= self._truncate_at
        )
        return (
            {a: None for a in self.agents},
            {a: 0.0 for a in self.agents},
            {a: False for a in self.agents},
            {a: truncated for a in self.agents},
            infos,
        )


def _hs_info(
    deaths=0, damage=0, taken=0, ammo=1, dead=0, yaw_mdeg=10000, fired=0, over=0
):
    return {
        "DEATHCOUNT": float(deaths),
        "DAMAGECOUNT": float(damage),
        "DAMAGE_TAKEN": float(taken),
        "SELECTED_WEAPON_AMMO": float(ammo),
        "DEAD": float(dead),
        "USER53": float(over),
        "USER54": 0.0 if dead else 1.0,
        "USER55": 1.0,
        "USER56": float(fired),
        "USER59": float(yaw_mdeg),
    }


def _hs_pair(shooter, hider):
    return {"agent_0": shooter, "agent_1": hider}


def test_hide_and_seek_round_outcome_is_zero_sum_with_damage_and_yaw_shaping():
    gamma, yaw_scale, dmg = 0.99, 0.05, 0.0025
    env = _ScriptedRoundEnv(
        [_hs_pair(_hs_info(yaw_mdeg=18000), _hs_info(ammo=0))],
        [
            # shooter turns towards the hider (18 -> 9 deg) and fires (USER56)
            _hs_pair(_hs_info(yaw_mdeg=9000, ammo=0, fired=1), _hs_info(ammo=0)),
            # splash for 40 hp
            _hs_pair(
                _hs_info(yaw_mdeg=9000, damage=40, ammo=0, fired=1),
                _hs_info(ammo=0, taken=40),
            ),
            # kill: hider death ends the round
            _hs_pair(
                _hs_info(yaw_mdeg=9000, damage=140, ammo=0, fired=2),
                _hs_info(ammo=0, taken=140, deaths=1),
            ),
        ],
    )
    wrapped = HideAndSeekRewardWrapper(env, gamma=gamma, yaw_shaping=yaw_scale)
    _, infos = wrapped.reset()
    assert infos["agent_0"]["hide_and_seek_role_code"] == 0.0
    assert infos["agent_1"]["hide_and_seek_role_code"] == 1.0
    assert infos["agent_0"]["hide_and_seek_spawn_retries"] == 0.0

    _, r1, term, trunc, infos = wrapped.step({})
    phi_prev, phi_now = -18000 / 1000 / 180, -9000 / 1000 / 180
    assert r1["agent_0"] == pytest.approx(yaw_scale * (gamma * phi_now - phi_prev))
    assert r1["agent_1"] == 0.0
    assert infos["agent_0"]["hide_and_seek_rocket_shots"] == 1.0
    assert not any(term.values())

    _, r2, *_ = wrapped.step({})
    assert r2["agent_0"] == pytest.approx(
        40 * dmg + yaw_scale * (gamma * phi_now - phi_now)
    )
    assert r2["agent_1"] == pytest.approx(-40 * dmg)

    _, r3, term, trunc, infos = wrapped.step({})
    # terminal: phi(s') = 0, so the potential term is -phi(s)
    assert r3["agent_0"] == pytest.approx(1.0 + 100 * dmg + yaw_scale * (0.0 - phi_now))
    assert r3["agent_1"] == pytest.approx(-1.0 - 100 * dmg)
    assert all(term.values()) and not any(trunc.values())
    assert infos["agent_0"]["hide_and_seek_outcome_code"] == 1.0  # shooter_win
    assert infos["agent_0"]["hide_and_seek_rocket_shots"] == 2.0
    assert wrapped.agents == []


def test_hide_and_seek_budget_flag_ends_round_as_hider_escape():
    idle = _hs_info(yaw_mdeg=0)
    env = _ScriptedRoundEnv(
        [_hs_pair(idle, _hs_info(ammo=0))],
        [
            _hs_pair(_hs_info(yaw_mdeg=0, fired=20, ammo=0), _hs_info(ammo=0)),
            _hs_pair(_hs_info(yaw_mdeg=0, fired=20, ammo=0, over=1), _hs_info(ammo=0)),
        ],
    )
    wrapped = HideAndSeekRewardWrapper(env, yaw_shaping=0.0)
    wrapped.reset()
    _, r1, term, *_ = wrapped.step({})
    assert r1 == {"agent_0": 0.0, "agent_1": 0.0}
    assert not any(term.values())
    _, r2, term, trunc, infos = wrapped.step({})
    assert r2 == {"agent_0": -1.0, "agent_1": 1.0}
    assert all(term.values()) and not any(trunc.values())
    assert infos["agent_1"]["hide_and_seek_outcome_code"] == 3.0  # hider_escape


def test_hide_and_seek_timeout_is_hider_escape_and_shooter_death_is_hider_win():
    idle = _hs_info(yaw_mdeg=0)
    env = _ScriptedRoundEnv(
        [_hs_pair(idle, idle), _hs_pair(idle, idle)],
        [
            _hs_pair(idle, idle),
            _hs_pair(_hs_info(yaw_mdeg=0, deaths=1, dead=1), idle),
        ],
        truncate_at=1,
    )
    wrapped = HideAndSeekRewardWrapper(env, yaw_shaping=0.0)
    wrapped.reset()
    _, rewards, term, trunc, infos = wrapped.step({})
    assert rewards == {"agent_0": -1.0, "agent_1": 1.0}
    assert all(trunc.values()) and not any(term.values())
    assert infos["agent_0"]["hide_and_seek_outcome_code"] == 3.0

    wrapped.reset()
    _, rewards, term, trunc, infos = wrapped.step({})
    assert rewards == {"agent_0": -1.0, "agent_1": 1.0}
    assert all(term.values())
    assert infos["agent_0"]["hide_and_seek_outcome_code"] == 2.0  # hider_win


def test_hide_and_seek_rerolls_a_reset_with_a_telefragged_player():
    stomped = _hs_pair(_hs_info(deaths=1, dead=1, taken=100), _hs_info(damage=100))
    clean = _hs_pair(_hs_info(), _hs_info(ammo=0))
    env = _ScriptedRoundEnv([stomped, clean], [_hs_pair(_hs_info(), _hs_info(ammo=0))])
    wrapped = HideAndSeekRewardWrapper(env, yaw_shaping=0.0)
    _, infos = wrapped.reset()
    assert env.reset_calls == 2
    assert infos["agent_0"]["DEATHCOUNT"] == 0.0
    assert infos["agent_0"]["hide_and_seek_spawn_retries"] == 1.0
    # the stomp is not counted as a death of the new round
    _, rewards, term, *_ = wrapped.step({})
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert not any(term.values())


def test_hide_and_seek_pit_death_is_hider_suicide_and_pit_damage_is_not_shaped():
    idle = _hs_info(yaw_mdeg=0)
    env = _ScriptedRoundEnv(
        [_hs_pair(idle, _hs_info(ammo=0))],
        # the hider takes 100 damage from the pit and dies; the shooter dealt nothing
        [_hs_pair(idle, _hs_info(ammo=0, taken=100, deaths=1, dead=1))],
    )
    wrapped = HideAndSeekRewardWrapper(env, yaw_shaping=0.0)
    wrapped.reset()
    _, rewards, term, _, infos = wrapped.step({})
    assert rewards == {"agent_0": 1.0, "agent_1": -1.0}
    assert all(term.values())
    assert infos["agent_0"]["hide_and_seek_outcome_code"] == 5.0  # hider_suicide


def test_hide_and_seek_falls_back_to_ammo_drop_without_acs_counter():
    def bare(ammo, deaths=0):
        return {"DEATHCOUNT": float(deaths), "SELECTED_WEAPON_AMMO": float(ammo)}

    env = _ScriptedRoundEnv(
        [_hs_pair(bare(1), bare(0))],
        [
            _hs_pair(bare(0), bare(0)),
            _hs_pair(bare(1), bare(0)),
            _hs_pair(bare(0), bare(0)),
        ],
    )
    wrapped = HideAndSeekRewardWrapper(env)
    wrapped.reset()
    shots = [
        wrapped.step({})[4]["agent_0"]["hide_and_seek_rocket_shots"] for _ in range(3)
    ]
    assert shots == [1.0, 1.0, 2.0]


def test_bot_eval_defaults_follow_training_respawn_and_scenario():
    config = BotEvalConfig()
    assert config.scenario is None
    assert config.respawn_delay == base_env_common.TRAINING_RESPAWN_DELAY
    assert base_env_common.TRAINING_RESPAWN_DELAY >= 1


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


def test_human_policy_duel_metrics_use_human_perspective():
    server_state = SimpleNamespace(
        players_names=("Policy", "Human", "Disconnected"),
        players_frags=(3, 5, 99),
        players_in_game=(True, True, False),
    )
    assert player_frags(server_state) == {"Policy": 3, "Human": 5}

    result = EpisodeResult(
        seed=42,
        tier="policy",
        valid=True,
        learner_frags=5,
        bot_frags=3,
        learner_deaths=2,
        learner_damage_made=100.0,
        learner_damage_taken=50.0,
        outcome="win",
    )
    summary = summary_payload([result], bootstrap_seed=42)
    assert summary["win_rate"] == 1.0
    assert summary["human_frags_mean"] == 5.0
    assert summary["policy_frags_mean"] == 3.0


def test_human_policy_duel_configures_wasd_and_mouse_look():
    game = Mock()

    _configure_human_controls(game)

    game.set_available_buttons.assert_called_once_with(list(HUMAN_BUTTONS))
    game.add_game_args.assert_called_once_with(HUMAN_CONTROL_ARGS)
    for command in (
        "use_mouse 1",
        "mouse_capturemode 2",
        "m_yaw 1",
        "m_pitch 1",
        "freelook 1",
    ):
        assert command in HUMAN_CONTROL_COMMANDS
        assert f"+{command}" in HUMAN_CONTROL_ARGS
    for key, command in (
        ("w", "+forward"),
        ("a", "+moveleft"),
        ("s", "+back"),
        ("d", "+moveright"),
    ):
        assert f'+bind {key} "{command}"' in HUMAN_CONTROL_ARGS


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


def test_base_env_continuous_actions_opt_in_for_binary_buttons(monkeypatch):
    monkeypatch.setattr(
        base_env_common,
        "discover_buttons",
        lambda _path: [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.ATTACK],
    )
    env = base_env_common.VizdoomParallelEnvBase(
        config_file="unused.cfg",
        num_agents=2,
        skip_frames=4,
        continuous_actions=True,
    )

    action_space = env.action_space("agent_0")
    assert isinstance(action_space, base_env_common.spaces.Box)
    assert action_space.shape == (3,)
    np.testing.assert_array_equal(action_space.low, np.zeros(3))
    np.testing.assert_array_equal(action_space.high, np.ones(3))
    np.testing.assert_array_equal(env._noop_action(), np.zeros(3))
    assert env._encode_env_action(np.asarray([0.2, 0.5, 0.9])) == [0.0, 1.0, 1.0]


class _ScriptedInfoEnv:
    """Two-agent env with fixed observations and scripted rewards/infos."""

    metadata = {}

    def __init__(self, observation, reset_infos, steps):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._observation = observation
        self._reset_infos = reset_infos
        self._steps = iter(steps)

    def observation_space(self, agent):
        if isinstance(self._observation, dict):
            return base_env_common.spaces.Dict(
                {
                    key: base_env_common.spaces.Box(
                        0, 255, shape=value.shape, dtype=np.uint8
                    )
                    for key, value in self._observation.items()
                }
            )
        return base_env_common.spaces.Box(
            0, 255, shape=self._observation.shape, dtype=np.uint8
        )

    def action_space(self, agent):
        return base_env_common.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        infos = {agent: dict(info) for agent, info in self._reset_infos.items()}
        return {agent: self._observation for agent in self.agents}, infos

    def step(self, actions):
        rewards, infos = next(self._steps)
        done = {agent: False for agent in self.agents}
        return (
            {agent: self._observation for agent in self.agents},
            dict(rewards),
            done,
            done,
            {agent: dict(info) for agent, info in infos.items()},
        )


def test_vector_obs_adds_vector_next_to_the_image_observation():
    spec = SCENARIO_VECTOR_OBS["health_gathering_multi_agent"]
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    env = VectorStateObservationWrapper(
        _ScriptedInfoEnv(
            image,
            {"agent_0": {"HEALTH": 100.0}, "agent_1": {"HEALTH": 50.0}},
            [({}, {"agent_0": {"HEALTH": 80.0}, "agent_1": {"HEALTH": 20.0}})],
        ),
        spec,
    )

    space = env.observation_space("agent_0")
    assert set(space.spaces) == {"observation", "vector"}
    assert space["vector"].shape == (2,)

    obs, _ = env.reset()
    assert obs["agent_0"]["observation"] is image
    # Own health first, then the other agents' in possible_agents order.
    np.testing.assert_allclose(obs["agent_0"]["vector"], [1.0, 0.5])
    np.testing.assert_allclose(obs["agent_1"]["vector"], [0.5, 1.0])

    obs, *_ = env.step({"agent_0": 0, "agent_1": 0})
    np.testing.assert_allclose(obs["agent_0"]["vector"], [0.8, 0.2])


def test_vector_obs_keeps_existing_dict_observation_keys():
    spec = SCENARIO_VECTOR_OBS["health_gathering_multi_agent"]
    observation = {
        "observation": np.zeros((4, 4, 3), dtype=np.uint8),
        "audio": np.zeros((4, 4, 2), dtype=np.uint8),
    }
    env = VectorStateObservationWrapper(
        _ScriptedInfoEnv(
            observation,
            {"agent_0": {"HEALTH": 100.0}, "agent_1": {"HEALTH": 100.0}},
            [],
        ),
        spec,
    )

    assert set(env.observation_space("agent_0").spaces) == {
        "observation",
        "audio",
        "vector",
    }
    obs, _ = env.reset()
    assert set(obs["agent_0"]) == {"observation", "audio", "vector"}


@pytest.mark.parametrize("agg, expected", [("sum", 3.0), ("mean", 1.5)])
def test_shared_reward_pools_rewards_and_keeps_individual_reward(agg, expected):
    env = SharedRewardWrapper(
        _ScriptedInfoEnv(
            np.zeros((4, 4, 3), dtype=np.uint8),
            {"agent_0": {}, "agent_1": {}},
            [({"agent_0": 2.0, "agent_1": 1.0}, {"agent_0": {}, "agent_1": {}})],
        ),
        agg=agg,
    )

    _, infos = env.reset()
    assert infos["agent_0"]["individual_reward"] == 0.0
    _, rewards, _, _, infos = env.step({"agent_0": 0, "agent_1": 0})
    assert rewards == {"agent_0": expected, "agent_1": expected}
    assert infos["agent_0"]["individual_reward"] == 2.0
    assert infos["agent_1"]["individual_reward"] == 1.0


def test_oracle_packs_buttons_into_discrete_bitmask():
    assert BaseOracle.to_discrete(np.asarray([0.0, 0.0, 0.0])) == 0
    assert BaseOracle.to_discrete(np.asarray([1.0, 0.0, 1.0])) == 5
    assert BaseOracle.to_discrete(np.asarray([1.0, 1.0, 1.0])) == 7

    buttons = [Button.TURN_LEFT, Button.TURN_RIGHT, Button.MOVE_FORWARD]
    agents = ["agent_0", "agent_1"]
    assert HealthGatheringOracle(buttons=buttons, agents=agents).discrete
    assert not HealthGatheringOracle(
        buttons=buttons, agents=agents, discrete=False
    ).discrete


def test_health_gathering_vector_obs_with_continuous_actions():
    env = pettingzoo_wrapper.make(
        scenario="health_gathering_multi_agent",
        num_agents=2,
        resolution="160X120",
        timeout=300,
        barrier_timeout=10.0,
        skip_frames=4,
        async_mode=False,
        ticrate=35,
        seed=7,
        port=next(_PORTS),
        enable_video=False,
        verbose=False,
        vector_obs=True,
        continuous_actions=True,
    )
    try:
        action_space = env.action_space("agent_0")
        assert isinstance(action_space, base_env_common.spaces.Box)
        obs, infos = env.reset(seed=7)
        for agent in env.possible_agents:
            assert set(obs[agent]) == {"observation", "vector"}
            assert obs[agent]["vector"].shape == (2,)
            assert obs[agent]["vector"][0] == pytest.approx(
                infos[agent]["HEALTH"] / 100.0
            )
        actions = {
            agent: np.ones(action_space.shape, dtype=np.float32) for agent in env.agents
        }
        obs, rewards, _, _, infos = env.step(actions)
        assert set(rewards) == set(env.possible_agents)
        assert all("death_penalty" in infos[agent] for agent in env.possible_agents)
    finally:
        env.close()


def test_base_env_uses_checkpoint_buttons_over_current_scenario(monkeypatch):
    monkeypatch.setattr(
        base_env_common,
        "discover_buttons",
        lambda _path: [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.ATTACK],
    )
    checkpoint_buttons = [
        Button.MOVE_LEFT,
        Button.MOVE_RIGHT,
        Button.TURN_LEFT,
        Button.TURN_RIGHT,
        Button.ATTACK,
    ]

    env = base_env_common.VizdoomParallelEnvBase(
        config_file="unused.cfg",
        num_agents=2,
        available_buttons=checkpoint_buttons,
    )

    assert env.available_buttons == tuple(checkpoint_buttons)
    assert env.action_space("agent_0").n == 32


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


@pytest.mark.parametrize("frame_stack", [1, 3])
@pytest.mark.parametrize("audio", [False, True])
def test_base_env_modality_stacking_state_and_rendering(audio, frame_stack):
    env = base_env_common.VizdoomParallelEnvBase(
        config_file="simple_tag_audio.cfg" if audio else "unused.cfg",
        available_buttons=[Button.ATTACK],
        num_agents=2,
        resolution="4X3",
        frame_stack=frame_stack,
        render_mode="rgb_array",
    )
    channels = {"observation": 3, "audio": 2} if audio else {"observation": 3}

    def modalities(observation):
        if audio:
            assert isinstance(observation, dict)
            assert set(observation) == set(channels)
            return observation
        assert isinstance(observation, np.ndarray)
        return {"observation": observation}

    def frame(value):
        result = {
            key: np.full((3, 4, count), value + index, dtype=np.uint8)
            for index, (key, count) in enumerate(channels.items())
        }
        return result if audio else result["observation"]

    assert env.state_space.contains(env.state())
    for value in modalities(env.state()).values():
        assert not value.any()
    first = {agent: frame(10 + index) for index, agent in enumerate(env.agents)}
    second = {agent: frame(20 + index) for index, agent in enumerate(env.agents)}
    reset_obs = env._stack_observations(first, reset=True)
    step_obs = env._stack_observations(second, reset=False)
    assert env.state_space.contains(env.state())
    for agent in env.agents:
        assert env.observation_space(agent).contains(step_obs[agent])
        for key, count in channels.items():
            reset_value = modalities(reset_obs[agent])[key]
            step_value = modalities(step_obs[agent])[key]
            assert step_value.shape == (3, 4, count * frame_stack)
            np.testing.assert_array_equal(
                reset_value,
                np.concatenate([modalities(first[agent])[key]] * frame_stack, axis=-1),
            )
            np.testing.assert_array_equal(
                step_value,
                np.concatenate(
                    [modalities(first[agent])[key]] * (frame_stack - 1)
                    + [modalities(second[agent])[key]],
                    axis=-1,
                ),
            )
        np.testing.assert_array_equal(
            env.rgb_observation(step_obs[agent]),
            modalities(second[agent])["observation"],
        )
    for key, count in channels.items():
        state_value = modalities(env.state())[key]
        assert state_value.shape == (3, 4, count * frame_stack * 2)
        np.testing.assert_array_equal(
            state_value,
            np.concatenate(
                [modalities(step_obs[agent])[key] for agent in env.agents], axis=-1
            ),
        )
    np.testing.assert_array_equal(
        env.render(),
        np.concatenate(
            [modalities(second[agent])["observation"] for agent in env.agents], axis=1
        ),
    )
    repeated = env._stack_observations(second, reset=True)
    for agent in env.agents:
        for key in channels:
            np.testing.assert_array_equal(
                modalities(repeated[agent])[key],
                np.concatenate([modalities(second[agent])[key]] * frame_stack, axis=-1),
            )


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("notifications", [False, True])
def test_worker_transports_separate_modalities_and_copies_snapshots(
    monkeypatch, audio, notifications
):
    monkeypatch.setattr(
        backend.VizdoomParallelEnv, "_spawn_agent_worker_process", lambda self: None
    )
    env = backend.VizdoomParallelEnv(
        config_file="simple_tag_audio.cfg" if audio else "unused.cfg",
        available_buttons=[Button.ATTACK],
        num_agents=2,
        resolution="16X12",
        frame_stack=2,
    )
    games = []

    def configure_game(**kwargs):
        game = Mock()
        rgb = np.full((3, 12, 16), 10 + kwargs["agent_idx"], dtype=np.uint8)
        pcm = np.full((512, 2), 16000, dtype=np.int16)
        pcm[::2] *= -1
        game.get_state.return_value = SimpleNamespace(
            screen_buffer=rgb,
            audio_buffer=pcm,
            notifications_buffer=f"Player{kwargs['agent_idx']}: RED",
        )
        game.is_notifications_buffer_enabled.return_value = notifications
        game.get_available_game_variables.return_value = []
        game.get_available_buttons_size.return_value = 1
        game.is_player_dead.return_value = False
        game.is_episode_finished.return_value = False
        game.get_last_reward.return_value = 0.0
        games.append(game)
        return game

    def initialize(coordinator):
        coordinator._spawn_agent_worker_threads()
        coordinator._dispatch([backend._Task("init", port=5029)] * 2, 3)

    monkeypatch.setattr(backend, "configure_doom_game", configure_game)
    monkeypatch.setattr(
        backend._AgentWorkerCoordinator, "_initialize_agent_workers", initialize
    )
    coordinator = None
    names = [shm.name for shm in env._shms.values()]
    try:
        assert len(names) == (2 if audio else 1)
        assert len(set(names)) == len(names)
        if audio:
            assert env._shm_frames["observation"].shape == (2, 12, 16, 3)
            assert env._shm_frames["audio"].shape == (2, 12, 16, 2)
            assert not np.shares_memory(
                env._shm_frames["observation"], env._shm_frames["audio"]
            )
        else:
            assert env._shm_frames.shape == (2, 12, 16, 3)
        coordinator = backend._AgentWorkerCoordinator(
            config_path=env.config_file,
            resolution=env.resolution,
            timeout=None,
            skip_frames=1,
            num_agents=2,
            host_address=env.host_address,
            port=env.port,
            slot_index=0,
            async_mode=False,
            netmode=0,
            ticrate=35,
            seed=1,
            verbose=False,
            audio_diagnostics=False,
            available_buttons=env.available_buttons,
            frames=env._shm_frames,
        )
        parent_conn = Mock()
        env._parent_conn = parent_conn

        def receive(**kwargs):
            command = parent_conn.send.call_args.args[0]
            return (
                coordinator.reset()
                if command["cmd"] == "reset"
                else coordinator.step(command["actions"])
            )

        monkeypatch.setattr(env, "_recv_message", receive)
        observations, infos = env.reset()
        for index, agent in enumerate(env.agents):
            obs = observations[agent]
            assert env.observation_space(agent).contains(obs)
            if notifications:
                assert infos[agent]["_notifications"] == [f"Player{index}: RED"]
            else:
                assert "_notifications" not in infos[agent]
            rgb = obs["observation"] if audio else obs
            np.testing.assert_array_equal(rgb, 10 + index)
            if audio:
                assert obs["audio"].any()
                assert obs["audio"].shape == (12, 16, 4)
        for game in games:
            game.get_state.return_value = None
            game.is_episode_finished.return_value = True
            game.is_episode_timeout_reached.return_value = True
        terminal, _, terminated, truncated, _ = env.step({})
        assert not any(terminated.values())
        assert all(truncated.values())
        for agent in env.agents:
            assert env.observation_space(agent).contains(terminal[agent])
            before = observations[agent]
            after = terminal[agent]
            before = before if audio else {"observation": before}
            after = after if audio else {"observation": after}
            for key, count in (
                {"observation": 3, "audio": 2} if audio else {"observation": 3}
            ).items():
                assert before[key].any()
                np.testing.assert_array_equal(
                    after[key][..., :count], before[key][..., -count:]
                )
                assert not after[key][..., -count:].any()
        env._hidden_reset_after_recovery()
        assert env.state_space.contains(env.state())
        values = env.state().values() if audio else [env.state()]
        assert all(not value.any() for value in values)
    finally:
        if coordinator is not None:
            coordinator.close()
        env._parent_conn = None
        env.close()
        env.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            backend.shared_memory.SharedMemory(name=name)


@pytest.mark.parametrize("failure", ["allocation", "startup"])
def test_audio_shared_memory_is_cleaned_up_on_init_failure(monkeypatch, failure):
    allocated = []
    shared_memory_class = backend.shared_memory.SharedMemory

    def allocate(**kwargs):
        if failure == "allocation" and allocated:
            raise RuntimeError("allocation failed")
        shm = shared_memory_class(**kwargs)
        allocated.append(shm.name)
        return shm

    def spawn(self):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(backend.shared_memory, "SharedMemory", allocate)
    monkeypatch.setattr(
        backend.VizdoomParallelEnv, "_spawn_agent_worker_process", spawn
    )
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        backend.VizdoomParallelEnv(
            config_file="simple_tag_audio.cfg",
            resolution="4X3",
            available_buttons=[Button.ATTACK],
        )
    assert len(allocated) == (1 if failure == "allocation" else 2)
    for name in allocated:
        with pytest.raises(FileNotFoundError):
            shared_memory_class(name=name)


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
