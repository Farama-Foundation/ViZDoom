from typing import Any, Dict, Optional

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper


class SimpleTagRewardWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        if env.possible_agents != [f"agent_{i}" for i in range(4)]:
            raise ValueError(
                "Simple Tag requires exactly four agents (three hunters, one prey)"
            )
        self.agents = env.agents[:]

    def _annotate(self, infos, outcome=0):
        for index, agent in enumerate(self.possible_agents):
            info = infos[agent]
            if not all(key in info for key in ("USER50", "USER51", "USER52")):
                raise ValueError("Simple Tag config requires USER50, USER51 and USER52")
            info["tag_role_code"] = float(index == 3)
            info["tag_outcome_code"] = float(outcome)
            info["tag_round_seconds"] = float(info["USER52"]) / 35.0

    def reset(self, seed=None, options=None):
        for attempt in range(4):
            obs, infos = self.env.reset(
                seed=seed if attempt == 0 else None, options=options
            )
            self.agents = self.env.agents[:]
            self._annotate(infos)
            if not any(
                info.get("player_dead") or info["USER50"] for info in infos.values()
            ):
                return obs, infos
        raise RuntimeError(
            "Simple Tag could not obtain a live, uncaught four-player reset"
        )

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        obs, _, terminations, truncations, infos = self.env.step(actions)
        self._annotate(infos)
        recovered = any(info.get("_hidden_reset") for info in infos.values())
        caught = any(float(info["USER50"]) > 0 for info in infos.values())
        timed_out = bool(truncations) and all(truncations.values())
        unexpected_end = any(terminations.values()) or any(
            info.get("player_dead") for info in infos.values()
        )
        outcome = 0
        if recovered:
            outcome = 3
        elif caught:
            outcome = 1
        elif timed_out:
            outcome = 2
        elif unexpected_end:
            outcome = 3
        team_reward = 10.0 if outcome == 1 else -10.0 if outcome == 2 else 0.0
        rewards = {
            agent: team_reward if index < 3 else -team_reward
            for index, agent in enumerate(self.possible_agents)
        }
        self._annotate(infos, outcome)
        if outcome:
            terminations = dict.fromkeys(self.possible_agents, outcome == 1)
            truncations = dict.fromkeys(self.possible_agents, outcome != 1)
            self.agents = []
        return obs, rewards, terminations, truncations, infos


def _reset_info(info):
    reset_info = info.get("reset_info")
    if isinstance(reset_info, dict):
        return reset_info
    return None


_DEATHMATCH_DELTA_REWARDS = {
    "FRAGCOUNT": 1.0,
    "DEATHCOUNT": -1.0,
    "DAMAGECOUNT": 0.0025,  # 0.0025/HP, but we use rocket which deals 100HP per hit
    "DAMAGE_TAKEN": -0.0025,
}
_DEATHMATCH_SIGNED_COUNTERS = frozenset({"FRAGCOUNT"})
_DEATHMATCH_AMMO_COST = 0.005
_DEATHMATCH_AMMO_VARIABLE = "SELECTED_WEAPON_AMMO"


class DeathmatchRewardWrapper(ParallelEnv):
    def __init__(
        self,
        env: ParallelEnv,
        *,
        delta_rewards: Optional[Dict[str, float]] = None,
        ammo_cost: float = _DEATHMATCH_AMMO_COST,
    ):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.delta_rewards = dict(
            _DEATHMATCH_DELTA_REWARDS if delta_rewards is None else delta_rewards
        )
        self.ammo_cost = float(ammo_cost)
        self.prev_vars: Dict[str, Dict[str, float]] = {}
        self._counters_checked = False

    @property
    def _tracked_variables(self) -> tuple:
        names = list(self.delta_rewards)
        if self.ammo_cost:
            names.append(_DEATHMATCH_AMMO_VARIABLE)
        return tuple(names)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def state_space(self):
        return self.env.state_space

    def state(self):
        return self.env.state()

    def state_observation(self, agent: str):
        return self.env.state_observation(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self.check_ctr(infos)
        # Frst step
        self.prev_vars = {
            agent: {
                name: infos[agent][name]
                for name in self._tracked_variables
                if isinstance(infos.get(agent), dict) and name in infos[agent]
            }
            for agent in self.agents
        }
        return obs, infos

    def check_ctr(self, infos: Dict[str, Any]) -> None:
        if self._counters_checked:
            return
        for agent in self.agents:
            info = infos.get(agent)
            if not isinstance(info, dict):
                return
            present = [n for n in self.delta_rewards if n in info]
            missing = [n for n in self.delta_rewards if n not in info]
            if not present:
                return
            if missing:
                raise ValueError(
                    f"DeathmatchRewardWrapper needs {sorted(missing)} in the "
                    f"scenario's available_game_variables (agent {agent!r} "
                    f"reported {sorted(info)})"
                )
        self._counters_checked = True

    def step(self, actions: Dict[str, Any]):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        for agent in self.agents:
            info = infos[agent]
            if _reset_info(info) is not None:
                self.prev_vars[agent] = {}

            shaping_reward = 0.0
            previous = self.prev_vars[agent]
            for name, weight in self.delta_rewards.items():
                if name not in previous or name not in info:
                    continue
                delta = info[name] - previous[name]
                if delta > 1e-8 or (
                    delta < -1e-8 and name in _DEATHMATCH_SIGNED_COUNTERS
                ):
                    shaping_reward += delta * weight
            ammo_known = (
                _DEATHMATCH_AMMO_VARIABLE in info
                and info[_DEATHMATCH_AMMO_VARIABLE] >= 0
            )
            if self.ammo_cost and ammo_known and _DEATHMATCH_AMMO_VARIABLE in previous:
                spent = (
                    previous[_DEATHMATCH_AMMO_VARIABLE]
                    - info[_DEATHMATCH_AMMO_VARIABLE]
                )
                if spent > 1e-8:
                    shaping_reward -= spent * self.ammo_cost

            rewards[agent] = float(rewards.get(agent, 0.0)) + shaping_reward
            # Carry last known value forward for anything absent this step
            self.prev_vars[agent] = {
                name: (
                    info[name]
                    if name in info
                    and (name != _DEATHMATCH_AMMO_VARIABLE or ammo_known)
                    else self.prev_vars[agent].get(name, info.get(name))
                )
                for name in self._tracked_variables
                if name in info or name in self.prev_vars[agent]
            }

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


_HIDE_AND_SEEK_DAMAGE_SHAPING = 0.0025  # per hp, as in the symmetric duel
_HIDE_AND_SEEK_YAW_SHAPING = 0  # scale of the shooter's yaw-error (was 0.05)
_HS_ROUND_OVER = "USER53"  # budget spent + grace elapsed -> hider_escape
_HS_SHOOTER_ALIVE = "USER54"
_HS_HIDER_ALIVE = "USER55"
_HS_ROCKETS_FIRED = "USER56"
_HS_SHOOTER_YAW_ERROR_MDEG = "USER59"


class HideAndSeekRewardWrapper(ParallelEnv):
    roles = {"agent_0": "shooter", "agent_1": "hider"}
    role_codes = {"agent_0": 0.0, "agent_1": 1.0}
    outcome_codes = {
        "ongoing": 0.0,
        "shooter_win": 1.0,
        "hider_win": 2.0,
        "hider_escape": 3.0,
        "draw": 4.0,
        "hider_suicide": 5.0,
    }
    max_spawn_retries = 3

    def __init__(
        self,
        env: ParallelEnv,
        *,
        win_reward: float = 1.0,
        damage_shaping: float = _HIDE_AND_SEEK_DAMAGE_SHAPING,
        yaw_shaping: float = _HIDE_AND_SEEK_YAW_SHAPING,
        gamma: float = 0.99,
    ):
        self.env = env
        self.win_reward = float(win_reward)
        self.damage_shaping = float(damage_shaping)
        self.yaw_shaping = float(yaw_shaping)
        self.gamma = float(gamma)
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self._previous_deaths: Dict[str, float] = {}
        self._previous_shooter_ammo: Optional[float] = None
        self._previous_damage: Optional[float] = None
        self._previous_yaw_potential: Optional[float] = None
        self._rocket_shots = 0.0
        self._spawn_retries = 0

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def state_space(self):
        return self.env.state_space

    def state(self):
        return self.env.state()

    def state_observation(self, agent: str):
        return self.env.state_observation(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def _validate_and_annotate(self, infos: Dict[str, Any]) -> None:
        if set(self.agents) != set(self.roles):
            raise ValueError(
                "HideAndSeekRewardWrapper requires exactly agent_0 (shooter) "
                "and agent_1 (hider)"
            )
        for agent in self.roles:
            if "DEATHCOUNT" not in infos.get(agent, {}):
                raise ValueError(
                    "HideAndSeekRewardWrapper requires DEATHCOUNT in the "
                    f"scenario's available_game_variables ({agent!r})"
                )
            infos[agent]["hide_and_seek_role_code"] = self.role_codes[agent]
            infos[agent]["hide_and_seek_outcome_code"] = self.outcome_codes["ongoing"]
            infos[agent]["hide_and_seek_rocket_shots"] = self._rocket_shots
            infos[agent]["hide_and_seek_spawn_retries"] = float(self._spawn_retries)

    @staticmethod
    def _dead_at_reset(infos: Dict[str, Any]) -> bool:
        for info in infos.values():
            if not isinstance(info, dict):
                continue
            if info.get("player_dead") or float(info.get("DEAD", 0.0) or 0.0) > 0:
                return True
        return False

    @staticmethod
    def _yaw_potential(info: Dict[str, Any]) -> Optional[float]:
        """phi(s) = -|yaw error| / 180 for the shooter, None when not measurable."""
        yaw = info.get(_HS_SHOOTER_YAW_ERROR_MDEG)
        if yaw is None:
            return None
        if _HS_SHOOTER_ALIVE in info and _HS_HIDER_ALIVE in info:
            if not (
                float(info[_HS_SHOOTER_ALIVE]) > 0 and float(info[_HS_HIDER_ALIVE]) > 0
            ):
                return None
        return -min(max(float(yaw) / 1000.0, 0.0), 180.0) / 180.0

    def _start_round(self, infos: Dict[str, Any]) -> None:
        self._previous_deaths = {
            agent: float(infos[agent]["DEATHCOUNT"]) for agent in self.agents
        }
        shooter_ammo = infos["agent_0"].get("SELECTED_WEAPON_AMMO")
        self._previous_shooter_ammo = (
            None if shooter_ammo is None else float(shooter_ammo)
        )
        self._previous_damage = float(infos["agent_0"].get("DAMAGECOUNT", 0.0))
        self._previous_yaw_potential = self._yaw_potential(infos["agent_0"])
        self._rocket_shots = 0.0
        for info in infos.values():
            info["hide_and_seek_rocket_shots"] = self._rocket_shots
            info["hide_and_seek_spawn_retries"] = float(self._spawn_retries)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._validate_and_annotate(infos)
        self._spawn_retries = 0
        while (
            self._dead_at_reset(infos) and self._spawn_retries < self.max_spawn_retries
        ):
            self._spawn_retries += 1
            obs, infos = self.env.reset(seed=None, options=options)
            self.agents = self.env.agents[:]
            self._validate_and_annotate(infos)
        self._start_round(infos)
        return obs, infos

    def _update_rocket_shots(self, infos: Dict[str, Any]) -> None:
        fired = infos["agent_0"].get(_HS_ROCKETS_FIRED)
        if fired is not None:
            self._rocket_shots = float(fired)
            return
        shooter_ammo = infos["agent_0"].get("SELECTED_WEAPON_AMMO")
        if shooter_ammo is not None:
            shooter_ammo = float(shooter_ammo)
            if self._previous_shooter_ammo is not None:
                self._rocket_shots += max(
                    0.0, self._previous_shooter_ammo - shooter_ammo
                )
            self._previous_shooter_ammo = shooter_ammo

    def _shooter_damage_delta(self, infos: Dict[str, Any]) -> float:
        """Damage dealt by the shooter this step"""
        dealt = infos["agent_0"].get("DAMAGECOUNT")
        if dealt is None:
            return 0.0
        dealt = float(dealt)
        previous = dealt if self._previous_damage is None else self._previous_damage
        self._previous_damage = dealt
        delta = dealt - previous
        # the counter only grows within a round; a drop means the engine reset it
        return delta if delta > 1e-8 else 0.0

    def _damage_shaping(self, dealt: float) -> Dict[str, float]:
        """Zero sum"""
        if not self.damage_shaping or dealt <= 0.0:
            return {"agent_0": 0.0, "agent_1": 0.0}
        return {
            "agent_0": dealt * self.damage_shaping,
            "agent_1": -dealt * self.damage_shaping,
        }

    def _yaw_shaping(self, infos: Dict[str, Any], terminal: bool) -> float:
        """For shooter: scale * (gamma * phi(s') - phi(s))"""
        if not self.yaw_shaping:
            return 0.0
        previous = self._previous_yaw_potential
        current = None if terminal else self._yaw_potential(infos["agent_0"])
        self._previous_yaw_potential = current
        if previous is None:
            return 0.0
        next_potential = 0.0 if current is None else current
        return self.yaw_shaping * (self.gamma * next_potential - previous)

    def step(self, actions: Dict[str, Any]):
        obs, _, terminations, truncations, infos = self.env.step(actions)
        self._validate_and_annotate(infos)

        self._update_rocket_shots(infos)
        for info in infos.values():
            info["hide_and_seek_rocket_shots"] = self._rocket_shots

        died = {}
        for agent in self.agents:
            reset_info = _reset_info(infos[agent])
            if reset_info is not None and "DEATHCOUNT" in reset_info:
                self._previous_deaths[agent] = float(reset_info["DEATHCOUNT"])
            current = float(infos[agent]["DEATHCOUNT"])
            died[agent] = bool(infos[agent].get("just_died", False)) or (
                current > self._previous_deaths[agent]
            )
            self._previous_deaths[agent] = current

        shooter_died = died["agent_0"]
        hider_died = died["agent_1"]
        timed_out = bool(truncations) and all(truncations.values())
        budget_spent = float(infos["agent_0"].get(_HS_ROUND_OVER, 0.0) or 0.0) > 0
        dealt = self._shooter_damage_delta(infos)
        rewards = {agent: 0.0 for agent in self.agents}

        if shooter_died and hider_died:
            outcome = "draw"
        elif hider_died:
            # a kill always comes with shooter damage in the same step; a death
            # without it is the instant-death pit
            outcome = "shooter_win" if dealt > 0.0 else "hider_suicide"
            rewards = {"agent_0": self.win_reward, "agent_1": -self.win_reward}
        elif shooter_died:
            outcome = "hider_win"
            rewards = {"agent_0": -self.win_reward, "agent_1": self.win_reward}
        elif timed_out or budget_spent:
            outcome = "hider_escape"
            rewards = {"agent_0": -self.win_reward, "agent_1": self.win_reward}
        else:
            outcome = "ongoing"

        shaped = self._damage_shaping(dealt)
        shaped["agent_0"] += self._yaw_shaping(infos, terminal=outcome != "ongoing")
        for agent in self.agents:
            rewards[agent] = float(rewards.get(agent, 0.0)) + shaped.get(agent, 0.0)

        if shooter_died or hider_died or (budget_spent and not timed_out):
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
        for info in infos.values():
            info["hide_and_seek_outcome_code"] = self.outcome_codes[outcome]
        if outcome != "ongoing":
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class HealthGatheringRewardWrapper(ParallelEnv):
    """
    Dense shaping on HEALTH. Optional sparse goal on reaching max health.
    """

    def __init__(
        self,
        env: ParallelEnv,
        *,
        death_penalty: float = -10.0,
        medkit_reward: float = 1.0,
        health_key: str = "HEALTH",
        dead_key: str = "DEAD",
    ):
        self.env = env
        self.death_penalty = death_penalty
        self.medkit_reward = medkit_reward
        self.health_key = health_key
        self.dead_key = dead_key
        self.prev_health: Dict[str, Optional[float]] = {}
        self.possible_agents = env.possible_agents
        self.agents = env.agents

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def state_space(self):
        return self.env.state_space

    def state(self):
        return self.env.state()

    def state_observation(self, agent: str):
        return self.env.state_observation(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self.prev_health = {a: infos[a][self.health_key] for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        for a in self.agents:
            reset_info = _reset_info(infos[a])
            if reset_info is not None:
                self.prev_health[a] = reset_info.get(
                    self.health_key, self.prev_health[a]
                )

            r = rewards[a]
            h_cur = infos[a].get(self.health_key, 0.0)
            h_prev = self.prev_health[a]

            if h_cur > h_prev:
                r += self.medkit_reward

            if infos[a]["just_died"]:
                r += self.death_penalty

            rewards[a] = r
            self.prev_health[a] = h_cur

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class RemedyRushRewardWrapper(ParallelEnv):
    """
    Reward shaping for remedy_rush. Health pickups give a positive reward;
    armor pickups give a negative reward (penalty).
    """

    def __init__(
        self,
        env: ParallelEnv,
        *,
        health_key: str = "HEALTH",
        armor_key: str = "ARMOR",
    ):
        self.env = env
        self.health_key = health_key
        self.armor_key = armor_key
        self.prev_health: Dict[str, Optional[float]] = {}
        self.prev_armor: Dict[str, Optional[float]] = {}
        self.possible_agents = env.possible_agents
        self.agents = env.agents

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def state_space(self):
        return self.env.state_space

    def state(self):
        return self.env.state()

    def state_observation(self, agent: str):
        return self.env.state_observation(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self.prev_health = {a: infos[a].get(self.health_key, 0.0) for a in self.agents}
        self.prev_armor = {a: infos[a].get(self.armor_key, 0.0) for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        for a in self.agents:
            reset_info = _reset_info(infos[a])
            if reset_info is not None:
                self.prev_health[a] = reset_info.get(
                    self.health_key, self.prev_health[a]
                )
                self.prev_armor[a] = reset_info.get(self.armor_key, self.prev_armor[a])

            r = rewards[a]
            h_cur = infos[a].get(self.health_key, 0.0)
            h_prev = self.prev_health[a]
            armor_cur = infos[a].get(self.armor_key, 0.0)
            armor_prev = self.prev_armor[a]

            r += h_cur - h_prev
            r -= armor_cur - armor_prev

            rewards[a] = r
            self.prev_health[a] = h_cur
            self.prev_armor[a] = armor_cur

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class PitfallRewardWrapper(ParallelEnv):
    """
    Dense shaping on POSITION_X (forward-only). Optional sparse goal on crossing goal_x.
    """

    def __init__(
        self,
        env: ParallelEnv,
        *,
        scaler: float = 0.01,
        death_penalty: float = -1.0,
        keep_lb: bool = True,
        goal_x: Optional[float] = None,
        goal_reward: float = 10.0,
        pos_key: str = "POSITION_X",
        dead_key: str = "DEAD",
        x_start: float = 32.0,
    ):
        self.env = env
        self.scaler = float(scaler)
        self.death_penalty = float(death_penalty)
        self.keep_lb = bool(keep_lb)
        self.goal_x = goal_x
        self.goal_reward = float(goal_reward)
        self.pos_key = pos_key
        self.dead_key = dead_key
        self.x_start = float(x_start)

        self._prev_x: Dict[str, Optional[float]] = {}
        self._best_x: Dict[str, float] = {}
        self._goal_given: Dict[str, bool] = {}

        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = env.possible_agents
        self.agents = env.agents

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def state_space(self):
        return self.env.state_space

    def state(self):
        return self.env.state()

    def state_observation(self, agent: str):
        return self.env.state_observation(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._prev_x = {a: infos[a].get(self.pos_key, None) for a in self.agents}
        self._best_x = {
            a: (self._prev_x[a] if self._prev_x[a] is not None else -np.inf)
            for a in self.agents
        }
        self._goal_given = {a: False for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, base_rewards, terminations, truncations, infos = self.env.step(actions)

        shaped: Dict[str, float] = {}
        for a in self.agents:
            reset_info = _reset_info(infos[a])
            if reset_info is not None:
                reset_x = reset_info.get(self.pos_key, None)
                self._prev_x[a] = reset_x
                self._best_x[a] = reset_x if reset_x is not None else self.x_start
                self._goal_given[a] = False

            r = float(base_rewards.get(a, 0.0))
            x_cur = infos[a].get(self.pos_key, None)
            x_prev = self._prev_x.get(a, None)

            if x_prev is not None and x_cur is not None:
                if self.keep_lb:
                    inc = max(0.0, x_cur - max(self._best_x[a], x_prev))
                    if inc > 0:
                        r += self.scaler * inc
                    self._best_x[a] = max(self._best_x[a], x_cur)
                else:
                    dx = max(0.0, x_cur - x_prev)
                    if dx > 0:
                        r += self.scaler * dx

            info_a = infos.get(a, {})
            just_died = bool(info_a.get("just_died", False))
            dead_now = bool(info_a.get(self.dead_key, 0))
            if just_died or (dead_now and not just_died):
                r += self.death_penalty
                self._best_x[a] = self.x_start  # reset best_x on death

            if self.goal_x is not None and not self._goal_given.get(a, False):
                if x_cur is not None and x_cur > float(self.goal_x):
                    r += self.goal_reward
                    self._goal_given[a] = True

            shaped[a] = r
            self._prev_x[a] = x_cur

        return obs, shaped, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
