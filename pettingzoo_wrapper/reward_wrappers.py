from typing import Any, Dict, Optional

import numpy as np
from pettingzoo import ParallelEnv


def _reset_info(info):
    reset_info = info.get("reset_info")
    if isinstance(reset_info, dict):
        return reset_info
    return None


_DEATHMATCH_DELTA_REWARDS = {
    "FRAGCOUNT": 1.0,
    "DEATHCOUNT": -1.0,
    "DAMAGECOUNT": 0.01,
    "DAMAGE_TAKEN": -0.01,
}


class DeathmatchRewardWrapper(ParallelEnv):
    def __init__(self, env: ParallelEnv):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.prev_vars: Dict[str, Dict[str, float]] = {}
        self._counters_checked = False

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
        self.prev_vars = {agent: {} for agent in self.agents}
        self.check_ctr(infos)
        return obs, infos

    def check_ctr(self, infos: Dict[str, Any]) -> None:
        if self._counters_checked:
            return
        for agent in self.agents:
            info = infos.get(agent)
            if not isinstance(info, dict):
                return
            present = [n for n in _DEATHMATCH_DELTA_REWARDS if n in info]
            missing = [n for n in _DEATHMATCH_DELTA_REWARDS if n not in info]
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
            for name, weight in _DEATHMATCH_DELTA_REWARDS.items():
                if name not in previous or name not in info:
                    continue
                delta = info[name] - previous[name]
                if delta > 1e-8:
                    shaping_reward += delta * weight

            rewards[agent] = float(rewards.get(agent, 0.0)) + shaping_reward
            # Carry last known value forward for anything absent this step
            self.prev_vars[agent] = {
                name: info[name] if name in info else self.prev_vars[agent][name]
                for name in _DEATHMATCH_DELTA_REWARDS
                if name in info or name in self.prev_vars[agent]
            }

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class HideAndSeekRewardWrapper(ParallelEnv):
    roles = {"agent_0": "shooter", "agent_1": "hider"}
    role_codes = {"agent_0": 0.0, "agent_1": 1.0}
    outcome_codes = {
        "ongoing": 0.0,
        "shooter_win": 1.0,
        "hider_win": 2.0,
        "hider_escape": 3.0,
        "draw": 4.0,
    }

    def __init__(self, env: ParallelEnv, *, win_reward: float = 1.0):
        self.env = env
        self.win_reward = float(win_reward)
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self._previous_deaths: Dict[str, float] = {}

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

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._validate_and_annotate(infos)
        self._previous_deaths = {
            agent: float(infos[agent]["DEATHCOUNT"]) for agent in self.agents
        }
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, _, terminations, truncations, infos = self.env.step(actions)
        self._validate_and_annotate(infos)

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
        rewards = {agent: 0.0 for agent in self.agents}

        if shooter_died and hider_died:
            outcome = "draw"
        elif hider_died:
            outcome = "shooter_win"
            rewards = {"agent_0": self.win_reward, "agent_1": -self.win_reward}
        elif shooter_died:
            outcome = "hider_win"
            rewards = {"agent_0": -self.win_reward, "agent_1": self.win_reward}
        elif timed_out:
            outcome = "hider_escape"
            rewards = {"agent_0": -self.win_reward, "agent_1": self.win_reward}
        else:
            outcome = "ongoing"

        if shooter_died or hider_died:
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
