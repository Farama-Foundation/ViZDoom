"""
Scenario-agnostic machinery for rule-based oracle agents.

An oracle reads engine ground truth (medkit and player positions, exact health)
instead of pixels. Oracles exist to determine how well a scenario can be played
(the ceiling a learned policy is measured against).

This module holds everything that does not depend on the scenario: reading the
privileged state, mapping the movement buttons, steering towards a point,
getting unstuck, and the per-agent bookkeeping. A scenario oracle subclasses
`BaseOracle` and implements `decide()`, which says where each agent should go
and whether it should stop short of the target.

Every oracle supports the two modes a social dilemma needs:

  "defector"     - every agent acts in the most greedy manner possible.
  "collaborator" - every agent acts in the most collaborative manner possible.

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

import vizdoom as vzd
from vizdoom.pettingzoo_wrapper.base_pettingzoo_env import PRIVILEGED_INFO_KEY


ORACLE_MODES = ("collaborator", "defector")

# Measured engine rates, per tic.
HEALTH_DRAIN_PER_TIC = 0.25
MOVE_UNITS_PER_TIC = 2.84
TURN_DEG_PER_TIC = 2.42


@dataclass
class OracleConfig:
    """
    Tuning shared by every oracle. Distances are Doom map units and angles
    degrees. The per step rates are derived from the env's skip_frames.
    """

    skip_frames: int = 4
    max_health: float = 100.0
    # Steering
    turn_deadzone_deg: float = 6.0
    forward_cone_deg: float = 60.0
    # Keep the current target unless another is this much cheaper, so two
    # agents do not swap targets back and forth every step.
    switch_hysteresis: float = 0.8
    # Stuck detection and escape
    stuck_window: int = 6
    stuck_eps: float = 8.0
    escape_steps: int = 6

    @property
    def drain_per_step(self) -> float:
        return HEALTH_DRAIN_PER_TIC * self.skip_frames

    @property
    def move_per_step(self) -> float:
        return MOVE_UNITS_PER_TIC * self.skip_frames

    @property
    def turn_per_step(self) -> float:
        return TURN_DEG_PER_TIC * self.skip_frames


@dataclass
class OracleDecision:
    """
    What an oracle chose for one agent this step. Scenario oracles fill in the
    target and the flags. The base class turns them into buttons, and the
    runner reports them.
    """

    agent: str
    health: float
    target: tuple | None = None  # (x, y) to walk to
    distance: float = float("inf")
    eta_steps: float = float("inf")
    steps_to_live: float = float("inf")
    # Stop short of the target instead of walking onto it.
    holding: bool = False
    # Gave way to another agent's claim.
    deferred: bool = False
    escaping: bool = False
    dead: bool = False


@dataclass
class AgentMemory:
    """Per-agent state the controller keeps between steps."""

    positions: deque = field(default_factory=lambda: deque(maxlen=16))
    escape_until: int = -1
    escape_left: bool = True
    target_id: int | None = None


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def find_env_attr(env, name: str):
    """
    Look up an attribute down the wrapper chain. The reward wrappers are plain
    ParallelEnvs without attribute forwarding, so getattr on the outermost env
    is not enough.
    """
    seen = set()
    while env is not None and id(env) not in seen:
        seen.add(id(env))
        value = getattr(env, name, None)
        if value is not None:
            return value
        env = getattr(env, "env", None)
    raise AttributeError(f"no {name!r} anywhere in the env wrapper chain")


class BaseOracle(ABC):
    """
    Privileged rule-based policy. Call `reset()` at episode start, then
    `act(infos)` with the infos dict the env returned.

    Subclasses implement `decide(states)`, returning one `OracleDecision` per
    agent. Everything below that (steering, waiting, getting unstuck) is
    handled here.
    """

    def __init__(
        self,
        env=None,
        *,
        mode: str = "collaborator",
        config: OracleConfig | None = None,
        buttons: list[Any] | None = None,
        agents: list[str] | None = None,
        discrete: bool | None = None,
    ) -> None:
        if mode not in ORACLE_MODES:
            raise ValueError(f"mode must be one of {ORACLE_MODES}, got {mode!r}")
        if env is None and (buttons is None or agents is None):
            raise ValueError("pass an env, or both buttons and agents")
        self.mode = mode
        self.config = config or OracleConfig()
        if agents is None:
            assert env is not None
            agents = env.possible_agents
        self.possible_agents = list(agents)

        buttons = list(
            buttons if buttons is not None else find_env_attr(env, "available_buttons")
        )
        self._act_len = len(buttons)
        try:
            self._turn_left = buttons.index(vzd.Button.TURN_LEFT)
            self._turn_right = buttons.index(vzd.Button.TURN_RIGHT)
            self._forward = buttons.index(vzd.Button.MOVE_FORWARD)
        except ValueError as exc:
            raise ValueError(
                f"{type(self).__name__} needs TURN_LEFT, TURN_RIGHT and "
                f"MOVE_FORWARD in the scenario buttons, got {buttons}"
            ) from exc

        # Binary-only scenarios default to a Discrete button-combination action
        # space (see base_env_common.encode_env_action), so the button vector is
        # packed into its bitmask index. continuous_actions=True keeps the vector.
        if discrete is None:
            if env is not None:
                discrete = isinstance(
                    env.action_space(self.possible_agents[0]), spaces.Discrete
                )
            else:
                discrete = not any(vzd.is_delta_button(b) for b in buttons)
        self.discrete = bool(discrete)

        self._memory: dict[str, AgentMemory] = {}
        self._step = 0
        self.last_decisions: dict[str, OracleDecision] = {}
        self.reset()

    # ------------------------------------------------------------------ api

    def reset(self) -> None:
        self._memory = {agent: AgentMemory() for agent in self.possible_agents}
        self._step = 0
        self.last_decisions = {}

    def act(self, infos: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Map the env's infos onto one action (button vector or bitmask) per agent."""
        self._step += 1
        states = self.read_states(infos)
        decisions = self.decide(states)
        self.last_decisions = decisions

        actions: dict[str, np.ndarray] = {}
        for agent in infos:
            state = states.get(agent)
            decision = decisions.get(agent)
            if state is None or decision is None or decision.dead:
                actions[agent] = self.noop()
                continue
            actions[agent] = self.control(state, decision)
        if self.discrete:
            return {
                agent: self.to_discrete(action) for agent, action in actions.items()
            }
        return actions

    @staticmethod
    def to_discrete(action: np.ndarray) -> int:
        """Pack a binary button vector into the env's Discrete bitmask index."""
        return int(
            sum(1 << index for index, value in enumerate(action) if value >= 0.5)
        )

    @abstractmethod
    def decide(self, states: dict[str, dict]) -> dict[str, OracleDecision]:
        """Choose a target (and whether to hold short of it) for every agent."""

    # --------------------------------------------------------------- state

    def read_states(self, infos: dict[str, dict[str, Any]]) -> dict[str, dict]:
        states: dict[str, dict] = {}
        for agent, info in infos.items():
            privileged = (
                info.get(PRIVILEGED_INFO_KEY) if isinstance(info, dict) else None
            )
            if privileged is not None:
                states[agent] = privileged
        if not states:
            raise KeyError(
                f"No '{PRIVILEGED_INFO_KEY}' in infos. Build the env with "
                "privileged_info=True to use an oracle."
            )
        return states

    def memory(self, agent: str) -> AgentMemory:
        return self._memory.setdefault(agent, AgentMemory())

    def objects_named(self, states: dict[str, dict], names) -> dict[int, dict]:
        """Union of the objects the engine reports to any agent, by object id."""
        found: dict[int, dict] = {}
        for state in states.values():
            for obj in state.get("objects", ()):
                if obj["name"] in names:
                    found[obj["id"]] = obj
        return found

    # -------------------------------------------------------------- travel

    def bearing_to(self, state: dict, x: float, y: float) -> float:
        return math.degrees(math.atan2(y - state["y"], x - state["x"]))

    def distance_to(self, state: dict, x: float, y: float) -> float:
        return math.hypot(x - state["x"], y - state["y"])

    def eta_steps(self, state: dict, x: float, y: float) -> float:
        """Rough number of env steps to walk somewhere, turning included."""
        cfg = self.config
        turn = abs(wrap_deg(self.bearing_to(state, x, y) - state["angle"]))
        return self.distance_to(state, x, y) / cfg.move_per_step + (
            turn / cfg.turn_per_step
        )

    def arrival_health(
        self, health: float, eta_steps: float, safety: float = 1.0
    ) -> float:
        """Health left on arrival, the quantity restraint has to protect."""
        return health - safety * eta_steps * self.config.drain_per_step

    # ------------------------------------------------------------- control

    def noop(self) -> np.ndarray:
        return np.zeros((self._act_len,), dtype=np.float32)

    def buttons(
        self, *, turn_left: bool, turn_right: bool, forward: bool
    ) -> np.ndarray:
        action = self.noop()
        action[self._turn_left] = float(turn_left)
        action[self._turn_right] = float(turn_right)
        action[self._forward] = float(forward)
        return action

    def steer(
        self, state: dict, desired_deg: float, *, forward: bool = True
    ) -> np.ndarray:
        """Turn towards a heading, walking while roughly facing it."""
        cfg = self.config
        heading_error = wrap_deg(desired_deg - state["angle"])
        return self.buttons(
            turn_left=heading_error > cfg.turn_deadzone_deg,
            turn_right=heading_error < -cfg.turn_deadzone_deg,
            forward=forward and abs(heading_error) < cfg.forward_cone_deg,
        )

    def control(self, state: dict, decision: OracleDecision) -> np.ndarray:
        """
        Turn a decision into buttons: hold station, escape a wall, or walk to
        the target. `desired_heading` lets a scenario bend the path.
        """
        memory = self.memory(decision.agent)
        target = decision.target

        waiting = target is not None and decision.holding and self.should_wait(decision)
        if waiting:
            # Standing still on purpose: keep the stuck detector out of it,
            # otherwise its escape manoeuvre would walk us onto the target.
            memory.positions.clear()
            return self.hold_station(state, decision)

        memory.positions.append((state["x"], state["y"]))
        if self.is_stuck(memory):
            memory.escape_until = self._step + self.config.escape_steps
            memory.escape_left = not memory.escape_left
            memory.positions.clear()
        if self._step <= memory.escape_until:
            decision.escaping = True
            return self.buttons(
                turn_left=memory.escape_left,
                turn_right=not memory.escape_left,
                forward=True,
            )

        if target is None:
            return self.idle(state, decision)
        return self.steer(state, self.desired_heading(state, decision))

    def desired_heading(self, state: dict, decision: OracleDecision) -> float:
        """Heading to the target. Scenarios override this to avoid obstacles."""
        target = decision.target
        assert target is not None
        return self.bearing_to(state, target[0], target[1])

    def should_wait(self, decision: OracleDecision) -> bool:
        """Whether a holding agent is close enough to stop walking."""
        return True

    def hold_station(self, state: dict, decision: OracleDecision) -> np.ndarray:
        """Default holding behaviour: face the target and take no step."""
        target = decision.target
        assert target is not None
        return self.steer(
            state, self.bearing_to(state, target[0], target[1]), forward=False
        )

    def idle(self, state: dict, decision: OracleDecision) -> np.ndarray:
        """No target: sweep the room so we are already moving when one appears."""
        return self.buttons(turn_left=True, turn_right=False, forward=True)

    def is_stuck(self, memory: AgentMemory) -> bool:
        cfg = self.config
        if len(memory.positions) < cfg.stuck_window:
            return False
        recent = list(memory.positions)[-cfg.stuck_window :]
        x0, y0 = recent[0]
        return all(math.hypot(x - x0, y - y0) < cfg.stuck_eps for x, y in recent[1:])


def summarize_decisions(decisions: dict[str, OracleDecision]) -> str:
    """One line of per-agent status, for --log-every and debugging."""
    parts: list[str] = []
    for agent in sorted(decisions):
        decision = decisions[agent]
        if decision.dead:
            state = "dead"
        elif decision.escaping:
            state = "escape"
        elif decision.holding:
            state = "hold"
        elif decision.deferred:
            state = "defer"
        else:
            state = "go"
        parts.append(f"{agent}:{decision.health:.0f}hp/{state}")
    return " ".join(parts)


class SchellingOracleAdapter:
    """
    Per-agent wrapper around an oracle, matching the policy interface that
    scripts/schelling_diagram.py expects (`act(agent, obs, info, rng)`).

    An oracle decides for all agents at once, so this caches the latest info of
    every agent and recomputes on each call. Agents not yet queried this step
    contribute their previous info, which is one step stale and does not matter
    at this control rate.

    The env must be built with `privileged_info=True`.
    """

    def __init__(self, oracle: BaseOracle) -> None:
        self.oracle = oracle
        self._infos: dict[str, dict[str, Any]] = {}

    def reset(self, agents=None, rng=None) -> None:
        self.oracle.reset()
        self._infos = {}

    def act(self, agent: str, obs: Any, info: dict[str, Any], rng=None) -> Any:
        self._infos[agent] = info
        return self.oracle.act(self._infos)[agent]


def build_schelling_adapter(
    oracle_cls, config_cls, mode: str, **context
) -> SchellingOracleAdapter:
    """Factory helper shared by the scenario oracles' cooperator/defector hooks."""
    agents = list(
        context.get("agents")
        or [f"agent_{i}" for i in range(int(context.get("num_agents", 2)))]
    )
    config = config_cls(skip_frames=int(context.get("skip_frames", 4)))
    continuous = context.get("continuous_actions")
    return SchellingOracleAdapter(
        oracle_cls(
            mode=mode,
            config=config,
            buttons=list(context["buttons"]),
            agents=agents,
            discrete=None if continuous is None else not continuous,
        )
    )
