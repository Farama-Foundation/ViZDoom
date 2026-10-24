"""
Oracle🔮 for the health gathering scenario.

The oracle has two modes, guided by game-theoretic perspectives:

  "defector"     - every agent walks to its own nearest medkit and takes it on
                   sight, at any health.
  "collaborator" - an agent only obtains a kit once its health has room for
                   the full heal (`consume_health`, 50 by default). Until then,
                   it waits beside the kit. Kits are allocated to whomever
                   runs out of health first. However, this claim is ignored
                   when the owner of this "claim" cannot reach the kit alive.

Waiting is done at arm's length. An agent comfortably parks itself at around
150 units from the selected kit. Note that kits might spawn right in front or
on top of the agent, making a pick-up unavoidable. In these rare circumstances,
even the oracles might have to succumb to the acid floor.

The map is designed to provide a surplus of medkits. For instance, 48 kits will
spawn throughout a 500-step episode with 2 agents. At the wad's current 50
health per kit that is 2400 health on offer, against a collective drain of 1000
health points for the two agents. With reasonable navigation and teamwork,
death should not be an option. In the defector mode, it is a certainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vizdoom.pettingzoo_wrapper.oracle_base import (
    BaseOracle,
    OracleConfig,
    OracleDecision,
    build_schelling_adapter,
    wrap_deg,
)
from vizdoom.pettingzoo_wrapper.utils import HEALTH_ITEM_NAMES


@dataclass
class HealthGatheringOracleConfig(OracleConfig):
    """Kit-specific tuning on top of the shared steering config."""

    # How much a medkit heals.
    kit_value: float = 50.0
    # The amount of health from which it worth to obtain a medkit for a
    # collaborator agent. Note that health caps at max_health, so consuming
    # a kit above max_health-kit_value throws the surplus away.
    consume_health: float | None = None
    # Restraint is dropped when the health left on arrival at the kit would
    # fall below this reserve. ETA is inflated by eta_safety first, since the
    # straight-line estimate ignores detours.
    survival_reserve_health: float = 25.0
    eta_safety: float = 1.3
    # Where to wait beside a kit that is not yet worth taking, and how close is
    # too close to stand still.
    hold_radius: float = 150.0
    min_hold_distance: float = 90.0
    # Doom picks items up on contact, so an agent that must not take a kit has
    # to steer around it: how far away to care, how wide to pass, and how far
    # past the target to keep clearing.
    avoid_radius: float = 250.0
    avoid_clearance: float = 64.0
    avoid_lookahead: float = 60.0
    # Detours are capped so stacked pushes in a kit-dense corner cannot turn
    # the agent away from its own target.
    avoid_max_push_deg: float = 60.0

    def __post_init__(self) -> None:
        if self.consume_health is None:
            self.consume_health = self.max_health - self.kit_value

    # Assignment: how many kits per agent enter the search, and the agent count
    # above which the exhaustive search gives way to a greedy pass.
    assignment_candidates: int = 4
    max_exact_agents: int = 4
    # Arriving dead is not an option. Cost for arriving below the reserve per
    # missing health point.
    death_cost: float = 10_000.0
    # Being left without a kit costs this much per point of missing health.
    urgency_weight: float = 4.0
    # Going without a kit is a last resort. It is only ever chosen when there
    # are fewer kits than agents. The emptier the health bar, the worse it
    # is to be the one left out.
    no_kit_cost: float = 1_000.0
    idle_penalty_per_hp: float = 0.5
    # Discount for keeping the kit we are already walking to.
    assignment_stickiness: float = 6.0


@dataclass
class KitDecision(OracleDecision):
    """An `OracleDecision` that also names the kit involved."""

    target_id: int | None = None
    # Kits this agent must walk around rather than consume.
    avoid_ids: tuple = ()
    # Too low on health to spend any of it detouring around kits.
    urgent: bool = False


class HealthGatheringOracle(BaseOracle):
    """Privileged medkit-commons policy. See the module docstring for modes."""

    def __init__(self, env=None, **kwargs) -> None:
        kwargs.setdefault("config", HealthGatheringOracleConfig())
        # Kits seen this step, cached by read_states for the control hooks.
        self._kits: dict[int, dict] = {}
        super().__init__(env, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._kits = {}

    # ---------------------------------------------------------- assignment

    def decide(self, states: dict[str, dict]) -> dict[str, KitDecision]:
        cfg = self.config
        kits = self.objects_named(states, HEALTH_ITEM_NAMES)
        decisions = {
            agent: KitDecision(
                agent=agent,
                health=state["health"],
                dead=state["health"] <= 0.0,
                steps_to_live=state["health"] / cfg.drain_per_step,
            )
            for agent, state in states.items()
        }

        live = [agent for agent, decision in decisions.items() if not decision.dead]
        if not kits or not live:
            return decisions

        if self.mode == "defector":
            for agent in live:
                memory = self.memory(agent)
                kit = self._best_kit(
                    states[agent], list(kits.values()), memory.target_id
                )
                self._take(decisions[agent], states[agent], kit)
                memory.target_id = kit["id"]
            return decisions

        self._allocate(states, kits, decisions, live)
        return decisions

    def _allocate(
        self,
        states: dict[str, dict],
        kits: dict[int, dict],
        decisions: dict[str, KitDecision],
        live: list[str],
    ) -> None:
        """
        Collaborator allocation: one kit per agent, chosen so that nobody runs
        out of health on the way. Assigning greedily agent by agent is what
        makes two agents race for the same kit.
        """
        cfg = self.config
        assignment = self._assign_kits(states, decisions, kits, live)

        for agent in live:
            state = states[agent]
            decision = decisions[agent]
            memory = self.memory(agent)
            kit = assignment.get(agent)
            if kit is None:
                # No kit of our own: go wait where the next one will spawn
                # rather than shadowing someone else's.
                decision.deferred = True
                memory.target_id = None
                continue

            self._take(decision, state, kit)
            nearest = self._best_kit(state, list(kits.values()), memory.target_id)
            decision.deferred = kit["id"] != nearest["id"]
            starving = (
                self._arrival_health(state, decision.health, kit)
                < cfg.survival_reserve_health
            )
            decision.urgent = starving
            # Wait until the kit's full heal fits in our health bar, unless
            # waiting would cost a life.
            decision.holding = decision.health > cfg.consume_health and not starving
            memory.target_id = kit["id"]

        # Every kit assigned to somebody else must be walked around, not over.
        # An agent too healthy to use a kit walks around all of them, since
        # the surplus heal would be lost.
        assigned = {
            kit["id"]: agent for agent, kit in assignment.items() if kit is not None
        }
        for agent in live:
            decision = decisions[agent]
            avoid = {kit_id for kit_id, owner in assigned.items() if owner != agent}
            if decision.holding:
                avoid |= {kit_id for kit_id in kits if kit_id != decision.target_id}
            decision.avoid_ids = tuple(avoid)

    # ------------------------------------------------------------ matching

    def _assign_kits(
        self,
        states: dict[str, dict],
        decisions: dict[str, KitDecision],
        kits: dict[int, dict],
        live: list[str],
    ) -> dict[str, dict | None]:
        """
        One kit per agent, at most. Solved exactly while the problem is small
        (the usual 2-4 agents), greedily above that.
        """
        cfg = self.config
        candidates: dict[str, list[dict]] = {}
        for agent in live:
            state = states[agent]
            ranked = sorted(
                kits.values(), key=lambda kit: self.eta_steps(state, kit["x"], kit["y"])
            )
            candidates[agent] = ranked[: cfg.assignment_candidates]

        pool: list[dict] = []
        seen: set = set()
        for agent in live:
            for kit in candidates[agent]:
                if kit["id"] not in seen:
                    seen.add(kit["id"])
                    pool.append(kit)

        if len(live) <= cfg.max_exact_agents:
            return self._exact_assignment(states, decisions, live, pool)
        return self._greedy_assignment(states, decisions, live, pool)

    def _exact_assignment(
        self,
        states: dict[str, dict],
        decisions: dict[str, KitDecision],
        live: list[str],
        pool: list[dict],
    ) -> dict[str, dict | None]:
        """Cheapest injective agent -> kit matching, searched exhaustively."""
        order = sorted(live, key=lambda agent: decisions[agent].steps_to_live)
        best: dict[str, dict | None] = {}
        best_cost = float("inf")

        def search(index: int, used: frozenset, cost: float, chosen: dict) -> None:
            nonlocal best, best_cost
            if cost >= best_cost:
                return
            if index == len(order):
                best_cost = cost
                best = dict(chosen)
                return
            agent = order[index]
            for kit in pool:
                if kit["id"] in used:
                    continue
                chosen[agent] = kit
                search(
                    index + 1,
                    used | {kit["id"]},
                    cost
                    + self._match_cost(states[agent], decisions[agent], agent, kit),
                    chosen,
                )
            chosen[agent] = None
            search(
                index + 1,
                used,
                cost + self._unassigned_cost(decisions[agent]),
                chosen,
            )
            chosen.pop(agent, None)

        search(0, frozenset(), 0.0, {})
        return best

    def _greedy_assignment(
        self,
        states: dict[str, dict],
        decisions: dict[str, KitDecision],
        live: list[str],
        pool: list[dict],
    ) -> dict[str, dict | None]:
        """Need-ordered fallback for agent counts too large to search."""
        assignment: dict[str, dict | None] = {}
        taken: set = set()
        for agent in sorted(live, key=lambda a: decisions[a].steps_to_live):
            free = [kit for kit in pool if kit["id"] not in taken]
            if not free:
                assignment[agent] = None
                continue
            kit = min(
                free,
                key=lambda kit: self._match_cost(
                    states[agent], decisions[agent], agent, kit
                ),
            )
            assignment[agent] = kit
            taken.add(kit["id"])
        return assignment

    def _match_cost(
        self, state: dict, decision: KitDecision, agent: str, kit: dict
    ) -> float:
        """
        Cost of sending one agent to one kit, in steps. Arriving dead is
        effectively forbidden, arriving on fumes is discouraged, and keeping
        the kit we are already walking to gets a discount so two agents do not
        trade targets every step.
        """
        cfg = self.config
        eta = self.eta_steps(state, kit["x"], kit["y"])
        # Feasibility is judged on the plain estimate: padding it with
        # eta_safety here would write off trips the agent can actually make.
        arrival = self.arrival_health(decision.health, eta)
        cost = eta
        if arrival <= 0.0:
            cost += cfg.death_cost
        elif arrival < cfg.survival_reserve_health:
            cost += (cfg.survival_reserve_health - arrival) * cfg.urgency_weight
        if self.memory(agent).target_id == kit["id"]:
            cost -= cfg.assignment_stickiness
        return cost

    def _unassigned_cost(self, decision: KitDecision) -> float:
        """Cost of leaving an agent without a kit, worse the emptier its bar."""
        cfg = self.config
        if decision.health <= cfg.survival_reserve_health:
            # Walking to a kit that may be out of reach still beats standing
            # around: abandoning has to cost more than any attempt, or a dying
            # agent is left with no target at all.
            return cfg.death_cost * 100.0
        return cfg.no_kit_cost + (
            (cfg.max_health - decision.health) * cfg.idle_penalty_per_hp
        )

    def _arrival_health(self, state: dict, health: float, kit: dict) -> float:
        return self.arrival_health(
            health,
            self.eta_steps(state, kit["x"], kit["y"]),
            safety=self.config.eta_safety,
        )

    def _best_kit(
        self, state: dict, candidates: list[dict], current_id: int | None
    ) -> dict:
        """Cheapest kit to reach, with hysteresis so targets do not oscillate."""

        def eta(kit):
            return self.eta_steps(state, kit["x"], kit["y"])

        best = min(candidates, key=eta)
        if current_id is None or current_id == best["id"]:
            return best
        current = next((kit for kit in candidates if kit["id"] == current_id), None)
        if current is None:
            return best
        keep = eta(best) >= self.config.switch_hysteresis * eta(current)
        return current if keep else best

    def _take(self, decision: KitDecision, state: dict, kit: dict) -> None:
        decision.target_id = kit["id"]
        decision.target = (kit["x"], kit["y"])
        decision.distance = self.distance_to(state, kit["x"], kit["y"])
        decision.eta_steps = self.eta_steps(state, kit["x"], kit["y"])

    # ------------------------------------------------------------- control

    def should_wait(self, decision: KitDecision) -> bool:
        return decision.distance <= self.config.hold_radius

    def hold_station(self, state: dict, decision: KitDecision) -> np.ndarray:
        target = decision.target
        assert target is not None
        bearing = self.bearing_to(state, target[0], target[1])
        if decision.distance < self.config.min_hold_distance:
            # Too close to sit still (we drifted in, or a kit spawned on top of
            # us): walk back out of pickup range.
            return self.steer(state, bearing + 180.0)
        return self.steer(state, bearing, forward=False)

    def idle(self, state: dict, decision: KitDecision) -> np.ndarray:
        """
        Nothing to claim: head for the middle of the remaining kits so the next
        spawn is close, instead of standing around or circling.
        """
        cfg = self.config
        kits = self._kits
        if not kits:
            return super().idle(state, decision)
        target_x = sum(kit["x"] for kit in kits.values()) / len(kits)
        target_y = sum(kit["y"] for kit in kits.values()) / len(kits)
        if self.distance_to(state, target_x, target_y) < cfg.hold_radius:
            return self.buttons(turn_left=True, turn_right=False, forward=False)
        return self.steer(state, self.bearing_to(state, target_x, target_y))

    def desired_heading(self, state: dict, decision: KitDecision) -> float:
        """Bend the heading away from kits this agent must not consume."""
        cfg = self.config
        kits = self._kits
        assert decision.target is not None
        straight = self.bearing_to(state, decision.target[0], decision.target[1])
        desired = straight
        if decision.urgent:
            # Low on health: take the direct line and accept the odd stray
            # pickup, dying to protect a kit helps nobody.
            return straight
        for kit_id in decision.avoid_ids:
            kit = kits.get(kit_id)
            if kit is None:
                continue
            distance = self.distance_to(state, kit["x"], kit["y"])
            if distance > cfg.avoid_radius:
                continue
            offset = wrap_deg(self.bearing_to(state, kit["x"], kit["y"]) - desired)
            offset_rad = math.radians(offset)
            along = distance * math.cos(offset_rad)
            cross = abs(distance * math.sin(offset_rad))
            behind = along <= 0.0
            past_target = along > decision.distance + cfg.avoid_lookahead
            if behind or past_target or cross >= cfg.avoid_clearance:
                continue
            # Aim wide enough that the kit passes outside pickup range.
            needed = math.degrees(
                math.asin(min(1.0, cfg.avoid_clearance / max(distance, 1e-6)))
            )
            push = needed - abs(offset)
            if push > 0.0:
                desired -= math.copysign(push, offset if offset != 0.0 else 1.0)
        detour = wrap_deg(desired - straight)
        if abs(detour) > cfg.avoid_max_push_deg:
            desired = straight + math.copysign(cfg.avoid_max_push_deg, detour)
        return desired

    def read_states(self, infos):
        states = super().read_states(infos)
        # Cache the kits so the control hooks can see them without re-deriving.
        self._kits = self.objects_named(states, HEALTH_ITEM_NAMES)
        return states


def make_collaborator(**context):
    """Schelling-diagram factory: the need-priority (collaborating) oracle."""
    return build_schelling_adapter(
        HealthGatheringOracle, HealthGatheringOracleConfig, "collaborator", **context
    )


def make_defector(**context):
    """Schelling-diagram factory: the greedy (defecting) oracle."""
    return build_schelling_adapter(
        HealthGatheringOracle, HealthGatheringOracleConfig, "defector", **context
    )
