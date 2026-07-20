from collections.abc import Mapping, Sequence

from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper

import vizdoom as vzd


Button = vzd.Button
Action = Mapping[Button, float]


def action(*buttons: Button, turn: float = 0.0) -> dict[Button, float]:
    values = {button: 1.0 for button in buttons}
    if turn:
        values[Button.TURN_LEFT_RIGHT_DELTA] = turn
    return values


class DiscreteActionWrapper(BaseParallelWrapper):
    """Use those as discrete actions available for simplification"""

    def __init__(self, env, actions: Sequence[Action]):
        super().__init__(env)
        if not actions:
            raise ValueError("actions must not be empty")

        available_buttons = tuple(self.unwrapped.available_buttons)
        available_set = set(available_buttons)
        compiled_actions = []
        for action in actions:
            unavailable = [button for button in action if button not in available_set]
            if unavailable:
                names = ", ".join(button.name for button in unavailable)
                verb = "is" if len(unavailable) == 1 else "are"
                raise ValueError(f"{names} {verb} unavailable for this scenario")
            compiled_actions.append(
                tuple(float(action.get(button, 0.0)) for button in available_buttons)
            )

        self._actions = tuple(compiled_actions)
        self._action_space = spaces.Discrete(len(self._actions))

    def action_space(self, agent):
        return self._action_space

    def step(self, actions):
        mapped = {}
        for agent, action in actions.items():
            if not self._action_space.contains(action):
                raise ValueError(f"Invalid discrete action for {agent}: {action!r}")
            mapped[agent] = list(self._actions[int(action)])
        return self.env.step(mapped)


SSL2_ACTIONS = (
    action(),
    action(Button.MOVE_FORWARD),
    action(Button.MOVE_BACKWARD),
    action(Button.MOVE_RIGHT),
    action(Button.MOVE_LEFT),
    action(turn=-3.125),
    action(turn=3.125),
    action(Button.ATTACK),
    action(Button.ATTACK, turn=-3.125),
    action(Button.ATTACK, turn=3.125),
    action(Button.ATTACK, Button.MOVE_FORWARD),
    action(Button.ATTACK, Button.MOVE_BACKWARD),
    action(Button.ATTACK, Button.MOVE_RIGHT),
    action(Button.ATTACK, Button.MOVE_LEFT),
    action(Button.MOVE_FORWARD, Button.SPEED),
    action(Button.USE),
)

MULTI_DUEL_ACTIONS = (
    action(),
    action(Button.MOVE_LEFT),
    action(Button.MOVE_RIGHT),
    action(Button.ATTACK),
    action(Button.MOVE_LEFT, Button.ATTACK),
    action(Button.MOVE_RIGHT, Button.ATTACK),
)

ACTION_SETS = {
    "multi_duel": MULTI_DUEL_ACTIONS,
    "ssl2": SSL2_ACTIONS,
}
