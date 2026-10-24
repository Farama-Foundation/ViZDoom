"""ViZDoom chat actions and history (bounded to each agent) for communication scenario."""

import re

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper


MOTOR_BUTTONS = (
    None,
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "USE",
)
TEXT_LENGTH = 256
DEFAULT_ALPHABET_SIZE = 8
# Policy features
FEATURE_NAMES = (
    "role_informer",
    "role_operator",
    "send_available",
    "send_cooldown",
)
FEATURE_LENGTH = len(FEATURE_NAMES)
FEATURE_ROLE_INFORMER = FEATURE_NAMES.index("role_informer")
FEATURE_ROLE_OPERATOR = FEATURE_NAMES.index("role_operator")
FEATURE_SEND_AVAILABLE = FEATURE_NAMES.index("send_available")
# For evaluation only
DELIVERY_MODES = ("natural", "silence", "swap", "random")


def make_chat_tokens(alphabet_size=DEFAULT_ALPHABET_SIZE):
    """Construct a bounded anonymous alphabet. Silence is an extra action."""
    if (
        isinstance(alphabet_size, bool)
        or not isinstance(alphabet_size, int)
        or not 1 <= alphabet_size <= 64
    ):
        raise ValueError("alphabet_size must be an integer between 1 and 64")
    return ("", *(f"S{i:02d}" for i in range(alphabet_size)))


def validate_chat_tokens(tokens):
    if isinstance(tokens, str):
        raise ValueError("chat_tokens must be a sequence, not a string")
    tokens = tuple(tokens)
    if (
        not 2 <= len(tokens) <= 65
        or tokens[0] != ""
        or any(
            not isinstance(token, str)
            or re.fullmatch(r"[A-Z0-9_]{1,16}", token) is None
            for token in tokens[1:]
        )
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("chat_tokens require silence then 1-64 unique safe tokens")
    return tokens


def clean_notifications(text):
    text = re.sub(r"\x1c(\[[^\]]*\]|.)", "", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip("\x1d\x1e\x1f \r")
        line = "".join(char for char in line if 32 <= ord(char) < 127)
        if line and not set(line) <= set("<->"):
            lines.append(line)
    return lines


class NotificationHistory:
    """Merge overlapping engine snapshots, retaining only local console text."""

    def __init__(self, length=TEXT_LENGTH):
        self.length = length
        self.previous = []
        self.text = ""

    def update(self, snapshot):
        lines = clean_notifications(snapshot)
        overlap = min(len(self.previous), len(lines))
        while overlap and self.previous[-overlap:] != lines[:overlap]:
            overlap -= 1
        new = lines[overlap:]
        if new:
            self.text = "\n".join(filter(None, (self.text, "\n".join(new))))[
                -self.length :
            ]
        self.previous = lines
        return new

    def encode(self):
        result = np.zeros(self.length, dtype=np.uint8)
        encoded = self.text.encode("ascii")
        result[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        return result


class CommunicationWrapper(BaseParallelWrapper):
    """
    Factorized actions [motor, message]. Zero/zero is idle silence.
    Movement and messaging are separate heads so policy sums 2 log-probs.
    Messages use engine `say` command, with 35 tics send cooldown. Movement can execute during chatting.

    `delivery` is an evaluation intervention on what the engine says for an accepted send:
        + natural (the chosen token)
        + silence (nothing)
        + swap (symbol order reversed, S00<->S01 for two symbols)
        + random (a uniform non-silent token from a delivery_seed stream)
    `observer_info` adds observer-only infos (secret, slot layout, first intended/delivered message). These are not inputs for policy.
    """

    def __init__(
        self,
        env,
        *,
        communication=True,
        chat_cooldown_tics=35,
        chat_tokens=None,
        delivery="natural",
        delivery_seed=None,
        observer_info=False,
    ):
        super().__init__(env)
        if env.possible_agents != ["agent_0", "agent_1"]:
            raise ValueError("Communication requires exactly two agents")
        if chat_cooldown_tics < 0:
            raise ValueError("chat_cooldown_tics must be nonnegative")
        if delivery not in DELIVERY_MODES:
            raise ValueError(f"delivery must be one of {DELIVERY_MODES}")
        self.communication = bool(communication)
        self.chat_cooldown_tics = int(chat_cooldown_tics)
        self.chat_tokens = validate_chat_tokens(
            make_chat_tokens() if chat_tokens is None else chat_tokens
        )
        self.delivery = delivery
        self.observer_info = bool(observer_info)
        self._delivery_rng = np.random.default_rng(delivery_seed)
        names = [button.name for button in env.available_buttons]
        if set(names) != set(MOTOR_BUTTONS[1:]):
            raise ValueError("Communication requires the seven movement/USE buttons")
        self.motor_actions = [0] + [
            1 << names.index(name) for name in MOTOR_BUTTONS[1:]
        ]
        self._action_space = spaces.MultiDiscrete(
            [len(MOTOR_BUTTONS), len(self.chat_tokens)]
        )
        self._observation_space = spaces.Dict(
            observation=env.observation_space("agent_0"),
            notifications=spaces.Box(0, 127, (TEXT_LENGTH,), dtype=np.uint8),
            features=spaces.Box(0.0, 1.0, (FEATURE_LENGTH,), dtype=np.float32),
        )
        self._observations = {}

    def action_space(self, agent):
        return self._action_space

    def observation_space(self, agent):
        return self._observation_space

    @property
    def state_space(self):
        fields = {}
        for key, space in self._observation_space.spaces.items():
            assert space.shape is not None
            fields[key] = spaces.Box(
                0, space.high.max(), (2, *space.shape), dtype=space.dtype
            )
        return spaces.Dict(fields)

    def state_observation(self, agent):
        return self._observations[agent]

    def state(self):
        return {
            key: np.stack([self._observations[a][key] for a in self.possible_agents])
            for key in self._observation_space.spaces
        }

    def _send_available(self, agent):
        return self.communication and self._tics[agent] >= self._next_send[agent]

    def _features(self, agent, informer_player):
        """
        ``features`` exports the current role of the agent and the channel state (send availability and remaining cooldown fraction)
        so a shared actor can mask actions that its role or channel cannot use, and so the centralized critic can see the channel it does not backprop through.
        """
        features = np.zeros(FEATURE_LENGTH, dtype=np.float32)
        agent_player = self.possible_agents.index(agent)
        features[FEATURE_ROLE_INFORMER] = float(agent_player == informer_player)
        features[FEATURE_ROLE_OPERATOR] = float(agent_player != informer_player)
        features[FEATURE_SEND_AVAILABLE] = float(self._send_available(agent))
        if self.chat_cooldown_tics:
            remaining = self._next_send[agent] - self._tics[agent]
            features[FEATURE_SEND_AVAILABLE + 1] = min(
                1.0, max(0.0, remaining / self.chat_cooldown_tics)
            )
        return features

    def _observe(self, observations, infos):
        result = {}
        for agent, image in observations.items():
            info = infos[agent]
            if info.get("_hidden_reset") or info.get("reset_info") is not None:
                self._histories[agent] = NotificationHistory()
                self._clue_first_seen[agent] = -1.0
            for snapshot in info.pop("_notifications", []):
                new_lines = self._histories[agent].update(snapshot)
                if self._clue_first_seen[agent] < 0 and any(
                    re.fullmatch(r"(RED|GREEN|BLUE)", line) for line in new_lines
                ):
                    self._clue_first_seen[agent] = (
                        max(0, int(info["step"]) - self._start_tics[agent]) / 35.0
                    )
                self._received[agent] += sum(
                    re.fullmatch(r"Player[01]: .+", line) is not None
                    for line in new_lines
                )
            # This diagnostic is not needed even in training infos.
            secret = info.pop("USER43", None)
            if self.observer_info:
                info["communication_secret"] = float(-1 if secret is None else secret)
                info["communication_slot0_color"] = float(info.pop("USER48", -1))
                first = self._first_message[agent]
                info["communication_first_intended_message"] = float(first[0])
                info["communication_first_delivered_message"] = float(first[1])
                info["communication_first_message_seconds"] = first[2]
                info["communication_messages_delivered"] = float(self._delivered[agent])
            info["communication_clue_seen"] = float(self._clue_first_seen[agent] >= 0)
            info["communication_clue_first_seen_seconds"] = self._clue_first_seen[agent]
            info["communication_messages_attempted"] = float(self._attempted[agent])
            info["communication_messages_sent"] = float(self._sent[agent])
            info["communication_chat_lines_received"] = float(self._received[agent])
            for symbol, count in enumerate(self._symbol_sent[agent], start=1):
                info[f"communication_symbol_{symbol}_sent"] = float(count)
            self._tics[agent] = int(info["step"])
            info["communication_round_seconds"] = (
                max(0, self._tics[agent] - self._start_tics[agent]) / 35.0
            )
            info["communication_send_available"] = float(self._send_available(agent))
            result[agent] = {
                "observation": image,
                "notifications": self._histories[agent].encode(),
                "features": self._features(agent, int(info.get("USER44", 0))),
            }
        self._observations = result
        return result

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._histories = {a: NotificationHistory() for a in self.agents}
        self._attempted = dict.fromkeys(self.agents, 0)
        self._sent = dict.fromkeys(self.agents, 0)
        self._received = dict.fromkeys(self.agents, 0)
        self._clue_first_seen = dict.fromkeys(self.agents, -1.0)
        self._symbol_sent = {a: [0] * (len(self.chat_tokens) - 1) for a in self.agents}
        self._delivered = dict.fromkeys(self.agents, 0)
        self._first_message = {a: (0, 0, -1.0) for a in self.agents}
        self._tics = dict.fromkeys(self.agents, 0)
        self._start_tics = {a: int(infos[a]["step"]) for a in self.agents}
        self._next_send = dict.fromkeys(self.agents, 0)
        return self._observe(observations, infos), infos

    def delivered_message(self, message):
        """Map an accepted token index to the index the engine will say."""
        if self.delivery == "natural":
            return message
        if self.delivery == "silence":
            return 0
        symbols = len(self.chat_tokens) - 1
        if self.delivery == "swap":
            return symbols + 1 - message
        return int(self._delivery_rng.integers(1, symbols + 1))

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError("Actions must contain exactly the active agents")
        decoded = {}
        for agent, action in actions.items():
            raw = np.asarray(action).reshape(-1)
            if raw.size != 2 or not self._action_space.contains(
                raw.astype(np.int64, copy=False)
            ):
                raise ValueError(f"Invalid communication action for {agent}: {action}")
            decoded[agent] = (int(raw[0]), int(raw[1]))
        motor = {}
        for agent, (movement, message) in decoded.items():
            motor[agent] = self.motor_actions[movement]
            self._attempted[agent] += int(message != 0)
            if message and self._send_available(agent):
                delivered = self.delivered_message(message)
                if delivered:
                    self.env.send_game_command(
                        f"say {self.chat_tokens[delivered]}",
                        self.possible_agents.index(agent),
                    )
                    self._delivered[agent] += 1
                if self._first_message[agent][0] == 0:
                    seconds = max(0, self._tics[agent] - self._start_tics[agent]) / 35
                    self._first_message[agent] = (message, delivered, seconds)
                self._sent[agent] += 1
                self._symbol_sent[agent][message - 1] += 1
                self._next_send[agent] = self._tics[agent] + self.chat_cooldown_tics
        obs, rewards, terminated, truncated, infos = self.env.step(motor)
        obs = self._observe(obs, infos)
        if all(terminated[a] or truncated[a] for a in self.agents):
            self.agents = []
        return obs, rewards, terminated, truncated, infos
