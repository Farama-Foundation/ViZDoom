from __future__ import annotations

import re
import textwrap

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _unbatched(td):
    if getattr(td, "batch_size", ()):
        raise ValueError("Communication diagnostics require an unbatched TensorDict")


def _scalar(td, group, index, count, name):
    value = td.get((group, "info", f"communication_{name}"), None)
    if value is None and name == "stage":
        value = td.get((group, "info", "USER40"), None)
    if value is None:
        return None
    values = _array(value).reshape(-1)
    if len(values) != count:
        raise ValueError(f"Unexpected per-agent info shape for {group}/{name}")
    result = float(values[index])
    return result if np.isfinite(result) else None


def agent_diagnostics(td, groups):
    """Return JSON-compatible local snapshots without accessing hidden USER43"""
    _unbatched(td)
    result = []
    for group, agents in groups.items():
        notifications = td.get((group, "notifications"), None)
        if notifications is not None:
            notifications = _array(notifications)
            if notifications.shape != (len(agents), 256):
                raise ValueError("Expected notifications shape [agents, 256]")
            if notifications.dtype.kind not in "iu":
                raise TypeError("Notifications must contain integer ASCII tokens")
            if np.any(notifications < 0) or np.any(notifications > 127):
                raise ValueError("Notification tokens must be in 0..127")
        for index, agent in enumerate(agents):
            text = None
            if notifications is not None:
                text = (
                    bytes(notifications[index].astype(np.uint8))
                    .rstrip(b"\0")
                    .decode("ascii")
                )
            result.append(
                dict(
                    group=group,
                    agent=agent,
                    notifications=text,
                    **{
                        name: _scalar(td, group, index, len(agents), name)
                        for name in (
                            "round_seconds",
                            "clue_seen",
                            "clue_first_seen_seconds",
                            "stage",
                            "wrong_presses",
                            "stages_completed",
                            "outcome_code",
                            "messages_attempted",
                            "messages_sent",
                            "chat_lines_received",
                        )
                    },
                )
            )
    return result


def panel_text(td, groups):
    """Human readable panels for each agent; private CODE lines are observer only"""

    def display(value):
        return "unknown" if value is None or value < 0 else f"{value:g}"

    panels = []
    for row in agent_diagnostics(td, groups):
        wrong, correct = row["wrong_presses"], row["stages_completed"]
        total = (
            wrong + correct
            if all(value is not None and value >= 0 for value in (wrong, correct))
            else None
        )
        panels.append(
            "\n".join(
                [
                    f"{row['agent']} ({row['group']}) - local view",
                    "Observer only (private clues)",
                    f"Round seconds: {display(row['round_seconds'])} | Stage: {display(row['stage'])}",
                    "Receiver switches (episode/team):",
                    f"Total: {display(total)} | Wrong: {display(wrong)} | Correct: {display(correct)}",
                    f"Clue seen: {display(row['clue_seen'])}",
                    f"First clue seconds: {display(row['clue_first_seen_seconds'])}",
                    f"Outcome code: {display(row['outcome_code'])}",
                    f"Attempted: {display(row['messages_attempted'])} | Sent/accepted: {display(row['messages_sent'])}",
                    f"Received chat lines: {display(row['chat_lines_received'])}",
                    "Local bounded notifications (snapshot):",
                    row["notifications"] or "(none available)",
                ]
            )
        )
    return panels


def render_communication_frame(frame, td, groups, *, panel_count=None, panel_offset=0):
    """Append readable local panels to an RGB uint8 frame"""
    from PIL import Image, ImageDraw, ImageFont

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 frame with shape [height, width, 3]")
    panels = panel_text(td, groups)
    if not panels:
        return frame.copy()
    if panel_count is None:
        panel_count = len(panels)
    if (
        isinstance(panel_count, bool)
        or not isinstance(panel_count, int)
        or panel_count < len(panels)
    ):
        raise ValueError("panel_count must fit all diagnostics panels")
    if (
        isinstance(panel_offset, bool)
        or not isinstance(panel_offset, int)
        or panel_offset < 0
        or panel_offset + len(panels) > panel_count
    ):
        raise ValueError("panel_offset must place all panels within panel_count")
    font = ImageFont.load_default(size=18)
    width = max(frame.shape[1], 600 * panel_count)
    view_height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    height = view_height + 640
    height += height % 2
    canvas = Image.new("RGB", (width, height), (18, 21, 28))
    canvas.paste(
        Image.fromarray(frame).resize((width, view_height), Image.Resampling.NEAREST)
    )
    draw = ImageDraw.Draw(canvas)
    column = width // panel_count
    wrap_width = max(1, int((column - 32) / font.getlength("W")))
    for index, panel in enumerate(panels):
        panel_index = panel_offset + index
        x, y = panel_index * column + 16, view_height + 14
        draw.line(
            (
                panel_index * column,
                view_height,
                panel_index * column,
                canvas.height,
            ),
            fill="gray",
        )
        header, history = panel.split("Local bounded notifications (snapshot):\n", 1)
        lines = [
            wrapped
            for line in header.splitlines()
            for wrapped in (textwrap.wrap(line, wrap_width) or [""])
        ]
        lines.append("Local notifications (snapshot):")
        available = max(1, 25 - len(lines))
        history_lines = [
            wrapped
            for line in history.splitlines()
            for wrapped in (textwrap.wrap(line, wrap_width) or [""])
        ]
        if len(history_lines) > available:
            history_lines = (
                ["... " + history_lines[-1]]
                if available == 1
                else ["... older text omitted ...", *history_lines[-(available - 1) :]]
            )
        for line in [*lines, *history_lines]:
            draw.text((x, y), line, font=font, fill=(235, 239, 246))
            y += 24
    return np.array(canvas, dtype=np.uint8)


class CommunicationTrace:
    """Requests, accepted commands, and local snapshots"""

    def __init__(self, groups, chat_tokens=None):
        self.groups = groups
        self.chat_tokens = tuple(chat_tokens) if chat_tokens is not None else None
        self.events = []
        self._previous = {}
        self._round_seconds = None

    def reset(self, td):
        self.events = []
        self._previous = {}
        self._round_seconds = None
        self._observe(td, 0)

    def _observe(self, td, step):
        for row in agent_diagnostics(td, self.groups):
            if row["round_seconds"] is not None:
                self._round_seconds = row["round_seconds"]
            key = row["group"], row["agent"]
            previous = self._previous.get(key, {})
            text = row["notifications"]
            stamp = dict(step=step, round_seconds=row["round_seconds"])
            if text is not None and text != previous.get("notifications"):
                self.events.append(
                    dict(
                        kind="notification_snapshot",
                        **stamp,
                        receiver=row["agent"],
                        group=row["group"],
                        text=text,
                        observer_only=True,
                        private_clue=bool(re.search(r"(?m)^CODE\s+", text)),
                        semantics="Full bounded local history, not a list of new deliveries",
                    )
                )
            state = {
                k: v
                for k, v in row.items()
                if k not in ("notifications", "round_seconds")
            }
            if state != previous.get("state"):
                self.events.append(dict(kind="agent_state", **stamp, **state))
            self._previous[key] = dict(
                notifications=text, state=state, round_seconds=row["round_seconds"]
            )

    def step(self, current_td, next_td, step):
        current = agent_diagnostics(current_td, self.groups)
        following = agent_diagnostics(next_td, self.groups)
        for before, after in zip(current, following):
            group, agent = before["group"], before["agent"]
            # step_mdp may retain root info
            previous = self._previous.get((group, agent))
            if previous is not None:
                before = dict(
                    previous["state"], round_seconds=previous["round_seconds"]
                )
            action = current_td.get((group, "action"), None)
            requested = symbol = None
            if action is not None:
                actions = _array(action)
                agents = self.groups[group]
                if actions.ndim == 0 or actions.shape[0] != len(agents):
                    raise ValueError(
                        "Expected per-agent categorical or one-hot actions"
                    )
                local = actions[agents.index(agent)].reshape(-1)
                if local.size == 2:
                    requested = [int(local[0]), int(local[1])]
                    if self.chat_tokens:
                        symbol = self.chat_tokens[requested[1]]
                elif np.all((local == 0) | (local == 1)) and local.sum() == 1:
                    requested = int(local.argmax())
                else:
                    raise ValueError("Expected factorized or primitive motor actions")
            sent_before, sent_after = before["messages_sent"], after["messages_sent"]
            delta = None
            if (
                sent_before is not None
                and sent_after is not None
                and sent_after >= sent_before
            ):
                delta = sent_after - sent_before
            self.events.append(
                dict(
                    kind="message_request",
                    step=step,
                    round_seconds=before["round_seconds"],
                    observed_round_seconds=after["round_seconds"],
                    sender=agent,
                    group=group,
                    action=requested,
                    requested_symbol=symbol,
                    accepted_commands=delta,
                    accepted_symbol=symbol if delta == 1 and symbol else None,
                    status=(
                        "unknown"
                        if delta is None
                        else "accepted" if delta > 0 else "not_accepted"
                    ),
                    semantics="Sent counter measures accepted commands, not delivery",
                )
            )
        self._observe(next_td, step)

    def finish(self, step, outcome_code, step_limit_reached=False):
        self.events.append(
            dict(
                kind="episode_end",
                step=step,
                round_seconds=self._round_seconds,
                outcome_code=outcome_code,
                step_limit_reached=step_limit_reached,
            )
        )
