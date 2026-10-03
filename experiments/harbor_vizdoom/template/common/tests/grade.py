"""Grade a generated task using evidence collected from its ViZDoom sidecar."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

RESULT_PATH = (
    Path(os.environ.get("VIZDOOM_RESULTS_DIR", "/vizdoom-results")) / "result.json"
)
EVENTS_PATH = RESULT_PATH.with_name("events.jsonl")
VIDEO_PATH = RESULT_PATH.with_name("recording.mp4")
VERIFIER_LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
REWARD_PATH = VERIFIER_LOG_DIR / "reward.txt"
METRICS_PATH = VERIFIER_LOG_DIR / "metrics.json"
EXPECTED_ENVIRONMENT_ID = os.environ.get(
    "EXPECTED_ENVIRONMENT_ID", "VizdoomBasic-MultiBinary-v1"
)
EXPECTED_MAX_DECISIONS = int(os.environ.get("EXPECTED_MAX_DECISIONS", "75"))
EXPECTED_EPISODE_TIMEOUT_SECONDS = int(
    os.environ.get("EXPECTED_EPISODE_TIMEOUT_SECONDS", "0")
)
EXPECTED_OUTCOME_KIND = os.environ.get("EXPECTED_OUTCOME_KIND", "goal")
NATURAL_STATUSES = {"success", "dead", "timeout", "episode_ended"}


def _write_score(score: float, message: str, result: dict[str, Any] | None) -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(f"{score}\n", encoding="utf-8")
    action_history = result.get("action_history") if result else None
    actions = action_history if isinstance(action_history, list) else []
    action_sequences = [
        action
        for action in actions
        if isinstance(action, dict) and action.get("repeat_index") == 1
    ]
    narrated_sequences = [
        action
        for action in action_sequences
        if isinstance(action.get("situation"), str)
        and bool(action["situation"].strip())
        and isinstance(action.get("agent_decision"), str)
        and bool(action["agent_decision"].strip())
    ]
    elapsed_game_tics = result.get("elapsed_game_tics") if result else None
    metrics = {
        "reward": score,
        "message": message,
        "outcome_kind": result.get("outcome_kind") if result else None,
        "status": result.get("status") if result else None,
        "natural_end": result.get("status") in NATURAL_STATUSES if result else False,
        "episode_return": result.get("episode_return") if result else None,
        "elapsed_game_tics": elapsed_game_tics,
        "elapsed_game_tics_exact": (
            result.get("elapsed_game_tics_exact") if result else None
        ),
        "survival_seconds": (
            elapsed_game_tics / 35
            if isinstance(elapsed_game_tics, int)
            and elapsed_game_tics >= 0
            and result is not None
            and result.get("elapsed_game_tics_exact") is True
            else None
        ),
        "decisions": result.get("decisions") if result else None,
        "seed": result.get("seed") if result else None,
        "action_sequences": len(action_sequences),
        "narrated_action_sequences": len(narrated_sequences),
        "narration_coverage": (
            len(narrated_sequences) / len(action_sequences)
            if action_sequences
            else None
        ),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fail(message: str, result: dict[str, Any] | None = None) -> NoReturn:
    _write_score(0, message, result)
    print(f"Invalid verifier evidence: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not RESULT_PATH.is_file():
        _fail(f"missing sidecar artifact {RESULT_PATH}")
    try:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read result: {exc}")

    if not isinstance(result, dict):
        _fail("result must be a JSON object")
    if result.get("environment") != EXPECTED_ENVIRONMENT_ID:
        _fail("unexpected environment", result)
    if result.get("outcome_kind") != EXPECTED_OUTCOME_KIND:
        _fail("unexpected outcome kind", result)
    if result.get("frame_skip") != 4:
        _fail("frame_skip must be 4", result)
    if result.get("video_finalized") is not True:
        _fail("video was not finalized", result)

    decisions = result.get("decisions")
    action_history = result.get("action_history")
    if not isinstance(decisions, int) or not 0 <= decisions <= EXPECTED_MAX_DECISIONS:
        _fail(
            "decisions must be an integer between 0 and " f"{EXPECTED_MAX_DECISIONS}",
            result,
        )
    if not isinstance(action_history, list) or len(action_history) != decisions:
        _fail("action_history length does not match decisions", result)
    elapsed_game_tics = result.get("elapsed_game_tics")
    if not isinstance(elapsed_game_tics, int) or elapsed_game_tics < 0:
        _fail("elapsed_game_tics must be a non-negative integer", result)
    if not isinstance(result.get("elapsed_game_tics_exact"), bool):
        _fail("elapsed_game_tics_exact must be boolean", result)
    if result.get("episode_timeout_seconds", 0) != EXPECTED_EPISODE_TIMEOUT_SECONDS:
        _fail("unexpected in-game time limit", result)
    if EXPECTED_EPISODE_TIMEOUT_SECONDS:
        if elapsed_game_tics > EXPECTED_EPISODE_TIMEOUT_SECONDS * 35:
            _fail("episode exceeded the in-game time limit", result)
        if result.get("status") == "timeout" and (
            not result["elapsed_game_tics_exact"]
            or elapsed_game_tics != EXPECTED_EPISODE_TIMEOUT_SECONDS * 35
        ):
            _fail("timeout did not reach the configured in-game time limit", result)
    for action in action_history:
        if not isinstance(action, dict) or not isinstance(action.get("buttons"), list):
            _fail("malformed action history", result)
        buttons = action["buttons"]
        if len(buttons) != len(set(buttons)) or not all(
            isinstance(button, str) for button in buttons
        ):
            _fail("action history contains invalid buttons", result)
        for field in ("situation", "agent_decision"):
            if action.get(field) is not None and not isinstance(action[field], str):
                _fail(f"action history contains invalid {field}", result)
        repeat_index = action.get("repeat_index")
        requested_repeat = action.get("requested_repeat")
        if (
            not isinstance(repeat_index, int)
            or not isinstance(requested_repeat, int)
            or not 1 <= repeat_index <= requested_repeat <= 2
        ):
            _fail("action history contains invalid repetition metadata", result)

    if not EVENTS_PATH.is_file():
        _fail(f"missing event log {EVENTS_PATH}", result)
    try:
        event_lines = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in event_lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read event log: {exc}", result)
    if not events or not all(isinstance(event, dict) for event in events):
        _fail("event log must contain JSON objects", result)
    event_names = [event.get("event") for event in events]
    if event_names.count("reset") != 1:
        _fail("event log must contain exactly one reset", result)
    if event_names.count("action") != decisions:
        _fail("event log action count does not match decisions", result)
    if event_names.count("finalize") < 1:
        _fail("event log does not contain finalization", result)

    if not VIDEO_PATH.is_file():
        _fail(f"missing video {VIDEO_PATH}", result)
    try:
        video_size = VIDEO_PATH.stat().st_size
        with VIDEO_PATH.open("rb") as video_file:
            video_header = video_file.read(32)
    except OSError as exc:
        _fail(f"could not read video: {exc}", result)
    if video_size < 1_000 or b"ftyp" not in video_header:
        _fail("recording is not a valid non-empty MP4", result)
    if result.get("video_size_bytes") != video_size:
        _fail("reported video size does not match recording", result)

    status = result.get("status")
    episode_return = result.get("episode_return")
    if not isinstance(episode_return, (int, float)):
        _fail("episode_return must be numeric", result)
    if EXPECTED_OUTCOME_KIND == "goal":
        success = result.get("success") is True and status == "success"
        score = float(success)
        message = "goal completed" if success else "goal not completed"
    elif EXPECTED_OUTCOME_KIND == "combat":
        score = float(episode_return) if status in NATURAL_STATUSES else 0.0
        message = (
            "combat episode ended"
            if status in NATURAL_STATUSES
            else "combat episode interrupted"
        )
    elif EXPECTED_OUTCOME_KIND == "survival":
        if result["elapsed_game_tics_exact"] is not True:
            _fail("survival time must be exact", result)
        score = elapsed_game_tics / 35 if status in NATURAL_STATUSES else 0.0
        message = (
            "survival episode ended"
            if status in NATURAL_STATUSES
            else "survival episode interrupted"
        )
    else:
        _fail("unknown outcome kind", result)
    _write_score(score, message, result)
    print(
        f"score={score} status={result.get('status')} "
        f"return={result.get('episode_return')} decisions={decisions}"
    )


if __name__ == "__main__":
    main()
