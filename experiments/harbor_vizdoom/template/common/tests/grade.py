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


def _write_score(score: int, message: str, result: dict[str, Any] | None) -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(f"{score}\n", encoding="utf-8")
    metrics = {
        "reward": score,
        "message": message,
        "status": result.get("status") if result else None,
        "episode_return": result.get("episode_return") if result else None,
        "decisions": result.get("decisions") if result else None,
        "seed": result.get("seed") if result else None,
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
    for action in action_history:
        if not isinstance(action, dict) or not isinstance(action.get("buttons"), list):
            _fail("malformed action history", result)
        buttons = action["buttons"]
        if len(buttons) != len(set(buttons)) or not all(
            isinstance(button, str) for button in buttons
        ):
            _fail("action history contains invalid buttons", result)

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

    success = (
        result.get("success") is True
        and result.get("status") == "success"
        and isinstance(result.get("episode_return"), (int, float))
        and result["episode_return"] >= 1
    )
    message = "level completed" if success else "level was not completed"
    score = int(success)
    _write_score(score, message, result)
    print(
        f"score={score} status={result.get('status')} "
        f"return={result.get('episode_return')} decisions={decisions}"
    )


if __name__ == "__main__":
    main()
