"""Expose one manifest-configured ViZDoom episode as multimodal MCP tools."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import signal
import struct
import sys
import threading
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import gymnasium
import numpy as np

from vizdoom import gymnasium_wrapper  # noqa: F401  (register environments)

ENVIRONMENT_ID = os.environ.get("VIZDOOM_ENVIRONMENT_ID", "VizdoomBasic-MultiBinary-v1")
FRAME_SKIP = 4
MAX_DECISIONS = int(os.environ.get("VIZDOOM_MAX_DECISIONS", "75"))
MAX_MEMORIES = int(os.environ.get("VIZDOOM_MAX_MEMORIES", "32"))
RESULTS_DIR = Path(os.environ.get("VIZDOOM_RESULTS_DIR", "/vizdoom-results"))
RESULT_PATH = RESULTS_DIR / "result.json"
EVENTS_PATH = RESULTS_DIR / "events.jsonl"
VIDEO_WORK_DIR = RESULTS_DIR / "video-tmp"
VIDEO_PATH = RESULTS_DIR / "recording.mp4"
FINALIZED_PATH = RESULTS_DIR / "finalized"
OBJECTIVE = os.environ.get(
    "VIZDOOM_OBJECTIVE", "Kill the red monster before the episode times out."
)
IWAD_PATH = os.environ.get("VIZDOOM_IWAD_PATH", "").strip()
REQUIRE_AUTOMAP = os.environ.get("VIZDOOM_REQUIRE_AUTOMAP", "0") == "1"
MEMORY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

if MAX_DECISIONS < 1:
    raise RuntimeError("VIZDOOM_MAX_DECISIONS must be a positive integer")
if MAX_MEMORIES < 1:
    raise RuntimeError("VIZDOOM_MAX_MEMORIES must be a positive integer")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _choose_seed() -> int:
    raw_seed = os.environ.get("VIZDOOM_SEED", "").strip()
    if not raw_seed:
        return secrets.randbits(32)
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise RuntimeError("VIZDOOM_SEED must be an unsigned 32-bit integer") from exc
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise RuntimeError("VIZDOOM_SEED must be an unsigned 32-bit integer")
    return seed


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)
    )


def _image_to_png(image_buffer: np.ndarray) -> bytes:
    """Encode a Gymnasium HWC image buffer to PNG entirely in memory."""
    image = np.asarray(image_buffer)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[2] not in (1, 3):
        raise ValueError(f"Expected HxWx1 or HxWx3 image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    image = np.ascontiguousarray(image)
    height, width, channels = image.shape
    color_type = 0 if channels == 1 else 2
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=6)),
            _png_chunk(b"IEND", b""),
        )
    )


def _observation_image(
    observation: dict[str, Any], name: str, *, required: bool = True
) -> np.ndarray | None:
    image = observation.get(name)
    if image is None and not required:
        return None
    if not isinstance(image, np.ndarray):
        raise RuntimeError(f"ViZDoom observation did not contain a {name} buffer")
    return image


def _game_variables(observation: dict[str, Any]) -> list[float]:
    values = observation.get("gamevariables")
    if values is None:
        return []
    return np.asarray(values).astype(float).tolist()


@dataclass(frozen=True)
class Snapshot:
    metadata: dict[str, Any]
    screen_png: bytes
    automap_png: bytes | None


class Episode:
    """Own the environment, explicit visual memories, logs, and video."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.seed = _choose_seed()
        self.episode_return = 0.0
        self.action_history: list[dict[str, Any]] = []
        self.memories: dict[str, Snapshot] = {}
        self.current_snapshot: Snapshot | None = None
        self.status = "running"
        self.video_finalized = False

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        EVENTS_PATH.write_text("", encoding="utf-8")

        environment_options: dict[str, Any] = {
            "frame_skip": FRAME_SKIP,
            "render_mode": "rgb_array",
        }
        if IWAD_PATH:
            iwad_path = Path(IWAD_PATH)
            if not iwad_path.is_file():
                raise RuntimeError(f"IWAD file does not exist: {iwad_path}")
            environment_options["doom_game_path"] = str(iwad_path)
        base_env = gymnasium.make(ENVIRONMENT_ID, **environment_options)
        render_fps = int(base_env.metadata.get("render_fps", 30))
        video_fps = max(1, round(render_fps / FRAME_SKIP))
        self.env = gymnasium.wrappers.RecordVideo(
            base_env,
            video_folder=str(VIDEO_WORK_DIR),
            episode_trigger=lambda episode_id: episode_id == 0,
            video_length=sys.maxsize,
            name_prefix="recording",
            fps=video_fps,
            disable_logger=True,
        )

        unwrapped = cast(Any, self.env.unwrapped)
        buttons = unwrapped.game.get_available_buttons()
        binary_buttons = buttons[unwrapped.num_delta_buttons :]
        self.button_names = [button.name for button in binary_buttons]
        if not isinstance(self.env.action_space, gymnasium.spaces.MultiBinary):
            raise RuntimeError(
                f"{ENVIRONMENT_ID} did not expose a MultiBinary action space"
            )
        if self.env.action_space.shape != (len(self.button_names),):
            raise RuntimeError("Action space does not match available buttons")

        observation, _ = self.env.reset(seed=self.seed)
        if REQUIRE_AUTOMAP and "automap" not in observation:
            self.env.close()
            raise RuntimeError(f"{ENVIRONMENT_ID} did not expose an automap buffer")
        self._record_snapshot(
            observation,
            reward=0.0,
            buttons=[],
            terminated=False,
            truncated=False,
            event="reset",
        )

    @property
    def done(self) -> bool:
        return self.status != "running"

    def _metadata(
        self,
        observation: dict[str, Any],
        *,
        reward: float,
        buttons: list[str],
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        decision = len(self.action_history)
        image_names = ["screen"]
        if "automap" in observation:
            image_names.append("automap")
        return {
            "environment": ENVIRONMENT_ID,
            "objective": OBJECTIVE,
            "seed": self.seed,
            "frame_skip": FRAME_SKIP,
            "decision": decision,
            "max_decisions": MAX_DECISIONS,
            "remaining_decisions": max(0, MAX_DECISIONS - decision),
            "available_buttons": self.button_names,
            "image_order": image_names,
            "last_buttons": buttons,
            "last_reward": reward,
            "episode_return": self.episode_return,
            "game_variables": _game_variables(observation),
            "terminated": terminated,
            "truncated": truncated,
            "done": self.done,
            "status": self.status,
            "success": self.status == "success",
        }

    def _result(self) -> dict[str, Any]:
        video_exists = VIDEO_PATH.is_file()
        return {
            "environment": ENVIRONMENT_ID,
            "objective": OBJECTIVE,
            "seed": self.seed,
            "frame_skip": FRAME_SKIP,
            "max_decisions": MAX_DECISIONS,
            "decisions": len(self.action_history),
            "episode_return": self.episode_return,
            "status": self.status,
            "success": self.status == "success",
            "action_history": self.action_history,
            "stored_memories": {
                memory_id: snapshot.metadata["decision"]
                for memory_id, snapshot in sorted(self.memories.items())
            },
            "events_path": str(EVENTS_PATH),
            "video_path": str(VIDEO_PATH),
            "video_finalized": self.video_finalized,
            "video_size_bytes": VIDEO_PATH.stat().st_size if video_exists else 0,
            "updated_at": _utc_now(),
        }

    def _write_result(self) -> None:
        temporary_path = RESULT_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self._result(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(RESULT_PATH)

    def _append_event(self, event: str, metadata: dict[str, Any]) -> None:
        record = {"timestamp": _utc_now(), "event": event, **metadata}
        with EVENTS_PATH.open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(record, sort_keys=True) + "\n")
            event_file.flush()
            os.fsync(event_file.fileno())

    def _record_snapshot(
        self,
        observation: dict[str, Any],
        *,
        reward: float,
        buttons: list[str],
        terminated: bool,
        truncated: bool,
        event: str,
    ) -> Snapshot:
        metadata = self._metadata(
            observation,
            reward=reward,
            buttons=buttons,
            terminated=terminated,
            truncated=truncated,
        )
        screen = _observation_image(observation, "screen")
        assert screen is not None
        automap = _observation_image(observation, "automap", required=REQUIRE_AUTOMAP)
        snapshot = Snapshot(
            metadata=metadata,
            screen_png=_image_to_png(screen),
            automap_png=_image_to_png(automap) if automap is not None else None,
        )
        self.current_snapshot = snapshot
        self._append_event(event, metadata)
        self._write_result()
        return snapshot

    def current(self) -> Snapshot:
        with self.lock:
            if self.current_snapshot is None:
                raise RuntimeError("Episode has no current state")
            self._append_event(
                "get_current_state",
                {
                    "decision": self.current_snapshot.metadata["decision"],
                    "status": self.status,
                },
            )
            return self.current_snapshot

    def store(self, memory_id: str) -> dict[str, Any]:
        with self.lock:
            if not MEMORY_ID_PATTERN.fullmatch(memory_id):
                raise ValueError(
                    "memory_id must be 1-64 characters and contain only letters, "
                    "digits, periods, underscores, or hyphens"
                )
            if self.current_snapshot is None:
                raise RuntimeError("Episode has no current state")
            replaced = memory_id in self.memories
            if not replaced and len(self.memories) >= MAX_MEMORIES:
                raise ValueError(f"At most {MAX_MEMORIES} states may be stored")
            self.memories[memory_id] = self.current_snapshot
            result = {
                "memory_id": memory_id,
                "decision": self.current_snapshot.metadata["decision"],
                "replaced": replaced,
                "stored_memory_ids": sorted(self.memories),
            }
            self._append_event("store_state", result)
            self._write_result()
            return result

    def retrieve(self, memory_id: str) -> Snapshot:
        with self.lock:
            snapshot = self.memories.get(memory_id)
            if snapshot is None:
                raise ValueError(
                    f"Unknown memory_id {memory_id!r}; stored IDs are "
                    f"{sorted(self.memories)}"
                )
            self._append_event(
                "get_stored_state",
                {
                    "memory_id": memory_id,
                    "decision": snapshot.metadata["decision"],
                },
            )
            return snapshot

    def _finish_video(self) -> None:
        if self.video_finalized:
            return
        self.env.close()
        generated_video = VIDEO_WORK_DIR / "recording-episode-0.mp4"
        if not generated_video.is_file():
            raise RuntimeError(f"Gymnasium did not create {generated_video}")
        generated_video.replace(VIDEO_PATH)
        self.video_finalized = True

    def finalize(self, unfinished_status: str = "agent_stopped") -> None:
        with self.lock:
            if self.status == "running":
                self.status = unfinished_status
            self._finish_video()
            self._append_event(
                "finalize",
                {
                    "decision": len(self.action_history),
                    "status": self.status,
                    "video_path": str(VIDEO_PATH),
                    "video_size_bytes": VIDEO_PATH.stat().st_size,
                },
            )
            self._write_result()
            FINALIZED_PATH.write_text("ready\n", encoding="utf-8")

    def act(self, buttons: list[str]) -> Snapshot:
        with self.lock:
            if self.done:
                raise RuntimeError(
                    f"Episode is already done with status {self.status!r}"
                )
            if len(buttons) != len(set(buttons)):
                raise ValueError("buttons must not contain duplicates")
            unknown_buttons = sorted(set(buttons) - set(self.button_names))
            if unknown_buttons:
                raise ValueError(
                    f"Unknown buttons {unknown_buttons}; valid buttons are "
                    f"{self.button_names}"
                )

            action = np.array(
                [int(button in buttons) for button in self.button_names],
                dtype=np.int8,
            )
            observation, reward, terminated, truncated, _ = self.env.step(action)
            numeric_reward = float(reward)
            self.episode_return += numeric_reward
            decision = len(self.action_history) + 1

            unwrapped = cast(Any, self.env.unwrapped)
            player_dead = bool(unwrapped.game.is_player_dead())
            if self.episode_return > 0:
                self.status = "success"
            elif player_dead:
                self.status = "dead"
            elif terminated or truncated:
                self.status = "timeout"
            elif decision >= MAX_DECISIONS:
                self.status = "decision_limit"

            self.action_history.append(
                {
                    "decision": decision,
                    "buttons": list(buttons),
                    "reward": numeric_reward,
                    "episode_return": self.episode_return,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "status": self.status,
                }
            )
            snapshot = self._record_snapshot(
                observation,
                reward=numeric_reward,
                buttons=list(buttons),
                terminated=bool(terminated),
                truncated=bool(truncated),
                event="action",
            )
            if self.done:
                self.finalize()
            return snapshot


def _serve_mcp(episode: Episode) -> None:
    from fastmcp import FastMCP
    from fastmcp.tools.tool import ToolResult
    from fastmcp.utilities.types import Image
    from mcp.types import TextContent

    mcp = FastMCP(
        "vizdoom",
        instructions=(
            "Play the single ViZDoom episode through these tools. Call "
            "get_current_state first. State results label and return the screen "
            "image followed by the automap image. Store selected states under "
            "your own memory IDs if you need to revisit them. Continue until "
            "done is true."
        ),
    )

    def snapshot_result(snapshot: Snapshot, label: str) -> ToolResult:
        content: list[Any] = [
            TextContent(
                type="text",
                text=f"{label}\n{json.dumps(snapshot.metadata, sort_keys=True)}",
            ),
            TextContent(type="text", text="Screen image:"),
            Image(data=snapshot.screen_png, format="png").to_image_content(),
        ]
        if snapshot.automap_png is not None:
            content.extend(
                (
                    TextContent(type="text", text="Automap image:"),
                    Image(data=snapshot.automap_png, format="png").to_image_content(),
                )
            )
        return ToolResult(content=content, structured_content=snapshot.metadata)

    @mcp.tool()
    def get_current_state() -> ToolResult:
        """Get current metadata, then the labeled screen and automap images."""
        return snapshot_result(episode.current(), "Current state:")

    @mcp.tool()
    def make_action(buttons: list[str]) -> ToolResult:
        """Press named buttons together and return the resulting state.

        Args:
            buttons: Any subset of available_buttons. Use [] to wait.
        """
        return snapshot_result(episode.act(buttons), "Resulting state:")

    @mcp.tool()
    def store_state(memory_id: str) -> dict[str, Any]:
        """Store the current visual state under a caller-provided memory ID."""
        return episode.store(memory_id)

    @mcp.tool()
    def get_stored_state(memory_id: str) -> ToolResult:
        """Retrieve a stored state by an ID previously passed to store_state."""
        return snapshot_result(
            episode.retrieve(memory_id), f"Stored state {memory_id!r}:"
        )

    def finalize_from_signal(_signal_number: int, _frame: Any) -> None:
        episode.finalize()

    signal.signal(signal.SIGUSR1, finalize_from_signal)
    try:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
    finally:
        if not episode.video_finalized:
            episode.finalize("server_stopped")


def _run_random_agent(
    episode: Episode,
    *,
    max_actions: int,
    button_probability: float,
    policy_seed: int,
    memory_every: int,
) -> None:
    policy = random.Random(policy_seed)
    episode.current()
    try:
        for action_index in range(max_actions):
            if episode.done:
                break
            if memory_every and action_index % memory_every == 0:
                episode.store("random-latest")
                episode.retrieve("random-latest")
            buttons = [
                button
                for button in episode.button_names
                if policy.random() < button_probability
            ]
            episode.act(buttons)
    finally:
        if not episode.video_finalized:
            episode.finalize("random_agent_stopped")
    print(RESULT_PATH.read_text(encoding="utf-8"), end="")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-agent",
        action="store_true",
        help="Run a random policy locally instead of starting the MCP server",
    )
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--button-probability", type=float, default=0.15)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument(
        "--memory-every",
        type=int,
        default=4,
        help="Exercise store/retrieve every N actions; 0 disables it",
    )
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if not 0 <= args.button_probability <= 1:
        parser.error("--button-probability must be between 0 and 1")
    if args.memory_every < 0:
        parser.error("--memory-every must be non-negative")
    if args.policy_seed is not None and not 0 <= args.policy_seed <= 2**32 - 1:
        parser.error("--policy-seed must be an unsigned 32-bit integer")
    return args


def main() -> None:
    args = _parse_args()
    episode = Episode()
    if args.random_agent:
        policy_seed = episode.seed if args.policy_seed is None else args.policy_seed
        _run_random_agent(
            episode,
            max_actions=args.max_actions,
            button_probability=args.button_probability,
            policy_seed=policy_seed,
            memory_every=args.memory_every,
        )
    else:
        _serve_mcp(episode)


if __name__ == "__main__":
    main()
