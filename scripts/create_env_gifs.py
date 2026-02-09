#!/usr/bin/env python3

######################################################
# Script to create GIFs for ViZDoom Gymnasium environments
######################################################

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np

import vizdoom.gymnasium_wrapper  # noqa


PANEL_BG = 20
PANEL_TEXT = (240, 240, 240)
WAVE_COLOR = (255, 255, 255)


def _ensure_rgb_image(screen: np.ndarray) -> np.ndarray:
    """Convert any supported screen shape into HxWx3 uint8."""
    if screen.ndim == 2:
        return np.stack([screen] * 3, axis=2)
    if screen.ndim == 3 and screen.shape[2] == 1:
        return np.repeat(screen, 3, axis=2)
    return screen


def _wrap_text(
    text: str, max_width: int, font, font_scale: float, thickness: int
) -> list[str]:
    """Wrap text to fit target width for cv2 text rendering."""
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [text]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_wave(
    panel: np.ndarray,
    wave: np.ndarray,
    left: int,
    right: int,
    top: int,
    bottom: int,
    max_abs: float | None = None,
) -> None:
    """Draw one waveform in a specified panel region."""
    if wave.size == 0:
        return

    wave = wave.astype(np.float32)
    plot_left = max(0, left)
    plot_right = min(panel.shape[1] - 1, right)
    plot_top = top
    plot_bottom = bottom
    plot_width = max(2, plot_right - plot_left)
    center_y = plot_top + (plot_bottom - plot_top) // 2
    amplitude = max(1, (plot_bottom - plot_top) // 2 - 2)

    cv2.line(panel, (plot_left, center_y), (plot_right, center_y), (80, 80, 80), 1)

    sample_idx = np.linspace(0, wave.size - 1, plot_width, dtype=np.int32)
    sampled = wave[sample_idx]
    if max_abs is None:
        max_abs = float(np.max(np.abs(sampled)))
    if max_abs > 0:
        sampled = sampled / max_abs
    y = center_y - (sampled * amplitude).astype(np.int32)
    x = np.arange(plot_width, dtype=np.int32) + plot_left
    points = np.column_stack((x, y))
    cv2.polylines(panel, [points], False, WAVE_COLOR, 1, cv2.LINE_AA)


def _render_audio_panel(audio: np.ndarray, width: int, height: int = 92) -> np.ndarray:
    """Render a stereo audio waveform panel for one observation."""
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    samples = np.asarray(audio)
    if samples.size == 0:
        return panel

    if samples.ndim == 1:
        samples = samples[:, None]
    elif samples.ndim > 2:
        samples = samples.reshape(samples.shape[0], -1)
    samples = samples.astype(np.float32)

    pad = 8
    if samples.shape[1] >= 2:
        # Keep both channels on the same vertical scale for direct comparison.
        shared_max_abs = float(np.max(np.abs(samples[:, :2])))
        gap = 6
        usable_h = max(2, height - (2 * pad) - gap)
        chan_h = max(8, usable_h // 2)
        top1 = pad
        bot1 = min(height - pad - gap - 1, top1 + chan_h)
        top2 = bot1 + gap
        bot2 = min(height - pad - 1, top2 + chan_h)
        _draw_wave(
            panel, samples[:, 0], pad, width - pad, top1, bot1, max_abs=shared_max_abs
        )
        _draw_wave(
            panel, samples[:, 1], pad, width - pad, top2, bot2, max_abs=shared_max_abs
        )
    else:
        _draw_wave(panel, samples[:, 0], pad, width - pad, pad, height - pad)

    return panel


def _normalize_notification_text(notification: object) -> str:
    """Convert notification buffer payload to renderable text."""

    def _sanitize_text(text: str) -> str:
        text = text.replace("\r", "\n").replace("\x00", "")
        cleaned_chars = []
        for ch in text:
            code = ord(ch)
            if ch in {"\n", "\t"} or 32 <= code <= 126:
                cleaned_chars.append(ch)
        cleaned = "".join(cleaned_chars)
        lines = []
        for line in cleaned.splitlines():
            compact = " ".join(line.replace("\t", " ").split())
            if compact:
                lines.append(compact)
        return "\n".join(lines).strip()

    if notification is None:
        return ""
    if isinstance(notification, str):
        return _sanitize_text(notification)
    if isinstance(notification, (bytes, bytearray, memoryview)):
        raw = bytes(notification)
        parts = [
            chunk.decode("latin1", errors="ignore")
            for chunk in raw.split(b"\x00")
            if chunk
        ]
        return _sanitize_text("\n".join(parts))
    if isinstance(notification, np.ndarray):
        if notification.dtype.kind in {"U", "S"}:
            if notification.dtype.kind == "S":
                raw = notification.tobytes()
                parts = [
                    chunk.decode("latin1", errors="ignore")
                    for chunk in raw.split(b"\x00")
                    if chunk
                ]
                return _sanitize_text("\n".join(parts))
            return _sanitize_text("".join(str(ch) for ch in notification.reshape(-1)))
        if notification.dtype.kind in {"i", "u"}:
            raw = notification.astype(np.uint8, copy=False).reshape(-1).tobytes()
            parts = [
                chunk.decode("latin1", errors="ignore")
                for chunk in raw.split(b"\x00")
                if chunk
            ]
            return _sanitize_text("\n".join(parts))
    if isinstance(notification, (list, tuple)):
        return _sanitize_text("\n".join(str(item) for item in notification))
    return _sanitize_text(str(notification))


def _notification_lines(notification: object, width: int) -> list[str]:
    """Wrap notification text and return all lines."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    max_width = width - 16
    text = _normalize_notification_text(notification)
    if not text:
        return [""]

    lines = []
    for paragraph in text.splitlines() or [text]:
        wrapped = _wrap_text(paragraph, max_width, font, font_scale, thickness)
        lines.extend(wrapped if wrapped else [""])
    return lines if lines else [""]


def _notification_panel_height(observations: list[dict], width: int) -> int:
    """Compute panel height that can fit the longest notification text in this sequence."""
    line_height = 18
    top = 10
    bottom = 8
    max_lines = 1
    for obs in observations:
        if "notifications" in obs:
            max_lines = max(
                max_lines, len(_notification_lines(obs["notifications"], width))
            )
    return top + (line_height * max_lines) + bottom


def _render_notification_panel(
    notification: object, width: int, height: int
) -> np.ndarray:
    """Render in-game notification text panel for one observation."""
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = _notification_lines(notification, width)
    y = 22
    for line in lines:
        cv2.putText(panel, line, (8, y), font, 0.45, PANEL_TEXT, 1, cv2.LINE_AA)
        y += 18
    return panel


def _build_frame_from_observation(
    obs: dict, notification_height: int | None = None
) -> np.ndarray:
    """Build renderable frame with optional buffer panels below the screen."""
    frame = _ensure_rgb_image(obs["screen"])
    panels = []

    if "audio" in obs:
        panels.append(_render_audio_panel(obs["audio"], frame.shape[1]))
    if "notifications" in obs:
        if notification_height is None:
            notification_height = _notification_panel_height([obs], frame.shape[1])
        panels.append(
            _render_notification_panel(
                obs["notifications"], frame.shape[1], notification_height
            )
        )

    if panels:
        return np.concatenate([frame, *panels], axis=0)
    return frame


def save_gif(frames, gif_path: Path, fps: float = 20):
    """Save a list of frames as a GIF."""
    if len(frames) > 0:
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"GIF saved to {gif_path}")
    else:
        print("No frames to save for GIF.")


def _copy_obs_for_render(obs: dict) -> dict:
    """Create a compact copy of observation fields used for GIF rendering."""
    obs_copy = {"screen": np.array(obs["screen"], copy=True)}
    if "audio" in obs:
        obs_copy["audio"] = np.array(obs["audio"], copy=True)
    if "notifications" in obs:
        # Store normalized text immediately to avoid any mutable-buffer aliasing.
        obs_copy["notifications"] = _normalize_notification_text(obs["notifications"])
    return obs_copy


def _apply_sticky_notifications(observations: list[dict]) -> list[dict]:
    """Reuse the last non-empty notification for frames with empty notifications."""
    resolved = []
    last_non_empty_notification = None

    for obs in observations:
        obs_resolved = dict(obs)
        if "notifications" in obs_resolved:
            notification_text = _normalize_notification_text(
                obs_resolved["notifications"]
            )
            if notification_text:
                last_non_empty_notification = obs_resolved["notifications"]
            elif last_non_empty_notification is not None:
                obs_resolved["notifications"] = last_non_empty_notification
        resolved.append(obs_resolved)

    return resolved


def _build_frames_for_sequence(observations: list[dict]) -> list[np.ndarray]:
    """Build renderable frames with consistent panel sizes across a sequence."""
    if not observations:
        return []

    screen = _ensure_rgb_image(observations[0]["screen"])
    width = screen.shape[1]
    has_notifications = any("notifications" in obs for obs in observations)
    notification_height = (
        _notification_panel_height(observations, width) if has_notifications else None
    )

    return [
        _build_frame_from_observation(obs, notification_height=notification_height)
        for obs in observations
    ]


def create_env_gif(
    env_name,
    duration: int = 3,
    fps: int = 20,
    seed: int = 1993,
    drop: int = 4,
):
    """Create a GIF for a single ViZDoom environment."""
    try:
        print(f"Creating GIF for {env_name}...")
        if drop <= 0:
            raise ValueError("drop must be greater than 0")

        env_doom_graphics = gym.make(
            env_name,
            doom_game_path="doom2.wad",
        )
        obs_doom_graphics, _ = env_doom_graphics.reset(seed=seed)
        env_freedoom_graphics = gym.make(
            env_name,
            doom_game_path="freedoom2.wad",
        )
        obs_freedoom_graphics, _ = env_freedoom_graphics.reset(seed=seed)

        observations_doom_graphics = []
        observations_freedoom_graphics = []
        total_frames = duration * fps

        for _ in range(total_frames):
            observations_doom_graphics.append(_copy_obs_for_render(obs_doom_graphics))
            observations_freedoom_graphics.append(
                _copy_obs_for_render(obs_freedoom_graphics)
            )

            action = env_doom_graphics.action_space.sample()
            next_obs_doom, _, term_doom, trunc_doom, _ = env_doom_graphics.step(action)
            (
                next_obs_freedoom,
                _,
                term_freedoom,
                trunc_freedoom,
                _,
            ) = env_freedoom_graphics.step(action)

            if term_doom or trunc_doom:
                obs_doom_graphics, _ = env_doom_graphics.reset(seed=seed)
            else:
                obs_doom_graphics = next_obs_doom
            if term_freedoom or trunc_freedoom:
                obs_freedoom_graphics, _ = env_freedoom_graphics.reset(seed=seed)
            else:
                obs_freedoom_graphics = next_obs_freedoom

        env_doom_graphics.close()
        env_freedoom_graphics.close()
        # Fill empty notifications before frame dropping, so short messages are preserved.
        observations_doom_graphics = _apply_sticky_notifications(
            observations_doom_graphics
        )
        observations_freedoom_graphics = _apply_sticky_notifications(
            observations_freedoom_graphics
        )
        observations_doom_graphics = observations_doom_graphics[::drop]
        observations_freedoom_graphics = observations_freedoom_graphics[::drop]
        frames_doom_graphics = _build_frames_for_sequence(observations_doom_graphics)
        frames_freedoom_graphics = _build_frames_for_sequence(
            observations_freedoom_graphics
        )
        output_fps = fps / drop
        env_prefix = env_name.split("-", 1)[0]

        if len(frames_doom_graphics) > 0:
            gif_path = (
                Path(__file__).parent
                / "../docs/_static/img/envs"
                / f"{env_prefix}-Doom2.gif"
            )
            save_gif(frames_doom_graphics, gif_path, fps=output_fps)
        if len(frames_freedoom_graphics) > 0:
            gif_path = (
                Path(__file__).parent
                / "../docs/_static/img/envs"
                / f"{env_prefix}-Freedoom2.gif"
            )
            save_gif(frames_freedoom_graphics, gif_path, fps=output_fps)

    except Exception as e:
        print(f"Failed to create GIF for {env_name}: {e}")


def main():
    # List of ViZDoom Gymnasium environments for which to create GIFs
    vizdoom_envs = [
        "VizdoomBasic-v1",
        "VizdoomBasicAudio-v1",
        "VizdoomBasicNotifications-v1",
        "VizdoomDeadlyCorridor-v1",
        "VizdoomDefendCenter-v1",
        "VizdoomDefendLine-v1",
        "VizdoomHealthGathering-v1",
        "VizdoomMyWayHome-v1",
        "VizdoomPredictPosition-v1",
        "VizdoomTakeCover-v1",
        "VizdoomDeathmatch-v1",
        "VizdoomHealthGatheringSupreme-v1",
        # TODO?
        # "VizdoomDoomE1M1-S3-v0",
        # "VizdoomDoom2MAP01-S3-v0",
        # "VizdoomFreedoom1E1M1-S3-v0",
        # "VizdoomFreedoom2MAP01-S3-v0",
    ]

    parser = argparse.ArgumentParser(description="Create docs GIFs for ViZDoom envs.")
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Seconds per GIF.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=35,
        help="GIF frame rate.",
    )
    parser.add_argument(
        "--drop",
        "-d",
        type=int,
        default=4,
        help="Drop every n frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1993 + 42,
        help="Random seed for action sampling.",
    )
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create GIFs for all environments
    for env_name in vizdoom_envs:
        create_env_gif(
            env_name,
            duration=args.duration,
            fps=args.fps,
            seed=args.seed,
            drop=args.drop,
        )


if __name__ == "__main__":
    main()
