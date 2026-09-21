#!/usr/bin/env python3
"""Let a vision-capable OpenRouter model play a default ViZDoom scenario.

Every supported scenario uses its ``-MultiBinary`` Gymnasium registration. For
each decision, the model receives the current RGB screen buffer as a
base64-encoded PNG and returns a JSON list of binary buttons to press. The
selected buttons are held for four ViZDoom tics by the Gymnasium wrapper. Each
request also retains the four preceding frame/action exchanges by default. Each
run stores its video, initial JSON configuration, and append-only JSONL
event log together in a timestamped directory below ``experiments/runs``.

Example:
    OPENROUTER_API_KEY=... python experiments/openrouter_vizdoom.py \
        --env VizdoomDeadlyCorridor-MultiBinary-v1 --justify-actions --render

Use ``--dry-run`` to initialize an environment and inspect its first request
without contacting OpenRouter.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import secrets
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gymnasium
import numpy as np

from vizdoom import gymnasium_wrapper  # noqa: F401  (registers the environments)

FRAME_SKIP = 4
DEFAULT_HISTORY_FRAMES = 4
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
MAX_COMPLETION_TOKENS = 512
DEFAULT_REASONING_EFFORT = "minimal"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DEFAULT_ENVIRONMENT_ID = "VizdoomBasic-MultiBinary-v1"
DESCRIPTIONS_PATH = Path(__file__).with_name("vizdoom_env_descriptions.json")
DEFAULT_RUN_DIRECTORY = Path(__file__).with_name("runs")


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns an invalid or unsuccessful response."""


def load_environment_descriptions() -> dict[str, dict[str, Any]]:
    try:
        with DESCRIPTIONS_PATH.open(encoding="utf-8") as descriptions_file:
            catalog = json.load(descriptions_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not load environment descriptions from {DESCRIPTIONS_PATH}: {exc}"
        ) from exc

    environments = catalog.get("environments")
    if not isinstance(environments, dict) or not environments:
        raise RuntimeError(
            f"{DESCRIPTIONS_PATH} must contain a non-empty 'environments' object"
        )

    required_fields = {"name", "description", "objective", "rewards", "episode_end"}
    for environment_id, description in environments.items():
        if not isinstance(description, dict):
            raise RuntimeError(f"Description for {environment_id!r} must be an object")
        missing_fields = required_fields - description.keys()
        if missing_fields:
            raise RuntimeError(
                f"Description for {environment_id!r} is missing: "
                f"{', '.join(sorted(missing_fields))}"
            )
    return environments


def parse_args(environment_ids: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenRouter vision model in a documented ViZDoom default "
            "scenario. One API request is made for every four ViZDoom tics."
        )
    )
    parser.add_argument(
        "--env",
        choices=environment_ids,
        default=DEFAULT_ENVIRONMENT_ID,
        help=f"MultiBinary Gymnasium environment (default: {DEFAULT_ENVIRONMENT_ID})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        help=(
            "OpenRouter vision model ID (default: OPENROUTER_MODEL or "
            f"{DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
        help=f"Model reasoning effort (default: {DEFAULT_REASONING_EFFORT})",
    )
    parser.add_argument(
        "--justify-actions",
        "--action-justification",
        action="store_true",
        help=(
            "Request and print a brief visual justification before each button "
            "action"
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes (default: 1)"
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=0,
        help="Maximum model decisions per episode; 0 uses the scenario timeout",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=DEFAULT_HISTORY_FRAMES,
        help=(
            "Completed frame/action exchanges retained before the current frame "
            f"(default: {DEFAULT_HISTORY_FRAMES})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base environment seed; omitted generates a random seed",
    )
    parser.add_argument(
        "--render", action="store_true", help="Display the Gymnasium environment"
    )
    parser.add_argument(
        "--log-dir",
        "--run-dir",
        "--video-dir",
        dest="run_dir",
        type=Path,
        default=DEFAULT_RUN_DIRECTORY,
        help=(
            "Base directory for per-run recordings, configuration, and event logs "
            f"(default: {DEFAULT_RUN_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable run video recording",
    )
    parser.add_argument(
        "--audio-input",
        "--send-audio",
        action="store_true",
        help=(
            "Send each available ViZDoom audio buffer as WAV input in addition "
            "to its text summary; requires a model accepting both image and audio input"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="OpenRouter request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-json-schema",
        action="store_true",
        help="Do not request structured outputs (for models without schema support)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt details for the first frame without calling OpenRouter",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.max_decisions < 0:
        parser.error("--max-decisions cannot be negative")
    if args.history_frames < 0:
        parser.error("--history-frames cannot be negative")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed cannot be negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)
    )


def screen_to_png_data_url(screen: np.ndarray) -> str:
    """Encode an HWC uint8 Gymnasium screen observation without extra packages."""
    image = np.asarray(screen)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[2] not in (1, 3):
        raise ValueError(f"Expected an HxWx1 or HxWx3 screen buffer, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 screen buffer, got {image.dtype}")

    image = np.ascontiguousarray(image)
    height, width, channels = image.shape
    color_type = 0 if channels == 1 else 2
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", header)
    png += _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
    png += _png_chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def get_binary_action_space(env: gymnasium.Env) -> gymnasium.spaces.MultiBinary:
    action_space = env.action_space
    if isinstance(action_space, gymnasium.spaces.MultiBinary):
        return action_space
    if isinstance(action_space, gymnasium.spaces.Dict):
        binary_space = action_space.spaces.get("binary")
        if isinstance(binary_space, gymnasium.spaces.MultiBinary):
            return binary_space
    raise TypeError(
        f"{env.spec.id if env.spec else 'Environment'} must expose a MultiBinary "
        f"action space, got {action_space!r}"
    )


def get_binary_button_names(env: gymnasium.Env) -> list[str]:
    """Return binary buttons in the order expected by the Gymnasium wrapper."""
    unwrapped = cast(Any, env.unwrapped)
    buttons = unwrapped.game.get_available_buttons()
    binary_buttons = buttons[unwrapped.num_delta_buttons :]
    button_names = [button.name for button in binary_buttons]
    binary_space = get_binary_action_space(env)
    if binary_space.shape != (len(button_names),):
        raise RuntimeError(
            f"Binary action shape {binary_space.shape} does not match buttons "
            f"{button_names!r}"
        )
    return button_names


def get_audio_sampling_rate(env: gymnasium.Env) -> int | None:
    """Return the configured sampling rate when the scenario exposes audio."""
    unwrapped = cast(Any, env.unwrapped)
    if not unwrapped.game.is_audio_buffer_enabled():
        return None
    return int(unwrapped.game.get_audio_sampling_rate())


def get_game_variable_names(env: gymnasium.Env) -> list[str]:
    unwrapped = cast(Any, env.unwrapped)
    return [variable.name for variable in unwrapped.game.get_available_game_variables()]


def uses_neutral_continuous_action(env: gymnasium.Env) -> bool:
    return isinstance(env.action_space, gymnasium.spaces.Dict)


def build_env_action(
    env: gymnasium.Env, button_names: list[str], pressed_buttons: list[str]
) -> np.ndarray | dict[str, np.ndarray]:
    """Translate a list of button names into the wrapper's action structure."""
    binary_space = get_binary_action_space(env)
    pressed_set = set(pressed_buttons)
    binary_action = np.asarray(
        [int(button in pressed_set) for button in button_names],
        dtype=binary_space.dtype,
    )

    if isinstance(env.action_space, gymnasium.spaces.Dict):
        continuous_space = env.action_space.spaces.get("continuous")
        if not isinstance(continuous_space, gymnasium.spaces.Box):
            raise TypeError("Dict action space has no continuous Box entry")
        action: np.ndarray | dict[str, np.ndarray] = {
            "binary": binary_action,
            "continuous": np.zeros(
                continuous_space.shape, dtype=continuous_space.dtype
            ),
        }
    else:
        action = binary_action

    if not env.action_space.contains(action):
        raise RuntimeError(
            f"Constructed action is outside the action space: {action!r}"
        )
    return action


def format_environment_description(description: dict[str, Any]) -> str:
    rewards = "\n".join(f"- {reward}" for reward in description["rewards"])
    return (
        f"{description['name']}: {description['description']}\n"
        f"Objective: {description['objective']}\n"
        f"Rewards:\n{rewards}\n"
        f"Episode end: {description['episode_end']}"
    )


def make_first_message(
    environment_id: str,
    description: dict[str, Any],
    button_names: list[str],
    history_frames: int,
    justify_actions: bool = False,
) -> dict[str, str]:
    buttons = "\n".join(f"- {button}" for button in button_names)
    if justify_actions:
        expected_output = (
            'Return only {"justification": "Brief observation supporting the '
            'action", "buttons": ["BUTTON_NAME"]}, with the justification first.'
        )
    else:
        expected_output = (
            'Return only {"buttons": ["BUTTON_NAME"]} for every new frame.'
        )

    return {
        "role": "system",
        "content": (
            f"You are the player in {environment_id}.\n\n"
            f"Environment description:\n{format_environment_description(description)}"
            "\n\nThe conversation is ordered from oldest to newest. Earlier user "
            f"messages contain up to {history_frames} historical frames, each "
            "followed by the assistant action chosen for that frame. The final user "
            "message contains the current frame; choose the next action for it."
            f"\n\nAvailable buttons:\n{buttons}\n\n"
            f"{expected_output} Use only listed button names, without duplicates. "
            "An empty list presses nothing."
        ),
    }


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.4f}"


def summarize_audio(audio: np.ndarray) -> str:
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.size == 0:
        return f"audio_shape={samples.shape}"
    normalized = samples / 32768.0
    rms = np.sqrt(np.mean(np.square(normalized), axis=0))
    peak = np.max(np.abs(normalized), axis=0)
    return (
        f"stereo_audio_rms(left={rms[0]:.5f}, right={rms[1]:.5f}), "
        f"stereo_audio_peak(left={peak[0]:.5f}, right={peak[1]:.5f})"
    )


def audio_to_wav_base64(audio: np.ndarray, sampling_rate: int) -> str:
    """Encode an interleaved int16 ViZDoom audio buffer as a base64 WAV."""
    samples = np.asarray(audio)
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.size == 0:
        raise ValueError(
            f"Expected a non-empty stereo audio buffer, got {samples.shape}"
        )
    if sampling_rate <= 0:
        raise ValueError(f"Audio sampling rate must be positive, got {sampling_rate}")

    pcm = np.ascontiguousarray(samples.astype("<i2", copy=False))
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(pcm.tobytes())
    return base64.b64encode(wav_bytes.getvalue()).decode("ascii")


def format_observation_context(
    observation: dict[str, Any],
    game_variable_names: list[str],
    latest_notification: str,
) -> str:
    context = []
    variables = observation.get("gamevariables")
    if variables is not None:
        values = np.asarray(variables).flat
        formatted_variables = ", ".join(
            f"{name}={format_number(float(value))}"
            for name, value in zip(game_variable_names, values)
        )
        context.append(f"game_variables({formatted_variables})")
    if "audio" in observation:
        context.append(summarize_audio(observation["audio"]))
    if "notifications" in observation:
        notification = latest_notification or "(none received yet)"
        context.append(f"latest_notification={json.dumps(notification)}")
    return "; ".join(context) if context else "no additional state values"


def make_frame_message(
    observation: dict[str, Any],
    game_variable_names: list[str],
    latest_notification: str,
    decision_number: int,
    previous_buttons: list[str] | None,
    previous_reward: float | None,
    episode_return: float,
    send_audio: bool = False,
    audio_sampling_rate: int | None = None,
) -> tuple[dict[str, Any], int]:
    image_url = screen_to_png_data_url(observation["screen"])
    context = format_observation_context(
        observation, game_variable_names, latest_notification
    )
    if previous_buttons is None or previous_reward is None:
        transition = "This is the initial frame; no previous buttons were pressed."
    else:
        transition = (
            f"Previous buttons: {json.dumps(previous_buttons)}. Reward since that "
            f"decision: {format_number(previous_reward)}."
        )
    text = (
        f"New frame {decision_number}. Current state: {context}. {transition} "
        f"Episode return so far: {format_number(episode_return)}. Select the next "
        "action."
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    if send_audio and "audio" in observation:
        if audio_sampling_rate is None:
            raise RuntimeError("The scenario provides audio without a sampling rate")
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_to_wav_base64(
                        observation["audio"], audio_sampling_rate
                    ),
                    "format": "wav",
                },
            }
        )
    message = {
        "role": "user",
        "content": content,
    }
    return message, len(image_url)


def make_response_format(
    button_names: list[str], justify_actions: bool = False
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required = []
    if justify_actions:
        properties["justification"] = {"type": "string"}
        required.append("justification")
    properties["buttons"] = {
        "type": "array",
        "items": {"type": "string", "enum": button_names},
    }
    required.append("buttons")

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vizdoom_buttons",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def extract_message_text(response: dict[str, Any]) -> str:
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        error = response.get("error", response)
        raise OpenRouterError(f"Response has no assistant message: {error!r}") from exc

    if content is None:
        diagnostics = {
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "refusal": message.get("refusal"),
            "reasoning_present": bool(
                message.get("reasoning") or message.get("reasoning_details")
            ),
            "usage": response.get("usage"),
        }
        raise OpenRouterError(
            "OpenRouter returned no assistant text. Response diagnostics: "
            f"{diagnostics!r}"
        )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    raise OpenRouterError(f"Unexpected assistant content: {content!r}")


def parse_action_response(
    message_text: str,
    button_names: list[str],
    justify_actions: bool = False,
) -> tuple[list[str], str | None]:
    text = message_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        output = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(
            f"Model did not return valid JSON: {message_text!r}"
        ) from exc

    expected_keys = {"buttons", "justification"} if justify_actions else {"buttons"}
    if not isinstance(output, dict) or set(output) != expected_keys:
        raise OpenRouterError(
            f"Model output must contain exactly {sorted(expected_keys)!r}: {output!r}"
        )

    justification = output.get("justification")
    if justify_actions and (
        not isinstance(justification, str) or not justification.strip()
    ):
        raise OpenRouterError(
            f'"justification" must be a non-empty string: {justification!r}'
        )
    buttons = output["buttons"]
    if not isinstance(buttons, list) or not all(
        isinstance(button, str) for button in buttons
    ):
        raise OpenRouterError(f'"buttons" must be a list of strings: {buttons!r}')
    if len(buttons) != len(set(buttons)):
        raise OpenRouterError(f"Model returned duplicate buttons: {buttons!r}")
    invalid_buttons = [button for button in buttons if button not in button_names]
    if invalid_buttons:
        raise OpenRouterError(
            f"Model returned unavailable buttons {invalid_buttons!r}; available "
            f"buttons are {button_names!r}"
        )
    return buttons, justification.strip() if isinstance(justification, str) else None


def parse_buttons(message_text: str, button_names: list[str]) -> list[str]:
    """Parse the original button-only response format."""
    buttons, _ = parse_action_response(message_text, button_names)
    return buttons


def request_buttons(
    api_key: str,
    model: str,
    first_message: dict[str, str],
    history_messages: list[dict[str, Any]],
    frame_message: dict[str, Any],
    button_names: list[str],
    request_timeout: float,
    use_json_schema: bool,
    reasoning_effort: str,
    justify_actions: bool = False,
) -> tuple[list[str], str | None, str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [first_message, *history_messages, frame_message],
        "temperature": 0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": reasoning_effort, "exclude": True},
    }
    if use_json_schema:
        payload["response_format"] = make_response_format(button_names, justify_actions)

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "ViZDoom OpenRouter experiment",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as result:
            response = json.loads(result.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise OpenRouterError(f"Could not reach OpenRouter: {reason}") from exc
    except json.JSONDecodeError as exc:
        raise OpenRouterError("OpenRouter returned a non-JSON response") from exc

    message_text = extract_message_text(response)
    buttons, justification = parse_action_response(
        message_text, button_names, justify_actions
    )
    usage = response.get("usage", {})
    return buttons, justification, message_text, usage


def append_frame_to_history(
    history_messages: list[dict[str, Any]],
    frame_message: dict[str, Any],
    assistant_output: str,
    history_frames: int,
) -> None:
    """Append one exchange and retain at most ``history_frames`` old images."""
    if history_frames == 0:
        history_messages.clear()
        return

    history_messages.extend(
        [frame_message, {"role": "assistant", "content": assistant_output}]
    )
    maximum_messages = history_frames * 2
    if len(history_messages) > maximum_messages:
        del history_messages[: len(history_messages) - maximum_messages]


def print_dry_run(
    first_message: dict[str, str],
    frame_message: dict[str, Any],
    encoded_image_characters: int,
    response_format: dict[str, Any],
) -> None:
    frame_text = frame_message["content"][0]["text"]
    screen_url = frame_message["content"][1]["image_url"]["url"]
    print("First API message:\n")
    print(first_message["content"])
    print("\nFirst frame message:\n")
    print(frame_text)
    print(
        f"\nImage: {screen_url[:22]}... "
        f"({encoded_image_characters:,} data-URL characters)"
    )
    audio_parts = [
        part for part in frame_message["content"] if part.get("type") == "input_audio"
    ]
    if audio_parts:
        audio_characters = len(audio_parts[0]["input_audio"]["data"])
        print(f"Audio: WAV ({audio_characters:,} base64 characters)")
    print("\nRequested response schema:\n")
    print(json.dumps(response_format, indent=2))


def record_first_episode(episode: int) -> bool:
    """Start one recording which remains open across every run episode."""
    return episode == 0


def choose_base_seed(configured_seed: int | None) -> int:
    return secrets.randbits(32) if configured_seed is None else configured_seed


def timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def filename_component(value: str) -> str:
    component = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    )
    component = component.strip("-_.")
    return component or "unknown"


def create_run_log_paths(
    run_directory: Path,
    environment_id: str,
    model: str,
    reasoning_effort: str,
    base_seed: int,
    justify_actions: bool = False,
) -> tuple[Path, Path]:
    """Create a directory containing the three artifacts from one run."""
    directory = run_directory.expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = (
        f"{filename_component(environment_id)}"
        f"__model-{filename_component(model)}"
        f"__reasoning-{filename_component(reasoning_effort)}"
        f"__seed-{base_seed}"
        f"__time-{run_id}-{os.getpid()}"
        f"__justification-{'enabled' if justify_actions else 'disabled'}"
    )
    directory = directory / run_name
    directory.mkdir(parents=True, exist_ok=False)
    return directory / "config.json", directory / "log.jsonl"


def write_initial_config(config_path: Path, config: dict[str, Any]) -> None:
    """Atomically write the immutable configuration for one run."""
    temporary_path = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")
        temporary_path.replace(config_path)
    except OSError as exc:
        raise RuntimeError(
            f"Could not write run configuration to {config_path}: {exc}"
        ) from exc


def append_jsonl_event(event_log_path: Path, event: dict[str, Any]) -> None:
    """Append and flush one self-contained event to the run's JSONL file."""
    try:
        with event_log_path.open("a", encoding="utf-8") as event_log:
            json.dump(event, event_log, ensure_ascii=False, separators=(",", ":"))
            event_log.write("\n")
            event_log.flush()
            os.fsync(event_log.fileno())
    except OSError as exc:
        raise RuntimeError(
            f"Could not append run event to {event_log_path}: {exc}"
        ) from exc


def finalize_recording(video_directory: Path) -> None:
    """Give Gymnasium's single run recording the stable artifact filename."""
    generated_recording = video_directory / "recording-episode-0.mp4"
    if generated_recording.exists():
        generated_recording.replace(video_directory / "recording.mp4")


def create_environment(
    args: argparse.Namespace,
    run_directory: Path,
) -> tuple[gymnasium.Env, Path | None]:
    """Create the base environment and optional Gymnasium video wrappers."""
    record_video = not args.no_video and not args.dry_run
    if record_video:
        render_mode = "rgb_array"
    else:
        render_mode = "human" if args.render else None

    env = gymnasium.make(
        args.env,
        frame_skip=FRAME_SKIP,
        render_mode=render_mode,
    )
    if not record_video:
        return env, None

    render_fps = int(env.metadata.get("render_fps", 30))
    video_fps = max(1, round(render_fps / FRAME_SKIP))
    video_directory = run_directory
    env = gymnasium.wrappers.RecordVideo(
        env,
        video_folder=str(video_directory),
        episode_trigger=record_first_episode,
        video_length=sys.maxsize,
        name_prefix="recording",
        fps=video_fps,
    )
    if args.render:
        env = gymnasium.wrappers.HumanRendering(env)
    return env, video_directory


def run(
    args: argparse.Namespace,
    descriptions: dict[str, dict[str, Any]] | None = None,
) -> None:
    if descriptions is None:
        descriptions = load_environment_descriptions()
    if args.env not in descriptions:
        raise RuntimeError(f"No description is available for {args.env!r}")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.dry_run and not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Export it in the environment, or use "
            "--dry-run to inspect the prompt without an API call."
        )

    started_at = timestamp_now()
    base_seed = choose_base_seed(args.seed)
    config_path, event_log_path = create_run_log_paths(
        args.run_dir,
        args.env,
        args.model,
        args.reasoning_effort,
        base_seed,
        args.justify_actions,
    )
    env: gymnasium.Env | None = None
    video_directory: Path | None = None
    event_logging_started = False
    current_episode: int | None = None
    current_decision: int | None = None
    completed_episodes = 0
    try:
        env, video_directory = create_environment(args, config_path.parent)
        button_names = get_binary_button_names(env)
        game_variable_names = get_game_variable_names(env)
        audio_sampling_rate = get_audio_sampling_rate(env)
        neutral_continuous_action = uses_neutral_continuous_action(env)
        first_message = make_first_message(
            args.env,
            descriptions[args.env],
            button_names,
            args.history_frames,
            args.justify_actions,
        )
        response_format = (
            make_response_format(button_names, args.justify_actions)
            if not args.no_json_schema
            else None
        )
        initial_config: dict[str, Any] = {
            "schema_version": 1,
            "created_at": started_at,
            "event_log": str(event_log_path),
            "recording": (
                str(config_path.parent / "recording.mp4")
                if not args.no_video and not args.dry_run
                else None
            ),
            "environment": args.env,
            "environment_description": descriptions[args.env],
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "justify_actions": args.justify_actions,
            "frame_skip": FRAME_SKIP,
            "history_frames": args.history_frames,
            "episodes": args.episodes,
            "max_decisions": args.max_decisions,
            "request_timeout_seconds": args.request_timeout,
            "json_schema": not args.no_json_schema,
            "response_format": response_format,
            "video_enabled": not args.no_video and not args.dry_run,
            "video_directory": str(video_directory) if video_directory else None,
            "audio_input_requested": args.audio_input,
            "audio_input_enabled": (
                args.audio_input and audio_sampling_rate is not None
            ),
            "audio_sampling_rate": audio_sampling_rate,
            "render_enabled": args.render,
            "dry_run": args.dry_run,
            "base_seed": base_seed,
            "base_seed_was_random": args.seed is None,
            "available_buttons": button_names,
            "continuous_delta_controls_held_at_zero": neutral_continuous_action,
            "system_prompt": first_message["content"],
        }
        write_initial_config(config_path, initial_config)
        append_jsonl_event(
            event_log_path,
            {
                "event": "run_started",
                "timestamp": started_at,
                "config_file": str(config_path),
            },
        )
        event_logging_started = True
        print(f"Run configuration: {config_path}")
        print(f"Appending events to: {event_log_path}")
        print(f"Environment: {args.env} (frame_skip={FRAME_SKIP})")
        print(f"Model: {args.model}")
        print(f"Reasoning effort: {args.reasoning_effort}")
        print(
            f"Action justification: {'enabled' if args.justify_actions else 'disabled'}"
        )
        if audio_sampling_rate is not None:
            audio_mode = (
                "WAV input and text summary" if args.audio_input else "text summary"
            )
            print(f"Audio: {audio_mode} ({audio_sampling_rate} Hz)")
        elif args.audio_input:
            print("Audio: requested, but this scenario has no audio buffer")
        print(f"Base seed: {base_seed}")
        print(f"History: {args.history_frames} completed frames plus the current frame")
        if video_directory is not None:
            print(f"Recording videos to: {video_directory}")
        print("Binary buttons: " + ", ".join(button_names))
        if neutral_continuous_action:
            print("Continuous delta controls: held at zero")

        for episode in range(args.episodes):
            current_episode = episode + 1
            episode_seed = base_seed + episode
            observation, _ = env.reset(seed=episode_seed)
            episode_return = 0.0
            previous_buttons = None
            previous_reward = None
            latest_notification = ""
            decision_number = 0
            history_messages: list[dict[str, Any]] = []
            append_jsonl_event(
                event_log_path,
                {
                    "event": "episode_started",
                    "timestamp": timestamp_now(),
                    "episode": current_episode,
                    "seed": episode_seed,
                },
            )

            while True:
                current_decision = decision_number
                notification = observation.get("notifications", "")
                if isinstance(notification, str) and notification.strip():
                    latest_notification = notification.strip()
                frame_message, image_characters = make_frame_message(
                    observation,
                    game_variable_names,
                    latest_notification,
                    decision_number,
                    previous_buttons,
                    previous_reward,
                    episode_return,
                    args.audio_input,
                    audio_sampling_rate,
                )
                if args.dry_run:
                    dry_run_response_format = make_response_format(
                        button_names, args.justify_actions
                    )
                    print_dry_run(
                        first_message,
                        frame_message,
                        image_characters,
                        dry_run_response_format,
                    )
                    append_jsonl_event(
                        event_log_path,
                        {
                            "event": "dry_run_preview",
                            "timestamp": timestamp_now(),
                            "episode": current_episode,
                            "decision": current_decision,
                            "frame_text": frame_message["content"][0]["text"],
                            "image_data_url_characters": image_characters,
                            "response_format": dry_run_response_format,
                        },
                    )
                    append_jsonl_event(
                        event_log_path,
                        {
                            "event": "run_finished",
                            "timestamp": timestamp_now(),
                            "status": "dry_run",
                            "episodes_completed": 0,
                        },
                    )
                    return

                append_jsonl_event(
                    event_log_path,
                    {
                        "event": "action_request_started",
                        "timestamp": timestamp_now(),
                        "episode": current_episode,
                        "decision": current_decision,
                        "frame_text": frame_message["content"][0]["text"],
                        "image_data_url_characters": image_characters,
                    },
                )
                request_started = time.perf_counter()
                pressed_buttons, justification, raw_output, usage = request_buttons(
                    api_key,
                    args.model,
                    first_message,
                    history_messages,
                    frame_message,
                    button_names,
                    args.request_timeout,
                    not args.no_json_schema,
                    args.reasoning_effort,
                    args.justify_actions,
                )
                api_latency = time.perf_counter() - request_started
                append_jsonl_event(
                    event_log_path,
                    {
                        "event": "action_selected",
                        "timestamp": timestamp_now(),
                        "episode": current_episode,
                        "decision": current_decision,
                        "justification": justification,
                        "buttons": pressed_buttons,
                        "raw_model_output": raw_output,
                        "usage": usage,
                        "api_latency_seconds": round(api_latency, 6),
                    },
                )
                append_frame_to_history(
                    history_messages,
                    frame_message,
                    raw_output,
                    args.history_frames,
                )
                if justification is not None:
                    print(
                        f"episode={episode + 1} decision={decision_number} "
                        f"justification={justification}",
                        flush=True,
                    )
                    print(
                        f"episode={episode + 1} decision={decision_number} "
                        f"applying_buttons={pressed_buttons!r}",
                        flush=True,
                    )
                action = build_env_action(env, button_names, pressed_buttons)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                append_jsonl_event(
                    event_log_path,
                    {
                        "event": "environment_step",
                        "timestamp": timestamp_now(),
                        "episode": current_episode,
                        "decision": current_decision,
                        "buttons": pressed_buttons,
                        "reward": float(reward),
                        "episode_return": episode_return,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    },
                )
                current_decision = None
                print(
                    f"episode={episode + 1} decision={decision_number} "
                    f"buttons={pressed_buttons!r} reward={float(reward):.4f} "
                    f"return={episode_return:.4f} model_output={raw_output!r} "
                    f"usage={usage}"
                )

                previous_buttons = pressed_buttons
                previous_reward = float(reward)
                decision_number += 1
                reached_limit = (
                    args.max_decisions > 0 and decision_number >= args.max_decisions
                )
                if terminated or truncated or reached_limit:
                    reason = (
                        "terminated"
                        if terminated
                        else "truncated" if truncated else "decision limit"
                    )
                    print(
                        f"Episode {episode + 1} finished ({reason}) after "
                        f"{decision_number} decisions; return={episode_return:.4f}"
                    )
                    append_jsonl_event(
                        event_log_path,
                        {
                            "event": "episode_finished",
                            "timestamp": timestamp_now(),
                            "episode": current_episode,
                            "end_reason": reason,
                            "decisions": decision_number,
                            "return": episode_return,
                        },
                    )
                    break
            completed_episodes += 1
            current_episode = None
        append_jsonl_event(
            event_log_path,
            {
                "event": "run_finished",
                "timestamp": timestamp_now(),
                "status": "completed",
                "episodes_completed": completed_episodes,
            },
        )
    except BaseException as exc:
        failure_status = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if event_logging_started:
            try:
                append_jsonl_event(
                    event_log_path,
                    {
                        "event": "run_finished",
                        "timestamp": timestamp_now(),
                        "status": failure_status,
                        "episodes_completed": completed_episodes,
                        "active_episode": current_episode,
                        "active_decision": current_decision,
                        "error": error,
                    },
                )
            except RuntimeError:
                pass
        raise
    finally:
        if env is not None:
            try:
                env.close()
            finally:
                if video_directory is not None:
                    finalize_recording(video_directory)


def main() -> None:
    try:
        descriptions = load_environment_descriptions()
        args = parse_args(sorted(descriptions))
        run(args, descriptions)
    except (OpenRouterError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
