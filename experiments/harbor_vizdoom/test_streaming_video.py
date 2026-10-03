"""Exercise bounded-memory encoding and RecordVideo's recording boundaries."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium
import imageio_ffmpeg
import numpy as np


def _load_module():
    server_dir = Path(__file__).parent / "template/common/environment/vizdoom-server"
    spec = importlib.util.spec_from_file_location(
        "streaming_video", server_dir / "streaming_video.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()
StreamingRecordVideo = module.StreamingRecordVideo


class ImageEnvironment(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 9}
    render_mode = "rgb_array"
    observation_space = gymnasium.spaces.Box(0, 255, (64, 128, 3), np.uint8)
    action_space = gymnasium.spaces.Discrete(1)

    def __init__(self) -> None:
        self.frame = np.zeros(self.observation_space.shape, np.uint8)
        self.frame[:, :64, 0] = 255
        self.frame[:, 64:, 2] = 255
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.frame, {}

    def step(self, action):
        self.steps += 1
        return self.frame, 0.0, False, False, {}

    def render(self):
        # Fresh render arrays expose accidental retention of all old frames.
        return self.frame.copy()


class StreamingVideoTests(unittest.TestCase):
    def test_streamed_episode_is_decodable_without_retaining_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "video"
            env = StreamingRecordVideo(
                ImageEnvironment(),
                video_folder=str(folder),
                episode_trigger=lambda episode: episode == 0,
                name_prefix="recording",
            )
            try:
                env.reset(seed=123)
                for _ in range(300):
                    env.step(0)
                    self.assertEqual(env.recorded_frames, [])
                    self.assertEqual(env.render_history, [])
                self.assertEqual(env.recorded_frame_count, 301)
            finally:
                env.close()
            video = folder / "recording-episode-0.mp4"
            self.assertTrue(video.is_file())
            self.assertEqual(list(folder.iterdir()), [video])
            self.assertGreater(video.stat().st_size, 0)
            decode = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(video),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                capture_output=True,
                check=True,
            )
            self.assertEqual(decode.stderr, b"")
            self.assertEqual(len(decode.stdout), 301 * 64 * 128 * 3)
            decoded = np.frombuffer(decode.stdout[: 64 * 128 * 3], np.uint8).reshape(
                64, 128, 3
            )
            self.assertGreater(decoded[:, :64, 0].mean(), 240)
            self.assertGreater(decoded[:, 64:, 2].mean(), 240)

    def test_finite_length_stops_at_recordvideo_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = StreamingRecordVideo(
                ImageEnvironment(),
                video_folder=str(Path(temporary) / "video"),
                episode_trigger=lambda episode: episode == 0,
                video_length=2,
            )
            try:
                env.reset()
                env.step(0)
                self.assertTrue(env.recording)
                env.step(0)
                self.assertFalse(env.recording)
                self.assertEqual(env.recorded_frame_count, 3)
                env.step(0)
                self.assertEqual(env.recorded_frame_count, 3)
            finally:
                env.close()

    def test_encoder_exit_failure_is_reported_on_finalize(self) -> None:
        encoder = MagicMock()
        encoder.stdin = io.BytesIO()
        encoder.wait.return_value = 1
        with tempfile.TemporaryDirectory() as temporary:
            env = StreamingRecordVideo(
                ImageEnvironment(),
                video_folder=str(Path(temporary) / "video"),
                episode_trigger=lambda episode: True,
            )
            with patch.object(module.subprocess, "Popen", return_value=encoder):
                env.reset()
                with self.assertRaisesRegex(RuntimeError, "encoder exited with code 1"):
                    env.close()
            self.assertFalse(env.recording)
            self.assertIsNone(env._encoder)
            self.assertIsNone(env._encoder_errors)


if __name__ == "__main__":
    unittest.main()
