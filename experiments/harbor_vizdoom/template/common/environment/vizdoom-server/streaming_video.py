"""A Gymnasium RecordVideo wrapper that encodes RGB frames as they arrive."""

from __future__ import annotations

import gc
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

import gymnasium
import imageio_ffmpeg
import numpy as np
from gymnasium.wrappers import RecordVideo


class StreamingRecordVideo(RecordVideo):
    """Keep RecordVideo's triggers and filenames without retaining an episode.

    ViZDoom uses ``rgb_array`` rendering, with screen and automap already
    concatenated by the environment. Each frame goes directly to FFmpeg's
    bounded stdin pipe. ``recorded_frames`` stays empty; ``recorded_frame_count``
    tracks the current recording and preserves RecordVideo's length semantics.
    """

    def __init__(self, env: gymnasium.Env, **kwargs: Any) -> None:
        if env.render_mode != "rgb_array":
            raise ValueError("StreamingRecordVideo requires rgb_array rendering")
        self._encoder: subprocess.Popen[bytes] | None = None
        self._encoder_errors: BinaryIO | None = None
        self._frame_shape: tuple[int, ...] | None = None
        self.recorded_frame_count = 0
        super().__init__(env, **kwargs)

    def start_recording(self, video_name: str) -> None:
        super().start_recording(video_name)
        self.recorded_frame_count = 0
        self._frame_shape = None

    def _open_encoder(self, frame: np.ndarray) -> None:
        assert self._video_name is not None
        height, width, _ = frame.shape
        self._frame_shape = frame.shape
        path = Path(self.video_folder) / f"{self._video_name}.mp4"
        # Stderr goes to a temporary file so encoder diagnostics cannot fill a
        # pipe and block gameplay. No individual image files are written.
        self._encoder_errors = tempfile.TemporaryFile()
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.frames_per_sec),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-threads",
            "1",
            "-crf",
            "23",
            str(path),
        ]
        try:
            self._encoder = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._encoder_errors,
            )
        except BaseException:
            self._encoder_errors.close()
            self._encoder_errors = None
            raise

    def _capture_frame(self) -> None:
        assert self.recording, "Cannot capture a frame before recording starts"
        frame = self.env.render()
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError("Video frames must be HxWx3 uint8 RGB arrays")
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            raise ValueError("H.264 video frame dimensions must be even")
        if self._encoder is None:
            self._open_encoder(frame)
        elif frame.shape != self._frame_shape:
            raise ValueError("Video frame dimensions changed during recording")
        assert self._encoder is not None and self._encoder.stdin is not None
        frame = np.ascontiguousarray(frame)
        try:
            self._encoder.stdin.write(memoryview(frame).cast("B"))
        except (BrokenPipeError, OSError):
            # Surface the encoder's error instead of marking a broken video as
            # finalized; stop_recording also closes all process resources.
            self.stop_recording()
            raise
        self.recorded_frame_count += 1
        if self.recorded_frame_count > self.video_length:
            self.stop_recording()

    def _close_encoder(self) -> None:
        encoder = self._encoder
        errors = self._encoder_errors
        self._encoder = None
        self._encoder_errors = None
        if encoder is None:
            return
        try:
            if encoder.stdin is not None:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                returncode = encoder.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                encoder.kill()
                encoder.wait()
                raise RuntimeError("Video encoder did not finalize within 30s") from exc
            if returncode:
                assert errors is not None
                errors.seek(0)
                diagnostics = errors.read(8192).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Video encoder exited with code {returncode}: {diagnostics}"
                )
        finally:
            if errors is not None:
                errors.close()

    def stop_recording(self) -> None:
        assert self.recording, "No recording was started"
        try:
            self._close_encoder()
        finally:
            self.recording = False
            self._video_name = None
            self.recorded_frames.clear()
            if self.gc_trigger and self.gc_trigger(self.episode_id):
                gc.collect()
