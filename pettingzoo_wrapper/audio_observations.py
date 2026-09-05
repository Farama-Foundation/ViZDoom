"""
Stereo sound: Each frame is [R, G, B, audio_left, audio_right]. It's easier to fit in the existing codebase this way.

Audio rows run from 80 Hz to 11025 Hz on geometric grid (log scale frequency), columns run oldest to newest across the last 8 tics (we chose this for STFT frames).

Spectrograms are computed directly from the raw int16 stereo audio buffer via
512 samples Hann STFT with 128 samples hops interpolated to the image dimensions. (https://medium.com/@ongzhixuan/exploring-the-short-time-fourier-transform-analyzing-time-varying-audio-signals-98157d1b9a12).
Both ears use the same fixed [-80, 0] dBFS scale mapped to [0, 255], with no
per-ear or per-frame normalization to preserve Interaural Level Differences (ILD) for stereo sound localization.
Silence or missing audio maps to zero.

Instead of STFT (short time fourier transform) we can raw 1D waveforms or Mel Spectrograms, but the former is hard to interpret, and the latter more "advanced"/complicated, which I think is overkill. Also STFT can be computed quite easily.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


# From setAudioSamplingRate in ViZDoomGame.cpp, vzd.SamplingRate.SR_22050
AUDIO_SAMPLE_RATE = 22050

# ticrate is 35, so each tic is 22050/35=630 samples, and 8 tics is 5040 samples (we use game.set_audio_buffer_size(8))
AUDIO_BUFFER_TICS = 8

# Fast Fourier Transform requires x^2 (2^9) (for radix-2 algo). At 22050 Hz, 512 samples is 512/22050 = 23.22 ms, with bins = 43.07 Hz apart.
_FFT_SIZE = 512

# 128-sample hop = 512 / 4 (75% window overlap, about 5.80 ms apart). 75% is good, for denser temporal sampling.
_HOP_SIZE = 128

# A geometric frequency grid needs a positive lower edge. Choose 80 Hz, but this may be set to lower as well idk.
_MIN_FREQUENCY = 80.0

# Full scale digital sine wave is defined as 0 dBFS (Decibels Relative to Full Scale). Ideal 16-bit linear PCM has full scale sine SQNR = 6.02 * 16 + 1.76 = 98.1 dB
# At -80 dBFS, amplitude ratio is 10^(-80/20) = 0.0001 (3.28 PCM quantization steps in peak amplitude), so mapping [-80, 0] dBFS linearly to [0, 255] gives 3.19 pixel intensity steps per dB in an 8-bit (uint8) plane.
_DB_FLOOR = -80.0

# If we cut raw audio chunks, the hard edges become square steps which cause noises. Thus we use Hann cosine window to tapers frame boundaries to zero.
_WINDOW = np.hanning(_FFT_SIZE).astype(np.float32)


def uses_audio_observations(config_path: str) -> bool:
    return Path(config_path).stem.lower() == "simple_tag_audio"


@lru_cache(maxsize=16)
def _frequency_interpolation(height: int):
    # Based on Nyquist limit, digital audio can only represent frequencies up to half the sample rate, so here it's f_max = sample_rate / 2 = 11025 Hz
    # Frequencies higher than this can't be represented without aliasing.
    frequencies = np.geomspace(_MIN_FREQUENCY, AUDIO_SAMPLE_RATE / 2, height)
    bins = frequencies * _FFT_SIZE / AUDIO_SAMPLE_RATE
    lower = np.floor(bins).astype(int)
    upper = np.minimum(lower + 1, _FFT_SIZE // 2)
    return lower, upper, (bins - lower)[:, None, None]


def stereo_spectrogram(
    audio_buffer: np.ndarray | None, height: int, width: int
) -> np.ndarray:
    """Return (h, w, 2) uint8 left/right log-frequency magnitude planes.

    Input is int16 stereo PCM sampled at AUDIO_SAMPLE_RATE of any length.
    Columns interpolate the supplied buffer's STFT frames from oldest to newest.
    Both ears share a fixed dBFS scale without automatic gain control (AGC).
    Short buffers are left-padded to one FFT window, wrong shapes are rejected.
    """
    if audio_buffer is None:
        return np.zeros((height, width, 2), dtype=np.uint8)
    audio = np.asarray(audio_buffer)
    if audio.dtype != np.int16 or audio.ndim != 2 or audio.shape[1] != 2:
        # ViZDoom audio buffer uses signed 16-bit stereo PCM in (N, 2) layout.
        raise ValueError(
            "audio_buffer must contain int16 samples in (N, 2) stereo layout"
        )
    if not audio.size or not np.any(audio):
        return np.zeros((height, width, 2), dtype=np.uint8)
    samples = audio.astype(np.float32) / 32768.0
    if len(samples) < _FFT_SIZE:
        samples = np.pad(samples, ((_FFT_SIZE - len(samples), 0), (0, 0)))
    starts = np.arange(0, len(samples) - _FFT_SIZE + 1, _HOP_SIZE)
    # Include the newest samples even when the buffer is not hop-aligned.
    if starts[-1] != len(samples) - _FFT_SIZE:
        starts = np.append(starts, len(samples) - _FFT_SIZE)
    windows = samples[starts[:, None] + np.arange(_FFT_SIZE)[None, :]]
    spectrum = np.abs(np.fft.rfft(windows * _WINDOW[None, :, None], axis=1))
    spectrum *= 2.0 / _WINDOW.sum()
    spectrum[:, (0, -1), :] *= 0.5
    magnitude = spectrum.transpose(1, 0, 2)
    lower, upper, weight = _frequency_interpolation(height)
    magnitude = magnitude[lower] * (1 - weight) + magnitude[upper] * weight
    times = np.linspace(starts[0], starts[-1], width)
    positions = np.interp(times, starts, np.arange(len(starts)))
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, len(starts) - 1)
    weight = (positions - left)[None, :, None]
    magnitude = magnitude[:, left] * (1 - weight) + magnitude[:, right] * weight
    # Without per-ear normalization or AGC (Automatic Gain Control), preserving relative left/right channel amplitude is essential for horizontal sound localization via Interaural Level Differences (ILD)
    db = 20 * np.log10(np.maximum(magnitude, 10 ** (_DB_FLOOR / 20)))
    return np.rint(np.clip((db - _DB_FLOOR) / -_DB_FLOOR, 0, 1) * 255).astype(np.uint8)
