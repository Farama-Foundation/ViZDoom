"""
Stereo sound: Each frame is [R, G, B, audio_left, audio_right]. It's easier to fit in the existing codebase this way.

Audio rows run from 80 Hz to 11025 Hz on geometric grid (log scale frequency), columns run oldest to newest across the last 8 tics (we chose this for STFT frames).

Spectrograms are computed directly from the raw int16 stereo audio buffer via
512 samples Hann STFT with 128 samples hops interpolated to the image dimensions. (https://medium.com/@ongzhixuan/exploring-the-short-time-fourier-transform-analyzing-time-varying-audio-signals-98157d1b9a12).
Both ears use the same fixed [-60, 0] dBFS magnitude scale mapped to [0, 255], with no per-ear or per-frame normalization. Distance attenuation and Interaural Level Differences (ILD) become additive intensity differences above the floor.
Silence, missing audio, and magnitudes <= -60 dBFS map to zero.

We append this to observation CNN so it scales these planes to [0, 1], so the level of frequency is visible/learnable but difference between phases/distances is not.
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
# However, I tried -80 dB before and the frequency was eaten by normalization, so we shouldn't set the maximum of an observation or ear.
# One uint8 step is 60 / 255 = 0.235 dB, which 4x amplitude adds about 51 steps.
_DB_FLOOR = -60.0

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
    """Return (h, w, 2) uint8 left/right log-frequency dBFS planes.

    Input is int16 stereo PCM sampled at AUDIO_SAMPLE_RATE of any length.
    Columns interpolate the supplied buffer's STFT frames from oldest to newest.
    Both ears share a fixed [-60, 0] dBFS scale without automatic gain control.
    A bin centered full scale sine has 0 dBFS magnitude before interpolation.
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
    # We compress with log here, it shows weak components without erasing absolute levels.
    db = 20 * np.log10(np.maximum(magnitude, 10 ** (_DB_FLOOR / 20)))
    return np.rint(np.clip((db - _DB_FLOOR) / -_DB_FLOOR, 0, 1) * 255).astype(np.uint8)


def plot_distance_comparison(
    audio_buffers: [np.ndarray],
    distances: [float],
    output_path: str | Path,
) -> Path:
    """Plot raw waveforms and processed STFT observations over distance.

    The three detailed rows are selected from audible buffers near the far,
    middle, and close distance terciles. All waveform and spectrogram axes use
    shared fixed scales, so attenuation remains directly visible.
    """
    if len(audio_buffers) != len(distances) or not audio_buffers:
        raise ValueError("audio_buffers and distances must have equal non-zero length")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Audio visualization requires matplotlib (pip install matplotlib)"
        ) from exc

    buffers = [np.asarray(buffer) for buffer in audio_buffers]
    for buffer in buffers:
        if buffer.dtype != np.int16 or buffer.ndim != 2 or buffer.shape[1] != 2:
            raise ValueError("each audio buffer must be int16 with shape (N, 2)")
    distance_values = np.asarray(distances, dtype=np.float64)
    if not np.all(np.isfinite(distance_values)):
        raise ValueError("distances must be finite")

    rms = np.asarray(
        [
            np.sqrt(np.mean((buffer.astype(np.float64) / 32768.0) ** 2))
            for buffer in buffers
        ]
    )
    audible = np.flatnonzero(rms > 0)
    if len(audible) < 3:
        raise ValueError("at least three non-silent audio buffers are required")

    distance_edges = np.linspace(
        distance_values[audible].min(), distance_values[audible].max(), 4
    )
    selected = []
    for edge_index in range(2, -1, -1):
        in_band = audible[
            (distance_values[audible] >= distance_edges[edge_index])
            & (distance_values[audible] <= distance_edges[edge_index + 1])
        ]
        if not len(in_band):
            raise ValueError("audio buffers must cover three distinct distance bands")
        strong = in_band[rms[in_band] >= 0.9 * rms[in_band].max()]
        selected.append(strong[np.argmin(distance_values[strong])])

    figure, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    axes[0, 0].scatter(distance_values[audible], rms[audible], s=14, alpha=0.7)
    axes[0, 0].set(
        title="Raw audio level as prey approaches",
        xlabel="Prey distance (map units)",
        ylabel="Stereo RMS amplitude",
    )
    axes[0, 0].invert_xaxis()

    stft_levels = np.asarray(
        [stereo_spectrogram(buffer, 120, 160).max() for buffer in buffers]
    )
    axes[0, 1].scatter(distance_values[audible], stft_levels[audible], s=14, alpha=0.7)
    axes[0, 1].set(
        title="Processed STFT level as prey approaches",
        xlabel="Prey distance (map units)",
        ylabel="Peak uint8 dBFS intensity",
        ylim=(0, 255),
    )
    axes[0, 1].invert_xaxis()

    image = None
    for row, index in enumerate(selected, start=1):
        buffer = buffers[index]
        times_ms = (
            (np.arange(len(buffer), dtype=np.float64) - len(buffer) + 1)
            / AUDIO_SAMPLE_RATE
            * 1000
        )
        axes[row, 0].plot(times_ms, buffer[:, 0] / 32768.0, label="left", lw=0.8)
        axes[row, 0].plot(
            times_ms, buffer[:, 1] / 32768.0, label="right", lw=0.8, alpha=0.8
        )
        axes[row, 0].set(
            title=f"Raw waveform — distance {distance_values[index]:.1f}",
            xlabel="Time before observation (ms)",
            ylabel="Amplitude",
            ylim=(-1, 1),
        )
        axes[row, 0].legend(loc="upper right")

        spectrogram = stereo_spectrogram(buffer, 120, 160).mean(axis=2)
        # Plot the actual policy grid at STFT window-center times.
        first_center = _FFT_SIZE / 2 - max(0, _FFT_SIZE - len(buffer))
        last_center = len(buffer) - _FFT_SIZE / 2
        last_center = max(first_center, last_center)
        stft_times = (
            (np.linspace(first_center, last_center, 160) - len(buffer))
            / AUDIO_SAMPLE_RATE
            * 1000
        )
        image = axes[row, 1].pcolormesh(
            stft_times,
            np.geomspace(_MIN_FREQUENCY, AUDIO_SAMPLE_RATE / 2, 120),
            spectrogram,
            shading="nearest",
            vmin=0,
            vmax=255,
            cmap="magma",
        )
        axes[row, 1].set_yscale("log")
        axes[row, 1].set(
            title=f"Mean stereo STFT — distance {distance_values[index]:.1f}",
            xlabel="Time before observation (ms)",
            ylabel="Frequency (Hz)",
        )

    assert image is not None
    figure.colorbar(
        image, ax=axes[1:, 1], label="Fixed dBFS intensity: 0 = ≤−60 dBFS, 255 = 0 dBFS"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
