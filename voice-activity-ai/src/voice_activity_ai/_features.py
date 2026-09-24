"""Per-frame measurements: how loud, how busy, how noise-like.

Energy on its own is a bad voice detector. A slammed door is loud and is not
speech; a whisper is quiet and is. So every frame is measured three ways and the
decision in :mod:`voice_activity_ai._detect` uses all three:

* **short-time energy** - how loud the frame is, in dBFS.
* **zero-crossing rate** - how often the waveform changes sign, which separates
  hiss and fricatives from voiced sound and from low rumble.
* **spectral flatness** - the ratio of the geometric to the arithmetic mean of
  the power spectrum, in dB. Noise spreads its power evenly and lands near 0 dB;
  speech has harmonics and formants, so its power is concentrated and it lands
  far below.

Frames are laid end to end with no overlap, so frame ``i`` covers samples
``[i * frame_length, (i + 1) * frame_length)`` and nothing is counted twice.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

EPS = 1e-20
"""Added inside every logarithm so digital silence gives a number, not ``-inf``."""

SILENT_DBFS = -100.0
"""Energy reported for a frame of exact zeros, and the clamp for anything below."""


def frame_signal(samples: np.ndarray, frame_length: int) -> np.ndarray:
    """Cut a 1-D signal into non-overlapping frames.

    Args:
        samples: 1-D float array.
        frame_length: Samples per frame; must be at least 1.

    Returns:
        A ``(n_frames, frame_length)`` view-backed array. A tail shorter than one
        frame is dropped, so a signal shorter than ``frame_length`` yields a
        ``(0, frame_length)`` array rather than an error.
    """
    if frame_length < 1:
        raise ValueError("frame_length must be at least 1 sample")
    n_frames = int(samples.size // frame_length)
    if n_frames == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    usable = n_frames * frame_length
    return np.ascontiguousarray(samples[:usable]).reshape(n_frames, frame_length)


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Root-mean-square level of every frame, in dBFS (``0`` = full scale)."""
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    power = np.mean(np.square(frames), axis=1)
    db = 10.0 * np.log10(power + EPS)
    return np.maximum(db, SILENT_DBFS)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Share of neighbouring sample pairs that change sign, in ``[0, 1]``.

    Each frame has its own mean removed first, so a DC offset or a slow drift
    cannot hide the crossings that are actually there.
    """
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)
    centred = frames - frames.mean(axis=1, keepdims=True)
    signs = np.signbit(centred)
    changes = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    return changes.astype(np.float64) / float(frames.shape[1] - 1)


def spectral_flatness(frames: np.ndarray) -> np.ndarray:
    """Spectral flatness of every frame in dB: ``0`` is noise-like, lower is tonal.

    The frame is Hann-windowed before the transform so that the frame edges do
    not smear energy across the whole spectrum and make everything look flat.
    """
    n_frames, frame_length = frames.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    if frame_length < 4:
        # Two or three samples carry no usable spectrum; call them noise-like
        # rather than inventing structure out of a two-bin transform.
        return np.zeros(n_frames, dtype=np.float64)
    window = np.hanning(frame_length)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.square(np.abs(spectrum))
    # The DC bin carries offset, not timbre, and would drag the flatness down
    # for any signal that is not perfectly centred.
    power = power[:, 1:]
    power = power + EPS
    log_mean = np.mean(np.log(power), axis=1)
    arithmetic = np.mean(power, axis=1)
    flatness_db = 10.0 * (log_mean / np.log(10.0)) - 10.0 * np.log10(arithmetic)
    # A frame of exact zeros is perfectly flat by this definition; keep that,
    # but never let rounding push the value above the 0 dB ceiling.
    return np.minimum(flatness_db, 0.0)


def frame_features(
    samples: np.ndarray, frame_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure every frame of ``samples`` three ways.

    Args:
        samples: 1-D mono float samples.
        frame_length: Samples per frame.

    Returns:
        ``(frames, energy_db, zcr, flatness_db)``; the last three are 1-D arrays
        with one entry per frame.
    """
    frames = frame_signal(samples, frame_length)
    return (
        frames,
        short_time_energy(frames),
        zero_crossing_rate(frames),
        spectral_flatness(frames),
    )
