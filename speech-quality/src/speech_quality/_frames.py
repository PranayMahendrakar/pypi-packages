"""Cutting a recording into short frames, and taking spectra of some of them.

Every measure that is about *when* something happens - silence at the edges,
speech in the middle, a noise floor between words - works on short overlapping
frames rather than on the whole signal, because a single number over a whole
recording hides exactly what the caller wants to know.

Frame levels come from a running sum of squares, so the cost is linear in the
number of samples and a minute of 48 kHz audio is framed in milliseconds.
Spectra cost far more, so at most ``max_spectral_frames`` of them are taken,
spread evenly across the recording; evenly, not randomly, so two runs on the
same input always agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._scoring import dbfs_array
from .thresholds import Thresholds

__all__ = ["Frames", "Spectra", "frame_signal", "take_spectra"]

MIN_FRAME_LENGTH = 16
"""No frame is shorter than this, whatever the sample rate works out to."""


@dataclass
class Frames:
    """Short overlapping windows of the recording, with a level for each.

    Attributes:
        starts: first sample index of each frame.
        length: frame length in samples.
        hop: step between frame starts, in samples.
        rms: per-frame RMS amplitude, full scale at 1.0.
        dbfs: per-frame level in dBFS, floored rather than negative infinity.
        peak: per-frame peak absolute amplitude.
        sample_rate: samples per second.
        n_samples: length of the recording the frames came from.
    """

    starts: np.ndarray
    length: int
    hop: int
    rms: np.ndarray
    dbfs: np.ndarray
    peak: np.ndarray
    sample_rate: int
    n_samples: int

    @property
    def count(self) -> int:
        """How many frames there are; always at least one."""
        return int(self.starts.size)

    @property
    def seconds_per_frame(self) -> float:
        """How much time one frame step stands for."""
        return float(self.hop) / float(self.sample_rate)

    def active_mask(self, thresholds: Thresholds) -> np.ndarray:
        """Which frames carry something rather than silence.

        A frame is active when it is both within ``silence_drop_db`` of the
        loudest frame and above the absolute ``silence_floor_dbfs``. The
        relative half keeps a quiet but clean recording from reading as all
        silence; the absolute half keeps a recording of nothing but hiss from
        reading as all speech.

        Args:
            thresholds: the limits to apply.

        Returns:
            A boolean array, one entry per frame.
        """
        loudest = float(np.max(self.dbfs))
        relative = loudest - float(thresholds.silence_drop_db)
        cutoff = max(relative, float(thresholds.silence_floor_dbfs))
        return self.dbfs > cutoff


@dataclass
class Spectra:
    """Power spectra of a sample of frames, and what they say per frame.

    Attributes:
        indices: which frames were analysed, ascending.
        freqs: bin centre frequencies in Hz.
        power: ``(len(indices), len(freqs))`` power in each bin.
        centroid: per-analysed-frame spectral centroid in Hz.
        band_share: per-analysed-frame share of energy inside the speech band.
        total: per-analysed-frame total power, for weighting.
    """

    indices: np.ndarray
    freqs: np.ndarray
    power: np.ndarray
    centroid: np.ndarray
    band_share: np.ndarray
    total: np.ndarray

    @property
    def count(self) -> int:
        """How many frames were analysed."""
        return int(self.indices.size)

    def mean_power(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Average spectrum over the analysed frames, or a subset of them.

        Args:
            mask: boolean array over the analysed frames. A mask that selects
                nothing is ignored, so the caller always gets a usable spectrum.

        Returns:
            One power value per frequency bin.
        """
        if self.power.size == 0:
            return np.zeros(self.freqs.size, dtype=np.float64)
        if mask is not None and bool(np.any(mask)):
            return np.asarray(self.power[mask].mean(axis=0), dtype=np.float64)
        return np.asarray(self.power.mean(axis=0), dtype=np.float64)


def _frame_geometry(n_samples: int, sample_rate: int, thresholds: Thresholds) -> Tuple[int, int]:
    """Frame length and hop in samples, kept sane for very short recordings."""
    length = int(round(float(thresholds.frame_seconds) * float(sample_rate)))
    length = max(MIN_FRAME_LENGTH, length)
    length = min(length, max(1, n_samples))
    hop = int(round(float(thresholds.hop_seconds) * float(sample_rate)))
    hop = max(1, min(hop, length))
    return length, hop


def frame_signal(samples: np.ndarray, sample_rate: int, thresholds: Thresholds) -> Frames:
    """Cut the signal into overlapping frames and measure the level of each.

    Args:
        samples: one-dimensional float samples.
        sample_rate: samples per second.
        thresholds: supplies the frame and hop length.

    Returns:
        The frames and their levels. A recording shorter than one frame becomes
        a single frame holding all of it.
    """
    signal = np.asarray(samples, dtype=np.float64)
    n_samples = int(signal.size)
    length, hop = _frame_geometry(n_samples, sample_rate, thresholds)

    if n_samples <= length:
        starts = np.zeros(1, dtype=np.int64)
        divisor = np.array([float(max(n_samples, 1))], dtype=np.float64)
        energy = np.array([float(np.dot(signal, signal))], dtype=np.float64)
        peak = np.array([float(np.max(np.abs(signal)))], dtype=np.float64)
    else:
        last = n_samples - length
        starts = np.arange(0, last + 1, hop, dtype=np.int64)
        if int(starts[-1]) != last:
            starts = np.append(starts, last)
        # A running sum of squares makes every frame energy one subtraction.
        cumulative = np.concatenate(([0.0], np.cumsum(signal * signal)))
        energy = np.maximum(cumulative[starts + length] - cumulative[starts], 0.0)
        divisor = np.full(starts.size, float(length), dtype=np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(np.abs(signal), length)
        peak = np.asarray(windows[starts].max(axis=1), dtype=np.float64)

    rms = np.sqrt(energy / divisor)
    return Frames(
        starts=starts,
        length=int(length),
        hop=int(hop),
        rms=rms,
        dbfs=dbfs_array(rms),
        peak=peak,
        sample_rate=int(sample_rate),
        n_samples=n_samples,
    )


def _pick_frames(frames: Frames, limit: int) -> np.ndarray:
    """Up to ``limit`` frame indices, spread evenly and in order."""
    count = frames.count
    if limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    positions = np.linspace(0.0, float(count - 1), num=int(limit))
    return np.unique(np.round(positions).astype(np.int64))


def take_spectra(samples: np.ndarray, frames: Frames, thresholds: Thresholds) -> Spectra:
    """Take Hann-windowed power spectra of an evenly spread sample of frames.

    Args:
        samples: the signal the frames came from.
        frames: the framing to follow.
        thresholds: supplies the speech band and the frame budget.

    Returns:
        The spectra, plus the centroid and speech-band share of each analysed
        frame. A recording too short for a usable transform comes back with
        empty arrays, which every caller checks for.
    """
    signal = np.asarray(samples, dtype=np.float64)
    length = frames.length
    indices = _pick_frames(frames, int(thresholds.max_spectral_frames))
    if length < 8 or indices.size == 0:
        return Spectra(
            indices=np.zeros(0, dtype=np.int64),
            freqs=np.zeros(0, dtype=np.float64),
            power=np.zeros((0, 0), dtype=np.float64),
            centroid=np.zeros(0, dtype=np.float64),
            band_share=np.zeros(0, dtype=np.float64),
            total=np.zeros(0, dtype=np.float64),
        )

    starts = frames.starts[indices]
    offsets = np.arange(length, dtype=np.int64)
    block = signal[starts[:, None] + offsets[None, :]]
    window = np.hanning(length)
    spectrum = np.fft.rfft(block * window[None, :], axis=1)
    power = (spectrum.real * spectrum.real) + (spectrum.imag * spectrum.imag)
    freqs = np.fft.rfftfreq(length, d=1.0 / float(frames.sample_rate))

    # The DC bin is a recording offset, not a frequency anyone hears.
    if power.shape[1] > 1:
        power[:, 0] = 0.0

    total = power.sum(axis=1)
    positive = total > 0.0
    safe_total = np.where(positive, total, 1.0)
    centroid = np.where(positive, (power * freqs[None, :]).sum(axis=1) / safe_total, 0.0)

    in_band = (freqs >= float(thresholds.speech_band_low_hz)) & (
        freqs <= float(thresholds.speech_band_high_hz)
    )
    band_share = np.where(positive, power[:, in_band].sum(axis=1) / safe_total, 0.0)

    return Spectra(
        indices=indices,
        freqs=freqs,
        power=power,
        centroid=centroid,
        band_share=band_share,
        total=total,
    )
