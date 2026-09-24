"""Framing, the band spectrum, and the profile a recording is compared against.

Everything downstream works on two small matrices produced here: the level of
each frequency band in every frame, in dB, and the overall level of every frame,
also in dB. Working in dB is deliberate - it makes "how far from normal" a
number a person can argue with, and it keeps a quiet machine and a loud one on
the same footing.

Only :func:`numpy.fft.rfft` is used. There is no scipy and no librosa here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

#: How many frequency bands the spectrum is reduced to. Fixed, so a profile
#: saved by :func:`audio_anomaly.spectral_profile` is always the same length and
#: can be recognised as a profile rather than a waveform.
N_BANDS = 32

#: Lowest frequency a band covers. Below this is mains rumble and DC drift.
F_MIN_HZ = 20.0

#: Highest frequency a band covers, clamped to just under Nyquist for low
#: sample rates. Above this, most recordings hold anti-alias filter roll-off
#: rather than anything about the machine.
F_MAX_HZ = 12000.0

#: Bins per band, minimum. One bin per band makes a band as noisy as a single
#: FFT bin, which is very noisy indeed; two is the cheapest useful averaging.
MIN_BINS_PER_BAND = 2

#: Added to a power spectrum before taking its logarithm, so digital silence
#: lands on a large negative number instead of ``-inf``.
POWER_EPS = 1e-20

#: A frame quieter than this is treated as silence rather than as signal.
SILENCE_DBFS = -80.0

#: Floor for a per-band robust spread, in dB. A band whose level never moves at
#: all would otherwise turn a rounding error into an infinite z-score, which is
#: exactly how a detector ends up flagging healthy machinery.
MIN_BAND_SPREAD_DB = 1.0

#: 1/Phi^-1(3/4): turns a median absolute deviation into a standard-deviation
#: equivalent for normally distributed data.
MAD_TO_SIGMA = 1.4826


@dataclass
class Frames:
    """One recording, reduced to per-frame band levels and overall levels.

    Attributes:
        band_db: ``(n_frames, n_bands)`` level of each band in each frame, dB.
        rms_db: ``(n_frames,)`` overall level of each frame, dBFS.
        times: ``(n_frames,)`` start time of each frame, in seconds.
        band_hz: ``(n_bands,)`` centre frequency of each band, in Hz.
    """

    band_db: np.ndarray
    rms_db: np.ndarray
    times: np.ndarray
    band_hz: np.ndarray
    sample_rate: int
    frame_len: int
    hop: int
    n_samples: int
    notes: List[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        """How many frames the recording was cut into."""
        return int(self.band_db.shape[0])

    @property
    def n_bands(self) -> int:
        """How many frequency bands each frame was reduced to."""
        return int(self.band_db.shape[1])

    @property
    def frame_seconds(self) -> float:
        """Length of one frame, in seconds."""
        return self.frame_len / float(self.sample_rate)

    @property
    def duration(self) -> float:
        """Length of the whole recording, in seconds."""
        return self.n_samples / float(self.sample_rate)

    @property
    def silent(self) -> np.ndarray:
        """Boolean mask of frames quiet enough to count as silence."""
        return self.rms_db < SILENCE_DBFS


def frame_plan(sample_rate: int, frame_ms: float) -> Tuple[int, int]:
    """Return ``(frame_len, hop)`` in samples for a frame of ``frame_ms``."""
    if not np.isfinite(frame_ms) or frame_ms <= 0:
        raise ValueError(
            "frame_ms must be a positive number of milliseconds, got %r" % (frame_ms,)
        )
    frame_len = int(round(sample_rate * float(frame_ms) / 1000.0))
    frame_len = max(frame_len, 2)
    hop = max(frame_len // 2, 1)
    return frame_len, hop


def band_ranges(sample_rate: int, n_fft: int, n_bands: int = N_BANDS) -> Tuple[np.ndarray, np.ndarray]:
    """Bin ranges for log-spaced bands, each at least a couple of bins wide.

    Bands are spaced logarithmically because that is how both hearing and
    machine noise are organised: an octave matters, a fixed 500 Hz does not.
    """
    n_bins = n_fft // 2 + 1
    nyquist = sample_rate / 2.0
    f_max = min(F_MAX_HZ, 0.95 * nyquist)
    f_min = min(F_MIN_HZ, f_max / 2.0)
    f_min = max(f_min, 1e-6)
    edges = np.geomspace(f_min, max(f_max, f_min * 2.0), n_bands + 1)
    bin_hz = sample_rate / float(n_fft)
    raw = np.rint(edges / bin_hz).astype(np.int64)
    # Bin 0 is DC, which carries no information about a sound.
    raw = np.clip(raw, 1, n_bins - 1)
    starts = np.empty(n_bands, dtype=np.int64)
    stops = np.empty(n_bands, dtype=np.int64)
    for b in range(n_bands):
        start = int(raw[b])
        stop = max(int(raw[b + 1]), start + MIN_BINS_PER_BAND)
        if stop > n_bins:
            # Ran off the top: slide the window down so the band keeps its width.
            stop = n_bins
            start = max(1, stop - MIN_BINS_PER_BAND)
        starts[b] = start
        stops[b] = stop
    return starts, stops


def _window(frame_len: int) -> np.ndarray:
    """A periodic Hann window, or a flat one when a frame is too short to taper."""
    if frame_len < 4:
        return np.ones(frame_len, dtype=np.float64)
    n = np.arange(frame_len, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / frame_len)


def _next_pow2(value: int) -> int:
    """Smallest power of two that is at least ``value``."""
    return 1 << max(int(value) - 1, 0).bit_length()


def analyse(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 50.0,
    n_bands: int = N_BANDS,
) -> Frames:
    """Cut ``samples`` into frames and measure each one.

    Args:
        samples: 1-D float64 mono audio.
        sample_rate: samples per second.
        frame_ms: length of one analysis frame in milliseconds. Frames overlap
            by half, so an event is never lost at a frame boundary.
        n_bands: how many log-spaced bands the spectrum is reduced to.

    Returns:
        A :class:`Frames`. When the recording is shorter than a single frame it
        holds zero frames, and every array in it is empty rather than absent.
    """
    frame_len, hop = frame_plan(sample_rate, frame_ms)
    n_fft = _next_pow2(frame_len)
    starts, stops = band_ranges(sample_rate, n_fft, n_bands)
    centre_bins = (starts + stops - 1) / 2.0
    band_hz = centre_bins * (sample_rate / float(n_fft))

    if samples.size < frame_len:
        return Frames(
            band_db=np.zeros((0, n_bands), dtype=np.float64),
            rms_db=np.zeros(0, dtype=np.float64),
            times=np.zeros(0, dtype=np.float64),
            band_hz=band_hz,
            sample_rate=int(sample_rate),
            frame_len=frame_len,
            hop=hop,
            n_samples=int(samples.size),
        )

    n_frames = 1 + (samples.size - frame_len) // hop
    # sliding_window_view gives a read-only view; multiplying by the window
    # allocates the working copy, so the caller's samples are never touched.
    windows = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop][:n_frames]
    win = _window(frame_len)

    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    rms_db = 20.0 * np.log10(rms + 1e-12)

    spectrum = np.fft.rfft(windows * win, n=n_fft, axis=1)
    power = np.square(spectrum.real) + np.square(spectrum.imag)
    # Normalise by the window's energy so band levels stay comparable across
    # different frame_ms settings and sample rates.
    power /= max(float(np.sum(np.square(win))), 1e-12)

    band_power = np.empty((n_frames, n_bands), dtype=np.float64)
    for b in range(n_bands):
        band_power[:, b] = power[:, starts[b] : stops[b]].mean(axis=1)
    band_db = 10.0 * np.log10(band_power + POWER_EPS)

    times = (np.arange(n_frames, dtype=np.float64) * hop) / float(sample_rate)
    return Frames(
        band_db=band_db,
        rms_db=rms_db,
        times=times,
        band_hz=band_hz,
        sample_rate=int(sample_rate),
        frame_len=frame_len,
        hop=hop,
        n_samples=int(samples.size),
    )


def robust_spread(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median absolute deviation, scaled to read like a standard deviation."""
    centre = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - centre), axis=axis)
    return MAD_TO_SIGMA * mad


def profile_of(frames: Frames, use: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(centre_db, spread_db)``: the typical level of each band and its wobble.

    Args:
        frames: the analysed recording.
        use: optional boolean mask of frames to learn from. Silent frames are
            normally excluded, because the level of a band that holds nothing
            but the dither floor says nothing about the machine.

    Returns:
        ``(centre_db, spread_db)``, both shaped ``(n_bands,)``. The spread is
        floored at :data:`MIN_BAND_SPREAD_DB` so a perfectly steady band cannot
        turn a rounding error into an enormous score.
    """
    band_db = frames.band_db
    if use is not None and use.any():
        band_db = band_db[use]
    if band_db.shape[0] == 0:
        centre = np.zeros(frames.n_bands, dtype=np.float64)
        spread = np.full(frames.n_bands, MIN_BAND_SPREAD_DB, dtype=np.float64)
        return centre, spread
    centre = np.median(band_db, axis=0)
    spread = np.maximum(robust_spread(band_db, axis=0), MIN_BAND_SPREAD_DB)
    return centre, spread
