"""Framing, the band spectrum, and the running statistics the detector is built on.

Everything downstream works on two small matrices produced here: the level of
each frequency band in every frame, and the overall level of every frame, both
in dBFS (a full-scale sine reads 0 dBFS). Working in dB is deliberate - it makes
"how far from normal" a number a person can argue with, and it keeps a quiet
machine and a loud one on the same footing.

Band levels are the *power in the band*, not the power per FFT bin, so a tone
reads the same whatever the sample rate or frame length, and so does a noise
floor. The band grid itself is fixed in hertz by ``frame_ms`` alone: a profile
taken at 44.1 kHz lines up band for band with a recording made at 16 kHz, which
simply has fewer bands at the top.

Only :func:`numpy.fft.rfft` is used. There is no scipy and no librosa here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

#: How many frequency bands the grid aims for. Very short frames get fewer,
#: because every band must span at least :data:`MIN_BINS_PER_BAND` FFT bins.
N_BANDS = 32

#: Lowest frequency the grid covers. Below this is DC drift and rumble.
F_MIN_HZ = 20.0

#: No band starts below this FFT bin: the first null of the Hann window.
LOWEST_BIN = 2

#: Highest frequency the grid covers. Above it, most recordings hold anti-alias
#: roll-off rather than anything about the machine.
F_MAX_HZ = 12000.0

#: Bands closer to Nyquist than this fraction are dropped: the converter's
#: anti-alias filter lives there, not the sound.
NYQUIST_FRACTION = 0.95

#: Bins per band, minimum. A one-bin band is as noisy as a single FFT bin; three
#: bins make even the lowest band steady enough to judge.
MIN_BINS_PER_BAND = 3

#: A frame quieter than this is digital silence rather than signal.
SILENCE_DBFS = -80.0

#: Levels are floored here so digital silence is a large negative number instead
#: of ``-inf``, and so a sound starting from silence has a finite rise.
FLOOR_DB = -120.0

#: A frame needs at least this many samples to hold a spectrum at all.
MIN_FRAME_SAMPLES = 16

#: Turns a median absolute deviation into a standard-deviation equivalent for
#: normally distributed data (1 / Phi^-1(3/4)).
MAD_TO_SIGMA = 1.4826

#: Adding this to 10*log10(power) makes a full-scale sine read 0 dBFS.
_FULL_SCALE_SINE_DB = 10.0 * np.log10(2.0)

#: Frames are transformed this many at a time, so memory stays bounded however
#: long the recording is.
_CHUNK_FRAMES = 2048


@dataclass
class BandLayout:
    """Which FFT bins make up each band, and where the band sits in hertz."""

    starts: np.ndarray
    stops: np.ndarray
    low_hz: np.ndarray
    high_hz: np.ndarray
    centre_hz: np.ndarray

    @property
    def n_bands(self) -> int:
        """How many bands this layout has."""
        return int(self.starts.size)

    @property
    def width_hz(self) -> np.ndarray:
        """Width of each band in hertz."""
        return self.high_hz - self.low_hz


@dataclass
class Frames:
    """One recording, reduced to per-frame band levels and overall levels.

    Attributes:
        band_db: ``(n_frames, n_bands)`` power in each band of each frame, dBFS.
        rms_db: ``(n_frames,)`` overall level of each frame, dBFS.
        starts: ``(n_frames,)`` first sample of each frame.
        layout: the band layout the columns of ``band_db`` follow.
    """

    band_db: np.ndarray
    rms_db: np.ndarray
    starts: np.ndarray
    layout: BandLayout
    sample_rate: int
    frame_len: int
    hop: int
    n_samples: int
    frame_ms: float
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
    def hop_seconds(self) -> float:
        """Time between the starts of consecutive frames."""
        return self.hop / float(self.sample_rate)

    @property
    def start_s(self) -> np.ndarray:
        """Start time of each frame, in seconds."""
        return self.starts / float(self.sample_rate)

    @property
    def end_s(self) -> np.ndarray:
        """End time of each frame, in seconds."""
        return (self.starts + self.frame_len) / float(self.sample_rate)

    @property
    def times(self) -> np.ndarray:
        """Centre time of each frame, in seconds."""
        return (self.starts + self.frame_len / 2.0) / float(self.sample_rate)

    @property
    def duration(self) -> float:
        """Length of the whole recording, in seconds."""
        return self.n_samples / float(self.sample_rate)

    @property
    def live(self) -> np.ndarray:
        """Boolean mask of frames loud enough to count as signal."""
        return self.rms_db >= SILENCE_DBFS


def frame_plan(sample_rate: int, frame_ms: float) -> Tuple[int, int]:
    """Return ``(frame_len, hop)`` in samples for frames of ``frame_ms``.

    Frames overlap by half, so an event is never lost at a frame boundary.
    """
    try:
        ms = float(frame_ms)
    except (TypeError, ValueError):
        raise ValueError("frame_ms must be a number of milliseconds, got %r" % (frame_ms,)) from None
    if not np.isfinite(ms) or ms <= 0:
        raise ValueError(
            "frame_ms must be a positive number of milliseconds, got %r" % (frame_ms,)
        )
    frame_len = int(round(sample_rate * ms / 1000.0))
    if frame_len < MIN_FRAME_SAMPLES:
        raise ValueError(
            "frame_ms=%g at %d Hz gives frames of only %d samples; a frame needs at "
            "least %d to hold a spectrum, so use frame_ms=%g or more"
            % (ms, sample_rate, frame_len, MIN_FRAME_SAMPLES,
               np.ceil(1000.0 * MIN_FRAME_SAMPLES / sample_rate))
        )
    return frame_len, max(frame_len // 2, 1)


def band_edges_hz(frame_ms: float) -> np.ndarray:
    """Nominal band edges in hertz: log-spaced, contiguous, each at least three bins wide.

    The grid depends only on ``frame_ms`` (which fixes the bin spacing), never on
    the sample rate, so two recordings analysed with the same ``frame_ms`` share
    their bands.
    """
    bin_hz = 1000.0 / float(frame_ms)
    # Bin 1 still sees DC and infrasonic drift through the Hann window's main
    # lobe (it is only 1 bin from DC; the first null is at 2), so no band starts
    # below bin 2. Without this, wind rumble and DC wander read as a "new tone".
    lo = max(LOWEST_BIN, int(round(F_MIN_HZ / bin_hz)))
    stop_all = max(lo + 1, int(round(F_MAX_HZ / bin_hz)) + 1)
    span = stop_all - lo
    count = max(1, min(N_BANDS, span // MIN_BINS_PER_BAND))
    edges = [lo]
    for i in range(count):
        remaining = count - i
        current = edges[-1]
        if remaining == 1:
            nxt = stop_all
        else:
            target = int(round(current * (stop_all / float(current)) ** (1.0 / remaining)))
            nxt = max(current + MIN_BINS_PER_BAND, target)
            # Leave room for the bands still to come.
            nxt = min(nxt, stop_all - MIN_BINS_PER_BAND * (remaining - 1))
        edges.append(nxt)
    # Bin k is centred on k * bin_hz, so it covers (k - 0.5) to (k + 0.5) bins.
    return (np.asarray(edges, dtype=np.float64) - 0.5) * bin_hz


def grid_centres_hz(frame_ms: float) -> np.ndarray:
    """Centre frequency of every band in the grid for ``frame_ms``."""
    edges = band_edges_hz(frame_ms)
    return np.sqrt(edges[:-1] * edges[1:])


def frame_ms_of_centres(centres: np.ndarray) -> Optional[float]:
    """The ``frame_ms`` whose band grid has these centres, or None if none does.

    A saved profile keeps only its band centres; this recovers the frame length
    it was made with, so a recording can be analysed on exactly the same bands.
    """
    c = np.asarray(centres, dtype=np.float64)
    if c.size == 0 or c[0] <= 0:
        return None
    # With the lowest band starting at bin 2 and three bins wide, its centre is
    # sqrt(1.5 * 4.5) bins: that pins frame_ms directly in the common case.
    direct = round(1000.0 * np.sqrt((LOWEST_BIN - 0.5) * (LOWEST_BIN + 2.5)) / c[0], 3)
    for candidate in [direct] + [x / 2.0 for x in range(10, 4001)]:
        if candidate <= 0:
            continue
        grid = grid_centres_hz(candidate)
        if grid.size >= c.size and np.allclose(grid[: c.size], c, rtol=1e-3):
            return float(candidate)
    return None


def band_layout(sample_rate: int, frame_len: int, frame_ms: float) -> BandLayout:
    """The bands of the fixed grid that fit below this sample rate's Nyquist limit."""
    edges_hz = band_edges_hz(frame_ms)
    f_top = NYQUIST_FRACTION * sample_rate / 2.0
    n_keep = int(np.sum(edges_hz[1:] <= f_top * (1.0 + 1e-9)))
    if n_keep == 0:
        raise ValueError(
            "at %d Hz nothing between %g Hz and %g Hz can be measured, which is "
            "where every frequency band lives; the sample rate is too low for "
            "audio analysis" % (sample_rate, F_MIN_HZ, F_MAX_HZ)
        )
    edges_hz = edges_hz[: n_keep + 1]
    bin_hz = sample_rate / float(frame_len)
    n_bins = frame_len // 2 + 1
    bins = np.rint(edges_hz / bin_hz + 0.5).astype(np.int64)
    bins = np.clip(bins, 1, n_bins)
    for i in range(1, bins.size):
        bins[i] = max(bins[i], bins[i - 1] + 1)
    if bins[-1] > n_bins:  # only reachable for degenerate, very short frames
        bins = np.minimum(bins, n_bins)
        keep = np.concatenate(([True], np.diff(bins) > 0))
        bins = bins[keep]
        edges_hz = edges_hz[: bins.size]
    low = edges_hz[:-1]
    high = edges_hz[1:]
    return BandLayout(
        starts=bins[:-1].copy(),
        stops=bins[1:].copy(),
        low_hz=low.copy(),
        high_hz=high.copy(),
        centre_hz=np.sqrt(low * high),
    )


def _hann(frame_len: int) -> np.ndarray:
    """A periodic Hann window."""
    n = np.arange(frame_len, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / frame_len)


def analyse(samples: np.ndarray, sample_rate: int, frame_ms: float = 50.0) -> Frames:
    """Cut ``samples`` into half-overlapping frames and measure each one.

    Args:
        samples: 1-D float64 mono audio. It is only read, never written.
        sample_rate: samples per second.
        frame_ms: length of one analysis frame in milliseconds.

    Returns:
        A :class:`Frames`. When the recording is shorter than a single frame it
        holds zero frames, and every array in it is empty rather than absent.
    """
    frame_len, hop = frame_plan(sample_rate, frame_ms)
    layout = band_layout(sample_rate, frame_len, frame_ms)
    n = int(samples.size)
    if n < frame_len:
        return Frames(
            band_db=np.zeros((0, layout.n_bands), dtype=np.float64),
            rms_db=np.zeros(0, dtype=np.float64),
            starts=np.zeros(0, dtype=np.int64),
            layout=layout,
            sample_rate=int(sample_rate),
            frame_len=frame_len,
            hop=hop,
            n_samples=n,
            frame_ms=float(frame_ms),
        )

    n_frames = 1 + (n - frame_len) // hop
    # sliding_window_view is a read-only view; multiplying by the window
    # allocates the working copy, so the caller's samples are never touched.
    windows = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop][:n_frames]
    win = _hann(frame_len)
    # Scale so the summed bins of a band give the mean-square power in it.
    norm = 2.0 / (frame_len * float(np.sum(win * win)))
    lo = int(layout.starts[0])
    hi = int(layout.stops[-1])
    offsets = layout.starts - lo

    band_power = np.empty((n_frames, layout.n_bands), dtype=np.float64)
    mean_square = np.empty(n_frames, dtype=np.float64)
    for a in range(0, n_frames, _CHUNK_FRAMES):
        b = min(n_frames, a + _CHUNK_FRAMES)
        block = windows[a:b]
        mean_square[a:b] = np.mean(block * block, axis=1)
        spectrum = np.fft.rfft(block * win, axis=1)[:, lo:hi]
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        band_power[a:b] = np.add.reduceat(power, offsets, axis=1) * norm

    tiny = 10.0 ** (FLOOR_DB / 10.0)
    band_db = 10.0 * np.log10(np.maximum(band_power, tiny)) + _FULL_SCALE_SINE_DB
    rms_db = 10.0 * np.log10(np.maximum(mean_square, tiny)) + _FULL_SCALE_SINE_DB
    np.maximum(band_db, FLOOR_DB, out=band_db)
    np.maximum(rms_db, FLOOR_DB, out=rms_db)
    return Frames(
        band_db=band_db,
        rms_db=rms_db,
        starts=np.arange(n_frames, dtype=np.int64) * hop,
        layout=layout,
        sample_rate=int(sample_rate),
        frame_len=frame_len,
        hop=hop,
        n_samples=n,
        frame_ms=float(frame_ms),
    )


def combined_db(band_db: np.ndarray) -> np.ndarray:
    """Total level over the bands (last axis), dBFS: the band powers summed."""
    power = np.power(10.0, (np.asarray(band_db) - _FULL_SCALE_SINE_DB) / 10.0)
    total = np.sum(power, axis=-1)
    tiny = 10.0 ** (FLOOR_DB / 10.0)
    return np.maximum(10.0 * np.log10(np.maximum(total, tiny)) + _FULL_SCALE_SINE_DB, FLOOR_DB)


def window_median(x: np.ndarray, offset: int, width: int) -> np.ndarray:
    """``out[t] = median(x[t + offset : t + offset + width])`` along axis 0.

    The ends are reflect-padded, so the first frames borrow their "before"
    context from just after them rather than from nothing.
    """
    n = int(x.shape[0])
    if n == 0:
        return np.array(x, dtype=np.float64, copy=True)
    width = max(1, int(width))
    pad_lo = max(0, -int(offset))
    pad_hi = max(0, int(offset) + width - 1)
    pad = [(pad_lo, pad_hi)] + [(0, 0)] * (x.ndim - 1)
    padded = np.pad(np.asarray(x, dtype=np.float64), pad, mode="reflect")
    view = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
    first = pad_lo + int(offset)
    out = np.empty(x.shape, dtype=np.float64)
    for a in range(0, n, _CHUNK_FRAMES):
        b = min(n, a + _CHUNK_FRAMES)
        out[a:b] = np.median(view[first + a : first + b], axis=-1)
    return out


def lag_spread(x: np.ndarray, lag: int) -> np.ndarray:
    """Robust spread of ``x`` along axis 0 from differences ``lag`` frames apart.

    A step change or a single burst touches only a few of the differences, so
    this measures how much the sound wobbles without being inflated by the very
    anomalies it will be used to find. Returns zeros when there are too few
    frames to difference.
    """
    lag = max(1, int(lag))
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] <= lag:
        return np.zeros(x.shape[1:], dtype=np.float64)
    diffs = np.abs(x[lag:] - x[:-lag])
    return MAD_TO_SIGMA * np.median(diffs, axis=0) / np.sqrt(2.0)


def residual_spread(x: np.ndarray, width: int) -> np.ndarray:
    """Robust spread along axis 0 of ``x`` around its own running median.

    The running median follows a step and ignores a burst, so what is left is
    the frame-to-frame wobble itself, however it is shaped.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 3:
        return np.zeros(x.shape[1:], dtype=np.float64)
    width = max(3, int(width) | 1)
    residual = x - window_median(x, -(width // 2), width)
    return MAD_TO_SIGMA * np.median(np.abs(residual), axis=0)


def robust_spread(x: np.ndarray) -> np.ndarray:
    """Median absolute deviation along axis 0, scaled to read like a standard deviation."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        return np.zeros(x.shape[1:], dtype=np.float64)
    centre = np.median(x, axis=0)
    return MAD_TO_SIGMA * np.median(np.abs(x - centre), axis=0)
