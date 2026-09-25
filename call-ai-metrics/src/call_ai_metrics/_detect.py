"""Decide, frame by frame, where each channel carries its own party speaking.

This is an energy heuristic built for the usual call-recording layout, where each
party has a channel of their own. Each channel gets a threshold set from its own
noise floor and its own loudest speech, so a quiet customer line and a loud agent
headset are judged the same way.

The one thing that breaks that layout is crosstalk: one party leaking into the
other's channel, through the recorder, the handset or line echo. Leaked speech is
a quieter copy of the other channel, so it rises and falls exactly when the other
channel does. That is what is measured here - how closely one channel's loudness
tracks the other's while the other is talking - and when it tracks closely the
leak is reported, its level is estimated, and a frame only counts as the quieter
channel's own speech if it stands clearly above the leak predicted for it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ._audio import SILENT_DB, CallAudio, power_to_db, smooth_rows

NOISE_PERCENTILE = 10.0
"""A channel's noise floor is this percentile of its frame levels."""

MIN_SPREAD_DB = 10.0
"""Speech must stand at least this far above the channel's noise floor."""

SILENT_CHANNEL_DB = -60.0
"""A channel whose loudest 100 ms is quieter than this carries no speech."""

DIGITAL_SILENCE_DB = -100.0
"""Frames quieter than this hold no signal at all, not even line noise."""

THRESHOLD_FLOOR_DB = -65.0
"""No speech threshold is set below this, whatever the noise floor."""

BLEED_MIN_CORRELATION = 0.6
"""Loudness tracking at least this closely means one channel leaks into another."""

SAME_AUDIO_CORRELATION = 0.95
SAME_AUDIO_DB = 3.0
"""Channels this alike, at this level difference, are one recording twice."""

MIN_EVIDENCE_FRAMES = 50
"""Half a second of speech is the least a crosstalk estimate is made from."""

MAX_LAG_S = 0.25
"""Leak delays up to this long (line echo, recorder skew) are searched."""

LAG_TIE_CORRELATION = 0.03
"""Lags scoring within this of the best are a tie, settled by the shortest."""

_DESPECKLE_FRAMES = 3


class ChannelLevel:
    """How one channel's speech threshold was set."""

    __slots__ = ("noise_db", "peak_db", "threshold_db")

    def __init__(self, noise_db: float, peak_db: float, threshold_db: Optional[float]) -> None:
        self.noise_db = noise_db
        self.peak_db = peak_db
        self.threshold_db = threshold_db


class BleedEstimate:
    """One channel leaking into another."""

    __slots__ = ("source", "target", "level_db", "correlation", "lag_s", "margin_db", "affected")

    def __init__(
        self,
        source: int,
        target: int,
        level_db: float,
        correlation: float,
        lag_s: float,
        margin_db: float,
        affected: bool,
    ) -> None:
        self.source = source
        self.target = target
        self.level_db = level_db
        self.correlation = correlation
        self.lag_s = lag_s
        self.margin_db = margin_db
        self.affected = affected


class Detection:
    """Per-channel speech intervals and everything that went into them."""

    __slots__ = ("intervals", "levels", "bleed", "same_audio")

    def __init__(
        self,
        intervals: List[List[Tuple[float, float]]],
        levels: List[ChannelLevel],
        bleed: List[BleedEstimate],
        same_audio: List[Tuple[int, int]],
    ) -> None:
        self.intervals = intervals
        self.levels = levels
        self.bleed = bleed
        self.same_audio = same_audio


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """``(start, stop)`` index pairs of every run of True in a 1-D bool array."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _drop_short(mask: np.ndarray, min_frames: int) -> None:
    for start, stop in runs(mask):
        if stop - start < min_frames:
            mask[start:stop] = False


def _fill_gaps(mask: np.ndarray, max_frames: int) -> None:
    size = mask.shape[0]
    for start, stop in runs(~mask):
        if start > 0 and stop < size and stop - start < max_frames:
            mask[start:stop] = True


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _aligned(values: np.ndarray, lag: int) -> Tuple[slice, slice]:
    """Slices pairing frame t of a source with frame t + lag of a target."""
    size = values.shape[0]
    if lag >= 0:
        return slice(0, size - lag), slice(lag, size)
    return slice(-lag, size), slice(0, size + lag)


def _estimate_bleed(
    src_db: np.ndarray, src_active: np.ndarray, dst_db: np.ndarray, max_lag: int
) -> Optional[Tuple[int, float, float, float]]:
    """Best ``(lag, correlation, level_db, spread_db)`` for src leaking into dst.

    Only frames where the source is talking and is the louder of the two are
    used: in those, a leak makes the target a quieter copy of the source, and
    genuine speech of the target's own makes it independent of the source.
    """
    size = src_db.shape[0]
    tried: List[Tuple[int, float, np.ndarray]] = []
    for lag in range(-max_lag, max_lag + 1):
        if abs(lag) >= size:
            continue
        src_part, dst_part = _aligned(src_db, lag)
        x = src_db[src_part]
        y = dst_db[dst_part]
        use = src_active[src_part] & (x > y)
        if int(np.count_nonzero(use)) < MIN_EVIDENCE_FRAMES:
            continue
        tried.append((lag, _pearson(x[use], y[use]), y[use] - x[use]))
    if not tried:
        return None
    # Speech loudness is only loosely periodic, so several lags can score almost
    # the same; a real leak is near-instant, so the shortest of those wins.
    top = max(r for _, r, _ in tried)
    lag, r, diff = min(
        (item for item in tried if item[1] >= top - LAG_TIE_CORRELATION),
        key=lambda item: abs(item[0]),
    )
    level = float(np.median(diff))
    spread = float(1.4826 * np.median(np.abs(diff - level)))
    return lag, r, level, spread


def detect(
    audio: CallAudio, *, min_speech_s: float, bridge_gap_s: float
) -> Detection:
    """Find each channel's own speech, with crosstalk detected and removed."""
    n_channels = audio.n_channels
    n_frames = int(audio.power.shape[1])
    hop_s = audio.hop_s
    db = power_to_db(smooth_rows(audio.power, 3))
    envelope = power_to_db(smooth_rows(audio.power, 10))  # 100 ms: a word, not a click

    levels: List[ChannelLevel] = []
    active = np.zeros((n_channels, n_frames), dtype=bool)
    for channel in range(n_channels):
        if n_frames == 0:
            levels.append(ChannelLevel(SILENT_DB, SILENT_DB, None))
            continue
        # Digital silence (VoIP silence suppression, a muted leg) says nothing
        # about the background a talker has to rise above; measure the floor
        # from the frames that carry any signal at all, when there are enough.
        audible = db[channel][db[channel] > DIGITAL_SILENCE_DB]
        floor_from = audible if audible.size >= MIN_EVIDENCE_FRAMES else db[channel]
        noise = float(np.percentile(floor_from, NOISE_PERCENTILE))
        peak = float(envelope[channel].max())
        silent_frames = int(np.count_nonzero(db[channel] <= DIGITAL_SILENCE_DB))
        if peak - noise < MIN_SPREAD_DB and silent_frames >= MIN_EVIDENCE_FRAMES:
            # The signal-bearing frames can be all speech when the pauses are
            # exact digital zero; then the digital silence itself is the floor.
            noise = DIGITAL_SILENCE_DB
        if peak < SILENT_CHANNEL_DB or peak - noise < MIN_SPREAD_DB:
            levels.append(ChannelLevel(noise, peak, None))
            continue
        threshold = max(
            noise + max(MIN_SPREAD_DB, 0.3 * (peak - noise)),
            peak - 40.0,
            THRESHOLD_FLOOR_DB,
        )
        levels.append(ChannelLevel(noise, peak, threshold))
        active[channel] = db[channel] > threshold

    same_audio: List[Tuple[int, int]] = []
    for first in range(n_channels):
        for second in range(first + 1, n_channels):
            either = active[first] | active[second]
            if int(np.count_nonzero(either)) < MIN_EVIDENCE_FRAMES:
                continue
            r = _pearson(db[first][either], db[second][either])
            gap = float(np.median(db[first][either] - db[second][either]))
            if r >= SAME_AUDIO_CORRELATION and abs(gap) <= SAME_AUDIO_DB:
                same_audio.append((first, second))
    duplicated = {index for pair in same_audio for index in pair}

    max_lag = max(0, int(round(MAX_LAG_S / hop_s)))
    bleed: List[BleedEstimate] = []
    for source in range(n_channels):
        if levels[source].threshold_db is None or source in duplicated:
            continue
        for target in range(n_channels):
            if target == source or target in duplicated:
                continue
            found = _estimate_bleed(db[source], active[source], db[target], max_lag)
            if found is None:
                continue
            lag, r, level, spread = found
            if r < BLEED_MIN_CORRELATION or level > -SAME_AUDIO_DB:
                continue
            margin = float(np.clip(3.0 * spread, 6.0, 15.0))
            target_threshold = levels[target].threshold_db
            affected = False
            if target_threshold is not None:
                talking = db[source][active[source]]
                loud_leak = np.count_nonzero(talking + level > target_threshold)
                affected = loud_leak >= 0.05 * max(talking.size, 1)
            bleed.append(
                BleedEstimate(source, target, level, r, lag * hop_s, margin, bool(affected))
            )

    for estimate in bleed:
        if not estimate.affected:
            continue
        lag = int(round(estimate.lag_s / hop_s))
        src_part, dst_part = _aligned(db[estimate.source], lag)
        predicted = np.full(n_frames, SILENT_DB + estimate.level_db)
        predicted[dst_part] = db[estimate.source][src_part] + estimate.level_db
        own = db[estimate.target] - predicted > estimate.margin_db
        active[estimate.target] &= own

    min_frames = max(1, int(round(min_speech_s / hop_s)))
    bridge_frames = max(0, int(round(bridge_gap_s / hop_s)))
    duration = audio.duration_s
    intervals: List[List[Tuple[float, float]]] = []
    for channel in range(n_channels):
        mask = active[channel].copy()
        _drop_short(mask, min(_DESPECKLE_FRAMES, min_frames))
        if bridge_frames:
            _fill_gaps(mask, bridge_frames)
        _drop_short(mask, min_frames)
        spans: List[Tuple[float, float]] = []
        for start, stop in runs(mask):
            begin = start * hop_s
            end = min(stop * hop_s, duration)
            if end > begin:
                spans.append((begin, end))
        intervals.append(spans)
    return Detection(intervals, levels, bleed, same_audio)
