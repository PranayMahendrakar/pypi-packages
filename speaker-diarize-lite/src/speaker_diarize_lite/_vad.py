"""An energy detector: which frames are loud enough to be someone talking.

This is deliberately simple and says so. It finds the loud parts of a
recording, not speech as such: music, a door or steady noise louder than the
talkers is "active" too, and speech quieter than the background is missed.

The threshold adapts to the recording. It sits a fraction of the way from the
quiet floor (10th percentile of frame levels) to the peak (loudest 100 ms), but
never more than ``max_below_peak_db`` under the peak and never under the
absolute ``silence_db``. A recording whose loudest 100 ms is below
``silence_db`` is treated as silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

SILENCE_DB = -60.0
FLOOR_FRACTION = 0.35
MAX_BELOW_PEAK_DB = 40.0
FLAT_SPREAD_DB = 10.0
#: How far a flat recording's peak must clear SILENCE_DB before it counts as active
#: rather than as quiet room tone that merely happens to graze the silence floor.
FLAT_MIN_ABOVE_SILENCE_DB = 6.0
BRIDGE_S = 0.3
MIN_BURST_S = 0.1
_PEAK_SPAN_S = 0.1


@dataclass
class Activity:
    """What the energy detector decided.

    Attributes:
        active: per frame, above the threshold (before gap bridging).
        regions: ``(first_frame, end_frame)`` stretches of activity after short
            gaps are bridged and very short bursts dropped.
        threshold_db: the level a frame had to exceed.
        floor_db: the quiet floor of the recording.
        peak_db: the loudest 100 ms of the recording.
        silent: True when nothing reached ``silence_db``.
        flat: True when the recording has no quiet stretches to learn a floor
            from, so everything above ``silence_db`` counted as active.
    """

    active: np.ndarray
    regions: List[Tuple[int, int]]
    threshold_db: float
    floor_db: float
    peak_db: float
    silent: bool
    flat: bool


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """``(start, end)`` index pairs of the True runs in a boolean array."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, changes.size, 2)]


def _loudest_span_db(level_db: np.ndarray, span: int) -> float:
    power = 10.0 ** (level_db / 10.0)
    if power.size <= span:
        return float(10.0 * np.log10(max(float(power.mean()), 1e-30)))
    kernel = np.ones(span) / span
    smoothed = np.convolve(power, kernel, mode="valid")
    return float(10.0 * np.log10(max(float(smoothed.max()), 1e-30)))


def detect_activity(level_db: np.ndarray, hop_s: float, silence_db: float = SILENCE_DB,
                    bridge_s: float = BRIDGE_S, min_burst_s: float = MIN_BURST_S) -> Activity:
    """Mark active frames and group them into regions."""
    n = int(level_db.shape[0])
    if n == 0:
        return Activity(np.zeros(0, dtype=bool), [], float(silence_db), float(silence_db),
                        float("-inf"), True, False)
    peak = _loudest_span_db(level_db, max(1, int(round(_PEAK_SPAN_S / hop_s))))
    floor = float(np.percentile(level_db, 10))
    if peak < silence_db:
        return Activity(np.zeros(n, dtype=bool), [], float(silence_db), floor, peak, True, False)
    spread = peak - floor
    flat = spread < FLAT_SPREAD_DB
    if flat:
        # A flat recording is not automatically active. Room tone and steady hiss are
        # ALSO flat (their own peak barely clears the floor), and marking everything
        # above the floor as active turned quiet room tone with no speech in it into a
        # continuous "speaker" for the whole clip. A flat recording only counts as
        # active talk when its own level is meaningfully above the silence floor - a
        # steady voice or tone sitting there, not room tone that happens to graze it.
        if peak < silence_db + FLAT_MIN_ABOVE_SILENCE_DB:
            return Activity(np.zeros(n, dtype=bool), [], float(silence_db), floor, peak, True, True)
        threshold = max(float(silence_db), floor - 1.0)
    else:
        threshold = max(float(silence_db), floor + FLOOR_FRACTION * spread, peak - MAX_BELOW_PEAK_DB)
    active = level_db > threshold

    bridged = active.copy()
    max_gap = int(round(bridge_s / hop_s))
    found = runs(active)
    for (_, end), (start, _) in zip(found[:-1], found[1:]):
        if start - end <= max_gap:
            bridged[end:start] = True
    min_len = max(1, int(round(min_burst_s / hop_s)))
    regions = [(s, e) for s, e in runs(bridged) if e - s >= min_len]
    return Activity(active, regions, float(threshold), floor, peak, False, bool(flat))
