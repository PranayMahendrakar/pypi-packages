"""Sensor faults that are not statistical outliers.

A dead sensor is not an outlier: it is boringly consistent. These scans catch the
hardware-shaped problems a z-score will always miss - a channel that stopped
moving, one pinned at zero, one railed against its measurement limit, a burst of
dropped samples, and a sudden step in level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd

#: Fault codes in the order they read best in a report.
FAULT_ORDER = (
    "all-missing",
    "no-numeric-data",
    "flatline",
    "stuck-at-zero",
    "railed-at-min",
    "railed-at-max",
    "step-change",
    "missing-burst",
)

#: Plain-words explanation for each fault code.
FAULT_HELP = {
    "all-missing": "no usable readings at all; treat this channel as a failed sensor",
    "no-numeric-data": "the column held no numbers; treat this channel as a failed sensor",
    "flatline": "the reading stopped changing for a long run, which a live sensor does not do",
    "stuck-at-zero": "the reading sat at exactly zero for a long run",
    "railed-at-min": "the reading sat against the lowest value this channel ever showed",
    "railed-at-max": "the reading sat against the highest value this channel ever showed",
    "step-change": "the level jumped and stayed there, usually a recalibration or a swap",
    "missing-burst": "a run of consecutive samples was missing",
}

#: At most this many step changes are reported per channel.
MAX_STEPS_PER_CHANNEL = 20


@dataclass
class FaultEvent:
    """One contiguous stretch of a channel that shows a fault."""

    start: int
    end: int
    kind: str
    severity: float


def runs_of(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """Contiguous ``True`` stretches of ``mask`` as inclusive ``(start, end)`` pairs."""
    flags = np.asarray(mask, dtype=bool)
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(0, edges.size, 2)]


def robust_sigma(values: np.ndarray) -> float:
    """A spread estimate that a handful of wild points cannot inflate.

    Uses the median absolute deviation, falling back to the mean absolute
    deviation when more than half the values are identical (a sensor that is
    mostly stuck but occasionally spikes). Returns ``0.0`` when there is no
    variation at all, which callers must treat as "no scoring possible".
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    centre = float(np.median(finite))
    mad = float(np.median(np.abs(finite - centre)))
    if mad > 0.0:
        return 1.4826 * mad
    mean_ad = float(np.mean(np.abs(finite - centre)))
    if mean_ad > 0.0:
        return 1.2533 * mean_ad
    return 0.0


def noise_sigma(values: np.ndarray) -> float:
    """The channel's own noise level, measured so that drift cannot inflate it.

    ``robust_sigma`` applied to rolling-median residuals under-reports the noise:
    a centred rolling median partly tracks the noise it is meant to remove, so the
    residuals are narrower than the readings really are. Scoring against that
    shrunken spread turns a 3-sigma test into roughly a 2.4-sigma one, and a clean
    channel then flags several times more often than ``sensitivity`` promises.

    The lag-one difference of a noisy signal has variance twice the noise variance,
    whatever slow trend sits underneath it, so ``median(|diff|)`` scaled by
    ``1.4826 / sqrt(2)`` recovers the noise level while ignoring drift. Taking the
    median keeps a handful of genuine spikes from inflating it.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 0.0
    steps = np.abs(np.diff(finite))
    steps = steps[np.isfinite(steps)]
    if steps.size == 0:
        return 0.0
    median_step = float(np.median(steps))
    if median_step > 0.0:
        return 1.4826 * median_step / math.sqrt(2.0)
    positive = steps[steps > 0.0]
    if positive.size == 0:
        return 0.0
    return 1.4826 * float(np.median(positive)) / math.sqrt(2.0)


def tolerance(values: np.ndarray) -> float:
    """How close two readings must be to count as the same reading."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1e-12
    span = float(np.max(finite) - np.min(finite))
    magnitude = float(np.max(np.abs(finite)))
    return max(1e-12, 1e-9 * max(span, magnitude, 1.0))


def min_run_length(n_valid: int) -> int:
    """How long a stretch must be before "it stopped moving" is a real finding."""
    return int(max(5, min(100, round(0.05 * max(n_valid, 0)))))


def _flat_runs(values: np.ndarray, tol: float, min_run: int) -> List[Tuple[int, int]]:
    """Inclusive spans over which the reading never moved by more than ``tol``."""
    if values.size < 2:
        return []
    with np.errstate(invalid="ignore"):
        same = np.abs(np.diff(values)) <= tol
    same &= np.isfinite(values[:-1]) & np.isfinite(values[1:])
    spans = []
    for start, end in runs_of(same):
        length = end - start + 2  # a run of k gaps covers k+1 samples
        if length >= min_run:
            spans.append((start, end + 1))
    return spans


def _step_changes(
    values: np.ndarray,
    tol: float,
    resid_scale: float,
    sensitivity: float,
) -> List[Tuple[int, float, float]]:
    """Positions where the level jumped and stayed jumped.

    Returns ``(position, jump, threshold)`` triples. A slow ramp is deliberately
    not a step: the jump must also be a real share of the channel's whole range.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return []
    span = float(np.max(finite) - np.min(finite))
    if span <= tol:
        return []

    n = values.size
    window = int(max(3, min(50, round(0.02 * n))))
    if n < 2 * window + 2:
        window = max(3, n // 4)
    min_periods = max(2, window // 2)

    series = pd.Series(values, dtype="float64").ffill().bfill()
    before = series.rolling(window, min_periods=min_periods).median().shift(1)
    after = series[::-1].rolling(window, min_periods=min_periods).median()[::-1]
    delta = (after - before).to_numpy(dtype="float64")

    sigma_delta = robust_sigma(delta)
    if sigma_delta <= 0.0:
        sigma_delta = resid_scale
    threshold = max(
        max(4.0, float(sensitivity) + 1.0) * sigma_delta,
        3.0 * resid_scale,
        0.10 * span,
    )
    if threshold <= 0.0:
        threshold = tol

    magnitude = np.abs(delta)
    with np.errstate(invalid="ignore"):
        hits = np.isfinite(delta) & (magnitude >= threshold)
    if not hits.any():
        return []

    found: List[Tuple[int, float, float]] = []
    for start, end in runs_of(hits):
        segment = magnitude[start : end + 1]
        peak = start + int(np.argmax(segment))
        found.append((peak, float(magnitude[peak]), float(threshold)))
    found.sort(key=lambda item: item[1], reverse=True)
    return found[:MAX_STEPS_PER_CHANNEL]


def scan_faults(
    values: np.ndarray,
    n_valid: int,
    resid_scale: float,
    sensitivity: float,
) -> Tuple[List[str], List[FaultEvent], List[str]]:
    """Look for hardware-shaped faults in one channel.

    Args:
        values: the channel as float64, with missing readings as NaN.
        n_valid: how many readings were usable numbers.
        resid_scale: robust spread of the channel's rolling residual, or 0.0.
        sensitivity: the caller's threshold, in robust sigmas.

    Returns:
        ``(fault codes, fault events, notes)``. Notes are one short phrase each.
    """
    faults: List[str] = []
    events: List[FaultEvent] = []
    notes: List[str] = []
    if n_valid == 0:
        return faults, events, notes

    valid = np.isfinite(values)
    tol = tolerance(values)
    min_run = min_run_length(n_valid)
    finite = values[valid]
    low = float(np.min(finite))
    high = float(np.max(finite))
    constant = (high - low) <= tol

    # --- stopped moving -------------------------------------------------
    spans = _flat_runs(values, tol, min_run)
    if constant and n_valid >= 3 and not spans:
        # Short but wholly constant: still a dead channel, not an outlier.
        positions = np.flatnonzero(valid)
        spans = [(int(positions[0]), int(positions[-1]))]
    longest_flat = 0
    for start, end in spans:
        level = float(values[start])
        length = end - start + 1
        longest_flat = max(longest_flat, length)
        severity = length / float(max(min_run, 1))
        if abs(level) <= tol:
            kind = "stuck-at-zero"
            codes = ["stuck-at-zero", "flatline"]
        elif not constant and abs(level - high) <= tol:
            kind, codes = "railed", ["railed-at-max"]
        elif not constant and abs(level - low) <= tol:
            kind, codes = "railed", ["railed-at-min"]
        else:
            kind, codes = "flatline", ["flatline"]
        events.append(FaultEvent(start, end, kind, severity))
        faults.extend(codes)
    if longest_flat:
        notes.append("held the same reading for %d samples in a row" % longest_flat)

    # --- pinned against a measurement limit -----------------------------
    if not constant:
        for code, rail in (("railed-at-min", low), ("railed-at-max", high)):
            with np.errstate(invalid="ignore"):
                at_rail = valid & (np.abs(values - rail) <= tol)
            count = int(at_rail.sum())
            if count >= 5 and count / float(n_valid) >= 0.10 and code not in faults:
                faults.append(code)
                notes.append(
                    "%.1f%% of readings sit exactly at %s"
                    % (100.0 * count / n_valid, describe_value(rail))
                )

    # --- dropped samples ------------------------------------------------
    burst_min = int(max(3, min(10, min_run)))
    gaps = [
        (start, end) for start, end in runs_of(~valid) if (end - start + 1) >= burst_min
    ]
    if gaps:
        faults.append("missing-burst")
        longest = max(end - start + 1 for start, end in gaps)
        notes.append(
            "%d burst%s of missing samples, longest %d rows"
            % (len(gaps), "" if len(gaps) == 1 else "s", longest)
        )
        for start, end in gaps:
            events.append(
                FaultEvent(
                    start, end, "missing", (end - start + 1) / float(max(burst_min, 1))
                )
            )

    # --- the level jumped and stayed ------------------------------------
    steps = _step_changes(values, tol, resid_scale, sensitivity)
    if steps:
        faults.append("step-change")
        notes.append(
            "%d sudden step change%s in level"
            % (len(steps), "" if len(steps) == 1 else "s")
        )
        for position, jump, threshold in steps:
            severity = jump / threshold if threshold > 0 else 1.0
            events.append(FaultEvent(position, position, "step", float(severity)))

    return order_faults(faults), events, notes


def order_faults(codes: Sequence[str]) -> List[str]:
    """Unique fault codes in the canonical report order."""
    seen = set()
    ordered: List[str] = []
    for code in FAULT_ORDER:
        if code in codes and code not in seen:
            seen.add(code)
            ordered.append(code)
    for code in codes:  # anything unexpected still gets reported
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def explain(codes: Sequence[str]) -> List[str]:
    """One plain sentence per fault code, for the report's notes."""
    return ["%s: %s" % (code, FAULT_HELP[code]) for code in codes if code in FAULT_HELP]


def describe_value(value: Optional[float]) -> str:
    """Format a reading for a message without scientific-notation surprises."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return "%g" % float(value)
