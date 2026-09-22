"""Unsupervised degradation scoring against a healthy baseline period."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._features import TINY, baseline_scale, choose_window, linear_fit, window_stats
from ._frame import Prepared, prepare
from ._io import TableLike
from ._results import HealthResult

logger = logging.getLogger(__name__)

BASELINE_FRACTION = 0.2
MIN_BASELINE_ROWS = 2

# Every axis is expressed in the same interpretable "deviation unit", so the
# three of them can be combined without one drowning the others out:
#   level    - one unit is one baseline standard deviation of the raw channel;
#   variance - one unit is one doubling (or halving) of the channel's noise;
#   spectral - one unit is a clear shift of the high-frequency band share,
#              measured against how much that share already wandered while the
#              equipment was healthy so ordinary noise does not read as damage.
VARIANCE_UNIT_LOG2 = 1.0
SPECTRAL_NOISE_MULTIPLE = 2.0
SPECTRAL_UNIT_SHARE = 0.05
MAX_UNITS = 50.0
SATURATION_UNITS = 4.0

# The health score that :func:`predictive_maintenance.estimate_rul` treats as the
# end of useful life by default; carried on the result as ``.threshold_hint``.
DEFAULT_THRESHOLD = 30.0

# Guards for degenerate channels. A window with essentially no movement has no
# meaningful spectrum, and a spread far below the channel's own scale is float
# dust rather than noise: both are measured against the baseline scale so the
# guards mean the same thing on a 1500 rpm channel as on a 1.0 g one.
SPECTRAL_ACTIVITY_FRACTION = 0.01
SPREAD_FLOOR_FRACTION = 1e-3

# A channel with no noise at all during the baseline has no calibration behind
# it, so however far it later moves it is capped at the middle of the scale: a
# "look at this" signal with a note attached, never a confident 100.
FLAT_MAX_UNITS = SATURATION_UNITS

# A trend is only called when it both moves the score a visible amount and fits
# a straight line well enough to be more than the wander of a healthy machine.
MIN_TREND_CHANGE = 3.0
MIN_TREND_FIT = 0.2


def _positional(data: Prepared, mask: np.ndarray, notes: List[str]) -> np.ndarray:
    """Carry a caller-supplied row mask through the sort ``prepare`` applied.

    ``baseline=mask`` names rows, not positions in whatever order they happened
    to arrive, so the mask has to travel with its rows when the frame was not in
    time order.
    """
    if not data.was_sorted:
        return mask
    notes.append(
        "the rows were not in time order, so the baseline selection was carried "
        "through the sort with them"
    )
    return np.asarray(data.align(mask), dtype=bool)


def _range_mask(data: Prepared, start: Any, end: Any) -> np.ndarray:
    """Rows whose time-axis value falls inside ``(start, end)``."""
    axis = data.index
    selector = np.ones(data.n_rows, dtype=bool)
    kind = "timestamps" if isinstance(axis, pd.DatetimeIndex) else "numbers"
    try:
        if start is not None:
            selector &= np.asarray(axis >= start)
        if end is not None:
            selector &= np.asarray(axis <= end)
    except (TypeError, ValueError):
        raise ValueError(
            "baseline=(start, end) must be expressed in the same units as the time "
            "axis, which holds {0}{1}; got start={2!r}, end={3!r}".format(
                kind,
                " (column {0!r})".format(data.time_name) if data.time_name else "",
                start,
                end,
            )
        ) from None
    return selector


def _baseline_mask(data: Prepared, baseline: Any, notes: List[str]) -> np.ndarray:
    """Boolean mask over the rows that count as the healthy reference period."""
    n_rows = data.n_rows
    default = max(MIN_BASELINE_ROWS, int(round(n_rows * BASELINE_FRACTION)))
    default = min(default, n_rows)

    mask = np.zeros(n_rows, dtype=bool)
    if baseline is None:
        mask[:default] = True
        return mask

    if isinstance(baseline, slice):
        mask[baseline] = True
    elif isinstance(baseline, (bool, np.bool_)):
        raise ValueError("baseline must be a row count, a fraction, a slice, a mask or a range")
    elif isinstance(baseline, (int, np.integer)):
        count = int(baseline)
        if count <= 0:
            raise ValueError("baseline row count must be positive, got {0}".format(baseline))
        mask[: min(count, n_rows)] = True
    elif isinstance(baseline, (float, np.floating)):
        fraction = float(baseline)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "baseline fraction must be between 0 and 1, got {0}".format(baseline)
            )
        mask[: max(1, int(round(n_rows * fraction)))] = True
    elif isinstance(baseline, tuple) and len(baseline) == 2:
        mask = _range_mask(data, baseline[0], baseline[1])
    else:
        candidate = np.asarray(baseline)
        if candidate.dtype == bool and candidate.size == n_rows:
            mask = _positional(data, candidate, notes)
        elif candidate.ndim == 1 and np.issubdtype(candidate.dtype, np.integer):
            positions = candidate[(candidate >= 0) & (candidate < n_rows)]
            mask[positions] = True
            mask = _positional(data, mask, notes)
        else:
            raise ValueError(
                "baseline must be a row count, a fraction in (0, 1], a slice, a boolean "
                "mask the length of the data, a list of row positions, or a (start, end) "
                "range on the time axis"
            )

    if int(mask.sum()) < MIN_BASELINE_ROWS:
        notes.append(
            "the requested baseline held {0} row(s); fell back to the first {1} rows".format(
                int(mask.sum()), default
            )
        )
        logger.warning("baseline selection too small; falling back to the first %d rows", default)
        mask = np.zeros(n_rows, dtype=bool)
        mask[:default] = True
    return mask


def _units(deviation: np.ndarray) -> np.ndarray:
    """Finite, non-negative deviation units, capped so no axis can swamp the rest."""
    clean = np.nan_to_num(np.asarray(deviation, dtype=float), nan=0.0, posinf=MAX_UNITS,
                          neginf=0.0)
    return np.clip(np.abs(clean), 0.0, MAX_UNITS)


def _channel_units(
    values: np.ndarray, window: int, baseline_mask: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """How far one channel has drifted from its baseline, in deviation units.

    Three independent axes are measured and combined as a root-mean-square, so a
    channel that moves on one axis alone still registers. A channel that never
    moves comes out as exactly zero on all three, never NaN. Returns the units
    and whether the baseline period held the channel perfectly constant, which
    is worth telling the caller about because the scale then has to be assumed.
    """
    stats = window_stats(values, window)
    raw = np.asarray(values, dtype=float)
    baseline_raw = raw[baseline_mask]

    # Level: how far the rolling mean has moved, in raw baseline sigmas. On a
    # channel that was held perfectly constant the scale is floored relative to
    # that held value instead of collapsing to zero.
    scale, flat = baseline_scale(baseline_raw, raw)
    scale = max(scale, TINY)
    reference_mean = stats["mean"][baseline_mask]
    centre = float(np.median(reference_mean)) if reference_mean.size else 0.0
    level = _units((stats["mean"] - centre) / scale)

    # Variance: how many doublings the rolling spread is away from baseline.
    # Both sides of the ratio are floored well below the channel's own scale, so
    # a dead-flat setpoint picking up float dust is not read as an explosion of
    # noise while an ordinary channel's spread is left untouched.
    reference_std = stats["std"][baseline_mask]
    base_std = float(np.median(reference_std)) if reference_std.size else 0.0
    spread_floor = max(SPREAD_FLOOR_FRACTION * scale, TINY)
    ratio = np.maximum(stats["std"], spread_floor) / max(base_std, spread_floor)
    spread = _units(np.log2(np.maximum(ratio, TINY)) / VARIANCE_UNIT_LOG2)

    # Spectral: how far the high-frequency share of the power has shifted, in
    # multiples of the wander that share already showed during the baseline. The
    # share is a ratio, so it stays meaningful however small the signal gets;
    # windows with no real movement in them are excluded rather than trusted.
    reference_share = stats["hf_share"][baseline_mask]
    base_share = float(np.median(reference_share)) if reference_share.size else 0.0
    share_scale = max(
        SPECTRAL_NOISE_MULTIPLE * float(np.std(reference_share)) if reference_share.size else 0.0,
        SPECTRAL_UNIT_SHARE,
    )
    active = np.asarray(stats["std"] > SPECTRAL_ACTIVITY_FRACTION * scale)
    spectral = np.where(active, _units((stats["hf_share"] - base_share) / share_scale), 0.0)

    combined = np.sqrt((level ** 2 + spread ** 2 + spectral ** 2) / 3.0)
    if flat:
        combined = np.minimum(combined, FLAT_MAX_UNITS)
    return _units(combined), flat


def _to_score(units: np.ndarray) -> np.ndarray:
    """Squash deviation units onto 0-100, saturating smoothly.

    ``SATURATION_UNITS`` of drift scores 50; twice that scores 80. The curve
    never reaches 100 exactly and never goes below 0.
    """
    squared = np.maximum(np.asarray(units, dtype=float), 0.0) ** 2
    return 100.0 * squared / (squared + SATURATION_UNITS ** 2)


def _trend_of(x: np.ndarray, score: np.ndarray, window: int) -> Tuple[str, float, float, int]:
    """Classify the recent direction of the score series."""
    n_rows = score.size
    segment = max(4, min(n_rows, max(2 * window, n_rows // 2)))
    start = n_rows - segment
    xs, ys = x[start:], score[start:]
    slope, _, r_squared = linear_fit(xs, ys)
    span = float(xs[-1] - xs[0]) if xs.size > 1 else 0.0
    change = slope * span
    scatter = float(np.std(ys)) * float(np.sqrt(max(0.0, 1.0 - r_squared))) if ys.size > 1 else 0.0
    tolerance = max(MIN_TREND_CHANGE, scatter)
    if r_squared < MIN_TREND_FIT or abs(change) <= tolerance:
        trend = "stable"
    elif change > 0.0:
        trend = "degrading"
    else:
        trend = "improving"
    return trend, slope, r_squared, start


def health_score(
    df: TableLike,
    *,
    time: Optional[str] = None,
    channels: Optional[Sequence[str]] = None,
    baseline: Any = None,
    window: Optional[int] = None,
) -> HealthResult:
    """Score how far equipment has drifted from healthy, 0 (healthy) to 100 (failed).

    Needs no failure labels. The first 20% of the rows (or whatever ``baseline``
    selects) is taken as healthy, and every later window is compared against it
    on three axes per channel: the level it settles at, how much it varies, and
    where its energy sits in the frequency band.

    A channel held perfectly constant through the baseline has no measured noise
    to compare against; its scale is assumed from the level it was held at, its
    contribution is capped mid-range, and a note records that. A positional
    ``baseline`` (mask or row positions) names rows, so it travels with them
    when the frame has to be sorted into time order.

    Args:
        df: sensor history as a DataFrame or a path to .csv / .parquet.
        time: the timestamp column; auto-detected when omitted.
        channels: the sensor columns; every numeric column when omitted.
        baseline: the healthy period - a row count, a fraction in (0, 1], a
            slice, a boolean mask, row positions, or a ``(start, end)`` range.
        window: rows per rolling window; about a tenth of the history by default.

    Returns:
        A :class:`HealthResult` with ``.score``, ``.series``, ``.trend`` and
        ``.contributors``.
    """
    data = prepare(df, time=time, channels=channels, min_rows=MIN_BASELINE_ROWS)
    notes = list(data.notes)
    n_rows = data.n_rows

    mask = _baseline_mask(data, baseline, notes)
    baseline_rows = int(mask.sum())

    chosen = choose_window(n_rows, window)
    if chosen > baseline_rows:
        notes.append(
            "window shortened from {0} to {1} rows so it fits inside the "
            "{2}-row baseline".format(chosen, max(2, baseline_rows), baseline_rows)
        )
        chosen = max(2, baseline_rows)
    chosen = int(max(2, min(chosen, n_rows)))

    per_channel: Dict[str, np.ndarray] = {}
    for name in data.channels:
        units, flat = _channel_units(data.values[name], chosen, mask)
        per_channel[name] = units
        if flat and float(units.max(initial=0.0)) > TINY:
            notes.append(
                "channel {0!r} never moved during the baseline, so it has no measured "
                "noise level; its scale was assumed from the value it was held at, and "
                "it is capped at a mid-range contribution rather than trusted to "
                "saturate the score".format(name)
            )
    stacked = np.vstack([per_channel[name] for name in data.channels])
    # Equipment is as sick as its sickest sensor, but broad drift is worse than
    # one noisy channel: half the worst channel, half the average of them all.
    combined = 0.5 * stacked.max(axis=0) + 0.5 * stacked.mean(axis=0)
    score = _to_score(combined)

    tail = max(1, min(n_rows, max(chosen, n_rows // 10)))
    weights = {name: float(per_channel[name][-tail:].sum()) for name in data.channels}
    total = float(sum(weights.values()))
    if total > TINY:
        contributors = {name: weights[name] / total for name in data.channels}
    else:
        contributors = {name: 0.0 for name in data.channels}
        notes.append("no channel shows any deviation from baseline; shares are all zero")

    trend, _, _, _ = _trend_of(data.x, score, chosen)
    baseline_score = float(np.mean(score[mask])) if baseline_rows else 0.0

    if n_rows < 10:
        notes.append(
            "only {0} rows of history; the score is directional rather than precise".format(n_rows)
        )

    series = pd.Series(score, index=data.index, name="health_score")
    return HealthResult(
        score=float(score[-1]),
        series=series,
        trend=trend,
        contributors=contributors,
        channels=list(data.channels),
        window=chosen,
        baseline_rows=baseline_rows,
        baseline_score=baseline_score,
        threshold_hint=DEFAULT_THRESHOLD,
        notes=notes,
    )
