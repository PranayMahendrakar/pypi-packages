"""Remaining useful life: extrapolate the degradation trend to the threshold."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._features import TINY, linear_fit
from ._health import health_score
from ._io import TableLike
from ._results import HealthResult, RULResult

logger = logging.getLogger(__name__)

NANOS_PER_SECOND = 1e9
MAX_TIMEDELTA_NANOS = 9.0e18
MIN_FIT_POINTS = 4


def _recent_segment(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The most recent half of the series, with a sane floor on its length."""
    n_rows = y.size
    length = max(MIN_FIT_POINTS, n_rows // 2)
    start = max(0, n_rows - length)
    return x[start:], y[start:]


def _row_step(x: np.ndarray) -> float:
    """The typical distance between rows on the time axis, in axis units.

    A cycle counter that advances 10 per row is ordinary in this domain, so the
    raw axis distance to the threshold has to be divided by this to be reported
    honestly as a number of rows.
    """
    axis = np.asarray(x, dtype=float)
    if axis.size < 2:
        return 1.0
    gaps = np.diff(axis)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0.0)]
    if not gaps.size:
        return 1.0
    spacing = float(np.median(gaps))
    return spacing if spacing > 0.0 else 1.0


def _as_threshold(threshold: Any) -> float:
    """The threshold as a health score, with the package's own error wording."""
    try:
        limit = float(threshold)
    except (TypeError, ValueError):
        raise ValueError(
            "threshold must be a health score between 0 and 100, got {0!r}".format(threshold)
        ) from None
    if not np.isfinite(limit) or not 0.0 < limit < 100.0:
        raise ValueError(
            "threshold must be a health score between 0 and 100, got {0!r}".format(threshold)
        )
    return limit


def _confidence(
    r_squared: float, n_points: int, reach: float, trend: str
) -> str:
    """How much to trust the extrapolation, from fit quality and how far it reaches."""
    if trend != "degrading":
        return "low"
    if r_squared >= 0.7 and n_points >= 20 and reach <= 1.0:
        return "high"
    if r_squared >= 0.3 and n_points >= 8 and reach <= 3.0:
        return "medium"
    return "low"


def _infinite(
    health: HealthResult,
    threshold: float,
    slope_reported: float,
    r_squared: float,
    is_temporal: bool,
    notes: List[str],
    reason: str,
) -> RULResult:
    notes.append(reason)
    notes.extend(health.notes)
    return RULResult(
        remaining=float("inf"),
        confidence="low",
        eta=None,
        score=health.score,
        threshold=threshold,
        trend=health.trend,
        slope_per_step=slope_reported,
        r_squared=r_squared,
        is_temporal=is_temporal,
        notes=notes,
    )


def estimate_rul(
    df: Union[TableLike, HealthResult],
    *,
    time: Optional[str] = None,
    channels: Optional[Sequence[str]] = None,
    threshold: float = 30.0,
    baseline: Any = None,
    window: Optional[int] = None,
) -> RULResult:
    """Estimate how long until the degradation trend reaches ``threshold``.

    Fits a straight line through the recent half of the health score and reads
    off where it crosses. A flat or improving trend never crosses, so the
    remaining life comes back as ``float('inf')`` with low confidence - never a
    negative number, and never a crossing date read off a line that is not
    rising, even when the score is already past the threshold. On a timestamped
    frame the answer is a ``Timedelta``; on a numeric axis it is a count of
    rows, whatever spacing that axis happens to use.

    Args:
        df: sensor history, a path to .csv / .parquet, or a
            :class:`HealthResult` you already computed.
        time: the timestamp column; auto-detected when omitted.
        channels: the sensor columns; every numeric column when omitted.
        threshold: the health score that counts as end of useful life.
        baseline: the healthy period, as in :func:`health_score`.
        window: rows per rolling window, as in :func:`health_score`.

    Returns:
        A :class:`RULResult` with ``.remaining``, ``.confidence`` and ``.eta``.
    """
    limit = _as_threshold(threshold)

    health = (
        df
        if isinstance(df, HealthResult)
        else health_score(df, time=time, channels=channels, baseline=baseline, window=window)
    )
    notes: List[str] = []

    index = health.series.index
    is_temporal = isinstance(index, pd.DatetimeIndex)
    if is_temporal:
        x = np.asarray(index.values, dtype="datetime64[ns]").astype("int64").astype(float)
    elif pd.api.types.is_numeric_dtype(index):
        x = np.asarray(index, dtype=float)
    else:  # pragma: no cover - prepare() never produces this
        x = np.arange(health.series.size, dtype=float)
    y = np.asarray(health.series.values, dtype=float)

    xs, ys = _recent_segment(x, y)
    slope, intercept, r_squared = linear_fit(xs, ys)
    span = float(xs[-1] - xs[0]) if xs.size > 1 else 0.0
    # On a numeric axis the fit is per axis unit, but the answer is promised in
    # rows, so both the rate and the distance are converted by the row spacing.
    step = 1.0 if is_temporal else _row_step(x)
    slope_reported = slope * NANOS_PER_SECOND if is_temporal else slope * step
    if not is_temporal and abs(step - 1.0) > TINY:
        notes.append(
            "the time axis advances about {0:g} per row, so the rate and the remaining "
            "life are reported per row rather than per axis unit".format(step)
        )

    past_threshold = bool(health.score >= limit)

    # A flat or improving trend never reaches the threshold, whatever the score
    # is now: projecting a crossing date off a line that is not rising would be
    # inventing one. The note carries the "already past it" fact instead.
    if health.trend != "degrading":
        if past_threshold:
            reason = (
                "the health score is already past the threshold ({0:.1f} vs {1:.1f}) but "
                "the trend is {2}, so no crossing date can be projected; judge this one on "
                "the score itself, which says maintenance is due".format(
                    health.score, limit, health.trend
                )
            )
        else:
            reason = (
                "the trend is {0}, so the threshold is never reached on current "
                "behaviour".format(health.trend)
            )
        return _infinite(health, limit, slope_reported, r_squared, is_temporal, notes, reason)

    if past_threshold:
        notes.append(
            "the health score is already at or past the threshold; maintenance is due now"
        )
        notes.extend(health.notes)
        remaining: Union[pd.Timedelta, float] = pd.Timedelta(0) if is_temporal else 0.0
        return RULResult(
            remaining=remaining,
            confidence="high",
            eta=index[-1],
            score=health.score,
            threshold=limit,
            trend=health.trend,
            slope_per_step=slope_reported,
            r_squared=r_squared,
            is_temporal=is_temporal,
            notes=notes,
        )

    # The slope is measured in score points per nanosecond on timestamped data,
    # so it is compared against zero, never against an absolute epsilon.
    if not np.isfinite(slope) or slope <= 0.0:
        return _infinite(
            health,
            limit,
            slope_reported,
            r_squared,
            is_temporal,
            notes,
            "the recent trend does not rise, so the threshold is never reached",
        )

    level = float(slope * xs[-1] + intercept)
    distance = limit - level
    if distance <= 0.0:
        notes.append("the fitted trend is already at the threshold; treating life as exhausted")
        notes.extend(health.notes)
        return RULResult(
            remaining=pd.Timedelta(0) if is_temporal else 0.0,
            confidence="medium",
            eta=index[-1],
            score=health.score,
            threshold=limit,
            trend=health.trend,
            slope_per_step=slope_reported,
            r_squared=r_squared,
            is_temporal=is_temporal,
            notes=notes,
        )

    delta = float(distance / slope)
    if not np.isfinite(delta) or delta < 0.0:  # pragma: no cover - guarded above
        return _infinite(
            health, limit, slope_reported, r_squared, is_temporal, notes,
            "the trend is too flat to project a crossing",
        )

    reach = delta / span if span > TINY else float("inf")
    confidence = _confidence(r_squared, int(ys.size), reach, health.trend)

    if is_temporal:
        if delta > MAX_TIMEDELTA_NANOS:
            return _infinite(
                health, limit, slope_reported, r_squared, is_temporal, notes,
                "the projected crossing is further out than a timedelta can express",
            )
        remaining = pd.Timedelta(int(round(delta)), unit="ns")
        if remaining >= pd.Timedelta(minutes=1):
            # Sub-second precision on a multi-day projection is false precision.
            remaining = remaining.round("s")
        eta: Any = index[-1] + remaining
    else:
        remaining = float(delta / step)
        eta = float(x[-1] + delta)

    if reach > 3.0:
        notes.append(
            "the projection reaches {0:.1f}x further than the data it was fitted "
            "on; treat it as a rough warning".format(reach)
        )
    if r_squared < 0.3:
        notes.append(
            "the degradation trend is noisy (r-squared {0:.2f}); the estimate is "
            "indicative only".format(r_squared)
        )
    notes.extend(health.notes)

    return RULResult(
        remaining=remaining,
        confidence=confidence,
        eta=eta,
        score=health.score,
        threshold=limit,
        trend=health.trend,
        slope_per_step=slope_reported,
        r_squared=r_squared,
        is_temporal=is_temporal,
        notes=notes,
    )
