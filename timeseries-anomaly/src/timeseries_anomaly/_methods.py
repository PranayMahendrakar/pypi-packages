"""
# NOTE: every .to_numpy() in this package asks for copy=True.
#
# Under pandas copy-on-write - the default from pandas 3 - to_numpy() returns a
# READ-ONLY view of the Series' own buffer, and this module writes into those arrays
# constantly. Without the copy, 52 of these tests fail with "assignment destination is
# read-only" on any current pandas, while passing on pandas 2. The arrays here are one
# column of a time series, so the copy costs nothing worth measuring.
The baselines a point can be compared against.

Each fitter answers one question: "what should this point have been?". It returns
a :class:`Baseline` holding that expectation for every point, the robust scale the
residuals are measured in, and a frozen level that a streaming detector can reuse
for later batches. Every fitter degrades to a simpler one rather than failing, and
says so in ``warnings``.
"""
from __future__ import annotations

import logging
import warnings as _pywarnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._robust import choose_scale, median as robust_median, roughness

LOG = logging.getLogger(__name__)

#: every method name accepted by ``detect(method=...)``
METHODS = ("auto", "all", "zscore", "iqr", "rolling", "seasonal", "ewma")
#: the methods that fit one baseline (i.e. everything except the meta-methods)
SINGLE_METHODS = ("zscore", "iqr", "rolling", "seasonal", "ewma")
#: candidate cycle lengths in seconds, smallest first
_CYCLE_SECONDS = (3600.0, 86400.0, 604800.0, 31557600.0)
#: below this many points a rolling baseline has nothing to roll over
MIN_ROLLING_POINTS = 7
#: autocorrelation a candidate cycle must clear before it is believed
MIN_SEASONAL_CORRELATION = 0.3
#: a side of a neighbourhood shorter than this can be swung by a single outlier
MIN_TRUSTED_SIDE = 3
#: below this, a residual is a smooth curve rather than scatter (see _robust.roughness)
SMOOTH_RESIDUAL = 0.3
#: a baseline that pushes more than this share of the series past the threshold
#: is not describing the series at all
DEGENERATE_SHARE = 0.2
#: too few points to judge whether a residual scatters or drifts
MIN_SHAPE_POINTS = 8


@dataclass
class Baseline:
    """What a method expected at each point, and how far off is far."""

    name: str
    expected: np.ndarray
    scale: Optional[float]
    scale_kind: str
    level: float = 0.0
    seasonal: Optional[np.ndarray] = None
    next_phase: int = 0
    seasonality: Optional[int] = None
    window: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


def auto_window(n: int) -> int:
    """The rolling window used when the caller does not choose one."""
    return max(MIN_ROLLING_POINTS, int(n) // 20)


def resolve_method(method: str, n: int) -> str:
    """Turn ``"auto"`` into the concrete method it means for a series of `n` points."""
    if method != "auto":
        return method
    return "rolling" if n >= 30 else "zscore"


def infer_period(time: Optional[np.ndarray], n: int) -> Optional[int]:
    """Guess a seasonal period from evenly spaced timestamps.

    Returns the number of points in the shortest natural cycle (hour, day, week,
    year) that fits at least twice into `n`, or ``None`` when nothing fits.
    """
    if time is None or n < 4:
        return None
    try:
        stamps = pd.to_datetime(pd.Series(time), errors="coerce")
    except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
        return None
    stamps = stamps.dropna().sort_values()
    if len(stamps) < 4:
        return None
    deltas = stamps.diff().dropna().dt.total_seconds().to_numpy(copy=True)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if deltas.size == 0:
        return None
    step = float(np.median(deltas))
    if step <= 0.0:
        return None
    for cycle in _CYCLE_SECONDS:
        period = int(round(cycle / step))
        if 2 <= period <= n // 2:
            return period
    return None


def infer_period_from_values(values: np.ndarray) -> Optional[int]:
    """Guess a repeating cycle length from the numbers alone, no timestamps needed.

    Scans the normalised autocorrelation of the mean-centred series and keeps the
    strongest lag that is a genuine local peak. A series that merely trends has a
    smoothly decaying autocorrelation and no peak, so it correctly yields ``None``
    rather than a bogus two-point season. Fully deterministic, no sampling.
    """
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    n = int(series.size)
    if n < 8:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        centred = series - float(np.mean(series))
        denominator = float(np.dot(centred, centred))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    max_lag = n // 2
    # the same sums of products as a lag-by-lag loop, but taken through a real
    # FFT so the cost is n log n rather than n squared: an hour of 1 Hz telemetry
    # used to take a minute here, and now takes a moment.
    size = 1 << int(2 * n - 1).bit_length()
    with np.errstate(over="ignore", invalid="ignore"):
        spectrum = np.fft.rfft(centred, size)
        covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), size)
    scores = np.zeros(max_lag + 2, dtype=float)
    scores[1 : max_lag + 1] = covariance[1 : max_lag + 1] / denominator
    scores[~np.isfinite(scores)] = 0.0
    best_lag: Optional[int] = None
    best_score = MIN_SEASONAL_CORRELATION
    for lag in range(2, max_lag + 1):
        score = float(scores[lag])
        if score <= best_score:
            continue
        if score > scores[lag - 1] and score >= scores[lag + 1]:
            best_lag, best_score = lag, score
    return best_lag


def _nan_median(values: np.ndarray, default: float = 0.0) -> float:
    """``np.nanmedian`` without the all-NaN warning."""
    kept = np.asarray(values, dtype=float)
    kept = kept[np.isfinite(kept)]
    if kept.size == 0:
        return default
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.median(kept))


def _neighbour_baseline(values: np.ndarray, size: int) -> np.ndarray:
    """What the neighbours of each point say it should have been.

    Every point is compared against the points around it and never against
    itself: the median of the `radius` points behind it and the median of the
    `radius` points ahead of it, averaged. On a steadily rising or falling signal
    those two medians sit the same distance below and above the point, so their
    average lands back on it and an ordinary ramp stays ordinary.

    Two details earn their keep. Leaving the point out of its own window matters:
    a centred median that includes it returns the point itself roughly one time
    in `size`, which piles a spike of exact zeros into the residuals, shrinks the
    spread they are measured against, and turns clean noise into false alarms.
    And at the two ends, where only one side exists, the missing side is replaced
    by the other one carried across the gap with the local slope - measured from
    the same two medians a full window away, where both sides are present - so
    that the first and last reading are judged like any other rather than being
    compared against themselves and never flagged.
    """
    n = int(values.size)
    radius = max(1, int(size) // 2)
    trusted = min(MIN_TRUSTED_SIDE, radius)
    behind_median = (
        pd.Series(values).rolling(radius, min_periods=1).median().shift(1).to_numpy(dtype=float, copy=True)
    )
    ahead_median = (
        pd.Series(values[::-1])
        .rolling(radius, min_periods=1)
        .median()
        .shift(1)
        .to_numpy(dtype=float, copy=True)[::-1]
    )
    behind = np.minimum(np.arange(n), radius).astype(float)
    ahead = np.minimum(np.arange(n - 1, -1, -1), radius).astype(float)
    behind_ok = np.isfinite(behind_median)
    ahead_ok = np.isfinite(ahead_median)
    # the two medians of a full window sit radius+1 apart, so their difference is
    # the local slope; ffill/bfill carries the nearest well-founded one into the ends
    full = (behind >= radius) & (ahead >= radius) & behind_ok & ahead_ok
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.where(full, (ahead_median - behind_median) / float(radius + 1), np.nan)
    # copy=True is not optional: with pandas copy-on-write, the default in pandas 3,
    # to_numpy() hands back a read-only view of the Series' own buffer, and the next
    # line writes into it. Without the copy this raises "assignment destination is
    # read-only" for every caller on a current pandas.
    slope = pd.Series(raw).ffill().bfill().fillna(0.0).to_numpy(dtype=float, copy=True)
    slope[~np.isfinite(slope)] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        from_behind = np.where(behind_ok, behind_median, 0.0) + slope * (behind + 1.0) / 2.0
        from_ahead = np.where(ahead_ok, ahead_median, 0.0) - slope * (ahead + 1.0) / 2.0
    # a side of one or two points is one outlier away from nonsense, so it is only
    # used when nothing better is available
    back_weight = np.where(behind_ok & (behind >= trusted), behind, 0.0)
    fore_weight = np.where(ahead_ok & (ahead >= trusted), ahead, 0.0)
    total = back_weight + fore_weight
    spare_back = np.where(behind_ok, behind, 0.0)
    spare_fore = np.where(ahead_ok, ahead, 0.0)
    thin = total <= 0.0
    back_weight = np.where(thin, spare_back, back_weight)
    fore_weight = np.where(thin, spare_fore, fore_weight)
    total = np.where(thin, spare_back + spare_fore, total)
    with np.errstate(over="ignore", invalid="ignore"):
        blended = (from_behind * back_weight + from_ahead * fore_weight) / np.where(
            total > 0.0, total, 1.0
        )
    return np.where(total > 0.0, blended, values)


def _fill(expected: np.ndarray, level: float) -> np.ndarray:
    """Replace any gap in a baseline with the overall level."""
    expected = np.asarray(expected, dtype=float)
    holes = ~np.isfinite(expected)
    if holes.any():
        expected = expected.copy()
        expected[holes] = level
    return expected


def fit_zscore(values: np.ndarray) -> Baseline:
    """Compare every point against the median of the whole series."""
    centre = robust_median(values)
    expected = np.full(values.shape, centre, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        scale, kind = choose_scale(values - centre, values)
    return Baseline("zscore", expected, scale, kind, level=centre)


def fit_iqr(values: np.ndarray) -> Baseline:
    """Same flat baseline as ``zscore``, but the spread comes from the quartiles."""
    centre = robust_median(values)
    expected = np.full(values.shape, centre, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        scale, kind = choose_scale(values - centre, values, prefer="iqr")
    return Baseline("iqr", expected, scale, kind, level=centre)


def fit_rolling(values: np.ndarray, window: Optional[int] = None) -> Baseline:
    """Compare every point against a centred rolling median of its neighbours.

    The straight line the series is travelling along is taken out before the
    neighbours are consulted and added back afterwards. A median has to pick one
    of the readings it is given, and on a slope steeper than the noise the wrong
    one of them is a whole step away - so a single spike inside a window drags
    that window's median by a step, and the handful of readings on either side of
    a real anomaly get measured against a baseline the anomaly itself moved. Level
    with the trend removed, those same neighbours sit on top of each other and
    picking any of them costs nothing.
    """
    n = int(values.size)
    size = int(window) if window else auto_window(n)
    size = max(2, size)
    if n < size or n < 3:
        fallback = fit_zscore(values)
        fallback.warnings.append(
            f"the series has {n} point(s), fewer than the rolling window of {size}; "
            "fell back to zscore"
        )
        fallback.window = size
        return fallback
    level = robust_median(values)
    line = _robust_line(values)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = _fill(_neighbour_baseline(values - line, size) + line, level)
        scale, kind = choose_scale(values - expected, values)
    tail = values[-size:]
    return Baseline(
        "rolling",
        expected,
        scale,
        kind,
        level=_nan_median(tail, level),
        window=size,
    )


def _robust_line(values: np.ndarray) -> np.ndarray:
    """The straight line the series is travelling along.

    The slope is the median rise across half the series - the middle of every
    ``x[i + half] - x[i]`` divided by `half` - and the intercept the median of what
    is left over. Taking the rise across a long gap rather than from one point to
    the next is what makes it safe on a repeating signal: the median step of a
    sawtooth that climbs four times and drops once is the climb, which would read
    a flat signal as a steady rise, while across half a series the ups and the
    downs have already cancelled. A spike contributes one high difference and one
    low one, so it cannot tilt the line either, and on a true ramp the estimate is
    exact.
    """
    n = int(values.size)
    position = np.arange(n, dtype=float)
    if n < 2:
        return np.full(n, robust_median(values), dtype=float)
    series = np.asarray(values, dtype=float)
    half = max(1, n // 2)
    with np.errstate(over="ignore", invalid="ignore"):
        slope = _nan_median(series[half:] - series[:-half], 0.0) / float(half)
        if not np.isfinite(slope):
            slope = 0.0
        intercept = _nan_median(series - slope * position, 0.0)
    if not np.isfinite(intercept):
        intercept = 0.0
    return intercept + slope * position


def _ewma_expected(values: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """The exponential average of everything before each point, and the raw average.

    The average is fed a trailing median of width 3 rather than the raw values, so
    one spike cannot drag the baseline up and then flag the perfectly normal points
    that follow it while the average decays back. The very first point has nothing
    before it to average, so rather than being handed a copy of itself - which can
    never look wrong - it is given the same average run backwards from the points
    that follow it.

    The two points that start the average get their guard from the first three
    readings together rather than from a window of one and of two, which are too
    short for a median to throw anything out. Otherwise an anomaly sitting on the
    very first reading becomes the value the whole average starts from, and the
    next span of perfectly ordinary readings are all measured against it while it
    decays - one real anomaly at the front turning into a dozen invented ones
    behind it.
    """
    guarded = pd.Series(values).rolling(3, min_periods=1).median()
    if values.size >= 3:
        guarded.iloc[:2] = _nan_median(np.asarray(values, dtype=float)[:3], float(guarded.iloc[0]))
    smoothed = guarded.ewm(span=width, adjust=False, ignore_na=True).mean()
    expected = smoothed.shift(1).to_numpy(dtype=float, copy=True)
    if values.size > 1:
        backward = (
            pd.Series(np.asarray(values, dtype=float)[::-1])
            .rolling(3, min_periods=1)
            .median()
            .ewm(span=width, adjust=False, ignore_na=True)
            .mean()
            .to_numpy(dtype=float, copy=True)[::-1]
        )
        expected[0] = backward[1]
    return expected, smoothed.to_numpy(dtype=float, copy=True)


def fit_ewma(values: np.ndarray, span: Optional[int] = None) -> Baseline:
    """Compare every point against the exponential moving average of the points before it.

    An average of past points always trails a rising or falling series, and that
    lag belongs to the smoother rather than to the data. So the smoother is not
    shown the trend at all: the straight line the series is travelling along
    (:func:`_robust_line`) is taken out first, the average runs over what is left,
    and the line is added back to make the baseline. Nothing then has to be
    corrected afterwards - there is no lag to correct, and no warm-up either,
    because the average now starts inside the scatter instead of climbing towards
    it from wherever the series began.

    Detrending first also keeps one anomaly from becoming several. The median that
    guards the average has to pick one of three neighbouring readings, and on a
    steep slope the wrong one of the three is a whole step away - on an energy
    meter climbing faster than it wobbles, that misplaced step is worth more than
    the noise, and the readings just after a spike get measured against a baseline
    nudged by it. On the detrended series those same three readings sit level with
    each other and picking any of them costs nothing.

    A robust median of whatever remains is taken out at the end, so a trend that
    is not a straight line is still centred before anything is scored.
    """
    n = int(values.size)
    width = int(span) if span else max(3, min(n if n else 3, auto_window(n)))
    width = max(2, width)
    if n < 2:
        fallback = fit_zscore(values)
        fallback.warnings.append(f"the series has {n} point(s); fell back to zscore")
        return fallback
    level = robust_median(values)
    line = _robust_line(values)
    with np.errstate(over="ignore", invalid="ignore"):
        detrended = np.asarray(values, dtype=float) - line
    expected, smoothed = _ewma_expected(detrended, width)
    expected = np.where(np.isfinite(expected), expected, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = _fill(expected + line, level)
        lag = robust_median(values - expected, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        if np.isfinite(lag) and lag != 0.0:
            expected = expected + lag
        else:
            lag = 0.0
        frozen = float(line[-1]) + _nan_median(smoothed[-1:], 0.0) + float(lag)
        scale, kind = choose_scale(values - expected, values)
    return Baseline("ewma", expected, scale, kind, level=frozen, window=width)


def _cycle_trend(values: np.ndarray, period: int, level: float) -> np.ndarray:
    """The slow movement under a seasonal signal: a rolling median of one whole cycle.

    A window of exactly one period holds the same set of phases wherever it sits,
    so the season itself cannot make the trend wobble. The first and last half
    cycle carry the nearest full-window value rather than a partial one, which
    would see only part of the season and drag the baseline away from the data.
    """
    rolled = pd.Series(values).rolling(period, center=True, min_periods=period).median()
    return _fill(rolled.ffill().bfill().to_numpy(dtype=float, copy=True), level)


def _phase_grid(detrended: np.ndarray, period: int) -> np.ndarray:
    """The series folded into one row per cycle, the last row padded with gaps."""
    n = int(detrended.size)
    padding = (-n) % period
    return np.concatenate(
        [np.asarray(detrended, dtype=float), np.full(padding, np.nan)]
    ).reshape(-1, period)


def _phase_medians(detrended: np.ndarray, period: int) -> np.ndarray:
    """The median of every point sharing a position in the cycle, as measured."""
    grid = _phase_grid(detrended, period)
    with _pywarnings.catch_warnings():
        _pywarnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over="ignore", invalid="ignore"):
            medians = np.nanmedian(grid, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def _phase_profile(detrended: np.ndarray, period: int) -> np.ndarray:
    """The repeating shape, centred on zero so the trend carries the level."""
    medians = _phase_medians(detrended, period)
    with np.errstate(over="ignore", invalid="ignore"):
        return medians - float(np.mean(medians))


def _phase_profile_per_point(detrended: np.ndarray, period: int) -> np.ndarray:
    """The seasonal shape for each point, measured without that point's own help.

    Same medians as :func:`_phase_profile`, except that the value being explained
    is left out of the median explaining it. It matters for the same reason it
    matters in :func:`_neighbour_baseline`: a point that happens to be the middle
    of its own phase gets a residual of exactly zero, and a series only five or
    ten cycles long hands out that exact zero to a fifth of its points. Those
    zeros pile up under the median absolute deviation, shrink the spread every
    residual is measured against, and turn ordinary cyclic noise into a few
    anomalies per hundred points.

    Every phase is folded into a column and sorted once, then the middle of the
    column with one entry skipped is read off by index: a sort for the whole
    series rather than a median per point, and no loop over the period, which a
    long inferred cycle would otherwise make expensive. A point with nobody else
    to ask - the only reading at its position in the cycle, or a gap that still
    wants a shape under it - falls back to the plain median of its phase.
    """
    n = int(detrended.size)
    grid = _phase_grid(detrended, period)
    rows = int(grid.shape[0])
    present = np.isfinite(grid)
    # gaps sort to the back, so the first `counts` entries of each sorted column
    # are exactly its finite values in order
    sortable = np.where(present, grid, np.inf)
    order = np.argsort(sortable, axis=0, kind="stable")
    ranked = np.take_along_axis(sortable, order, axis=0)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(rows)[:, None] + np.zeros_like(order), axis=0)
    counts = present.sum(axis=0)[None, :]
    shorter = counts - 1
    low = (shorter - 1) // 2
    high = shorter // 2
    with np.errstate(over="ignore", invalid="ignore"):
        below = np.take_along_axis(ranked, np.clip(np.where(rank <= low, low + 1, low), 0, rows - 1), axis=0)
        above = np.take_along_axis(ranked, np.clip(np.where(rank <= high, high + 1, high), 0, rows - 1), axis=0)
        folded = 0.5 * (below + above)
    folded = np.where(present & (counts > 1), folded, _phase_medians(detrended, period)[None, :])
    return folded.reshape(-1)[:n]


def fit_seasonal(values: np.ndarray, period: Optional[int]) -> Baseline:
    """Compare every point against a trend plus a repeating seasonal shape.

    A robust classical decomposition: a rolling-median trend, then the median of
    the detrended points sharing each position in the cycle. Falls back to a
    rolling baseline when there is not enough data for two full cycles.

    The decomposition is taken twice. A first pass gives the seasonal shape; the
    trend is then re-measured on the series with that shape subtracted out, and
    the shape re-measured against the new trend. The second pass is what keeps a
    single outlier local. On the raw signal the values inside a trend window are
    spread across the whole swing of the season, so they sit far apart and one
    extreme point is enough to move their median - dragging the baseline away
    from the data for a whole cycle either side and reporting a dozen innocent
    neighbours along with the real event. Once the season is subtracted, that
    same window holds a period's worth of points at one level, where an outlier
    is one vote among many and cannot shift the middle of them.
    """
    n = int(values.size)
    if not period or period < 2 or n < 2 * period:
        fallback = fit_rolling(values)
        if not period or period < 2:
            fallback.warnings.append(
                "no seasonality was given or could be inferred; fell back to "
                f"{fallback.name}"
            )
        else:
            fallback.warnings.append(
                f"a season of {period} points needs at least {2 * period} points, "
                f"the series has {n}; fell back to {fallback.name}"
            )
        return fallback
    level = robust_median(values)
    phase = np.arange(n) % period
    with np.errstate(over="ignore", invalid="ignore"):
        trend = _cycle_trend(values, period, level)
        profile = _phase_profile(values - trend, period)
        deseasonalised = values - profile[phase]
        trend = _fill(_neighbour_baseline(deseasonalised, period), level)
        detrended = values - trend
        medians = _phase_medians(detrended, period)
        centre = float(np.mean(medians))
        profile = medians - centre
        shape = _phase_profile_per_point(detrended, period)
        shape = np.where(np.isfinite(shape), shape, medians[phase]) - centre
        expected = _fill(trend + shape, level)
        scale, kind = choose_scale(values - expected, values)
    tail = trend[-period:]
    return Baseline(
        "seasonal",
        expected,
        scale,
        kind,
        level=_nan_median(tail, level),
        seasonal=profile,
        next_phase=n % period,
        seasonality=period,
        window=period,
    )


def _note_scale(baseline: Baseline) -> Baseline:
    """Say out loud when the spread estimate had to step down a rung."""
    if baseline.scale is None:
        baseline.warnings.append(
            "nothing varies from its baseline by a usable amount, "
            "so no point can be an anomaly"
        )
    elif baseline.scale_kind == "std":
        baseline.warnings.append(
            "the median absolute deviation was zero; used the standard deviation instead"
        )
    return baseline


def _guard_degenerate(
    baseline: Baseline, values: np.ndarray, sensitivity: float
) -> Baseline:
    """Refuse a spread that is the shape of the fit rather than the noise in the data.

    A baseline earns its spread by leaving behind scatter: most points land near
    it and a few do not. When instead it pushes a large share of the whole series
    past the threshold, and the residual it leaves drifts smoothly from point to
    point rather than scattering, the spread being divided by is the tail of the
    fit's own settling curve. That is how a perfectly ordinary ramp - an energy
    meter, a packet counter, an odometer - can come back with every reading
    reported at millions of sigmas. Saying nothing, and saying why, is the honest
    answer; the caller still gets ``expected`` to look at.

    Note this can only ever fire on a residual that is systematically offset from
    its baseline: at most half the points of a centred residual can exceed three
    times its own MAD-based sigma, whatever its shape.
    """
    if baseline.scale is None or not np.isfinite(sensitivity) or sensitivity <= 0.0:
        return baseline
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = np.asarray(values, dtype=float) - baseline.expected
    usable = np.isfinite(residuals)
    total = int(np.count_nonzero(usable))
    if total < MIN_SHAPE_POINTS:
        return baseline
    with np.errstate(over="ignore", invalid="ignore"):
        limit = float(sensitivity) * float(baseline.scale)
        flagged = int(np.count_nonzero(np.abs(residuals[usable]) > limit))
    if flagged <= DEGENERATE_SHARE * total:
        return baseline
    if roughness(residuals) >= SMOOTH_RESIDUAL:
        return baseline
    baseline.scale = None
    baseline.scale_kind = "none"
    baseline.warnings.append(
        f"the {baseline.name} baseline is {flagged} of {total} point(s) away from the "
        "series and drifts rather than scatters, so what is left of it is the shape "
        "of the fit and not a spread to measure against; no point can be an anomaly"
    )
    return baseline


def fit(
    name: str,
    values: np.ndarray,
    *,
    seasonality: Optional[int] = None,
    sensitivity: float = 3.0,
) -> Baseline:
    """Fit one named method, applying its own fallbacks."""
    if name == "zscore":
        baseline = fit_zscore(values)
    elif name == "iqr":
        baseline = fit_iqr(values)
    elif name == "rolling":
        baseline = fit_rolling(values)
    elif name == "ewma":
        baseline = fit_ewma(values)
    elif name == "seasonal":
        baseline = fit_seasonal(values, seasonality)
    else:
        raise ValueError(f"unknown method {name!r}; choose one of {', '.join(METHODS)}")
    return _note_scale(_guard_degenerate(baseline, values, sensitivity))


def applicable(n: int, seasonality: Optional[int]) -> List[str]:
    """The methods worth running on a series of `n` points, for the ``"all"`` vote."""
    names = ["zscore", "iqr"]
    if n >= 2:
        names.append("ewma")
    if n >= max(3, auto_window(n)):
        names.append("rolling")
    if seasonality and seasonality >= 2 and n >= 2 * seasonality:
        names.append("seasonal")
    return names


def describe(names: Sequence[str]) -> str:
    """A human list of method names."""
    return ", ".join(names)
