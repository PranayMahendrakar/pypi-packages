"""The baseline: what each period should have looked like, and why.

A Monday 9am is compared with other Monday 9ams, not with 3am. That is the whole
idea. When there is not enough history for that, the baseline drops to a coarser
phase (time of day, then a flat median) and says so, rather than pretending.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

#: smallest number of same-phase readings before a phase is trusted
MIN_PER_PHASE = 3
#: robust-sigma multiplier that turns a MAD into a standard deviation
MAD_TO_SIGMA = 1.4826


@dataclass
class Trend:
    """Which way the baseline itself is moving."""

    direction: str = "flat"
    slope_per_day: Optional[float] = None
    pct_per_week: Optional[float] = None
    pct_per_window: float = 0.0
    r2: float = 0.0
    confident: bool = False
    span_days: Optional[float] = None

    def __bool__(self) -> bool:
        return self.direction != "flat"

    def __str__(self) -> str:
        if self.direction == "flat":
            return "flat"
        if self.pct_per_week is not None:
            return f"{self.direction} about {abs(self.pct_per_week):.1f}% per week"
        return f"{self.direction} about {abs(self.pct_per_window):.1f}% across the window"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Trend({str(self)!r}, r2={self.r2:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the trend."""
        return {
            "direction": self.direction,
            "slope_per_day": _clean(self.slope_per_day),
            "pct_per_week": _clean(self.pct_per_week),
            "pct_per_window": _clean(self.pct_per_window),
            "r2": _clean(self.r2),
            "confident": bool(self.confident),
            "span_days": _clean(self.span_days),
            "text": str(self),
        }


@dataclass
class StepChange:
    """A sudden change in the level that then stayed."""

    when: Any
    before: float
    after: float
    change: float
    pct: Optional[float]

    def __str__(self) -> str:
        where = _when_text(self.when)
        direction = "up" if self.change > 0 else "down"
        size = "" if self.pct is None else f" by {abs(self.pct):.0f}%"
        return f"level stepped {direction}{size} at {where} and stayed there"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the step."""
        return {
            "when": _stamp(self.when),
            "before": _clean(self.before),
            "after": _clean(self.after),
            "change": _clean(self.change),
            "pct": _clean(self.pct),
            "text": str(self),
        }


@dataclass
class Baseline:
    """What every period was expected to be, and how it was worked out."""

    expected: np.ndarray
    scale: Optional[float]
    method: str
    level: float
    trend: Trend
    phase_kind: str = "flat"
    step_minutes: float = 60.0
    phase_medians: Dict[int, float] = field(default_factory=dict)
    fallback: float = 0.0
    slope_per_day: float = 0.0
    origin: Any = None
    n_used: int = 0

    def predict(self, index: Any) -> np.ndarray:
        """Expected consumption for any index, using the same phases and trend."""
        codes = phase_codes(index, self.phase_kind, self.step_minutes)
        base = np.array(
            [self.phase_medians.get(int(code), self.fallback) for code in codes], dtype=float
        )
        return base + self.drift(index)

    def drift(self, index: Any) -> np.ndarray:
        """The slow trend part of the baseline, on its own."""
        return self.slope_per_day * _elapsed_days(index, self.origin)


def _clean(value: Any) -> Optional[float]:
    """A JSON-safe float: NaN, infinities and non-numbers become ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _stamp(value: Any) -> Any:
    """A JSON-safe timestamp: ISO text for a date, an int for a position."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.isoformat()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return _clean(value)


def _when_text(value: Any) -> str:
    """How a period is written in plain sentences."""
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        if stamp.hour or stamp.minute:
            return stamp.strftime("%Y-%m-%d %H:%M")
        return stamp.strftime("%Y-%m-%d")
    return f"reading {int(value)}" if isinstance(value, (int, np.integer)) else str(value)


def _elapsed_days(index: Any, origin: Any) -> np.ndarray:
    """Distance from the origin in days, or in positions when there is no clock."""
    if isinstance(index, pd.DatetimeIndex):
        if origin is None:
            origin = index[0] if index.size else pd.Timestamp(0)
        return (index - pd.Timestamp(origin)).total_seconds().to_numpy() / 86400.0
    values = np.asarray(index, dtype=float)
    start = float(origin) if origin is not None else (values[0] if values.size else 0.0)
    return values - start


def phase_codes(index: Any, kind: str, step_minutes: float) -> np.ndarray:
    """The same-phase group each period belongs to, as integer codes."""
    if kind == "flat" or not isinstance(index, pd.DatetimeIndex):
        return np.zeros(len(index), dtype=int)
    width = max(1, int(round(step_minutes)))
    minutes = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    slot = (minutes // width).astype(int)
    dow = index.dayofweek.to_numpy().astype(int)
    slots_per_day = max(1, int(round(1440 / width)))
    if kind == "dow_slot":
        return dow * slots_per_day + slot
    if kind == "slot":
        return slot
    if kind == "hour":
        return index.hour.to_numpy().astype(int)
    if kind == "dow":
        return dow
    return np.zeros(len(index), dtype=int)  # pragma: no cover - defensive


def _candidates(index: Any, step: Optional[pd.Timedelta]) -> List[Tuple[str, str]]:
    """Phase levels to try, most specific first, as ``(kind, human name)``."""
    if not isinstance(index, pd.DatetimeIndex) or step is None:
        return [("flat", "flat median")]
    day = pd.Timedelta(days=1)
    if step < day:
        fine = step < pd.Timedelta(hours=1)
        specific = "day of week and time of day" if fine else "day of week and hour of day"
        coarse = "time of day" if fine else "hour of day"
        return [
            ("dow_slot", specific),
            ("slot", coarse),
            ("hour", "hour of day"),
            ("flat", "flat median"),
        ]
    if step < pd.Timedelta(days=7):
        return [("dow", "day of week"), ("flat", "flat median")]
    return [("flat", "flat median")]


def _group_medians(values: np.ndarray, codes: np.ndarray, usable: np.ndarray) -> Dict[int, float]:
    """Median of each phase group, over the usable points only."""
    frame = pd.Series(values[usable]).groupby(pd.Series(codes[usable])).median()
    return {int(code): float(value) for code, value in frame.items() if np.isfinite(value)}


def _pair_scale(
    residual: np.ndarray, codes: np.ndarray, usable: np.ndarray
) -> Optional[float]:
    """Robust sigma from the spread *within* each phase group.

    The median of a small group is itself noisy, so residuals taken around it are
    too tight and a plain MAD would call ordinary noise an anomaly. Differences
    between two readings of the same phase cancel that group median entirely, so
    they measure the real spread however few readings each phase has.
    """
    order = np.flatnonzero(usable)
    if order.size < 4:
        return None
    frame = pd.DataFrame({"code": codes[order], "value": residual[order]})
    pieces = [
        np.abs(np.diff(group.to_numpy(dtype=float)))
        for _, group in frame.groupby("code", sort=False)["value"]
        if group.size >= 2
    ]
    if not pieces:
        return None
    spread = np.concatenate(pieces)
    spread = spread[np.isfinite(spread)]
    if spread.size < 3:
        return None
    scale = MAD_TO_SIGMA * float(np.median(spread)) / np.sqrt(2.0)
    return scale if scale > 0 and np.isfinite(scale) else None


def _fit_trend(
    series: pd.Series, step: Optional[pd.Timedelta], has_time: bool, usable: np.ndarray
) -> Trend:
    """Straight-line trend through the daily level."""
    values = series.to_numpy(dtype=float).copy()
    values[~usable] = np.nan
    if has_time and step is not None and step < pd.Timedelta(days=1):
        coarse = pd.Series(values, index=series.index).resample(pd.Timedelta(days=1)).mean()
    else:
        coarse = pd.Series(values, index=series.index)
    y = coarse.to_numpy(dtype=float)
    keep = np.isfinite(y)
    if int(np.count_nonzero(keep)) < 4:
        return Trend()
    x = _elapsed_days(coarse.index, None)[keep]
    y = y[keep]
    if float(np.ptp(x)) <= 0:
        return Trend()
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    spread = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 0.0 if spread <= 0 else float(1.0 - np.sum((y - fitted) ** 2) / spread)
    r2 = max(0.0, min(1.0, r2))
    level = float(np.median(np.abs(y)))
    span = float(np.max(x) - np.min(x))
    pct_window = 0.0 if level <= 0 else float(slope * span / level * 100.0)
    pct_week = None if not has_time or level <= 0 else float(slope * 7.0 / level * 100.0)
    reference = abs(pct_week) if pct_week is not None else abs(pct_window)
    enough = span >= (14.0 if has_time else 20.0)
    confident = bool(enough and reference >= 1.0 and r2 >= 0.05 and np.isfinite(slope))
    direction = "flat"
    if confident:
        direction = "rising" if slope > 0 else "falling"
    return Trend(
        direction=direction,
        slope_per_day=float(slope) if has_time else None,
        pct_per_week=pct_week,
        pct_per_window=pct_window,
        r2=r2,
        confident=confident,
        span_days=span if has_time else None,
    )


def fit(
    series: pd.Series,
    step: Optional[pd.Timedelta],
    has_time: bool,
    exclude: Optional[np.ndarray] = None,
    warnings: Optional[List[str]] = None,
) -> Baseline:
    """Work out what each period should have been.

    `exclude` marks periods that must not shape the baseline (negative readings).
    Anything that had to be given up on is appended to `warnings`.
    """
    warnings = warnings if warnings is not None else []
    values = series.to_numpy(dtype=float)
    n = int(values.size)
    index = series.index
    step_minutes = 60.0 if step is None else max(1.0, step.total_seconds() / 60.0)
    exclude = np.zeros(n, dtype=bool) if exclude is None else np.asarray(exclude, dtype=bool)
    usable = np.isfinite(values) & ~exclude
    n_usable = int(np.count_nonzero(usable))

    if n_usable == 0:
        return Baseline(
            expected=np.full(n, np.nan),
            scale=None,
            method="none",
            level=float("nan"),
            trend=Trend(),
            step_minutes=step_minutes,
            origin=index[0] if n else None,
        )

    trend = _fit_trend(series, step, has_time, usable)
    origin = index[0] if n else None
    elapsed = _elapsed_days(index, origin)
    slope = float(trend.slope_per_day) if (trend.confident and trend.slope_per_day) else 0.0
    if not has_time and trend.confident:
        # no clock, so the slope is per reading and _elapsed_days counts readings
        slope = float(np.polyfit(elapsed[usable], values[usable], 1)[0]) if n_usable >= 4 else 0.0
    drift = slope * elapsed
    residual = values - drift

    options = _candidates(index, step)
    chosen_kind, chosen_name = options[-1]
    medians: Dict[int, float] = {}
    for position, (kind, name) in enumerate(options):
        codes = phase_codes(index, kind, step_minutes)
        counts = pd.Series(codes[usable]).value_counts()
        if counts.empty:
            continue
        if int(counts.min()) >= MIN_PER_PHASE and n_usable >= MIN_PER_PHASE * int(counts.size):
            chosen_kind, chosen_name = kind, name
            medians = _group_medians(residual, codes, usable)
            if position > 0 and options[0][0] != kind:
                warnings.append(
                    f"not enough history to compare each {options[0][1]} separately "
                    f"({n_usable} usable periods); the baseline uses {name} instead"
                )
            break
    else:
        chosen_kind, chosen_name = "flat", "flat median"
        medians = _group_medians(residual, np.zeros(n, dtype=int), usable)
        warnings.append(
            f"only {n_usable} usable period(s), less than one full seasonal cycle; "
            "the baseline is a flat median of everything"
        )

    fallback = float(np.median(residual[usable]))
    codes = phase_codes(index, chosen_kind, step_minutes)
    base = np.array([medians.get(int(code), fallback) for code in codes], dtype=float)
    expected = base + drift

    gap = residual[usable] - base[usable]
    centre = float(np.median(gap))
    mad = float(np.median(np.abs(gap - centre)))
    scale: Optional[float] = mad * MAD_TO_SIGMA
    if chosen_kind != "flat":
        within = _pair_scale(residual, codes, usable)
        if within is not None:
            scale = within if scale is None else max(float(scale), within)
    elif options[0][0] != "flat":
        # The baseline had to give up on the daily shape, so the residual still
        # contains it. Measuring noise with a MAD here would call every working
        # hour an anomaly, so the honest noise level is the whole spread.
        spread = float(np.std(gap)) if gap.size > 1 else 0.0
        if spread > 0 and np.isfinite(spread):
            scale = spread if scale is None else max(float(scale), spread)
            warnings.append(
                "against a flat baseline the normal daily shape counts as variation, "
                "so only a period well outside the whole spread is called unusual"
            )
    if not scale or not np.isfinite(scale) or scale <= 0:
        spread = float(np.std(gap)) if gap.size > 1 else 0.0
        scale = spread if spread > 0 and np.isfinite(spread) else None
    if scale is None:
        if n_usable < MIN_PER_PHASE:
            warnings.append(
                f"only {n_usable} usable reading(s): too few to measure what normal "
                "variation looks like, so nothing is flagged as unusual"
            )
        else:
            warnings.append(
                "every same-phase reading is identical, so nothing can stand out as unusual"
            )

    return Baseline(
        expected=expected,
        scale=scale,
        method=chosen_name,
        level=float(np.median(values[usable])),
        trend=trend,
        phase_kind=chosen_kind,
        step_minutes=step_minutes,
        phase_medians=medians,
        fallback=fallback,
        slope_per_day=slope,
        origin=origin,
        n_used=n_usable,
    )


def standby_load(
    series: pd.Series,
    step: Optional[pd.Timedelta],
    has_time: bool,
    exclude: Optional[np.ndarray] = None,
    night: Tuple[int, int] = (0, 5),
) -> Tuple[float, str]:
    """The always-on floor: the persistent overnight minimum.

    Returns ``(load_per_period, how_it_was_measured)``.
    """
    values = series.to_numpy(dtype=float)
    exclude = np.zeros(values.size, dtype=bool) if exclude is None else np.asarray(exclude, bool)
    usable = np.isfinite(values) & ~exclude & (values >= 0)
    if not usable.any():
        return 0.0, "no usable readings"
    index = series.index
    ceiling = float(np.median(values[usable]))
    if has_time and step is not None and step < pd.Timedelta(hours=12) and isinstance(index, pd.DatetimeIndex):
        hours = index.hour.to_numpy()
        start, end = int(night[0]), int(night[1])
        window = (hours >= start) & (hours < end) if start < end else (hours >= start) | (hours < end)
        overnight = usable & window
        if int(np.count_nonzero(overnight)) >= 2:
            dates = pd.Index(index.date)
            nightly = pd.Series(values[overnight]).groupby(pd.Index(dates[overnight])).min()
            nightly = nightly[np.isfinite(nightly.to_numpy(dtype=float))]
            if nightly.size >= 1:
                load = float(np.median(nightly.to_numpy(dtype=float)))
                how = f"overnight minimum ({start:02d}:00-{end:02d}:00), median of {nightly.size} night(s)"
                return float(min(max(load, 0.0), ceiling)), how
    load = float(np.percentile(values[usable], 5))
    how = "5th percentile of the readings (no overnight detail available)"
    return float(min(max(load, 0.0), ceiling)), how


def find_steps(
    residual: pd.Series,
    step: Optional[pd.Timedelta],
    has_time: bool,
    level: float,
    threshold: float = 5.0,
    max_steps: int = 3,
) -> List[StepChange]:
    """Sudden shifts in the level that then persisted."""
    if has_time and step is not None and step < pd.Timedelta(days=1):
        coarse = residual.resample(pd.Timedelta(days=1)).mean()
    else:
        coarse = residual
    y = coarse.to_numpy(dtype=float)
    keep = np.isfinite(y)
    if int(np.count_nonzero(keep)) < 8:
        return []
    stamps = coarse.index[keep]
    y = y[keep]
    jumps = np.abs(np.diff(y))
    jumps = jumps[np.isfinite(jumps)]
    noise = float(np.median(jumps)) * MAD_TO_SIGMA / np.sqrt(2.0) if jumps.size else 0.0
    floor = 0.01 * abs(level) if np.isfinite(level) else 0.0
    noise = max(noise, floor, 1e-12)
    found: List[StepChange] = []
    _segment(y, stamps, 0, y.size, noise, level, threshold, found, max_steps, 0)
    found.sort(key=lambda item: item.when)
    return found


def _beats_a_line(window: np.ndarray, cut: int) -> bool:
    """True when a step at `cut` explains `window` better than a straight line.

    A gradual ramp always has a "best" place to cut it in half, and the cut looks
    convincing on its own. Comparing against a straight line through the same
    window is what tells a real step apart from drift.
    """
    n = window.size
    if n < 6 or cut < 1 or cut >= n:  # pragma: no cover - guarded by the caller
        return False
    x = np.arange(n, dtype=float)
    line = np.polyval(np.polyfit(x, window, 1), x)
    sse_line = float(np.sum((window - line) ** 2))
    fitted = np.concatenate(
        [
            np.full(cut, float(np.mean(window[:cut]))),
            np.full(n - cut, float(np.mean(window[cut:]))),
        ]
    )
    sse_step = float(np.sum((window - fitted) ** 2))
    if sse_line <= 0:
        return False
    return sse_step <= 0.7 * sse_line


def _segment(
    y: np.ndarray,
    stamps: Any,
    lo: int,
    hi: int,
    noise: float,
    level: float,
    threshold: float,
    found: List[StepChange],
    max_steps: int,
    depth: int,
) -> None:
    """Binary segmentation: split where the mean shifts hardest, then recurse."""
    if len(found) >= max_steps or depth > 2 or hi - lo < 6:
        return
    window = y[lo:hi]
    n = window.size
    total = np.cumsum(window)
    best_score = 0.0
    best_at = -1
    for cut in range(3, n - 2):
        left = total[cut - 1] / cut
        right = (total[-1] - total[cut - 1]) / (n - cut)
        spread = noise * np.sqrt(1.0 / cut + 1.0 / (n - cut))
        score = abs(right - left) / spread if spread > 0 else 0.0
        if score > best_score:
            best_score, best_at = score, cut
    if best_at < 0 or best_score < threshold:
        return
    if not _beats_a_line(window, best_at):
        # a straight line explains this window about as well, so it is a slow
        # drift being cut in half, not a level that jumped and stayed
        return
    base = level if np.isfinite(level) else 0.0
    before = base + float(np.mean(window[:best_at]))
    after = base + float(np.mean(window[best_at:]))
    change = after - before
    if abs(change) < 0.05 * abs(level):
        return
    reference = abs(before) if abs(before) > 1e-12 else abs(base)
    pct = None if reference <= 1e-12 else float(change / reference * 100.0)
    found.append(
        StepChange(
            when=stamps[lo + best_at],
            before=before,
            after=after,
            change=change,
            pct=pct,
        )
    )
    _segment(y, stamps, lo, lo + best_at, noise, level, threshold, found, max_steps, depth + 1)
    _segment(y, stamps, lo + best_at, hi, noise, level, threshold, found, max_steps, depth + 1)
