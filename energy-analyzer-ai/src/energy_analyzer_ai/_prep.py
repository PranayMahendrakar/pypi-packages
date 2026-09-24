"""Get the readings onto a regular grid, honestly.

Three things happen here, and every one of them is recorded rather than done
quietly:

* a cumulative meter (a register that only ever counts up) is detected and
  differenced, so the total is what was actually used and not what the dial says;
* negative readings (export, or a meter that rolled over) are flagged;
* irregular readings are resampled onto a regular grid and the periods with no
  reading are reported as gaps, never filled in.
"""
from __future__ import annotations

import logging
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._io import LoadedSeries

LOG = logging.getLogger(__name__)

#: the grid sizes "auto" is allowed to snap to
LADDER: Tuple[Tuple[str, pd.Timedelta], ...] = (
    ("1min", pd.Timedelta(minutes=1)),
    ("5min", pd.Timedelta(minutes=5)),
    ("10min", pd.Timedelta(minutes=10)),
    ("15min", pd.Timedelta(minutes=15)),
    ("30min", pd.Timedelta(minutes=30)),
    ("1h", pd.Timedelta(hours=1)),
    ("2h", pd.Timedelta(hours=2)),
    ("3h", pd.Timedelta(hours=3)),
    ("6h", pd.Timedelta(hours=6)),
    ("12h", pd.Timedelta(hours=12)),
    ("1D", pd.Timedelta(days=1)),
    ("7D", pd.Timedelta(days=7)),
)

#: friendly words accepted by ``granularity=``
#: Offset aliases pandas 2 accepted with a warning and pandas 3 removed. A caller
#: writing '30T' or '2H' is not wrong; their pandas changed under them.
LEGACY_OFFSETS = {
    "H": "h", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns",
    "h": "h", "min": "min", "s": "s",
}
for _n in range(1, 61):
    for _old, _new in (("H", "h"), ("T", "min"), ("S", "s"), ("L", "ms"), ("U", "us")):
        LEGACY_OFFSETS[f"{_n}{_old}"] = f"{_n}{_new}"

WORDS = {
    "auto": None,
    "minute": pd.Timedelta(minutes=1),
    "minutely": pd.Timedelta(minutes=1),
    "quarter-hourly": pd.Timedelta(minutes=15),
    "halfhourly": pd.Timedelta(minutes=30),
    "half-hourly": pd.Timedelta(minutes=30),
    "hour": pd.Timedelta(hours=1),
    "hourly": pd.Timedelta(hours=1),
    "day": pd.Timedelta(days=1),
    "daily": pd.Timedelta(days=1),
    "week": pd.Timedelta(days=7),
    "weekly": pd.Timedelta(days=7),
}

#: never build a grid bigger than this, whatever the caller asks for
MAX_PERIODS = 500_000


@dataclass
class Prepared:
    """Readings on a regular grid, with everything that had to be decided."""

    series: pd.Series
    label: str
    time_label: Optional[str]
    has_time: bool
    step: Optional[pd.Timedelta]
    granularity: str
    cumulative: bool = False
    meter_resets: int = 0
    n_readings: int = 0
    n_gaps: int = 0
    longest_gap: int = 0
    negative: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def values(self) -> np.ndarray:
        """The consumption per period, NaN where there was no reading."""
        return self.series.to_numpy(dtype=float)

    @property
    def n_periods(self) -> int:
        """Length of the regular grid."""
        return int(self.series.size)

    @property
    def gap_share(self) -> float:
        """Share of the grid with no reading, in ``[0, 1]``."""
        if self.n_periods == 0:
            return 0.0
        return self.n_gaps / self.n_periods

    @property
    def periods_per_day(self) -> float:
        """How many periods make a day; 1.0 when there is no real time axis."""
        if self.step is None or self.step <= pd.Timedelta(0):
            return 1.0
        return float(pd.Timedelta(days=1) / self.step)


def label_for(step: pd.Timedelta) -> str:
    """A short human label for a grid size, e.g. ``'15min'``."""
    for name, width in LADDER:
        if width == step:
            return name
    total = step.total_seconds()
    if total <= 0:
        return "period"
    if total % 86400 == 0:
        return f"{int(total // 86400)}D"
    if total % 3600 == 0:
        return f"{int(total // 3600)}h"
    if total % 60 == 0:
        return f"{int(total // 60)}min"
    return f"{total:g}s"


def parse_granularity(granularity: Any) -> Optional[pd.Timedelta]:
    """Turn ``granularity=`` into a fixed width, or ``None`` for ``'auto'``."""
    if granularity is None:
        return None
    if isinstance(granularity, pd.Timedelta):
        return granularity
    if isinstance(granularity, pd.DateOffset):
        return pd.Timedelta(granularity)
    if isinstance(granularity, str):
        key = granularity.strip().lower()
        if key in WORDS:
            return WORDS[key]
        # Legacy aliases warned on pandas 2 and were REMOVED in pandas 3, so
        # granularity='H' - valid code for years, and still in plenty of scripts -
        # started raising "not a fixed period" on an upgrade. They are translated
        # here rather than passed through, so the caller's spelling keeps working
        # whichever pandas they have.
        spelled = LEGACY_OFFSETS.get(granularity.strip(), granularity)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                offset = pd.tseries.frequencies.to_offset(spelled)
            except (ValueError, TypeError, AttributeError):
                offset = None
            width = None
            if offset is not None:
                # `.nanos` is the reliable "is this a fixed duration, and how long"
                # question on both pandas 2 and 3. pd.Timedelta(offset) looks like the
                # obvious call and is a trap: on pandas 3 it refuses a Day offset, so
                # granularity='1D' and 'daily' - the commonest setting for meter data -
                # raised "not a fixed period" on upgrade. `.nanos` still refuses weeks
                # and months, which genuinely are not fixed durations, so the error
                # stays correct for those.
                try:
                    width = pd.Timedelta(offset.nanos, unit="ns")
                except (ValueError, TypeError, AttributeError):
                    try:
                        width = pd.Timedelta(offset)
                    except (ValueError, TypeError, AttributeError):
                        width = None
        if width is None or width <= pd.Timedelta(0):
            raise ValueError(
                f"granularity={granularity!r} is not a fixed period; use 'auto', "
                "'hourly', 'daily', 'weekly', '15min', '1h', '1D' or a pandas offset "
                "like '30min'"
            )
        return width
    raise TypeError(
        f"granularity must be a string or a Timedelta, not {type(granularity).__name__}"
    )


def infer_step(index: pd.DatetimeIndex, notes: List[str]) -> pd.Timedelta:
    """Guess the grid size from how far apart the readings actually are."""
    stamps = pd.DatetimeIndex(index).unique().sort_values()
    if stamps.size < 2:
        notes.append("only one timestamp, so an hourly grid was assumed")
        return pd.Timedelta(hours=1)
    deltas = np.diff(stamps.to_numpy()).astype("timedelta64[s]").astype(float)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:  # pragma: no cover - guarded by .unique() above
        return pd.Timedelta(hours=1)
    median = float(np.median(deltas))
    best = min(LADDER, key=lambda item: abs(np.log(item[1].total_seconds() / median)))
    return best[1]


def _looks_cumulative(values: np.ndarray) -> bool:
    """True when the readings are a register that only counts up."""
    finite = values[np.isfinite(values)]
    if finite.size < 6:
        return False
    steps = np.diff(finite)
    spread = float(np.nanmax(finite) - np.nanmin(finite))
    if spread <= 0:
        return False
    tolerance = 1e-9 * max(1.0, float(np.nanmax(np.abs(finite))))
    non_decreasing = float(np.mean(steps >= -tolerance))
    rising = float(np.mean(steps > tolerance))
    return non_decreasing >= 0.98 and rising >= 0.4


def _difference(
    series: pd.Series, notes: List[str], warnings: List[str]
) -> Tuple[pd.Series, int]:
    """Difference a cumulative register, dropping meter rollovers."""
    values = series.to_numpy(dtype=float)
    steps = np.diff(values)
    used = pd.Series(steps, index=series.index[1:], name=series.name)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    typical = float(np.percentile(positive, 95)) if positive.size else 0.0
    resets = np.isfinite(steps) & (steps < 0) & (np.abs(steps) > max(3.0 * typical, 1e-12))
    n_resets = int(np.count_nonzero(resets))
    if n_resets:
        used.iloc[np.flatnonzero(resets)] = np.nan
        warnings.append(
            f"{n_resets} meter reset or rollover detected; those intervals are left "
            "as gaps instead of being counted as consumption"
        )
    notes.append(
        "the readings only ever increase, so they were read as a cumulative meter and "
        "analysed as the difference between consecutive readings"
    )
    return used, n_resets


def prepare(
    loaded: LoadedSeries,
    granularity: Any = "auto",
    cumulative: Optional[bool] = None,
) -> Prepared:
    """Put `loaded` on a regular grid and report what had to change.

    `cumulative` forces or forbids the cumulative-meter reading; the default
    ``None`` detects it.
    """
    notes = list(loaded.notes)
    warnings = list(loaded.warnings)
    series = loaded.series
    n_readings = int(series.size)
    asked = parse_granularity(granularity)

    if n_readings == 0:
        return Prepared(
            series=pd.Series([], index=series.index[:0], name=loaded.label, dtype=float),
            label=loaded.label,
            time_label=loaded.time_label,
            has_time=loaded.has_time,
            step=asked,
            granularity=label_for(asked) if asked is not None else "period",
            n_readings=0,
            notes=notes,
            warnings=warnings,
        )

    is_cumulative = _looks_cumulative(series.to_numpy(dtype=float)) if cumulative is None else bool(cumulative)
    resets = 0
    if is_cumulative:
        if n_readings < 6:
            is_cumulative = False
            warnings.append(
                "too few readings to tell a cumulative meter from rising consumption; "
                "they were read as consumption per reading"
            )
        else:
            series, resets = _difference(series, notes, warnings)

    if not loaded.has_time:
        grid = pd.Series(
            series.to_numpy(dtype=float),
            index=pd.RangeIndex(series.size),
            name=loaded.label,
            dtype=float,
        )
        step = None
        granularity_label = "reading"
    else:
        step = asked if asked is not None else infer_step(series.index, notes)
        span = series.index.max() - series.index.min()
        estimate = int(span / step) + 1 if step > pd.Timedelta(0) else 0
        if estimate > MAX_PERIODS:
            raise ValueError(
                f"a {label_for(step)} grid over {span} would be {estimate:,} periods; "
                "pass a coarser granularity= such as 'hourly' or 'daily'"
            )
        grid = series.resample(step).sum(min_count=1)
        grid.name = loaded.label
        grid.index.name = loaded.time_label or "time"
        granularity_label = label_for(step)

    values = grid.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    n_gaps = int(np.count_nonzero(missing))
    longest = 0
    run = 0
    for flag in missing:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    if n_gaps and loaded.has_time:
        notes.append(
            f"{n_gaps} of {values.size} {granularity_label} periods have no reading; "
            "they are reported as gaps and are never filled in"
        )
    elif n_gaps:
        notes.append(f"{n_gaps} reading(s) are missing and are reported as gaps")
    if values.size and n_gaps / values.size > 0.5 and loaded.has_time:
        warnings.append(
            f"a {granularity_label} grid is finer than the readings: "
            f"{n_gaps / values.size:.0%} of periods have no reading. Pass a coarser "
            "granularity= to line the grid up with the meter"
        )

    negative = np.isfinite(values) & (values < 0)
    if negative.any():
        warnings.append(
            f"{int(np.count_nonzero(negative))} period(s) have negative consumption, "
            "which usually means export to the grid or a meter reset; they are "
            "reported but left out of the baseline"
        )

    return Prepared(
        series=grid,
        label=loaded.label,
        time_label=loaded.time_label,
        has_time=loaded.has_time,
        step=step,
        granularity=granularity_label,
        cumulative=is_cumulative,
        meter_resets=resets,
        n_readings=n_readings,
        n_gaps=n_gaps,
        longest_gap=longest,
        negative=negative,
        notes=notes,
        warnings=warnings,
    )
