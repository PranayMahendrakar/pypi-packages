"""Turn a caller's table into one regular grid of units made per interval.

Everything here works on copies. Arrays that are written into are created with
``np.array(..., copy)`` or ``np.zeros``; nothing writes into ``.to_numpy()`` output,
which is read-only under pandas 3 copy-on-write.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

log = logging.getLogger(__name__)

NS_PER_S = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_S
NS_PER_DAY = 24 * 60 * NS_PER_MIN
MAX_BINS = 5_000_000
NICE_SECONDS = (
    1, 2, 5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800,
    3600, 7200, 10800, 14400, 21600, 28800, 43200, 86400,
)
TIME_NAME = re.compile(r"(time|date|stamp|^ts$|^dt$|period|interval|hour|when)", re.I)
OUTPUT_WORDS = (
    "output", "units", "produced", "production", "count", "qty", "quantity", "pieces",
    "pcs", "parts", "good", "made", "throughput", "total", "counter", "yield",
)
NOT_OUTPUT_WORDS = (
    "target", "plan", "scrap", "reject", "defect", "id", "temp", "speed", "pressure",
    "shift", "line", "station", "operator",
)
_UNIT_WORDS = {
    "ms": 0.001, "millisecond": 0.001, "milliseconds": 0.001,
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "t": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
}


def label_text(label: Any) -> str:
    """A column label as text for messages and the report."""
    return str(label)


def find_label(frame: pd.DataFrame, wanted: Any) -> Any:
    """The real column label matching ``wanted`` exactly or by its text, else ``None``."""
    for label in frame.columns:
        if label == wanted:
            return label
    for label in frame.columns:
        if str(label) == str(wanted):
            return label
    return None


# ---------------------------------------------------------------- intervals
def interval_seconds(interval: Any) -> float:
    """``"5min"``, ``"1h"``, ``"30s"``, a number of minutes, or a timedelta -> seconds.

    Retired pandas aliases (``"H"``, ``"T"``, ``"S"``) are translated here rather than
    passed to pandas, where version 3 no longer accepts them.
    """
    if isinstance(interval, bool):
        raise ValueError(f"interval={interval!r} is not a duration")
    if isinstance(interval, (int, float, np.integer, np.floating)):
        seconds = float(interval) * 60.0
    elif isinstance(interval, pd.Timedelta):
        seconds = interval.total_seconds()
    elif hasattr(interval, "total_seconds"):
        seconds = float(interval.total_seconds())
    elif isinstance(interval, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)?\s*([A-Za-z]+)\s*", interval)
        if not match or match.group(2).lower() not in _UNIT_WORDS:
            raise ValueError(
                f"interval={interval!r} is not a duration; use text like '1min', '5min', "
                "'1h' or '30s', a number of minutes, or a timedelta"
            )
        amount = float(match.group(1)) if match.group(1) else 1.0
        seconds = amount * _UNIT_WORDS[match.group(2).lower()]
    else:
        raise ValueError(f"interval={interval!r} is not a duration")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"interval={interval!r} must be a positive duration")
    return seconds


def nice_interval_ns(median_ns: float) -> int:
    """Snap a measured spacing to a round value (1 min, 5 min, 1 h...) when within 5%."""
    seconds = median_ns / NS_PER_S
    if seconds <= 0:
        return NS_PER_MIN
    best = min(NICE_SECONDS, key=lambda n: abs(math.log(n / seconds)))
    if abs(best / seconds - 1.0) <= 0.05:
        return int(best * NS_PER_S)
    if seconds >= 1:
        return int(round(seconds) * NS_PER_S)
    return max(1, int(round(median_ns)))


def nearest_nice_ns(median_ns: float) -> int:
    """The round interval (1 s, 30 s, 1 min, 5 min...) nearest a measured spacing.

    Used for irregular timestamps, where the median spacing is only a rough guide and an
    odd grid (65 s, 32 s) would apply every minute-based threshold in odd multiples.
    """
    seconds = median_ns / NS_PER_S
    if seconds <= 0:
        return NS_PER_MIN
    if seconds < 0.5:
        return max(1, int(round(median_ns)))
    best = min(NICE_SECONDS, key=lambda n: abs(math.log(n / seconds)))
    return int(best * NS_PER_S)


def long_gap_seconds(gaps_s: np.ndarray, dt_s: float) -> float:
    """How long a gap between irregular readings must be to count as missing data.

    Normal irregular spacing is the bulk of the gaps; a logger that was offline leaves a gap
    far beyond it. The limit is taken from the spread of the gaps themselves (five times the
    median, three times the 95th percentile), never less than two intervals, so jittered
    sampling and event logs with random arrivals are both spread in full.
    """
    finite = gaps_s[np.isfinite(gaps_s) & (gaps_s > 0)]
    if finite.size == 0:
        return 2.0 * dt_s
    return max(
        2.0 * dt_s,
        5.0 * float(np.median(finite)),
        3.0 * float(np.quantile(finite, 0.95)),
    )


# ---------------------------------------------------------------- time column
def _epoch_to_datetime(values: pd.Series, name: str, notes: List[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    if finite.empty:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    peak = float(np.abs(finite).max())
    unit = "s" if peak < 1e11 else "ms" if peak < 1e14 else "us" if peak < 1e17 else "ns"
    notes.append(f"time column '{name}' is numeric; read it as epoch {unit} (UTC, naive).")
    scale = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}[unit]
    ns = np.array(numeric.to_numpy(dtype=float, na_value=np.nan), dtype=float) * scale
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    ok = np.isfinite(ns)
    if ok.any():
        stamps = pd.DatetimeIndex(ns[ok].astype(np.int64).astype("datetime64[ns]"))
        out = out.copy()
        out[ok] = stamps
    return out


def _parse_datetimes(values: pd.Series) -> Tuple[pd.Series, bool]:
    """Parse text into datetimes; returns (parsed, converted_to_utc)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = False
        try:
            parsed = pd.to_datetime(values, errors="coerce")
        except (ValueError, TypeError, OverflowError):
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            converted = True
        if not ptypes.is_datetime64_any_dtype(parsed):
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            converted = True
        missing = int(parsed.isna().sum()) - int(values.isna().sum())
        if missing > 0:
            try:
                mixed = pd.to_datetime(values, errors="coerce", format="mixed")
                if not ptypes.is_datetime64_any_dtype(mixed):
                    mixed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
                    if int(mixed.isna().sum()) < int(parsed.isna().sum()):
                        converted = True
                if int(mixed.isna().sum()) < int(parsed.isna().sum()):
                    parsed = mixed
            except (ValueError, TypeError, OverflowError):
                pass  # pandas < 2 has no format="mixed"; keep the first parse
    return parsed, converted


def to_datetime_series(values: pd.Series, name: str, notes: List[str], explicit: bool) -> pd.Series:
    """A column as datetimes, keeping any timezone it already has."""
    if ptypes.is_datetime64_any_dtype(values):
        return values
    if ptypes.is_bool_dtype(values):
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if ptypes.is_numeric_dtype(values):
        if not explicit and not TIME_NAME.search(name):
            return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        return _epoch_to_datetime(values, name, notes)
    parsed, converted = _parse_datetimes(values)
    if converted and parsed.notna().any():
        notes.append(
            f"time column '{name}' mixes UTC offsets; every timestamp was converted to UTC."
        )
    return parsed


def resolve_time(
    frame: pd.DataFrame, time: Any, notes: List[str]
) -> Tuple[pd.Series, Optional[str]]:
    """Find the timestamp column (or a DatetimeIndex) and parse it."""
    index_is_time = isinstance(frame.index, pd.DatetimeIndex)
    if time is not None:
        label = find_label(frame, time)
        if label is not None:
            name = label_text(label)
            parsed = to_datetime_series(frame[label], name, notes, explicit=True)
            return parsed.reset_index(drop=True), name
        if index_is_time and (str(time) in ("index", str(frame.index.name))):
            name = str(frame.index.name or "index")
            return pd.Series(frame.index).reset_index(drop=True), name
        raise ValueError(
            f"time column {time!r} not found; columns are: "
            f"{', '.join(repr(label_text(c)) for c in frame.columns) or '(none)'}"
        )

    for label in frame.columns:
        if ptypes.is_datetime64_any_dtype(frame[label]):
            name = label_text(label)
            others = [
                label_text(c)
                for c in frame.columns
                if label_text(c) != name and ptypes.is_datetime64_any_dtype(frame[c])
            ]
            if others:
                notes.append(
                    f"time not given; used '{name}', the first datetime column "
                    f"(also found: {', '.join(others)}). Pass time= to choose."
                )
            return frame[label].reset_index(drop=True), name
    if index_is_time:
        notes.append("time not given; used the DataFrame's DatetimeIndex.")
        return pd.Series(frame.index).reset_index(drop=True), str(frame.index.name or "index")

    named = [c for c in frame.columns if TIME_NAME.search(label_text(c))]
    rest = [
        c
        for c in frame.columns
        if c not in named and not ptypes.is_numeric_dtype(frame[c])
        and not ptypes.is_bool_dtype(frame[c])
    ]
    for label in named + rest:
        column = frame[label]
        present = int(column.notna().sum())
        if present == 0:
            continue
        scratch: List[str] = []
        parsed = to_datetime_series(column, label_text(label), scratch, explicit=False)
        if int(parsed.notna().sum()) >= 0.8 * present:
            notes.extend(scratch)
            notes.append(
                f"time not given; used '{label_text(label)}', which reads as timestamps."
            )
            return parsed.reset_index(drop=True), label_text(label)
    raise ValueError(
        "no time column found; pass time='<column>' naming the timestamp column. Columns are: "
        + (", ".join(repr(label_text(c)) for c in frame.columns) or "(none)")
    )


# ---------------------------------------------------------------- output column
def _numeric_share(column: pd.Series) -> float:
    present = column.dropna()
    if present.empty:
        return 0.0
    return float(pd.to_numeric(present, errors="coerce").notna().mean())


def _name_score(name: str) -> int:
    text = name.lower()
    score = 0
    for rank, word in enumerate(OUTPUT_WORDS):
        if word in text:
            score = max(score, 100 - rank)
    if any(re.search(rf"(^|[^a-z]){w}([^a-z]|$)", text) or text.startswith(w) for w in NOT_OUTPUT_WORDS):
        score -= 200
    return score


def resolve_output(frame: pd.DataFrame, output: Any, time_name: Optional[str], notes: List[str]) -> Any:
    """Find the units-produced column; an automatic pick is always recorded in ``notes``."""
    if output is not None:
        label = find_label(frame, output)
        if label is None:
            raise ValueError(
                f"output column {output!r} not found; columns are: "
                f"{', '.join(repr(label_text(c)) for c in frame.columns) or '(none)'}"
            )
        if time_name is not None and label_text(label) == time_name:
            raise ValueError(f"output={output!r} is the time column; name the units column")
        return label

    candidates = [c for c in frame.columns if label_text(c) != time_name]
    numeric = [
        c for c in candidates
        if ptypes.is_numeric_dtype(frame[c]) and not ptypes.is_bool_dtype(frame[c])
        and not ptypes.is_datetime64_any_dtype(frame[c])
    ]
    how = "numeric"
    if not numeric:
        numeric = [
            c for c in candidates
            if not ptypes.is_datetime64_any_dtype(frame[c]) and _numeric_share(frame[c]) >= 0.5
        ]
        how = "mostly numeric"
    if not numeric:
        raise ValueError(
            "no numeric column to use as output; pass output='<column>' naming the "
            "units-produced column. Columns are: "
            + (", ".join(repr(label_text(c)) for c in frame.columns) or "(none)")
        )
    if len(numeric) == 1:
        label = numeric[0]
        notes.append(f"output not given; used '{label_text(label)}', the only {how} column.")
        return label
    scored = sorted(numeric, key=lambda c: -_name_score(label_text(c)))
    label = scored[0]
    others = ", ".join(label_text(c) for c in numeric if label_text(c) != label_text(label))
    if _name_score(label_text(label)) > 0:
        notes.append(
            f"output not given; used '{label_text(label)}' (its name says units produced) "
            f"out of {len(numeric)} {how} columns (others: {others}). Pass output= to choose."
        )
    else:
        label = numeric[0]
        others = ", ".join(label_text(c) for c in numeric if label_text(c) != label_text(label))
        notes.append(
            f"WARNING: output not given and no column name says units produced; used "
            f"'{label_text(label)}', the first {how} column (others: {others}). "
            "Pass output= to choose."
        )
    return label


def to_float(column: pd.Series, name: str, notes: List[str]) -> np.ndarray:
    """The output column as a float array (a fresh, writable copy)."""
    if ptypes.is_bool_dtype(column):
        column = column.astype(float)
    numeric = pd.to_numeric(column, errors="coerce")
    values = np.array(numeric.to_numpy(dtype=float, na_value=np.nan), dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    unreadable = int(column.notna().sum()) - int(np.isfinite(values).sum())
    if unreadable > 0:
        notes.append(
            f"{unreadable} value(s) in '{name}' were not numbers (or infinite) and were "
            "treated as unknown."
        )
    return values


# ---------------------------------------------------------------- counters
def looks_like_counter(values: np.ndarray) -> bool:
    """True when the column behaves like a running total rather than units per interval.

    A running total only goes up (apart from the odd reset), and it climbs far beyond
    the size of one step. Units per interval go up and down.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return False
    steps = np.diff(finite)
    moving = steps[steps != 0]
    if moving.size < 3:
        return False
    falls = int((moving < 0).sum())
    if falls > max(1, int(0.05 * steps.size)) or falls / moving.size > 0.2:
        return False
    rises = moving[moving > 0]
    if rises.size == 0:
        return False
    return float(finite.max() - finite.min()) > 10.0 * float(np.median(rises))


# ---------------------------------------------------------------- grid
@dataclass
class Reset:
    """One place where a counter went backwards (or a per-interval count went negative)."""

    start_ns: int
    end_ns: int
    before: Optional[float]
    after: float


@dataclass
class Grid:
    """Units per regular interval; NaN where nothing usable is known."""

    ns: np.ndarray
    index: pd.DatetimeIndex
    units: np.ndarray
    reset_bins: np.ndarray
    dt_ns: int
    tz: Any
    wall_ns: np.ndarray
    resets: List[Reset] = field(default_factory=list)
    gap_units: float = 0.0
    resampled: bool = False


def datetime_index(ns: np.ndarray, tz: Any) -> pd.DatetimeIndex:
    """UTC-epoch nanoseconds -> DatetimeIndex in ``tz`` (naive when ``tz`` is None)."""
    index = pd.DatetimeIndex(np.array(ns, dtype=np.int64).astype("datetime64[ns]"))
    if tz is not None:
        index = index.tz_localize("UTC").tz_convert(tz)
    return index


def timestamp(ns: int, tz: Any) -> pd.Timestamp:
    """One UTC-epoch nanosecond value -> Timestamp in ``tz``."""
    if tz is None:
        return pd.Timestamp(int(ns))
    return pd.Timestamp(int(ns), tz="UTC").tz_convert(tz)


def epoch_ns(stamps: pd.Series) -> Tuple[np.ndarray, np.ndarray, Any]:
    """(UTC-epoch ns, local wall-clock ns, tz) for a datetime Series without NaT."""
    index = pd.DatetimeIndex(stamps)
    if hasattr(index, "as_unit"):
        index = index.as_unit("ns")
    tz = index.tz
    utc = np.array(index.asi8, dtype=np.int64, copy=True)
    wall_index = index.tz_localize(None) if tz is not None else index
    wall = np.array(wall_index.asi8, dtype=np.int64, copy=True)
    return utc, wall, tz


def aggregate_same_time(
    ns: np.ndarray, wall: np.ndarray, values: np.ndarray, counter: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collapse rows that share a timestamp: summed for counts, the largest for a counter."""
    unique, first, inverse = np.unique(ns, return_index=True, return_inverse=True)
    if unique.size == ns.size:
        return ns, wall, values, 0
    finite = np.isfinite(values)
    if counter:
        merged = np.full(unique.size, -np.inf)
        np.maximum.at(merged, inverse[finite], values[finite])
    else:
        merged = np.zeros(unique.size)
        np.add.at(merged, inverse[finite], values[finite])
    has_value = np.zeros(unique.size, dtype=bool)
    has_value[inverse[finite]] = True
    merged[~has_value] = np.nan
    return unique, wall[first], merged, int(ns.size - unique.size)


def build_grid(
    ns: np.ndarray,
    wall: np.ndarray,
    values: np.ndarray,
    tz: Any,
    counter: bool,
    dt_ns: Optional[int],
    notes: List[str],
) -> Grid:
    """Spread each reading's units over the time it covers and sum them per interval.

    For timestamps already on a regular grid this is an exact copy of the readings, with
    missing rows as unknown intervals. Irregular timestamps are resampled onto a regular
    grid at the typical spacing, and ``notes`` says so.
    """
    n = ns.size
    measured = dt_ns is None
    median_ns = float(np.median(np.diff(ns))) if n >= 2 else None
    if dt_ns is None:
        if median_ns is not None:
            dt_ns = nice_interval_ns(median_ns)
        else:
            dt_ns = 60 * NS_PER_MIN
            notes.append(
                "WARNING: the interval length cannot be measured from a single timestamp; "
                "assumed 1 hour. Use ProductionAnalyzer(interval=...) to set it."
            )
    dt_ns = int(dt_ns)

    offsets = ns - ns[0]
    remainder = np.mod(offsets, dt_ns)
    deviation = np.minimum(remainder, dt_ns - remainder)
    regular = bool(n < 2 or deviation.max() <= 0.01 * dt_ns)
    if regular:
        steps = np.round(offsets / dt_ns).astype(np.int64)
        ns = ns[0] + steps * dt_ns
        ns, wall, values, _ = aggregate_same_time(ns, wall, values, counter)
        anchor = int(ns[0])
    else:
        gaps = np.diff(ns) / NS_PER_S
        why = ""
        if measured and median_ns is not None:
            rounded = nearest_nice_ns(median_ns)
            if rounded != dt_ns:
                dt_ns = rounded
            why = (
                f", the round interval nearest the typical gap of "
                f"{_secs(median_ns / NS_PER_S)}"
            )
        notes.append(
            "timestamps are irregular (gaps between readings from "
            f"{_secs(gaps.min())} to {_secs(gaps.max())}); resampled onto a regular "
            f"{_secs(dt_ns / NS_PER_S)} grid{why}, spreading each reading's units over the "
            "time until the next reading."
        )
        anchor = int(ns[0] - np.mod(wall[0], dt_ns))

    rel = (ns - anchor).astype(float) / NS_PER_S
    dt_s = dt_ns / NS_PER_S
    resets: List[Reset] = []
    gap_units = 0.0

    if counter:
        finite = np.isfinite(values)
        rel_c, ns_c, vals_c = rel[finite], ns[finite], values[finite]
        if vals_c.size < 2:
            starts = np.zeros(0)
            spans = np.zeros(0)
            counts = np.zeros(0)
        else:
            starts = np.array(rel_c[:-1], dtype=float)
            spans = np.array(np.diff(rel_c), dtype=float)
            counts = np.array(np.diff(vals_c), dtype=float)
            for i in np.flatnonzero(counts < 0):
                resets.append(
                    Reset(int(ns_c[i]), int(ns_c[i + 1]), float(vals_c[i]), float(vals_c[i + 1]))
                )
            counts[counts < 0] = np.nan
            limit = 2.0 * dt_s if regular else long_gap_seconds(spans, dt_s)
            long_gap = spans > limit + 1e-9
            if long_gap.any():
                gap_units = float(np.nansum(counts[long_gap]))
                counts[long_gap] = np.nan
    else:
        starts = np.array(rel, dtype=float)
        counts = np.array(values, dtype=float)
        if regular or n < 2:
            spans = np.full(n, dt_s)
        else:
            # Each reading covers the time until the next one. Only a gap far beyond the
            # data's own spacing is missing data: that reading is given one typical gap
            # and the rest of the gap is unknown.
            gaps_s = np.diff(rel)
            typical_gap = float(np.median(gaps_s))
            following = np.append(gaps_s, typical_gap)
            limit = long_gap_seconds(gaps_s, dt_s)
            spans = np.where(following <= limit + 1e-9, following, typical_gap)
        for i in np.flatnonzero(counts < 0):
            resets.append(
                Reset(int(ns[i]), int(ns[i] + spans[i] * NS_PER_S), None, float(counts[i]))
            )
        counts[counts < 0] = np.nan

    if starts.size:
        ends = starts + spans
        total_s = float(ends.max())
    else:
        total_s = 0.0
    k = int(math.ceil(total_s / dt_s - 1e-9)) if total_s > 0 else 0
    if k > MAX_BINS:
        raise ValueError(
            f"the data spans {k:,} intervals of {_secs(dt_s)}, more than {MAX_BINS:,}; "
            "resample it to a coarser interval first or pass a larger interval"
        )

    units = np.full(k, np.nan)
    reset_bins = np.zeros(k, dtype=bool)
    if k:
        known = np.isfinite(counts)
        units_piece = np.where(known, counts, 0.0)
        cover_piece = np.where(known, spans, 0.0)
        before_u = np.cumsum(units_piece) - units_piece
        before_c = np.cumsum(cover_piece) - cover_piece
        xp = np.empty(2 * starts.size)
        xp[0::2], xp[1::2] = starts, starts + spans
        fu = np.empty_like(xp)
        fu[0::2], fu[1::2] = before_u, before_u + units_piece
        fc = np.empty_like(xp)
        fc[0::2], fc[1::2] = before_c, before_c + cover_piece
        edges = np.arange(k + 1, dtype=float) * dt_s
        made = np.diff(np.interp(edges, xp, fu))
        coverage = np.diff(np.interp(edges, xp, fc)) / dt_s
        ok = coverage >= 0.5 - 1e-9
        # Unknown pieces add no coverage, so an interval that is mostly unknown stays NaN.
        units[ok] = np.round(made[ok] / coverage[ok], 9)
        for reset in resets:
            first = int(max(0, math.floor((reset.start_ns - anchor) / dt_ns)))
            last = int(min(k, math.ceil((reset.end_ns - anchor) / dt_ns)))
            reset_bins[first:max(first + 1, last)] = True
        reset_bins &= ~np.isfinite(units)
        if not regular and ok.any():
            # Resampling starts and ends on round times; the partly covered intervals at the
            # two edges are not gaps in the data, so they are dropped rather than reported.
            first_ok, last_ok = int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1])
            units = units[first_ok:last_ok + 1]
            reset_bins = reset_bins[first_ok:last_ok + 1]
            anchor += first_ok * dt_ns
            k = units.size

    bin_ns = anchor + np.arange(k, dtype=np.int64) * dt_ns
    index = datetime_index(bin_ns, tz)
    wall_index = index.tz_localize(None) if tz is not None else index
    bin_wall = np.array(wall_index.asi8, dtype=np.int64, copy=True)
    return Grid(
        ns=bin_ns,
        index=index,
        units=units,
        reset_bins=reset_bins,
        dt_ns=dt_ns,
        tz=tz,
        wall_ns=bin_wall,
        resets=resets,
        gap_units=gap_units,
        resampled=not regular,
    )


# ---------------------------------------------------------------- low counts
def is_integer_counts(values: np.ndarray) -> bool:
    """True when every usable value is a whole number: parts counted, not a measured rate."""
    finite = values[np.isfinite(values)]
    return bool(finite.size) and bool(np.all(finite == np.round(finite)))


def _block_ids(grid: Grid, block_ns: int) -> Tuple[int, np.ndarray]:
    """(UTC ns of the first block start, block number of each interval), on local round times."""
    anchor = int(grid.ns[0] - np.mod(grid.wall_ns[0], block_ns))
    return anchor, np.floor_divide(grid.ns - anchor, block_ns).astype(np.int64)


def _typical_block(units: np.ndarray, usable: np.ndarray, ids: np.ndarray, per: int) -> Optional[float]:
    """Median units in fully measured, scheduled blocks that made anything, else None."""
    blocks = int(ids[-1]) + 1
    n_usable = np.bincount(ids, weights=usable.astype(float), minlength=blocks)
    sums = np.bincount(ids, weights=np.where(usable, units, 0.0), minlength=blocks)
    running = sums[(n_usable >= per - 1e-9) & (sums > 0)]
    return float(np.median(running)) if running.size else None


def _block_lengths(dt_ns: int, boundary_minutes: List[int]) -> List[int]:
    """Round multiples of the interval that also fit every shift boundary."""
    out = []
    for seconds in NICE_SECONDS:
        block = int(seconds * NS_PER_S)
        if block <= dt_ns or block % dt_ns:
            continue
        if seconds >= 60 and any((minute * 60) % seconds for minute in boundary_minutes):
            continue
        out.append(block)
    return out


def coarsen_low_counts(
    grid: Grid,
    shift_of_minute: np.ndarray,
    boundary_minutes: List[int],
    min_units: float,
    notes: List[str],
) -> Grid:
    """Merge intervals that hold only a few whole parts each into longer ones.

    A line that makes 1.3 parts a minute reads 1, 1, 2, 1, 1, 2 at 1-minute intervals: the
    2s look like double counts and a 0 between two slow parts looks like a stop, although
    the line is perfectly steady. Every threshold compares one interval with the typical
    one, so each interval has to hold enough parts (``min_units``) for one part more or
    less not to matter. The intervals are merged into the shortest round length that does,
    aligned to the local clock and to every shift boundary.
    """
    k = grid.units.size
    if k < 2:
        return grid
    dt = int(grid.dt_ns)
    minute_of_day = np.mod(np.floor_divide(grid.wall_ns, NS_PER_MIN), 24 * 60).astype(np.int64)
    finite = np.isfinite(grid.units)
    usable = (shift_of_minute[minute_of_day] >= 0) & finite
    base = _typical_block(grid.units, usable, np.arange(k, dtype=np.int64), 1)
    if base is None or base >= min_units:
        return grid

    span_ns = int(grid.ns[-1] - grid.ns[0]) + dt
    chosen: Optional[int] = None
    typical: Optional[float] = None
    for block in _block_lengths(dt, boundary_minutes):
        if 2 * block > span_ns:
            break
        _, ids = _block_ids(grid, block)
        value = _typical_block(grid.units, usable, ids, block // dt)
        if value is None:
            continue
        chosen, typical = block, value
        if value >= min_units:
            break
    if chosen is None or typical is None:
        notes.append(
            f"WARNING: counts are low (about {base:.3g} units per {_secs(dt / NS_PER_S)} "
            "interval) and the data is too short to merge intervals, so ordinary gaps "
            "between parts may be reported as stops."
        )
        return grid

    per = chosen // dt
    anchor, ids = _block_ids(grid, chosen)
    blocks = int(ids[-1]) + 1
    present = np.bincount(ids, minlength=blocks)
    n_known = np.bincount(ids, weights=finite.astype(float), minlength=blocks)
    sums = np.bincount(ids, weights=np.where(finite, grid.units, 0.0), minlength=blocks)
    coverage = n_known / per
    ok = coverage >= 0.5 - 1e-9
    units = np.full(blocks, np.nan)
    units[ok] = np.round(sums[ok] / coverage[ok], 9)
    reset_bins = (np.bincount(ids, weights=grid.reset_bins.astype(float), minlength=blocks) > 0) & ~ok

    # A block that the data only starts or ends inside, too little of it to measure, is not
    # a gap in the data: it is dropped, as the edges of a resampled grid are.
    first, last = 0, blocks
    left_out = 0.0
    if present[0] < per and not ok[0]:
        left_out += float(sums[0])
        first = 1
    if last - 1 > first and present[-1] < per and not ok[-1]:
        left_out += float(sums[-1])
        last -= 1
    units, reset_bins = units[first:last], reset_bins[first:last]
    anchor += first * chosen

    bin_ns = anchor + np.arange(units.size, dtype=np.int64) * chosen
    index = datetime_index(bin_ns, grid.tz)
    wall_index = index.tz_localize(None) if grid.tz is not None else index
    old, new = _secs(dt / NS_PER_S), _secs(chosen / NS_PER_S)
    text = (
        f"counts are low: the line makes about {base:.3g} units in a typical {old} interval, "
        "so one part more or less would look like a stop or a double count. The analysis "
        f"uses {new} intervals instead (about {typical:.3g} units each), so stops and slow "
        f"running are timed to the nearest {new}."
    )
    if typical < min_units:
        text = (
            "WARNING: " + text + f" Even that is under {min_units:.3g} units an interval, so "
            "ordinary gaps between parts may still be reported as stops."
        )
    notes.append(text)
    if left_out > 0:
        notes.append(
            f"{left_out:.6g} units in a partial {new} interval at the edge of the data were "
            "left out."
        )
    return Grid(
        ns=bin_ns,
        index=index,
        units=units,
        reset_bins=reset_bins,
        dt_ns=int(chosen),
        tz=grid.tz,
        wall_ns=np.array(wall_index.asi8, dtype=np.int64, copy=True),
        resets=list(grid.resets),
        gap_units=grid.gap_units,
        resampled=grid.resampled,
    )


def _secs(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.0f} s" if abs(seconds - round(seconds)) < 1e-6 else f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 120:
        return f"{minutes:.0f} min" if abs(minutes - round(minutes)) < 1e-6 else f"{minutes:.1f} min"
    hours = minutes / 60.0
    return f"{hours:.0f} h" if abs(hours - round(hours)) < 1e-6 else f"{hours:.1f} h"
