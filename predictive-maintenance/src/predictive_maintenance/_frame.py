"""Turn a user DataFrame into a sorted time axis plus clean numeric channels."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._io import TableLike, load_table

logger = logging.getLogger(__name__)

_TIME_NAMES = ("time", "timestamp", "date", "datetime", "ts", "cycle", "hours")


@dataclass
class Prepared:
    """A DataFrame reduced to what the estimators need."""

    values: Dict[str, np.ndarray]
    channels: List[str]
    index: pd.Index
    x: np.ndarray
    is_temporal: bool
    time_name: Optional[str]
    notes: List[str] = field(default_factory=list)
    order: Optional[np.ndarray] = None

    @property
    def n_rows(self) -> int:
        """How many rows survived preparation."""
        return int(self.x.size)

    @property
    def was_sorted(self) -> bool:
        """True when the rows had to be reordered to run in time order."""
        return self.order is not None

    def align(self, per_row: np.ndarray) -> np.ndarray:
        """Carry a caller-supplied per-row array through the same sort the rows took.

        Labels, baseline masks and anything else indexed by the caller's row
        positions must travel with their row, or they end up describing a
        different reading entirely.
        """
        values = np.asarray(per_row)
        if self.order is None:
            return values
        return values[self.order]


def check_duplicate_columns(df: pd.DataFrame, label: str = "df") -> None:
    """Raise a clear ValueError when column names repeat, naming the duplicates."""
    counts = pd.Index(df.columns).value_counts()
    dupes = sorted(str(name) for name, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(
            "{0} has duplicate column names: {1}. Rename them so every channel "
            "can be identified.".format(label, ", ".join(dupes))
        )


def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series)


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return str(column)
    for column in df.columns:
        if str(column).strip().lower() in _TIME_NAMES:
            return str(column)
    return None


def _time_axis(index: pd.Index) -> Tuple[np.ndarray, bool]:
    """Return a float axis for least-squares fits, and whether it is real time."""
    if isinstance(index, pd.DatetimeIndex):
        nanos = np.asarray(index.values, dtype="datetime64[ns]").astype("int64")
        return nanos.astype(float), True
    if pd.api.types.is_numeric_dtype(index) and not pd.api.types.is_bool_dtype(index):
        values = np.asarray(index, dtype=float)
        if np.all(np.isfinite(values)):
            return values, False
    return np.arange(len(index), dtype=float), False


def _clean_channel(series: pd.Series, name: str, notes: List[str]) -> np.ndarray:
    """Numeric values with inf and NaN filled forward, then backward, then to zero."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    missing = int(values.isna().sum())
    total = max(len(values), 1)
    if missing == total:
        notes.append("channel " + repr(name) + " is entirely missing; treated as constant zero")
        logger.warning("channel %r is entirely missing", name)
        return np.zeros(len(values), dtype=float)
    if missing:
        notes.append(
            "channel {0!r}: {1:.0%} of readings missing, carried forward".format(
                name, missing / total
            )
        )
    filled = values.ffill().bfill()
    return np.asarray(filled.fillna(0.0), dtype=float)


def _resolve_time_name(frame: pd.DataFrame, time: Any, label: str) -> Optional[str]:
    if time is False:
        return None
    if time is None:
        return _find_time_column(frame)
    if time not in frame.columns:
        raise ValueError(
            "time column {0!r} is not in {1}; available columns: {2}".format(
                time, label, ", ".join(str(c) for c in frame.columns) or "(none)"
            )
        )
    return str(time)


def _resolve_channels(
    frame: pd.DataFrame, channels: Optional[Sequence[str]], time_name: Optional[str], label: str
) -> List[str]:
    if channels is None:
        picked = [str(c) for c in frame.columns if str(c) != time_name and _is_numeric(frame[c])]
        if not picked:
            raise ValueError(
                "{0} has no numeric sensor channels to work with; pass channels=[...] "
                "with the columns that hold sensor readings".format(label)
            )
        return picked
    if isinstance(channels, str):
        channels = [channels]
    picked = [str(c) for c in channels]
    if not picked:
        raise ValueError("channels=[] selects nothing; pass at least one channel name")
    missing = [c for c in picked if c not in frame.columns]
    if missing:
        raise ValueError(
            "channels not found in {0}: {1}; available columns: {2}".format(
                label, ", ".join(missing), ", ".join(str(c) for c in frame.columns) or "(none)"
            )
        )
    non_numeric = [c for c in picked if not _is_numeric(frame[c])]
    if non_numeric:
        raise ValueError(
            "channels must hold numbers, but these do not: " + ", ".join(non_numeric)
        )
    return picked


def _fill_missing_stamps(
    stamps: "pd.Series", time_name: str, notes: List[str]
) -> Optional["pd.DatetimeIndex"]:
    """Interpolate a few unreadable timestamps, or return ``None`` if there are too many.

    Remaining useful life is measured against the time axis, so throwing that axis away
    turned a visibly dying machine into "infinite remaining life" - the worst failure
    this package had. A handful of missing stamps is a data-entry problem, not a reason
    to stop measuring time, so they are filled from their neighbours and the report says
    how many. When most of the column is unreadable there is nothing to interpolate
    between and row order really is the honest fallback.
    """
    total = len(stamps)
    readable = int(stamps.notna().sum())
    if readable == total:
        return pd.DatetimeIndex(stamps.values)
    if readable < 2 or readable < 0.8 * total:
        return None
    nanoseconds = stamps.astype("int64").astype("float64")
    nanoseconds[stamps.isna().to_numpy()] = np.nan
    nanoseconds = nanoseconds.interpolate(method="linear", limit_direction="both")
    filled = pd.to_datetime(nanoseconds.to_numpy(), unit="ns", errors="coerce")
    if bool(pd.Series(filled).notna().all()):
        notes.append(
            "time column {0!r} had {1} unreadable timestamp(s) of {2}; they were filled "
            "from their neighbours".format(time_name, total - readable, total)
        )
        return pd.DatetimeIndex(filled)
    return None


def _build_index(
    frame: pd.DataFrame, time_name: Optional[str], n_rows: int, notes: List[str]
) -> Tuple[pd.Index, Optional[str]]:
    if time_name is None:
        if isinstance(frame.index, pd.DatetimeIndex):
            return frame.index, None
        return pd.RangeIndex(n_rows), None
    raw = frame[time_name]
    if pd.api.types.is_datetime64_any_dtype(raw):
        repaired = _fill_missing_stamps(pd.Series(raw).reset_index(drop=True), time_name, notes)
        if repaired is not None:
            return repaired, time_name
        notes.append(
            "time column {0!r} is mostly missing; using row order".format(time_name)
        )
        return pd.RangeIndex(n_rows), None
    if _is_numeric(raw) and np.all(np.isfinite(np.asarray(raw, dtype=float))):
        return pd.Index(np.asarray(raw, dtype=float)), time_name
    with warnings.catch_warnings():
        # This is already the last-resort branch; pandas guessing per element is
        # exactly what is wanted here, and its warning is not the user's problem.
        warnings.simplefilter("ignore", UserWarning)
        converted = pd.to_datetime(raw, errors="coerce")
    if bool(converted.notna().all()):
        return pd.DatetimeIndex(converted.values), time_name
    repaired = _fill_missing_stamps(pd.Series(converted).reset_index(drop=True), time_name, notes)
    if repaired is not None:
        return repaired, time_name
    notes.append(
        "time column {0!r} is neither dates nor finite numbers; using row order".format(time_name)
    )
    return pd.RangeIndex(n_rows), None


def prepare(
    df: TableLike,
    *,
    time: Any = None,
    channels: Optional[Sequence[str]] = None,
    label: str = "df",
    min_rows: int = 2,
) -> Prepared:
    """Validate and reshape ``df`` into a :class:`Prepared` bundle.

    ``time`` names the timestamp column (pass ``False`` to force row positions);
    ``channels`` picks the sensor columns, defaulting to every numeric column
    that is not the time column.
    """
    frame = load_table(df, label=label)
    check_duplicate_columns(frame, label=label)
    notes: List[str] = []

    # Channel names are compared and returned as strings throughout, so a frame whose
    # labels are not strings - a DataFrame built from a numpy array has integer labels,
    # and that is an ordinary thing to hand us - looked up "0" against the integer 0 and
    # died with a bare KeyError. Renaming a copy once keeps every later lookup honest.
    if any(not isinstance(column, str) for column in frame.columns):
        renamed = [str(column) for column in frame.columns]
        if len(set(renamed)) != len(renamed):
            raise ValueError(
                "{0} has column labels that collide once written as text: {1}".format(
                    label, ", ".join(sorted({c for c in renamed if renamed.count(c) > 1}))
                )
            )
        frame = frame.copy()
        frame.columns = renamed
        # the caller names columns with the labels THEY have, so accept those too
        if time is not None and time is not False and not isinstance(time, str):
            time = str(time)
        if channels is not None and not isinstance(channels, str):
            channels = [str(c) for c in channels]
        elif isinstance(channels, str):
            channels = str(channels)

    time_name = _resolve_time_name(frame, time, label)
    picked = _resolve_channels(frame, channels, time_name, label)

    n_rows = len(frame)
    if n_rows < min_rows:
        raise ValueError(
            "{0} has {1} row(s); at least {2} rows of sensor history are needed to "
            "judge equipment health".format(label, n_rows, min_rows)
        )

    index, time_name = _build_index(frame, time_name, n_rows, notes)

    order = None
    if len(index) > 1 and not bool(pd.Index(index).is_monotonic_increasing):
        order = np.argsort(np.asarray(index), kind="stable")
        reordered = np.asarray(index)[order]
        index = pd.DatetimeIndex(reordered) if isinstance(index, pd.DatetimeIndex) else pd.Index(reordered)
        notes.append("rows were not in time order; sorted by the time axis first")

    values: Dict[str, np.ndarray] = {}
    for name in picked:
        column = frame[name]
        if order is not None:
            column = column.iloc[order]
        values[name] = _clean_channel(column.reset_index(drop=True), name, notes)

    x, is_temporal = _time_axis(pd.Index(index))
    if len(x) > 1 and float(np.max(x) - np.min(x)) <= 0.0:
        notes.append("every timestamp is identical; using row positions as the time axis")
        x = np.arange(len(index), dtype=float)
        is_temporal = False

    return Prepared(
        values=values,
        channels=picked,
        index=pd.Index(index),
        x=x,
        is_temporal=is_temporal,
        time_name=time_name,
        notes=notes,
        order=order,
    )
