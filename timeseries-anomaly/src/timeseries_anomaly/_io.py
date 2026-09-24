"""Turn whatever the caller passed in into one numeric series plus a time axis.

Everything downstream works on a plain float array in time order, so all of the
guessing about columns, timestamps and ordering happens here and is recorded as
warnings the caller can read back off the result.
"""
from __future__ import annotations

import logging
import os
import warnings as _pywarnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._robust import as_float_array

LOG = logging.getLogger(__name__)

#: column names that look like a time axis, lowercased
TIME_NAMES = ("time", "timestamp", "date", "datetime", "ds", "dt", "period")
#: column names that look like the measurement, lowercased
VALUE_NAMES = ("value", "values", "y", "val", "reading", "measurement", "metric", "signal", "count")


@dataclass
class LoadedSeries:
    """A numeric series in input order, plus the permutation that puts it in time order."""

    values: np.ndarray
    time: Optional[np.ndarray]
    order: np.ndarray
    label: str
    time_label: Optional[str] = None
    index: Optional[pd.Index] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Number of points, missing values included."""
        return int(self.values.size)

    @property
    def in_time_order(self) -> np.ndarray:
        """The values sorted by their timestamp."""
        return self.values[self.order]

    def restore(self, ordered: np.ndarray) -> np.ndarray:
        """Map an array computed in time order back to input order."""
        ordered = np.asarray(ordered)
        out = np.empty(ordered.shape, dtype=ordered.dtype)
        out[self.order] = ordered
        return out


def _read_path(path: Any) -> pd.DataFrame:
    """Read a .csv or .parquet file into a DataFrame."""
    text = str(path)
    lower = text.lower()
    if not os.path.exists(text):
        raise FileNotFoundError(f"{text!r} does not exist")
    if lower.endswith((".parquet", ".pq")):
        try:
            return pd.read_parquet(text)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "reading .parquet needs pyarrow: pip install timeseries-anomaly[parquet]"
            ) from exc
    if lower.endswith((".csv", ".csv.gz", ".csv.zip", ".tsv", ".txt")):
        separator = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(text, sep=separator)
    raise ValueError(f"unsupported file type {text!r}; use .csv or .parquet")


def _check_columns(frame: pd.DataFrame) -> None:
    """Refuse a frame with duplicate column names, naming them."""
    counts = Counter(str(column) for column in frame.columns)
    duplicates = sorted(name for name, seen in counts.items() if seen > 1)
    if duplicates:
        raise ValueError(
            "the table has duplicate column names: "
            + ", ".join(repr(name) for name in duplicates)
            + "; rename them so the value and time columns are unambiguous"
        )


def _as_time_key(raw: Any, name: str, warnings: List[str]) -> Optional[np.ndarray]:
    """Best effort conversion of a column or index into a sortable time axis."""
    series = pd.Series(raw)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.to_numpy(copy=True)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(copy=True)
    parsed = pd.to_datetime(series, errors="coerce")
    if bool(parsed.notna().any()):
        unreadable = int(parsed.isna().sum())
        if unreadable:
            warnings.append(f"{unreadable} timestamp(s) in {name!r} could not be read and sort last")
        return parsed.to_numpy(copy=True)
    return None


#: punctuation a column of text must contain before it is tried as timestamps
DATE_MARKS = ("-", "/", ":")


def _looks_like_time(column: pd.Series) -> Optional[np.ndarray]:
    """Parse a column of text as timestamps, but only when it really is one.

    Dates arrive from a ``.csv`` as ordinary strings, so a time axis is invisible
    to a dtype check. This tries parsing, and believes the result only when the
    text carries date punctuation, nearly all of it parses, and the stamps are
    not all the same - which keeps a column of plain words or bare numbers from
    being mistaken for a clock.
    """
    if column.empty:
        return None
    text = column.dropna().astype(str)
    if text.empty or not text.str.contains("|".join(DATE_MARKS), regex=True).all():
        return None
    try:
        with _pywarnings.catch_warnings():
            _pywarnings.simplefilter("ignore")
            parsed = pd.to_datetime(column, errors="coerce")
    except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
        return None
    readable = parsed.notna()
    if int(readable.sum()) < 2 or float(readable.mean()) < 0.9:
        return None
    if parsed[readable].nunique() < 2:
        return None
    return parsed.to_numpy(copy=True)


def _pick_time_column(
    frame: pd.DataFrame, time: Optional[str], warnings: List[str]
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Return ``(name, key_array)`` for the time axis, or ``(None, None)``."""
    if time is not None:
        if time not in frame.columns:
            raise ValueError(
                f"time column {time!r} is not in the table; available columns: "
                + ", ".join(repr(str(c)) for c in frame.columns)
            )
        key = _as_time_key(frame[time], str(time), warnings)
        if key is None:
            raise ValueError(f"time column {time!r} holds no readable dates or numbers")
        return str(time), key
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return str(column), frame[column].to_numpy(copy=True)
    for column in frame.columns:
        if str(column).strip().lower() in TIME_NAMES:
            key = _as_time_key(frame[column], str(column), warnings)
            if key is not None:
                return str(column), key
    index = frame.index
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    if isinstance(index, pd.DatetimeIndex):
        return str(index.name or "time"), index.to_numpy(copy=True)
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        key = _looks_like_time(frame[column])
        if key is not None:
            warnings.append(
                f"no time= given, reading {str(column)!r} as the time axis"
            )
            return str(column), key
    return None, None


def _pick_value_column(
    frame: pd.DataFrame, value: Optional[str], skip: Optional[str], warnings: List[str]
) -> str:
    """Return the name of the column holding the measurement."""
    if value is not None:
        if value not in frame.columns:
            raise ValueError(
                f"value column {value!r} is not in the table; available columns: "
                + ", ".join(repr(str(c)) for c in frame.columns)
            )
        return str(value)
    candidates = [
        str(column)
        for column in frame.columns
        if str(column) != skip and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not candidates:
        spare = [str(column) for column in frame.columns if str(column) != skip]
        if len(frame) == 0 and spare:
            return spare[0]
        raise ValueError(
            "no numeric column to analyse; pass value=<column> naming the measurement"
        )
    if len(candidates) == 1:
        return candidates[0]
    for column in candidates:
        if column.strip().lower() in VALUE_NAMES:
            warnings.append(f"no value= given, using the column named {column!r}")
            return column
    warnings.append(
        f"no value= given and {len(candidates)} numeric columns are present, using {candidates[0]!r}"
    )
    return candidates[0]


def _from_frame(
    frame: pd.DataFrame, value: Optional[str], time: Optional[str], warnings: List[str]
) -> LoadedSeries:
    _check_columns(frame)
    if frame.shape[1] == 0:
        warnings.append("the table has no columns")
        return LoadedSeries(
            values=np.zeros(0, dtype=float),
            time=None,
            order=np.zeros(0, dtype=int),
            label=str(value) if value else "value",
            time_label=None,
            index=frame.index,
            warnings=warnings,
        )
    time_label, time_key = _pick_time_column(frame, time, warnings)
    value_label = _pick_value_column(frame, value, time_label, warnings)
    values = as_float_array(frame[value_label].to_numpy(copy=True))
    return _finish(
        values, time_key, value_label, time_label, warnings, index=frame.index
    )


def _finish(
    values: np.ndarray,
    time_key: Optional[np.ndarray],
    label: str,
    time_label: Optional[str],
    warnings: List[str],
    index: Optional[pd.Index] = None,
) -> LoadedSeries:
    """Sort by time when there is one, recording what had to be fixed."""
    n = int(values.size)
    if time_key is None or n == 0:
        return LoadedSeries(
            values=values,
            time=None if time_key is None else np.asarray(time_key),
            order=np.arange(n),
            label=label,
            time_label=time_label,
            index=index,
            warnings=warnings,
        )
    time_key = np.asarray(time_key)
    order = np.argsort(time_key, kind="stable")
    ordered = time_key[order]
    if not np.array_equal(order, np.arange(n)):
        warnings.append("timestamps were not in order; the series was sorted before detecting")
    duplicates = int(np.sum(ordered[1:] == ordered[:-1])) if n > 1 else 0
    if duplicates:
        warnings.append(
            f"{duplicates} duplicate timestamp(s); every point is kept and scored in input order"
        )
    return LoadedSeries(
        values=values,
        time=time_key,
        order=order,
        label=label,
        time_label=time_label,
        index=index,
        warnings=warnings,
    )


def _frame_from_dict(data: dict) -> pd.DataFrame:
    """A dict of columns as a table, refusing a dict of single values in our own words."""
    lonely = [
        str(name)
        for name, column in data.items()
        if column is None or np.isscalar(column) or isinstance(column, (str, bytes))
    ]
    if lonely:
        raise ValueError(
            "a dict must map each column name to a sequence of values, but "
            + ", ".join(repr(name) for name in lonely)
            + " holds a single value; wrap each column in a list"
        )
    return pd.DataFrame(data)


def load(data: Any, value: Optional[str] = None, time: Optional[str] = None) -> LoadedSeries:
    """Load `data` into a numeric series plus an optional time axis.

    Accepts a pandas Series or DataFrame, a list/tuple/ndarray of numbers, a dict
    of columns, or a path to a ``.csv`` / ``.parquet`` file.
    """
    if data is None:
        raise ValueError("data is None; pass a series, a table, a list of numbers or a file path")
    warnings: List[str] = []
    if isinstance(data, (str, os.PathLike)):
        return _from_frame(_read_path(data), value, time, warnings)
    if isinstance(data, pd.DataFrame):
        return _from_frame(data, value, time, warnings)
    if isinstance(data, dict):
        return _from_frame(_frame_from_dict(data), value, time, warnings)
    if isinstance(data, pd.Series):
        label = str(data.name) if data.name is not None else (str(value) if value else "value")
        values = as_float_array(data.to_numpy(copy=True))
        index = data.index
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp()
        if isinstance(index, pd.DatetimeIndex):
            return _finish(
                values,
                index.to_numpy(copy=True),
                label,
                str(index.name or "time"),
                warnings,
                index=data.index,
            )
        return _finish(values, None, label, None, warnings, index=data.index)
    if isinstance(data, pd.Index):
        return _finish(as_float_array(data.to_numpy(copy=True)), None, str(value or "value"), None, warnings)
    if isinstance(data, (list, tuple, np.ndarray, range)):
        array = np.asarray(list(data) if isinstance(data, range) else data)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array.reshape(-1)
        return _finish(as_float_array(array), None, str(value or "value"), None, warnings)
    if np.isscalar(data):
        return _finish(as_float_array([data]), None, str(value or "value"), None, warnings)
    raise TypeError(
        f"unsupported data type {type(data).__name__}; pass a pandas Series/DataFrame, "
        "a list of numbers or a .csv/.parquet path"
    )
