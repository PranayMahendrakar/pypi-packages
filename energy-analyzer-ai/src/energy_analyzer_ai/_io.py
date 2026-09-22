"""Turn whatever the caller passed in into one numeric series on a time axis.

Everything downstream works on a pandas Series of floats whose index is either a
sorted ``DatetimeIndex`` (timezone preserved exactly as it arrived) or a plain
integer position when the data carried no timestamps at all. All of the guessing
about columns, timestamps and ordering happens here and is recorded as notes the
caller can read back off the report.
"""
from __future__ import annotations

import logging
import os
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

#: column names that look like a time axis, lowercased
TIME_NAMES = (
    "time",
    "timestamp",
    "datetime",
    "date",
    "ds",
    "dt",
    "period",
    "interval",
    "reading_time",
    "recorded_at",
    "read_at",
    "when",
    "start",
    "start_time",
)
#: column names that look like the measurement, lowercased
VALUE_NAMES = (
    "kwh",
    "kw",
    "wh",
    "consumption",
    "usage",
    "energy",
    "demand",
    "power",
    "load",
    "reading",
    "meter",
    "value",
    "values",
    "units",
    "y",
)


@dataclass
class LoadedSeries:
    """A float series in time order, plus what had to be guessed to get it."""

    series: pd.Series
    label: str
    time_label: Optional[str] = None
    has_time: bool = False
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Number of readings, missing values included."""
        return int(self.series.size)


def _read_path(path: Any) -> pd.DataFrame:
    """Read a ``.csv`` or ``.parquet`` file into a DataFrame."""
    text = str(path)
    lower = text.lower()
    if not os.path.exists(text):
        raise FileNotFoundError(f"{text!r} does not exist")
    if lower.endswith((".parquet", ".pq")):
        try:
            return pd.read_parquet(text)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "reading .parquet needs pyarrow: pip install energy-analyzer-ai[parquet]"
            ) from exc
    if lower.endswith((".csv", ".csv.gz", ".csv.zip", ".tsv", ".txt")):
        separator = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(text, sep=separator)
    raise ValueError(f"unsupported file type {text!r}; use .csv or .parquet")


def check_columns(frame: pd.DataFrame) -> None:
    """Refuse a frame with duplicate column names, naming them."""
    counts = Counter(str(column) for column in frame.columns)
    duplicates = sorted(name for name, seen in counts.items() if seen > 1)
    if duplicates:
        raise ValueError(
            "the table has duplicate column names: "
            + ", ".join(repr(name) for name in duplicates)
            + "; rename them so the value and time columns are unambiguous"
        )


def _to_datetime(raw: Any, name: str, notes: List[str]) -> Optional[pd.DatetimeIndex]:
    """Best effort conversion of a column or index into a DatetimeIndex.

    Keeps a single fixed offset as it arrived. Mixed offsets are normalised to UTC,
    because there is no other way to place them on one axis.
    """
    series = pd.Series(raw).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.DatetimeIndex(series)
    if pd.api.types.is_numeric_dtype(series):
        return None
    # Both of these are guesses about a column we may well reject, so pandas
    # grumbling about mixed offsets or an uninferrable format is our business,
    # not the caller's.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError):
            parsed = pd.Series([pd.NaT] * len(series))
        if not pd.api.types.is_datetime64_any_dtype(parsed):
            try:
                parsed = pd.to_datetime(series, errors="coerce", utc=True)
            except (ValueError, TypeError):  # pragma: no cover - very unusual input
                return None
            if pd.api.types.is_datetime64_any_dtype(parsed):
                notes.append(
                    f"timestamps in {name!r} mix time zones; they were converted to UTC"
                )
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        return None
    if not bool(parsed.notna().any()):
        return None
    return pd.DatetimeIndex(parsed)


def _pick_time_column(
    frame: pd.DataFrame, time: Optional[str], notes: List[str]
) -> Tuple[Optional[str], Optional[pd.DatetimeIndex]]:
    """Return ``(name, DatetimeIndex)`` for the time axis, or ``(None, None)``."""
    if time is not None:
        if time not in frame.columns:
            raise ValueError(
                f"time column {time!r} is not in the table; available columns: "
                + ", ".join(repr(str(c)) for c in frame.columns)
            )
        stamps = _to_datetime(frame[time], str(time), notes)
        if stamps is None:
            raise ValueError(f"time column {time!r} holds no readable timestamps")
        return str(time), stamps
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return str(column), pd.DatetimeIndex(pd.Series(frame[column]).reset_index(drop=True))
    index = frame.index
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    if isinstance(index, pd.DatetimeIndex):
        return str(index.name or "time"), index
    for column in frame.columns:
        if str(column).strip().lower() in TIME_NAMES:
            stamps = _to_datetime(frame[column], str(column), notes)
            if stamps is not None:
                return str(column), stamps
    # Nothing was named like a clock. A csv written in another language still has
    # a timestamp column, so try to read one: a column that parses as dates and
    # is not really numbers is a time axis whatever it is called.
    for column in frame.columns:
        raw = pd.Series(frame[column])
        if pd.api.types.is_numeric_dtype(raw) or _numeric_score(raw) >= 0.5:
            continue
        stamps = _to_datetime(raw, str(column), notes)
        if stamps is None or len(stamps) == 0:
            continue
        if float(pd.Series(stamps).notna().mean()) < 0.8:
            continue
        if int(pd.Series(stamps).nunique(dropna=True)) < min(2, len(stamps)):
            continue
        notes.append(
            f"no time= given; the entries in {str(column)!r} read as timestamps, "
            "so that column was used as the time axis"
        )
        return str(column), stamps
    return None, None


def _numeric_score(series: pd.Series) -> float:
    """Share of entries that read as a number; 0.0 when nothing does."""
    if len(series) == 0:
        return 0.0
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean())


def _pick_value_column(
    frame: pd.DataFrame, value: Optional[str], skip: Optional[str], notes: List[str]
) -> str:
    """Return the name of the column holding the measurement."""
    if value is not None:
        if value not in frame.columns:
            raise ValueError(
                f"value column {value!r} is not in the table; available columns: "
                + ", ".join(repr(str(c)) for c in frame.columns)
            )
        return str(value)
    others = [str(column) for column in frame.columns if str(column) != skip]
    numeric = [name for name in others if pd.api.types.is_numeric_dtype(frame[name])]
    usable = [name for name in numeric if bool(frame[name].notna().any())]
    candidates = usable or numeric
    if not candidates:
        candidates = [name for name in others if _numeric_score(frame[name]) >= 0.5]
        if candidates:
            notes.append(
                f"column {candidates[0]!r} is stored as text; the entries that read as "
                "numbers were used and the rest treated as missing"
            )
    if not candidates:
        raise ValueError(
            "no numeric column to analyse; pass value=<column> naming the meter reading. "
            "available columns: " + ", ".join(repr(str(c)) for c in frame.columns)
        )
    if len(candidates) == 1:
        return candidates[0]
    for column in candidates:
        if column.strip().lower() in VALUE_NAMES:
            notes.append(f"no value= given, using the column named {column!r}")
            return column
    for column in candidates:
        lowered = column.strip().lower()
        if any(word in lowered for word in VALUE_NAMES):
            notes.append(f"no value= given, using the column named {column!r}")
            return column
    notes.append(
        f"no value= given and {len(candidates)} numeric columns are present, "
        f"using {candidates[0]!r}"
    )
    return candidates[0]


def _coerce(series: pd.Series, name: str, notes: List[str]) -> np.ndarray:
    """Turn one column into floats, counting whatever could not be read."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    coerced = pd.to_numeric(series, errors="coerce")
    unreadable = int(coerced.isna().sum() - pd.Series(series).isna().sum())
    if unreadable > 0:
        notes.append(
            f"{unreadable} reading(s) in {name!r} were not numbers and are treated as missing"
        )
    return coerced.to_numpy(dtype=float)


def _assemble(
    values: np.ndarray,
    stamps: Optional[pd.DatetimeIndex],
    label: str,
    time_label: Optional[str],
    notes: List[str],
    warnings: List[str],
) -> LoadedSeries:
    """Build the time-ordered series, recording anything that had to be fixed."""
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    if stamps is None or n == 0:
        if stamps is None and n > 0:
            warnings.append(
                "no timestamp column was found, so the readings are treated as evenly "
                "spaced periods in the order given; pass time=<column> for a real time axis"
            )
        series = pd.Series(values, index=pd.RangeIndex(n), name=label, dtype=float)
        return LoadedSeries(
            series=series,
            label=label,
            time_label=None,
            has_time=False,
            notes=notes,
            warnings=warnings,
        )
    stamps = pd.DatetimeIndex(stamps)
    unreadable = int(stamps.isna().sum())
    if unreadable:
        notes.append(f"{unreadable} unreadable timestamp(s) and their readings were dropped")
        keep = ~stamps.isna()
        stamps = stamps[keep]
        values = values[np.asarray(keep)]
    if stamps.size == 0:
        warnings.append("every timestamp was unreadable, so nothing could be placed in time")
        return LoadedSeries(
            series=pd.Series([], index=pd.RangeIndex(0), name=label, dtype=float),
            label=label,
            time_label=time_label,
            has_time=False,
            notes=notes,
            warnings=warnings,
        )
    order = np.argsort(stamps.to_numpy(), kind="stable")
    if not np.array_equal(order, np.arange(stamps.size)):
        notes.append("timestamps were out of order; the readings were sorted before analysis")
        stamps = stamps[order]
        values = values[order]
    duplicates = int(stamps.duplicated().sum())
    if duplicates:
        notes.append(
            f"{duplicates} duplicate timestamp(s); readings sharing a timestamp are added together"
        )
    series = pd.Series(values, index=stamps, name=label, dtype=float)
    series.index.name = time_label or "time"
    return LoadedSeries(
        series=series,
        label=label,
        time_label=time_label,
        has_time=True,
        notes=notes,
        warnings=warnings,
    )


def _from_frame(
    frame: pd.DataFrame,
    value: Optional[str],
    time: Optional[str],
    notes: List[str],
    warnings: List[str],
) -> LoadedSeries:
    check_columns(frame)
    if frame.shape[1] == 0:
        warnings.append("the table has no columns, so there is nothing to analyse")
        return _assemble(np.zeros(0), None, str(value or "value"), None, notes, warnings)
    time_label, stamps = _pick_time_column(frame, time, notes)
    value_label = _pick_value_column(frame, value, time_label, notes)
    values = _coerce(frame[value_label], value_label, notes)
    return _assemble(values, stamps, value_label, time_label, notes, warnings)


def _is_column(item: Any) -> bool:
    """True when a dict value looks like a column of readings, not one value."""
    if isinstance(item, (str, bytes)):
        return False
    return hasattr(item, "__len__") or hasattr(item, "__iter__")


def _frame_from_dict(data: dict) -> pd.DataFrame:
    """Build a frame from a dict of columns, refusing a dict of single values.

    pandas has its own words for both failures here ("If using all scalar values,
    you must pass an index", "All arrays must be of the same length") and they name
    nothing the caller can act on, so they never reach them.
    """
    if data and not any(_is_column(column) for column in data.values()):
        named = ", ".join(repr(str(key)) for key in list(data)[:3])
        holds = "holds" if len(data) == 1 else "each hold"
        raise ValueError(
            f"dict values must be columns of readings, not single values ({named} "
            f"{holds} one value); pass {{'kwh': [1.0, 2.0, ...]}} or "
            "{'time': [...], 'kwh': [...]}"
        )
    try:
        return pd.DataFrame(data)
    except (ValueError, TypeError):
        lengths = {
            str(key): len(column)
            for key, column in data.items()
            if _is_column(column) and hasattr(column, "__len__")
        }
        detail = ""
        if len(set(lengths.values())) > 1:
            detail = " (" + ", ".join(
                f"{name} has {size}" for name, size in lengths.items()
            ) + ")"
        raise ValueError(
            "the dict could not be read as columns of readings"
            + detail
            + "; pass one value per reading in every column, e.g. "
            "{'time': [...], 'kwh': [1.0, 2.0, ...]}"
        ) from None


def load(data: Any, value: Optional[str] = None, time: Optional[str] = None) -> LoadedSeries:
    """Load `data` into a float series on a time axis.

    Accepts a pandas DataFrame or Series, a dict of columns, a list/array of
    numbers, or a path to a ``.csv`` / ``.parquet`` file.
    """
    if data is None:
        raise ValueError(
            "data is None; pass meter readings as a DataFrame, a Series, a list of "
            "numbers or a .csv/.parquet path"
        )
    notes: List[str] = []
    warnings: List[str] = []
    if isinstance(data, (str, os.PathLike)):
        return _from_frame(_read_path(data), value, time, notes, warnings)
    if isinstance(data, pd.DataFrame):
        return _from_frame(data, value, time, notes, warnings)
    if isinstance(data, dict):
        return _from_frame(_frame_from_dict(data), value, time, notes, warnings)
    if isinstance(data, pd.Series):
        label = str(data.name) if data.name is not None else str(value or "value")
        index = data.index
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp()
        values = _coerce(data.reset_index(drop=True), label, notes)
        if isinstance(index, pd.DatetimeIndex):
            return _assemble(
                values, index, label, str(index.name or "time"), notes, warnings
            )
        return _assemble(values, None, label, None, notes, warnings)
    if isinstance(data, pd.Index):
        return _assemble(
            _coerce(pd.Series(data), str(value or "value"), notes),
            None,
            str(value or "value"),
            None,
            notes,
            warnings,
        )
    if isinstance(data, (list, tuple, np.ndarray, range)):
        array = np.asarray(list(data) if isinstance(data, range) else data)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array.reshape(-1)
        if array.ndim > 1:
            raise ValueError(
                f"expected one column of readings, got an array with shape {array.shape}; "
                "pass a DataFrame and name the column with value="
            )
        return _assemble(
            _coerce(pd.Series(array), str(value or "value"), notes),
            None,
            str(value or "value"),
            None,
            notes,
            warnings,
        )
    if np.isscalar(data):
        return _assemble(
            np.asarray([float(data)]), None, str(value or "value"), None, notes, warnings
        )
    raise TypeError(
        f"unsupported data type {type(data).__name__}; pass a pandas DataFrame/Series, "
        "a list of numbers or a .csv/.parquet path"
    )
