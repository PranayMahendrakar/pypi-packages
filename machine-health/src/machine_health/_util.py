"""Small shared helpers: JSON conversion, channel selection, time handling."""

from __future__ import annotations

import datetime as _dt
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


def jsonable(obj: Any) -> Any:
    """Recursively turn numpy/pandas scalars and containers into JSON-safe Python values."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset, np.ndarray, pd.Index, pd.Series)):
        return [jsonable(v) for v in list(obj)]
    if obj is pd.NaT:
        return None
    if isinstance(obj, np.datetime64):
        return None if np.isnat(obj) else pd.Timestamp(obj).isoformat()
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, _dt.timedelta)):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Return ``value`` pulled inside [low, high]; non-finite values become ``low``."""
    number = float(value)
    if not math.isfinite(number):
        return low
    if number < low:
        return low
    if number > high:
        return high
    return number


def column_map(df: pd.DataFrame) -> Dict[str, Any]:
    """Map the string form of every column name to the real column label."""
    return {str(label): label for label in df.columns}


def is_channel_dtype(series: pd.Series) -> bool:
    """True when a column holds numbers (or booleans) usable as a sensor channel."""
    if is_datetime64_any_dtype(series):
        return False
    if is_bool_dtype(series):
        return True
    return bool(is_numeric_dtype(series))


def as_float(series: pd.Series) -> np.ndarray:
    """Channel values as a float64 array; booleans become 0.0/1.0, bad values become NaN."""
    if is_bool_dtype(series):
        return series.astype("float64").to_numpy(dtype="float64")
    values = pd.to_numeric(series, errors="coerce")
    return np.asarray(values, dtype="float64")


def describe_names(names: Sequence[Any], limit: int = 25) -> str:
    """'a, b, c' for an error message, truncated when a table is very wide."""
    text = [str(n) for n in names]
    if not text:
        return "<none>"
    if len(text) > limit:
        return ", ".join(text[:limit]) + f", ... ({len(text)} in total)"
    return ", ".join(text)


def select_channels(
    df: pd.DataFrame,
    channels: Optional[Sequence[Any]],
    exclude: Sequence[str] = (),
) -> Tuple[List[str], List[str]]:
    """Return (channel names, names skipped because they are not numeric).

    With ``channels=None`` every numeric column is used. An explicit list is checked:
    an unknown name raises ValueError listing what is available, and a known but
    non-numeric name raises ValueError saying so.
    """
    names = column_map(df)
    skipped: List[str] = []
    if channels is None:
        chosen = []
        for name, label in names.items():
            if name in exclude:
                continue
            if is_channel_dtype(df[label]):
                chosen.append(name)
            else:
                skipped.append(name)
        return chosen, skipped
    if isinstance(channels, str):
        channels = [channels]
    chosen = []
    for raw in channels:
        name = str(raw)
        if name not in names:
            raise ValueError(
                f"unknown channel {name!r}; available channels: {describe_names(list(names))}"
            )
        if not is_channel_dtype(df[names[name]]):
            raise ValueError(
                f"channel {name!r} is not numeric (dtype {df[names[name]].dtype}); "
                "machine-health scores numeric sensor channels"
            )
        if name not in chosen:
            chosen.append(name)
    return chosen, skipped


def resolve_time(
    df: pd.DataFrame, time: Any
) -> Tuple[Optional[pd.Series], Optional[str], List[str]]:
    """Return (parsed time values, column name used, notes).

    ``time`` may be a column name, a Series/array of the right length, or None.
    Unusable timestamps are tolerated: they become a note and sort to the end
    rather than raising, because one bad clock reading should not stop a score.
    """
    notes: List[str] = []
    if time is None:
        return None, None, notes
    names = column_map(df)
    if isinstance(time, str) or not isinstance(
        time, (pd.Series, pd.Index, np.ndarray, list, tuple)
    ):
        name = str(time)
        if name not in names:
            raise ValueError(
                f"unknown time column {name!r}; available columns: {describe_names(list(names))}"
            )
        raw = df[names[name]]
        label: Optional[str] = name
    else:
        values = list(time)
        if len(values) != len(df):
            raise ValueError(
                f"time has {len(values)} values but df has {len(df)} rows; they must match"
            )
        raw = pd.Series(values, index=df.index)
        label = None
    parsed = parse_time(raw, label or "time")
    n_missing = int(parsed.isna().sum())
    if n_missing:
        notes.append(
            f"time has {n_missing} unusable value(s); those rows sort to the end of the window"
        )
    return parsed, label, notes


def parse_time(series: pd.Series, name: str) -> pd.Series:
    """Parse a time column to datetime64 or float64, turning bad values into NaT/NaN."""
    if is_datetime64_any_dtype(series):
        parsed = series
    elif is_bool_dtype(series):
        raise ValueError(f"time column {name!r} is boolean; expected dates or numbers")
    elif is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    else:
        parsed = _to_datetime(series)
        if parsed is None or parsed.isna().all():
            numeric = pd.to_numeric(series, errors="coerce").astype("float64")
            if numeric.notna().any():
                return numeric
            raise ValueError(f"time column {name!r} could not be read as dates or numbers")
    if is_datetime64_any_dtype(parsed) and getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def _to_datetime(series: pd.Series) -> Optional[pd.Series]:
    values = series if is_object_dtype(series) else series.astype(object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(values, errors="coerce")
        except (ValueError, TypeError, OverflowError):
            pass
        try:  # pandas >= 2.0 can parse per-element formats
            return pd.to_datetime(values, errors="coerce", format="mixed")
        except (ValueError, TypeError, OverflowError):
            return None


def sort_key(parsed: pd.Series) -> np.ndarray:
    """A sortable float array for a parsed time column; unusable values sort last."""
    if is_datetime64_any_dtype(parsed):
        raw = parsed.to_numpy(dtype="datetime64[ns]").astype("float64")
    else:
        raw = np.asarray(parsed.to_numpy(dtype="float64"), dtype="float64")
    return np.where(np.isnan(raw), np.inf, raw)


def fmt(value: Any, digits: int = 2) -> str:
    """Compact plain-ASCII rendering of a number for summary text."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.{digits}f}"
