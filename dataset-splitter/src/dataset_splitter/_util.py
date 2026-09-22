"""Small shared helpers: JSON conversion, time parsing, target kind and strata."""
from __future__ import annotations

import datetime as _dt
import math
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
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
    if isinstance(obj, (pd.Timedelta, _dt.timedelta, pd.Interval)):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def parse_time(series: pd.Series, name: Any) -> pd.Series:
    """Return a time column as datetime64 or float64 values.

    Raises ValueError when values are missing or cannot be interpreted as dates or numbers.
    """
    if is_datetime64_any_dtype(series):
        parsed = series
    elif is_bool_dtype(series):
        raise ValueError(f"time column {name!r} is boolean; expected dates or numbers")
    elif is_numeric_dtype(series):
        parsed = pd.to_numeric(series, errors="coerce").astype("float64")
    else:
        parsed = _to_datetime(series, name)
    n_missing = int(parsed.isna().sum())
    if n_missing:
        raise ValueError(
            f"time column {name!r} has {n_missing} missing value(s); fill or drop them before splitting"
        )
    if is_datetime64_any_dtype(parsed) and getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def _to_datetime(series: pd.Series, name: Any) -> pd.Series:
    values = series.astype(object) if not is_object_dtype(series) else series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(values, errors="raise")
        except (ValueError, TypeError, OverflowError):
            pass
        try:  # pandas >= 2.0 can parse per-element formats
            return pd.to_datetime(values, errors="raise", format="mixed")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(
                f"time column {name!r} could not be parsed as dates or numbers"
            ) from exc


def time_key(parsed: pd.Series) -> np.ndarray:
    """A sortable numeric array for a column returned by parse_time()."""
    if is_datetime64_any_dtype(parsed):
        return parsed.to_numpy(dtype="datetime64[ns]").astype("int64")
    return parsed.to_numpy(dtype="float64")


def target_kind(y: pd.Series, max_categories: int = 20) -> str:
    """'categorical' for non-numeric or low-cardinality targets, otherwise 'numeric'."""
    if (
        isinstance(y.dtype, pd.CategoricalDtype)
        or is_bool_dtype(y)
        or is_object_dtype(y)
        or is_string_dtype(y)
    ):
        return "categorical"
    if is_datetime64_any_dtype(y):
        return "numeric"
    if is_numeric_dtype(y):
        return "categorical" if y.nunique(dropna=True) <= max_categories else "numeric"
    return "categorical"


def strata(y: pd.Series, kind: str, n_bins: int) -> Tuple[np.ndarray, List[str]]:
    """Integer stratum code per row plus the label of each code.

    Categorical targets use their classes; numeric targets use quantile bins.
    Missing target values become a stratum of their own, labelled '<missing>'.
    """
    if kind == "categorical":
        codes, uniques = pd.factorize(y)
        labels = [str(u) for u in uniques]
    else:
        categories = None
        try:
            binned = pd.qcut(y, q=max(1, int(n_bins)), duplicates="drop")
            categories = list(binned.cat.categories)
        except (ValueError, TypeError):
            binned = None
        if binned is None or not categories:
            codes = np.zeros(len(y), dtype=np.int64)
            labels = ["all"]
        else:
            codes = binned.cat.codes.to_numpy()
            labels = [str(iv) for iv in categories]
    codes = np.asarray(codes, dtype=np.int64)
    if (codes < 0).any():
        codes = np.where(codes < 0, len(labels), codes)
        labels = labels + ["<missing>"]
    return codes, labels
