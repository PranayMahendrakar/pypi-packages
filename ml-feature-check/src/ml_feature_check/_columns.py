"""Per-column profiling and dtype helpers shared by every check."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Hashable, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

NUMERIC = "numeric"
BOOL = "bool"
DATETIME = "datetime"
CATEGORICAL = "categorical"

#: kinds that have a meaningful float view (see :func:`to_float`)
FLOAT_KINDS = (NUMERIC, BOOL, DATETIME)


def column_kind(s: pd.Series) -> str:
    """Classify a series as numeric, bool, datetime or categorical."""
    dt = s.dtype
    if ptypes.is_bool_dtype(dt):
        return BOOL
    if ptypes.is_datetime64_any_dtype(dt) or ptypes.is_timedelta64_dtype(dt):
        return DATETIME
    if ptypes.is_numeric_dtype(dt) and not ptypes.is_complex_dtype(dt):
        return NUMERIC
    return CATEGORICAL


def to_float(s: pd.Series) -> np.ndarray:
    """Float64 view of a numeric/bool/datetime series; missing and infinite values become NaN."""
    if column_kind(s) == DATETIME:
        if ptypes.is_datetime64_any_dtype(s.dtype):
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            arr = s.to_numpy(dtype="datetime64[ns]")
        else:
            arr = s.to_numpy(dtype="timedelta64[ns]")
        out = arr.view("int64").astype("float64")
        out[np.isnat(arr)] = np.nan
        return out
    out = np.array(s.to_numpy(dtype="float64", na_value=np.nan), dtype="float64", copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def codes_of(s: pd.Series) -> Tuple[np.ndarray, int]:
    """Dense integer codes (``-1`` for missing) and the number of distinct non-missing values."""
    try:
        codes, uniques = pd.factorize(s, use_na_sentinel=True)
    except TypeError:  # unhashable cells (lists, dicts): fall back to their text
        codes, uniques = pd.factorize(s.astype(str).where(s.notna()), use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int64), int(len(uniques))


def canonical_object(s: pd.Series) -> pd.Series:
    """Object view of a categorical-ish series with every kind of missing value as ``np.nan``."""
    obj = s.astype(object)
    return obj.where(s.notna(), np.nan)


def jsonable(value: Any) -> Any:
    """Turn numpy / pandas scalars into plain Python so ``to_dict()`` is JSON-safe."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # np.float64 is a subclass of float, so it lands here rather than in the
        # np.floating branch below. numpy 2 renders it as "np.float64(2.5)", which
        # would leak that repr into every message quoting a float cell, so the
        # conversion to a plain float has to happen here too.
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return jsonable(float(value))
    if isinstance(value, (pd.Timestamp, pd.Timedelta, np.datetime64, np.timedelta64)):
        return str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if value is pd.NaT or value is pd.NA:
        return None
    return str(value)


@dataclass
class ColumnProfile:
    """Cheap facts about one column, computed once and shared by every check."""

    name: Hashable
    dtype: str
    kind: str
    n: int
    n_nonnull: int
    missing_frac: float
    nunique: int
    unique_ratio: float
    top_value: Any
    top_frac: float
    integer_like: bool
    span: Optional[float]  # max - min + 1 for integer-like columns, else None


def profile_column(name: Hashable, s: pd.Series) -> ColumnProfile:
    """Profile one column: missingness, cardinality, dominant value, integer-likeness."""
    n = int(len(s))
    kind = column_kind(s)
    nonnull = s.dropna()
    n_nonnull = int(len(nonnull))
    try:
        vc = nonnull.value_counts(dropna=True)
    except TypeError:  # unhashable cells
        nonnull = nonnull.astype(str)
        vc = nonnull.value_counts(dropna=True)
    vc = vc[vc > 0]  # categoricals also report unused categories
    nunique = int(len(vc))
    top_value = jsonable(vc.index[0]) if nunique else None
    top_frac = float(vc.iloc[0] / n_nonnull) if n_nonnull else 0.0

    integer_like = False
    span: Optional[float] = None
    if kind == NUMERIC and n_nonnull:
        arr = to_float(s)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if ptypes.is_integer_dtype(s.dtype):
                integer_like = True
            else:
                integer_like = bool(np.all(np.mod(finite, 1.0) == 0.0))
            if integer_like:
                span = float(finite.max() - finite.min()) + 1.0

    return ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        kind=kind,
        n=n,
        n_nonnull=n_nonnull,
        missing_frac=(1.0 - n_nonnull / n) if n else 0.0,
        nunique=nunique,
        unique_ratio=(nunique / n_nonnull) if n_nonnull else 0.0,
        top_value=top_value,
        top_frac=top_frac,
        integer_like=integer_like,
        span=span,
    )


# --------------------------------------------------------------------------- names

_SUSPICIOUS_TOKENS = {
    "id": "id",
    "ids": "id",
    "uuid": "uuid",
    "guid": "uuid",
    "index": "index",
    "idx": "index",
    "key": "key",
    "timestamp": "timestamp",
    "row": "row",
    "rownum": "row",
    "rowid": "row",
    "rownumber": "row",
    "unnamed": "unnamed",
}
_GLUED_ID = re.compile(
    r"^(?:user|customer|cust|client|account|acct|session|order|item|product|record|"
    r"txn|transaction|event|device|member|employee|emp|patient|student|obj|object|"
    r"entity|doc|document|msg|message|ticket|invoice|shipment|vehicle|visitor|"
    r"subscriber|merchant|store|shop|trip|ride|booking|reservation|loan|policy|claim|"
    r"case|lead|contact|company|org|vendor|supplier|payment|card|house|listing|property|"
    r"tenant|driver|passenger|flight|node|job|task|run|batch|file|image|video|article|"
    r"post|thread|comment|review|campaign|click|request|trace|span|sample|person)id$"
)


def suspicious_name_token(name: Hashable) -> Optional[str]:
    """Return the identifier-ish token found in a column name, or ``None``."""
    text = str(name)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)  # camelCase -> camel_Case
    for token in re.split(r"[^a-zA-Z0-9]+", spaced.lower()):
        if token in _SUSPICIOUS_TOKENS:
            return _SUSPICIOUS_TOKENS[token]
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    if _GLUED_ID.match(compact):
        return "id"
    return None


# --------------------------------------------------------------------------- dates

_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE_RE = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"  # 2021-03-04, 04/03/2021, 4.3.21
    r"|\d{1,2}:\d{2}"  # 13:45
    r"|" + _MONTHS + r",?\s+\d{1,4}"  # March 4, 2021 / Mar 2021
    r"|\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTHS,  # 4 March 2021
    re.IGNORECASE,
)


def looks_like_dates(nonnull: pd.Series, *, sample: int = 200, min_frac: float = 0.9) -> Optional[str]:
    """If a text column is mostly parseable dates, return an example value, else ``None``."""
    if len(nonnull) == 0:
        return None
    probe = nonnull if len(nonnull) <= sample else nonnull.sample(n=sample, random_state=0)
    values = [v.strip() for v in probe.tolist() if isinstance(v, str)]
    if len(values) < max(1, int(np.ceil(min_frac * len(probe)))):
        return None
    values = [v for v in values if v]
    if not values:
        return None
    hits = [v for v in values if _DATE_RE.search(v)]
    if len(hits) < min_frac * len(values):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(pd.Series(values, dtype=object), errors="coerce")
        frac = float(parsed.notna().mean())
        if frac < min_frac:
            try:
                parsed = pd.to_datetime(pd.Series(values, dtype=object), errors="coerce", format="mixed")
                frac = float(parsed.notna().mean())
            except (TypeError, ValueError):  # pandas < 2 has no "mixed"
                pass
    if frac < min_frac:
        return None
    return hits[0]
