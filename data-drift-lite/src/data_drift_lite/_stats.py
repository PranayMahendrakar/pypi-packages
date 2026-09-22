"""Statistical building blocks: column typing, value extraction, binning, PSI, KS, chi-square.

Everything here works on plain numpy arrays and returns plain Python numbers (or
``None`` when a quantity cannot be computed), so the report stays JSON-friendly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats as sps

PSI_EPS = 1e-4  # floor for bin shares so log(0) never happens
LOW_CARDINALITY = 10  # numeric columns with this many distinct values or fewer: one bin per value
QUANTILE_BINS = 10  # otherwise: quantile bins built from the reference
TOP_CATEGORIES = 5  # categories listed in the per-column stats

_NUMERIC_FAMILIES = frozenset({"number", "datetime", "timedelta"})


def family_of(dtype: Any) -> str:
    """Map a pandas dtype to one of ``number``, ``datetime``, ``timedelta``, ``category``."""
    if pdt.is_bool_dtype(dtype):
        return "category"
    if pdt.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pdt.is_timedelta64_dtype(dtype):
        return "timedelta"
    if pdt.is_numeric_dtype(dtype):
        return "number"
    return "category"


def kind_of(family: str) -> str:
    """Public kind label for a family: ``numeric`` or ``categorical``."""
    return "numeric" if family in _NUMERIC_FAMILIES else "categorical"


def share(part: int, whole: int) -> float:
    """``part / whole`` as a float, 0.0 when ``whole`` is 0."""
    return float(part) / float(whole) if whole else 0.0


@dataclass
class ColumnValues:
    """The usable values of one column plus how much was missing."""

    family: str
    n_rows: int
    n_missing: int
    values: np.ndarray  # number: finite float64; datetime/timedelta: int64 ns; category: object array of str
    n_nonfinite: int = 0  # +/-inf in a number column; counted inside n_missing

    @property
    def missing_share(self) -> float:
        return share(self.n_missing, self.n_rows)


def extract(series: pd.Series) -> ColumnValues:
    """Pull the usable values out of a Series according to its dtype family."""
    family = family_of(series.dtype)
    n_rows = int(len(series))
    if family == "number":
        arr = np.asarray(series.to_numpy(dtype="float64", na_value=np.nan), dtype="float64")
        finite = np.isfinite(arr)
        n_present = int(finite.sum())
        n_nan = int(np.isnan(arr).sum())
        return ColumnValues(
            family, n_rows, n_rows - n_present, arr[finite], n_nonfinite=n_rows - n_present - n_nan
        )
    if family in ("datetime", "timedelta"):
        s = series
        if family == "datetime" and getattr(s.dtype, "tz", None) is not None:
            s = s.dt.tz_convert(None)
        unit = "datetime64[ns]" if family == "datetime" else "timedelta64[ns]"
        arr = s.to_numpy(dtype=unit)
        present = ~np.isnat(arr)
        return ColumnValues(family, n_rows, n_rows - int(present.sum()), arr[present].astype(np.int64))
    missing = series.isna().to_numpy(dtype=bool)
    present_values = series[~missing].astype(str).to_numpy(dtype=object)
    return ColumnValues("category", n_rows, int(missing.sum()), present_values)


# --------------------------------------------------------------------------- numeric bins


def numeric_edges(values: np.ndarray) -> np.ndarray:
    """Right-closed interior boundaries built from the reference values.

    Bin ``i`` covers ``(edges[i-1], edges[i]]``; the outer bins extend to -inf and
    +inf so every current value lands somewhere. Columns with at most
    ``LOW_CARDINALITY`` distinct values get one bin per value (boundaries at the
    midpoints), everything else gets ``QUANTILE_BINS`` quantile bins. A constant
    column yields no boundaries, i.e. a single bin, so its PSI is 0 rather than NaN.
    """
    if values.size == 0:
        return np.empty(0, dtype="float64")
    uniq = np.unique(values)
    if uniq.size <= LOW_CARDINALITY:
        lo = uniq[:-1].astype("float64")
        hi = uniq[1:].astype("float64")
        return (lo + hi) / 2.0
    probs = np.linspace(0.0, 1.0, QUANTILE_BINS + 1)[1:-1]
    return np.unique(np.quantile(values, probs))


def bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count values per bin for the boundaries from :func:`numeric_edges`."""
    idx = np.searchsorted(edges, values, side="left")
    return np.bincount(idx, minlength=edges.size + 1)


# --------------------------------------------------------------------------- categories


def value_counts(values: np.ndarray) -> Dict[str, int]:
    """Counts per category, ordered by count descending then name (deterministic)."""
    if values.size == 0:
        return {}
    uniq, counts = np.unique(values, return_counts=True)
    pairs = sorted(zip(uniq.tolist(), counts.tolist()), key=lambda pair: (-pair[1], pair[0]))
    return {str(cat): int(n) for cat, n in pairs}


# --------------------------------------------------------------------------- tests


def psi(reference_counts: Any, current_counts: Any) -> Optional[float]:
    """Population Stability Index between two aligned count vectors.

    Shares are floored at ``PSI_EPS`` so an empty bin on one side contributes a
    large but finite amount. Returns ``None`` when either side has no counts.
    """
    ref = np.asarray(reference_counts, dtype="float64")
    cur = np.asarray(current_counts, dtype="float64")
    if ref.size == 0 or ref.sum() <= 0 or cur.sum() <= 0:
        return None
    r = np.clip(ref / ref.sum(), PSI_EPS, None)
    c = np.clip(cur / cur.sum(), PSI_EPS, None)
    value = float(np.sum((c - r) * np.log(c / r)))
    if not math.isfinite(value):
        return None
    return max(value, 0.0)


def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Two-sample Kolmogorov-Smirnov test; ``(None, None)`` when a side is empty."""
    if reference.size == 0 or current.size == 0:
        return None, None
    result = sps.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def chi2_test(table: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Chi-square test of independence on a 2 x K count table.

    Returns ``(None, None)`` when a row is empty, and ``(0.0, 1.0)`` for a single
    category (zero degrees of freedom).
    """
    table = np.asarray(table, dtype="float64")
    if table.ndim != 2 or table.shape[1] == 0 or np.any(table.sum(axis=1) <= 0):
        return None, None
    if table.shape[1] == 1:
        return 0.0, 1.0
    result = sps.chi2_contingency(table)
    return float(result[0]), float(result[1])


# --------------------------------------------------------------------------- stats


def _format_datetime(ns: Any) -> str:
    return pd.Timestamp(int(ns)).isoformat()


def _format_timedelta(ns: Any) -> str:
    return str(pd.Timedelta(int(ns)))


def numeric_stats(cv: ColumnValues, dtype: str) -> Dict[str, Any]:
    """Descriptive stats for a numeric column (datetimes and timedeltas as strings)."""
    v = cv.values
    out: Dict[str, Any] = {
        "dtype": dtype,
        "count": int(v.size),
        "missing_share": round(cv.missing_share, 6),
    }
    if cv.family == "number":
        keys = ("mean", "std", "min", "median", "max")
        if v.size:
            vals: Tuple[Any, ...] = (
                float(v.mean()),
                float(v.std()),
                float(v.min()),
                float(np.median(v)),
                float(v.max()),
            )
        else:
            vals = (None,) * len(keys)
    else:
        fmt = _format_datetime if cv.family == "datetime" else _format_timedelta
        keys = ("min", "median", "max")
        vals = (fmt(v.min()), fmt(np.median(v)), fmt(v.max())) if v.size else (None,) * len(keys)
    out.update(zip(keys, vals))
    return out


def category_stats(cv: ColumnValues, counts: Dict[str, int], dtype: str) -> Dict[str, Any]:
    """Descriptive stats for a categorical column, including its top categories."""
    total = sum(counts.values())
    top = {cat: round(n / total, 6) for cat, n in list(counts.items())[:TOP_CATEGORIES]} if total else {}
    return {
        "dtype": dtype,
        "count": int(cv.values.size),
        "missing_share": round(cv.missing_share, 6),
        "n_categories": len(counts),
        "top": top,
    }
