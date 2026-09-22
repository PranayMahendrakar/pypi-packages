"""Fidelity of a synthetic table against the real one: ``evaluate(real, synthetic)``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ._columns import to_float_array
from ._io import TableLike, load_table


@dataclass
class ColumnFidelity:
    """Distance between the real and synthetic distribution of one column.

    ``value`` is the KS statistic (numeric, datetime) or the total variation distance
    (categorical, boolean, text): 0 means identical, 1 means disjoint. ``score`` is
    ``100 * (1 - value)``.
    """

    name: Any
    kind: str
    metric: str
    value: float
    score: float
    missing_rate_real: float
    missing_rate_synthetic: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return {
            "name": str(self.name),
            "kind": self.kind,
            "metric": self.metric,
            "value": float(self.value),
            "score": float(self.score),
            "missing_rate_real": float(self.missing_rate_real),
            "missing_rate_synthetic": float(self.missing_rate_synthetic),
        }


@dataclass
class FidelityReport:
    """How faithful a synthetic table is to the real one.

    ``score`` (0-100) averages the marginal score (mean over columns of
    ``100 * (1 - distance)``) and the correlation score (``100 * (1 - MAD)`` where
    MAD is the mean absolute difference between the two Spearman correlation
    matrices). With fewer than two non-constant columns the score is the marginal
    score alone and ``correlation_score`` is None.
    """

    score: float
    marginal_score: float
    correlation_score: Optional[float]
    correlation_mad: Optional[float]
    columns: Dict[Any, ColumnFidelity]
    n_real: int
    n_synthetic: int
    missing_columns: List[Any] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable report."""
        lines = [
            f"Fidelity score: {self.score:.1f} / 100  "
            f"({self.n_real} real rows, {self.n_synthetic} synthetic rows, "
            f"{len(self.columns)} columns compared)",
            f"  marginals:    {self.marginal_score:.1f} / 100  "
            "(mean over columns; 100 = identical distributions)",
        ]
        if self.correlation_score is None:
            lines.append("  correlations: n/a  (needs at least two non-constant columns)")
        else:
            lines.append(
                f"  correlations: {self.correlation_score:.1f} / 100  "
                f"(mean abs. difference of Spearman correlations: {self.correlation_mad:.3f})"
            )
        lines.append("")
        width = max([len(str(c.name)) for c in self.columns.values()] + [6])
        lines.append(f"{'column':<{width}}  {'kind':<11}  {'metric':<6}  {'value':>6}  {'score':>5}")
        for c in self.columns.values():
            lines.append(
                f"{str(c.name):<{width}}  {c.kind:<11}  {c.metric.upper():<6}  "
                f"{c.value:>6.3f}  {c.score:>5.1f}"
            )
        if self.missing_columns:
            lines.append("")
            lines.append(
                "Columns missing from the synthetic table: "
                + ", ".join(str(c) for c in self.missing_columns)
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return {
            "score": float(self.score),
            "marginal_score": float(self.marginal_score),
            "correlation_score": None
            if self.correlation_score is None
            else float(self.correlation_score),
            "correlation_mad": None if self.correlation_mad is None else float(self.correlation_mad),
            "n_real": int(self.n_real),
            "n_synthetic": int(self.n_synthetic),
            "columns": {str(name): c.to_dict() for name, c in self.columns.items()},
            "missing_columns": [str(c) for c in self.missing_columns],
        }


# ----------------------------------------------------------------- metrics
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max distance between empirical CDFs)."""
    a = np.sort(a)
    b = np.sort(b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _frequencies(series: pd.Series) -> Dict[Any, float]:
    values = series.dropna()
    if len(values) == 0:
        return {}
    try:
        counts = values.astype(object).value_counts(normalize=True)
        return {key: float(freq) for key, freq in counts.items()}
    except TypeError:  # unhashable values such as lists: compare their text form
        counts = values.astype(str).value_counts(normalize=True)
        return {key: float(freq) for key, freq in counts.items()}


def total_variation_distance(real: pd.Series, synthetic: pd.Series) -> float:
    """Half the L1 distance between the two category frequency tables (0 = same, 1 = disjoint)."""
    p = _frequencies(real)
    q = _frequencies(synthetic)
    if not p and not q:
        return 0.0
    if not p or not q:
        return 1.0
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in keys)


def _metric_kind(dtype: Any) -> str:
    if ptypes.is_bool_dtype(dtype):
        return "categorical"
    if (
        ptypes.is_datetime64_any_dtype(dtype)
        or ptypes.is_timedelta64_dtype(dtype)
        or ptypes.is_numeric_dtype(dtype)
    ):
        return "numeric"
    return "categorical"


def _numeric_values(series: pd.Series, like_dtype: Any) -> np.ndarray:
    """float64 values of ``series`` read the way a column of ``like_dtype`` would be."""
    dtype = series.dtype
    if ptypes.is_datetime64_any_dtype(like_dtype):
        if not ptypes.is_datetime64_any_dtype(dtype):
            utc = getattr(like_dtype, "tz", None) is not None
            series = pd.to_datetime(series, errors="coerce", utc=utc)
    elif ptypes.is_timedelta64_dtype(like_dtype):
        if not ptypes.is_timedelta64_dtype(dtype):
            series = pd.to_timedelta(series, errors="coerce")
    elif not (ptypes.is_numeric_dtype(dtype) or ptypes.is_bool_dtype(dtype)):
        series = pd.to_numeric(series, errors="coerce")
    return to_float_array(series)


def _column_fidelity(name: Any, real: pd.Series, synthetic: pd.Series) -> ColumnFidelity:
    kind = _metric_kind(real.dtype)
    if kind == "numeric":
        a = _numeric_values(real, real.dtype)
        b = _numeric_values(synthetic, real.dtype)
        value = ks_statistic(a[np.isfinite(a)], b[np.isfinite(b)])
        metric = "ks"
    else:
        value = total_variation_distance(real, synthetic)
        metric = "tvd"
    value = float(min(max(value, 0.0), 1.0))
    return ColumnFidelity(
        name=name,
        kind=kind,
        metric=metric,
        value=value,
        score=100.0 * (1.0 - value),
        missing_rate_real=float(real.isna().mean()) if len(real) else 0.0,
        missing_rate_synthetic=float(synthetic.isna().mean()) if len(synthetic) else 0.0,
    )


def _ordinal_codes(real: pd.Series, synthetic: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    counts = real.dropna().astype(object).value_counts()
    code = {category: float(i) for i, category in enumerate(counts.index)}

    def encode(series: pd.Series) -> np.ndarray:
        mapped = series.astype(object).map(code)
        return pd.to_numeric(mapped, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)

    return encode(real), encode(synthetic)


def _n_finite_unique(values: np.ndarray) -> int:
    return int(len(np.unique(values[np.isfinite(values)])))


def _correlation_mad(real: pd.DataFrame, synthetic: pd.DataFrame) -> Optional[float]:
    encoded_real: Dict[Any, np.ndarray] = {}
    encoded_synthetic: Dict[Any, np.ndarray] = {}
    for name in real.columns:
        r = real[name]
        s = synthetic[name]
        if _metric_kind(r.dtype) == "numeric":
            a = _numeric_values(r, r.dtype)
            b = _numeric_values(s, r.dtype)
        else:
            try:
                a, b = _ordinal_codes(r, s)
            except TypeError:  # unhashable values carry no usable ordering
                continue
        if _n_finite_unique(a) < 2:  # constant in the real data: no correlation to preserve
            continue
        encoded_real[name] = a
        encoded_synthetic[name] = b
    if len(encoded_real) < 2:
        return None
    corr_real = pd.DataFrame(encoded_real).corr(method="spearman").to_numpy(dtype="float64")
    corr_synthetic = (
        pd.DataFrame(encoded_synthetic).corr(method="spearman").to_numpy(dtype="float64")
    )
    upper = np.triu_indices(len(encoded_real), k=1)
    diff = np.abs(np.nan_to_num(corr_real[upper]) - np.nan_to_num(corr_synthetic[upper]))
    return float(diff.mean())


def evaluate(real: TableLike, synthetic: TableLike) -> FidelityReport:
    """Compare a synthetic table with the real one column by column and by correlation.

    Both arguments accept a DataFrame or a path to a .csv/.parquet file. Columns are
    matched by name; columns only present in ``real`` are listed in
    ``FidelityReport.missing_columns``.
    """
    real_df = load_table(real)
    synthetic_df = load_table(synthetic)
    for label, frame in (("real", real_df), ("synthetic", synthetic_df)):
        if not frame.columns.is_unique:
            raise ValueError(f"The {label} table has duplicate column names")
    common = [name for name in real_df.columns if name in synthetic_df.columns]
    if not common:
        raise ValueError("The real and synthetic tables share no columns")

    columns = {
        name: _column_fidelity(name, real_df[name], synthetic_df[name]) for name in common
    }
    marginal = float(np.mean([c.score for c in columns.values()]))
    mad = _correlation_mad(real_df[common], synthetic_df[common])
    if mad is None:
        correlation_score: Optional[float] = None
        score = marginal
    else:
        correlation_score = 100.0 * max(0.0, 1.0 - mad)
        score = (marginal + correlation_score) / 2.0
    return FidelityReport(
        score=score,
        marginal_score=marginal,
        correlation_score=correlation_score,
        correlation_mad=mad,
        columns=columns,
        n_real=int(len(real_df)),
        n_synthetic=int(len(synthetic_df)),
        missing_columns=[name for name in real_df.columns if name not in synthetic_df.columns],
    )
