"""Per-column marginal models used by the Gaussian copula.

Each model knows how to turn its column into standard-normal scores for fitting
(``scores``) and how to turn uniform draws back into values (``inverse``), or, for
columns that stay outside the copula, how to draw values directly (``resample``).
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats

# Candidate time resolutions, coarsest first, in nanoseconds.
_TIME_STEPS_NS: Tuple[Tuple[str, int], ...] = (
    ("day", 86_400_000_000_000),
    ("hour", 3_600_000_000_000),
    ("minute", 60_000_000_000),
    ("second", 1_000_000_000),
    ("millisecond", 1_000_000),
    ("microsecond", 1_000),
    ("nanosecond", 1),
)


def temporal_int64(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Datetime/timedelta values as int64 nanoseconds (UTC when tz-aware) plus a validity mask."""
    dtype = series.dtype
    if ptypes.is_datetime64_any_dtype(dtype):
        if getattr(dtype, "tz", None) is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        values = series.to_numpy(dtype="datetime64[ns]")
    else:
        values = series.to_numpy(dtype="timedelta64[ns]")
    return values.astype("int64"), ~np.isnat(values)


def to_float_array(series: pd.Series) -> np.ndarray:
    """Numeric, boolean, datetime or timedelta values as float64 with NaN where missing."""
    dtype = series.dtype
    if ptypes.is_datetime64_any_dtype(dtype) or ptypes.is_timedelta64_dtype(dtype):
        raw, valid = temporal_int64(series)
        out = raw.astype("float64")
        out[~valid] = np.nan
        return out
    return series.to_numpy(dtype="float64", na_value=np.nan)


def normal_scores(x: np.ndarray) -> np.ndarray:
    """Rank-transform the finite entries of ``x`` to standard-normal scores (NaN stays NaN)."""
    z = np.full(x.shape, np.nan)
    mask = np.isfinite(x)
    m = int(mask.sum())
    if m:
        ranks = stats.rankdata(x[mask], method="average")
        z[mask] = stats.norm.ppf(ranks / (m + 1.0))
    return z


def infer_decimals(values: np.ndarray, max_decimals: int = 6) -> Optional[int]:
    """Fewest decimals that reproduce every value, or None when more than ``max_decimals`` are needed."""
    if len(values) == 0:
        return None
    tolerance = 1e-9 * np.maximum(1.0, np.abs(values))
    for decimals in range(max_decimals + 1):
        if np.all(np.abs(np.round(values, decimals) - values) <= tolerance):
            return decimals
    return None


def infer_time_step(relative_ns: np.ndarray) -> Tuple[str, int]:
    """Coarsest resolution (name, nanoseconds) that every value is a whole multiple of."""
    for name, step in _TIME_STEPS_NS:
        if step == 1 or not np.any(relative_ns % step):
            return name, step
    return "nanosecond", 1


class ColumnModel:
    """Base class holding a column's name, original dtype and missing rate."""

    kind: str = "column"
    modeled: bool = False  # True when the column takes part in the copula

    def __init__(self, name: Any, series: pd.Series) -> None:
        self.name = name
        self.dtype = series.dtype
        self.missing_rate = float(series.isna().mean()) if len(series) else 0.0
        self.scores: Optional[np.ndarray] = None

    def describe(self) -> str:
        """One-line human description of what was learned."""
        return self.kind


class NumericColumn(ColumnModel):
    """Numeric column: empirical quantile marginal (or a fitted normal when ``smooth``)."""

    kind = "numeric"
    modeled = True

    def __init__(self, name: Any, series: pd.Series, *, smooth: bool = False) -> None:
        super().__init__(name, series)
        self.smooth = smooth
        x = self._encode(series)
        observed = x[np.isfinite(x)]
        self.sorted_values = np.sort(observed)
        self.vmin = float(self.sorted_values[0])
        self.vmax = float(self.sorted_values[-1])
        self.mean = float(observed.mean())
        self.std = float(observed.std()) or 1.0
        self.n_unique = int(len(np.unique(observed)))
        self.decimals: Optional[int] = self._infer_decimals(observed)
        self.scores = normal_scores(x)

    def _encode(self, series: pd.Series) -> np.ndarray:
        return series.to_numpy(dtype="float64", na_value=np.nan)

    def _infer_decimals(self, observed: np.ndarray) -> Optional[int]:
        if ptypes.is_integer_dtype(self.dtype):
            return 0
        return infer_decimals(observed)

    def inverse(self, u: np.ndarray) -> Any:
        """Map uniform draws in (0, 1) back to column values."""
        if self.smooth:
            values = self.mean + self.std * stats.norm.ppf(u)
        else:
            m = len(self.sorted_values)
            values = np.interp(u * (m - 1), np.arange(m), self.sorted_values)
        return self._decode(np.clip(values, self.vmin, self.vmax))

    def _decode(self, values: np.ndarray) -> Any:
        if self.decimals is not None:
            values = np.round(values, self.decimals)
        return np.clip(values, self.vmin, self.vmax)

    def describe(self) -> str:
        if self.decimals == 0:
            shape = "integer"
        elif self.decimals is None:
            shape = "float"
        else:
            shape = f"{self.decimals} decimals"
        return f"{shape}, {self.n_unique} unique, range {self.vmin:g} to {self.vmax:g}"


class TemporalColumn(NumericColumn):
    """Datetime or timedelta column, modelled as nanoseconds since its earliest value."""

    def __init__(self, name: Any, series: pd.Series, *, smooth: bool = False) -> None:
        self.kind = "datetime" if ptypes.is_datetime64_any_dtype(series.dtype) else "timedelta"
        self.tz = getattr(series.dtype, "tz", None)
        super().__init__(name, series, smooth=smooth)
        decoded = self._decode(np.array([self.vmin, self.vmax]))
        self.first = decoded.iloc[0]
        self.last = decoded.iloc[1]

    def _encode(self, series: pd.Series) -> np.ndarray:
        raw, valid = temporal_int64(series)
        self.offset = int(raw[valid].min()) if valid.any() else 0
        relative = raw - self.offset
        self.step_name, self.step = infer_time_step(relative[valid])
        out = relative.astype("float64")
        out[~valid] = np.nan
        return out

    def _infer_decimals(self, observed: np.ndarray) -> Optional[int]:
        return None

    def _decode(self, values: np.ndarray) -> pd.Series:
        ticks = np.rint(values / self.step).astype("int64") * self.step + self.offset
        if self.kind == "timedelta":
            return pd.Series(pd.to_timedelta(ticks, unit="ns"))
        index = pd.to_datetime(ticks, unit="ns")
        if self.tz is not None:
            index = index.tz_localize("UTC").tz_convert(self.tz)
        return pd.Series(index)

    def describe(self) -> str:
        return (
            f"{self.step_name} resolution, {self.n_unique} unique, "
            f"range {self.first} to {self.last}"
        )


class CategoricalColumn(ColumnModel):
    """Categorical or boolean column: categories ordered by frequency, sampled by thresholds."""

    modeled = True

    def __init__(self, name: Any, series: pd.Series, *, kind: str = "categorical") -> None:
        super().__init__(name, series)
        self.kind = kind
        present = ~series.isna().to_numpy()
        non_null = series[present]
        counts = non_null.value_counts(sort=False)
        counts = counts[counts > 0]
        order = np.lexsort((np.arange(len(counts)), -counts.to_numpy(dtype="int64")))
        self.categories: List[Any] = [counts.index[i] for i in order]
        probabilities = counts.to_numpy(dtype="float64")[order]
        self.probabilities = probabilities / probabilities.sum()
        self.cumulative = np.cumsum(self.probabilities)
        self.cumulative[-1] = 1.0
        self._values = np.empty(len(self.categories), dtype=object)
        for i, category in enumerate(self.categories):
            self._values[i] = category
        codes = pd.Categorical(non_null, categories=self.categories).codes
        x = np.full(len(series), np.nan)
        x[present] = codes
        self.scores = normal_scores(x)

    def inverse(self, u: np.ndarray) -> np.ndarray:
        """Map uniform draws in (0, 1) to categories via cumulative frequency thresholds."""
        idx = np.searchsorted(self.cumulative, u, side="right")
        idx = np.minimum(idx, len(self.categories) - 1)
        values = self._values[idx]
        if self.kind == "bool":
            return values.astype(bool)
        return values

    def describe(self) -> str:
        top = self.categories[0]
        if isinstance(top, np.generic):  # plain Python repr, not np.True_ / np.int64(3)
            top = top.item()
        return (
            f"{len(self.categories)} categories, most common {top!r} "
            f"({self.probabilities[0]:.0%})"
        )


class TextColumn(ColumnModel):
    """High-cardinality text column: values are resampled with replacement from the originals."""

    kind = "text"

    def __init__(self, name: Any, series: pd.Series, *, unique_ratio: float) -> None:
        super().__init__(name, series)
        self.values = series.dropna().to_numpy(dtype=object)
        self.unique_ratio = float(unique_ratio)

    def resample(self, n: int, rng: np.random.Generator) -> Any:
        if len(self.values) == 0:
            return pd.Series([None] * n, dtype=object)
        return self.values[rng.integers(0, len(self.values), size=n)]

    def describe(self) -> str:
        ratio = "unhashable values" if np.isnan(self.unique_ratio) else f"{self.unique_ratio:.0%} unique"
        return f"{ratio}, resampled from the {len(self.values)} original values"


class ConstantColumn(ColumnModel):
    """Column with at most one distinct value (possibly all missing)."""

    kind = "constant"

    def __init__(self, name: Any, series: pd.Series, *, value: Any = None) -> None:
        super().__init__(name, series)
        self.value = value

    def resample(self, n: int, rng: np.random.Generator) -> pd.Series:
        return pd.Series([self.value] * n, dtype=object)

    def describe(self) -> str:
        return "all missing" if self.value is None else f"constant {self.value!r}"


def build_column_model(
    name: Any, series: pd.Series, *, smooth: bool = False, text_threshold: float = 0.5
) -> ColumnModel:
    """Pick and fit the right model for one column."""
    dtype = series.dtype
    non_null = series.dropna()
    if ptypes.is_bool_dtype(dtype):
        kind = "bool"
    elif ptypes.is_datetime64_any_dtype(dtype) or ptypes.is_timedelta64_dtype(dtype):
        kind = "temporal"
    elif ptypes.is_numeric_dtype(dtype):
        kind = "numeric"
    elif isinstance(dtype, pd.CategoricalDtype):
        kind = "categorical"
    else:
        kind = "categorical"
        try:
            n_unique = int(non_null.nunique())
        except TypeError:  # unhashable values such as lists: only resampling makes sense
            return TextColumn(name, series, unique_ratio=float("nan"))
        if n_unique > 1 and ptypes.infer_dtype(non_null, skipna=True) == "string":
            ratio = n_unique / len(non_null)
            if ratio > text_threshold:
                return TextColumn(name, series, unique_ratio=ratio)

    if kind == "numeric":
        values = to_float_array(series)
        finite = values[np.isfinite(values)]
        if len(np.unique(finite)) <= 1:
            if len(finite):
                value: Any = finite[0]
            else:
                value = non_null.iloc[0] if len(non_null) else None
            return ConstantColumn(name, series, value=value)
        return NumericColumn(name, series, smooth=smooth)

    if non_null.nunique() <= 1:
        value = non_null.iloc[0] if len(non_null) else None
        return ConstantColumn(name, series, value=value)
    if kind == "temporal":
        return TemporalColumn(name, series, smooth=smooth)
    return CategoricalColumn(name, series, kind=kind)
