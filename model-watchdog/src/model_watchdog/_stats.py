"""Small statistical helpers: PSI, quantiles, run lengths, scoring.

Everything here is pure numpy/pandas and returns plain Python numbers or
``None`` when a value cannot be computed. Nothing raises on odd input.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Additive smoothing so an empty bin never produces ``inf``.
_SMOOTH = 0.5

#: A numeric column with this many distinct values or fewer is binned per value
#: instead of by deciles, so a 95/5 to 5/95 flip in a 0/1 output is caught.
MAX_DISCRETE = 10

#: Up to this many distinct integer-like values still counts as class labels.
MAX_CLASSES = 20

#: Current records a PSI bin needs before its share means anything. A decile
#: PSI over ten bins holding one point each is noise, not drift: on two
#: samples of the *same* distribution it reads about 0.43, twice the usual
#: 0.2 alarm line. The bin count is scaled down to honour this.
PSI_MIN_PER_BIN = 20

_MISSING = "__missing__"


def as_series(values: Any) -> pd.Series:
    """Any sequence as an object Series, with missing values kept as NaN.

    A DataFrame is rejected rather than coerced: ``pd.Series(list(frame))``
    would quietly hand back the *column names*, which reads as a perfectly
    valid two-value distribution and fabricates drift out of nothing.
    """
    if isinstance(values, pd.DataFrame):
        raise TypeError(
            "expected a single column of values, got a DataFrame with columns %s "
            "(duplicate column names are the usual cause)" % list(values.columns)
        )
    if isinstance(values, pd.Series):
        return values.astype("object")
    if values is None:
        return pd.Series([], dtype="object")
    if isinstance(values, (str, bytes)):
        return pd.Series([values], dtype="object")
    return pd.Series(list(values), dtype="object")


def numeric_array(values: Any) -> np.ndarray:
    """Finite float values only; anything non-numeric, NaN or inf is dropped."""
    series = as_series(values)
    if series.empty:
        return np.empty(0, dtype="float64")
    numbers = pd.to_numeric(series, errors="coerce")
    array = np.asarray(numbers, dtype="float64")
    return array[np.isfinite(array)]


def looks_numeric(values: Any, min_share: float = 0.8) -> bool:
    """True when most non-missing values parse as finite numbers."""
    series = as_series(values)
    present = series[series.notna()]
    if present.empty:
        return False
    return numeric_array(present).size >= min_share * len(present)


def is_discrete(values: Any, max_values: int = MAX_CLASSES) -> bool:
    """True when the values look like a small set of integer-like labels."""
    array = numeric_array(values)
    if array.size == 0:
        return False
    unique = np.unique(array)
    return unique.size <= max_values and bool(np.all(unique == np.round(unique)))


def _key(value: Any) -> Any:
    """Hashable, type-stable bucket key so 1 and 1.0 land in the same bin."""
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return _MISSING
        if pd.isna(value):
            return _MISSING
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)


def _keys(values: Any) -> pd.Series:
    series = as_series(values)
    if series.empty:
        return series
    return series.map(_key)


def _psi_from_counts(reference_counts: Any, current_counts: Any) -> Optional[float]:
    reference_counts = np.asarray(reference_counts, dtype="float64")
    current_counts = np.asarray(current_counts, dtype="float64")
    bins = reference_counts.size
    if bins == 0 or reference_counts.sum() <= 0 or current_counts.sum() <= 0:
        return None
    reference_share = (reference_counts + _SMOOTH) / (reference_counts.sum() + _SMOOTH * bins)
    current_share = (current_counts + _SMOOTH) / (current_counts.sum() + _SMOOTH * bins)
    value = float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))
    if not np.isfinite(value):
        return None
    return max(value, 0.0)


def _categorical_psi(reference: Any, current: Any) -> Optional[float]:
    reference_keys = _keys(reference)
    current_keys = _keys(current)
    if reference_keys.empty or current_keys.empty:
        return None
    reference_counts = reference_keys.value_counts()
    current_counts = current_keys.value_counts()
    categories = list(reference_counts.index)
    seen = float(sum(float(current_counts.get(category, 0.0)) for category in categories))
    reference_bins = [float(reference_counts.get(category, 0.0)) for category in categories] + [0.0]
    current_bins = [float(current_counts.get(category, 0.0)) for category in categories]
    current_bins.append(max(float(current_counts.sum()) - seen, 0.0))
    return _psi_from_counts(np.array(reference_bins), np.array(current_bins))


#: How a PSI comparison is made: by reference deciles, or one bin per value.
PSI_BY_QUANTILE = "deciles"
PSI_BY_VALUE = "values"


def psi_plan(reference: Any, current: Any, bins: int = 10) -> Optional[Tuple[str, int]]:
    """How a PSI comparison would be made, as ``(kind, bins)``.

    ``kind`` is :data:`PSI_BY_QUANTILE` for a continuous numeric column binned
    by reference quantiles, or :data:`PSI_BY_VALUE` when the values look like
    labels and get one bin each. For the quantile path the bin count is capped
    so that each bin is backed by at least :data:`PSI_MIN_PER_BIN` current
    records; ten bins over thirty points would measure sampling noise, not
    drift. ``None`` when either side is empty.
    """
    reference_series = as_series(reference)
    current_series = as_series(current)
    reference_series = reference_series[reference_series.notna()]
    current_series = current_series[current_series.notna()]
    if reference_series.empty or current_series.empty:
        return None
    if looks_numeric(reference_series) and looks_numeric(current_series):
        reference_values = numeric_array(reference_series)
        current_values = numeric_array(current_series)
        if reference_values.size == 0 or current_values.size == 0:
            return None
        if np.unique(reference_values).size > MAX_DISCRETE:
            affordable = min(
                int(current_values.size // PSI_MIN_PER_BIN),
                int(reference_values.size // PSI_MIN_PER_BIN),
            )
            used = max(2, min(max(int(bins), 2), affordable))
            return PSI_BY_QUANTILE, used
    return PSI_BY_VALUE, 0


def psi(reference: Any, current: Any, bins: int = 10) -> Optional[float]:
    """Population Stability Index of ``current`` against ``reference``.

    Numeric values are binned by reference quantiles - at most ``bins`` of
    them, and fewer when there are not enough records to fill that many; values
    that look like labels (or a numeric column with few distinct values) are
    binned per value, with everything the reference never saw pooled into one
    extra bin. Returns ``None`` when either side is empty.
    """
    plan = psi_plan(reference, current, bins)
    if plan is None:
        return None
    kind, used = plan
    reference_series = as_series(reference)
    current_series = as_series(current)
    reference_series = reference_series[reference_series.notna()]
    current_series = current_series[current_series.notna()]
    if kind == PSI_BY_QUANTILE:
        reference_values = numeric_array(reference_series)
        current_values = numeric_array(current_series)
        quantiles = np.quantile(reference_values, np.linspace(0.0, 1.0, used + 1))
        edges = np.unique(quantiles)
        if edges.size >= 3:
            edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
            reference_counts, _ = np.histogram(reference_values, bins=edges)
            current_counts, _ = np.histogram(current_values, bins=edges)
            return _psi_from_counts(reference_counts, current_counts)
    return _categorical_psi(reference_series, current_series)


def psi_label(value: Optional[float]) -> str:
    """The usual reading of a PSI number, as plain words."""
    if value is None:
        return "not computable"
    if value < 0.1:
        return "no real shift"
    if value < 0.25:
        return "moderate shift"
    return "major shift"


def to_datetime_utc(values: Any) -> pd.Series:
    """Parse a column of timestamps to timezone-aware UTC.

    Mixed ISO shapes are the normal case here: a live record carries
    microseconds (``...T10:00:00.123456+00:00``) while a backfilled one, or one
    logged with an explicit ``ts=``, lands on a whole second
    (``...T10:00:00+00:00``). pandas 2.x infers one strict format from the
    first value and coerces everything that does not match it to ``NaT``, which
    would silently drop half the log. ``format="ISO8601"`` parses each value on
    its own; pandas 1.5 has no such option and is lenient already, so the
    fallback keeps that version working.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    except (ValueError, TypeError):  # pragma: no cover - pandas < 2.0
        return pd.to_datetime(series, utc=True, errors="coerce")


def quantile(values: Any, q: float) -> Optional[float]:
    """A single quantile of the finite numeric values, or ``None``."""
    array = numeric_array(values)
    if array.size == 0:
        return None
    return float(np.quantile(array, q))


def longest_run(values: Any) -> Tuple[int, Any]:
    """Longest run of consecutive identical values, and the value repeated."""
    keys = list(_keys(values))
    if not keys:
        return 0, None
    best_length, best_value = 1, keys[0]
    length = 1
    for previous, current in zip(keys, keys[1:]):
        length = length + 1 if current == previous else 1
        if length > best_length:
            best_length, best_value = length, current
    return best_length, best_value


#: Euler-Mascheroni and the Gumbel spread, used by :func:`expected_longest_run`.
_EULER = 0.5772156649015329
_GUMBEL_SD = 1.2825498301618641

#: How many standard deviations above the expected longest run still counts as
#: chance. Four keeps the false-alarm rate near zero on simulated traffic while
#: a genuinely stuck model - whose run is the whole window - trips easily.
RUN_SIGMA = 4.0


def value_share(values: Any) -> Optional[float]:
    """Share of the most common value, ignoring missing ones. ``None`` if empty."""
    keys = _keys(values)
    if keys.empty:
        return None
    keys = keys[keys != _MISSING]
    if keys.empty:
        return None
    counts = keys.value_counts()
    return float(counts.iloc[0]) / float(len(keys))


def expected_longest_run(values: Any, n: int) -> Optional[float]:
    """The longest run of one value ``values``' own mix would give by chance.

    A 95/5 classifier is *supposed* to repeat itself: in 1000 predictions the
    longest run of the majority class is about 87 long, and comparing that to a
    fixed limit calls every healthy fraud model stuck. The longest run of a
    value that turns up with probability ``p`` over ``n`` draws is the classic
    ``log(n(1-p)) / log(1/p)``, Gumbel-distributed around it; this returns that
    mean plus :data:`RUN_SIGMA` standard deviations.

    ``None`` when ``values`` is empty, is a single constant value (nothing to
    compare a constant window against), or ``n`` is below two.
    """
    share = value_share(values)
    if share is None or n < 2:
        return None
    if share <= 0.0 or share >= 1.0:
        return None
    rate = math.log(1.0 / share)
    if rate <= 0.0 or not math.isfinite(rate):
        return None
    expected = math.log(max(float(n) * (1.0 - share), 1.0)) / rate + _EULER / rate - 0.5
    spread = _GUMBEL_SD / rate
    limit = expected + RUN_SIGMA * spread
    if not math.isfinite(limit):
        return None
    return max(limit, 1.0)


def typical_magnitude(values: Any) -> Optional[float]:
    """Mean absolute value of the finite numbers in ``values``, or ``None``."""
    array = numeric_array(values)
    if array.size == 0:
        return None
    return float(np.mean(np.abs(array)))


def missing_share(frame: pd.DataFrame) -> Optional[float]:
    """Share of missing cells across the whole frame, or ``None`` if empty."""
    if frame is None or frame.empty or frame.shape[1] == 0:
        return None
    total = float(frame.shape[0] * frame.shape[1])
    if total <= 0:
        return None
    return float(frame.isna().to_numpy().sum() / total)


def task_of(predictions: Any, actuals: Any) -> str:
    """Classification when both sides look like labels, else regression."""
    prediction_labels = (not looks_numeric(predictions)) or is_discrete(predictions)
    actual_labels = (not looks_numeric(actuals)) or is_discrete(actuals)
    return "classification" if prediction_labels and actual_labels else "regression"


def score(predictions: Any, actuals: Any) -> Optional[Dict[str, Any]]:
    """Accuracy (classification) or mean absolute error (regression).

    Returns a dict with ``task``, ``accuracy``, ``error`` and ``n``, or
    ``None`` when no prediction/actual pair survives.
    """
    prediction_series = as_series(predictions).reset_index(drop=True)
    actual_series = as_series(actuals).reset_index(drop=True)
    size = min(len(prediction_series), len(actual_series))
    if size == 0:
        return None
    prediction_series = prediction_series.iloc[:size]
    actual_series = actual_series.iloc[:size]
    keep = prediction_series.notna() & actual_series.notna()
    prediction_series = prediction_series[keep]
    actual_series = actual_series[keep]
    if prediction_series.empty:
        return None
    task = task_of(prediction_series, actual_series)
    if task == "classification":
        matches = _keys(prediction_series).to_numpy() == _keys(actual_series).to_numpy()
        return {
            "task": task,
            "accuracy": float(np.mean(matches)),
            "error": None,
            "n": int(prediction_series.size),
        }
    predicted = pd.to_numeric(prediction_series, errors="coerce").to_numpy(dtype="float64")
    observed = pd.to_numeric(actual_series, errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(predicted) & np.isfinite(observed)
    if not finite.any():
        return None
    return {
        "task": task,
        "accuracy": None,
        "error": float(np.mean(np.abs(predicted[finite] - observed[finite]))),
        "n": int(finite.sum()),
    }


def rate_per_hour(timestamps: Sequence[Any], min_span_seconds: float = 60.0) -> Optional[float]:
    """Records per hour over the span of the timestamps.

    ``None`` when there are fewer than two timestamps or they span less than
    ``min_span_seconds``, because a rate over a few milliseconds is noise.
    """
    stamps = to_datetime_utc(timestamps).dropna()
    if stamps.size < 2:
        return None
    span_seconds = float((stamps.max() - stamps.min()).total_seconds())
    if span_seconds < float(min_span_seconds):
        return None
    return float(stamps.size) / (span_seconds / 3600.0)


def jsonable(value: Any) -> Any:
    """Coerce anything into something json.dumps accepts. Never raises."""
    try:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if np.isfinite(number) else None
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.ndarray):
            return [jsonable(item) for item in value.tolist()]
        if isinstance(value, pd.Series):
            return [jsonable(item) for item in value.tolist()]
        if isinstance(value, (list, tuple, set, frozenset)):
            return [jsonable(item) for item in value]
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return str(value)
    except Exception:  # pragma: no cover - a __str__ that raises
        return None


def iter_items(mapping: Optional[Dict[str, Any]]) -> Iterable[Tuple[str, Any]]:
    """Items of a mapping that may be ``None``."""
    if not mapping:
        return []
    return mapping.items()
