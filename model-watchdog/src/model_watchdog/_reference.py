"""What "normal" looked like: the reference the monitors compare against.

A reference can be a DataFrame of past traffic, a path to a ``.csv`` or
``.parquet`` file, a plain dict of values you already have, or just a list of
past predictions. Whatever comes in, a :class:`ReferenceProfile` comes out, and
anything it could not work out stays ``None`` so the monitor that needs it
reports itself inactive instead of failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from . import _stats
from ._records import CORE_COLUMNS

logger = logging.getLogger(__name__)

#: Column names that describe the record rather than a model feature.
_RESERVED = set(CORE_COLUMNS) | {"timestamp", "time", "features", "meta"}

_PREDICTION_KEYS = ("prediction", "predictions", "y_pred", "pred", "score")
_ACTUAL_KEYS = ("actual", "actuals", "y_true", "label", "target")
_LATENCY_KEYS = ("latency_ms", "latency", "latency_millis")
_TIME_KEYS = ("ts", "timestamp", "time")


def _read_table(path: Any) -> pd.DataFrame:
    """Read a ``.csv`` or ``.parquet`` file into a DataFrame."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        try:
            return pd.read_parquet(file_path)
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "reading %s needs pyarrow: pip install \"model-watchdog[parquet]\"" % file_path
            ) from exc
    if suffix in (".csv", ".txt", ""):
        return pd.read_csv(file_path)
    raise ValueError("unsupported reference file type %r: use .csv or .parquet" % suffix)


def _require_unique_columns(frame: pd.DataFrame, where: str = "reference") -> None:
    """Raise a clear ValueError naming any duplicated column name.

    Called again after the ``features`` column is exploded, because that is
    where duplicates are actually born: a log read back with
    ``pd.read_json(..., lines=True)`` carries both an ``age`` column and a
    ``features`` dict holding ``age``, and concatenating the two puts the name
    in twice. ``frame["age"]`` is then a DataFrame, not a Series, and the
    monitors would silently profile its *column names* as the feature's values.
    """
    names = [str(name) for name in frame.columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "%s has duplicate column names: %s" % (where, ", ".join(duplicates))
        )


def _first(mapping: Mapping[str, Any], keys: Any) -> Optional[Any]:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _series(values: Any, *, drop_missing: bool = True) -> Optional[pd.Series]:
    """The values as a Series, optionally with missing readings removed.

    ``drop_missing`` must be False for predictions and actuals. They are paired by
    position, so removing a gap from one and not the other slides every later
    prediction onto the wrong actual: one missing prediction in an alternating
    baseline turned a perfect 1.00 accuracy into 0.03, and a model that had
    completely collapsed then looked unchanged against it. Pairwise filtering is
    done in :func:`_stats.score`, which drops a row only when either side is missing.
    """
    if values is None:
        return None
    series = _stats.as_series(values)
    if drop_missing:
        series = series[series.notna()]
    elif series.notna().sum() == 0:
        return None
    return series if not series.empty else None


@dataclass
class ReferenceProfile:
    """The reference side of every comparison. Every field may be ``None``."""

    predictions: Optional[pd.Series] = None
    actuals: Optional[pd.Series] = None
    latency: Optional[pd.Series] = None
    features: Dict[str, pd.Series] = field(default_factory=dict)
    accuracy: Optional[float] = None
    error: Optional[float] = None
    task: Optional[str] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    rate_per_hour: Optional[float] = None
    null_rate: Optional[float] = None
    n: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """True when the reference carries nothing a monitor can use."""
        return not (
            self.predictions is not None
            or self.features
            or self.accuracy is not None
            or self.error is not None
            or self.latency_p95 is not None
            or self.rate_per_hour is not None
        )

    def describe(self) -> str:
        """One line naming what the reference can actually check."""
        if self.empty:
            return "no reference"
        parts: List[str] = []
        if self.predictions is not None:
            parts.append("%d predictions" % len(self.predictions))
        if self.features:
            parts.append("%d features" % len(self.features))
        if self.accuracy is not None:
            parts.append("accuracy %.4f" % self.accuracy)
        if self.error is not None:
            parts.append("error %.4f" % self.error)
        if self.latency_p95 is not None:
            parts.append("p95 %.1f ms" % self.latency_p95)
        if self.rate_per_hour is not None:
            parts.append("%.1f records/hour" % self.rate_per_hour)
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the reference, without the raw values."""
        return {
            "records": int(self.n),
            "has_predictions": self.predictions is not None,
            "features": sorted(self.features),
            "task": self.task,
            "accuracy": self.accuracy,
            "error": self.error,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "rate_per_hour": self.rate_per_hour,
            "null_rate": self.null_rate,
            "notes": list(self.notes),
        }

    def _finish(self) -> "ReferenceProfile":
        """Fill in the derived numbers once the raw values are in place."""
        if self.latency is not None:
            if self.latency_p50 is None:
                self.latency_p50 = _stats.quantile(self.latency, 0.50)
            if self.latency_p95 is None:
                self.latency_p95 = _stats.quantile(self.latency, 0.95)
        if self.predictions is not None and self.actuals is not None:
            scored = _stats.score(self.predictions, self.actuals)
            if scored:
                if self.accuracy is None:
                    self.accuracy = scored["accuracy"]
                if self.error is None:
                    self.error = scored["error"]
                if self.task is None:
                    self.task = scored["task"]
        if self.task is None:
            if self.accuracy is not None:
                self.task = "classification"
            elif self.error is not None:
                self.task = "regression"
        if self.null_rate is None and self.features:
            self.null_rate = _stats.missing_share(pd.DataFrame(dict(self.features)))
        if not self.n:
            if self.predictions is not None:
                self.n = int(len(self.predictions))
            elif self.features:
                self.n = int(max(len(values) for values in self.features.values()))
        return self

    @classmethod
    def from_any(cls, reference: Any) -> "ReferenceProfile":
        """Build a profile from whatever the caller passed as ``reference``."""
        if reference is None:
            return cls()
        if isinstance(reference, ReferenceProfile):
            return reference
        if isinstance(reference, (str, Path)):
            return cls.from_frame(_read_table(reference))
        if isinstance(reference, pd.DataFrame):
            return cls.from_frame(reference)
        if isinstance(reference, Mapping):
            return cls.from_mapping(reference)
        if isinstance(reference, pd.Series):
            return cls(predictions=_series(reference))._finish()
        if isinstance(reference, (list, tuple)) or hasattr(reference, "tolist"):
            return cls(predictions=_series(reference))._finish()
        raise TypeError(
            "reference must be a DataFrame, a path, a dict or a sequence of "
            "past predictions, got %s" % type(reference).__name__
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ReferenceProfile":
        """Build a profile from a table of past traffic."""
        if not isinstance(frame, pd.DataFrame):  # pragma: no cover - guarded above
            raise TypeError("expected a DataFrame")
        _require_unique_columns(frame)
        columns = {str(name).lower(): str(name) for name in frame.columns}
        profile = cls(n=int(len(frame)))

        expanded = frame
        feature_column = columns.get("features")
        if feature_column is not None:
            exploded = _expand_feature_column(frame[feature_column])
            if exploded is not None:
                expanded = pd.concat([frame.drop(columns=[feature_column]), exploded], axis=1)
                # The concat is where a duplicate can appear: a plain "age"
                # column beside an "age" key inside every features dict.
                _require_unique_columns(expanded, "reference (after expanding 'features')")
                columns.pop("features", None)

        prediction_column = _first(columns, _PREDICTION_KEYS)
        actual_column = _first(columns, _ACTUAL_KEYS)
        latency_column = _first(columns, _LATENCY_KEYS)
        time_column = _first(columns, _TIME_KEYS)

        if prediction_column is not None:
            profile.predictions = _series(expanded[prediction_column], drop_missing=False)
        elif len(expanded.columns) == 1:
            profile.predictions = _series(expanded[expanded.columns[0]], drop_missing=False)
        if actual_column is not None:
            profile.actuals = _series(expanded[actual_column], drop_missing=False)
        if latency_column is not None:
            profile.latency = _series(expanded[latency_column])
        if time_column is not None:
            profile.rate_per_hour = _stats.rate_per_hour(expanded[time_column])

        used = {prediction_column, actual_column, latency_column, time_column}
        feature_names = [
            str(name)
            for name in expanded.columns
            if str(name) not in used and str(name).lower() not in _RESERVED
        ]
        if profile.predictions is not None and len(expanded.columns) == 1:
            feature_names = []
        profile.features = {
            name: _stats.as_series(expanded[name]) for name in feature_names
        }
        if profile.features:
            profile.null_rate = _stats.missing_share(expanded[feature_names])
        return profile._finish()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ReferenceProfile":
        """Build a profile from a dict of values or already-known numbers."""
        lowered = {str(key).lower(): value for key, value in mapping.items()}
        profile = cls()
        profile.predictions = _series(_first(lowered, _PREDICTION_KEYS), drop_missing=False)
        profile.actuals = _series(_first(lowered, _ACTUAL_KEYS), drop_missing=False)
        profile.latency = _series(_first(lowered, _LATENCY_KEYS))
        features = lowered.get("features")
        if isinstance(features, pd.DataFrame):
            _require_unique_columns(features, "reference features")
            profile.features = {str(name): features[name] for name in features.columns}
        elif isinstance(features, Mapping):
            profile.features = {
                str(name): _stats.as_series(values) for name, values in features.items()
            }
        for name, attribute in (
            ("accuracy", "accuracy"),
            ("error", "error"),
            ("mae", "error"),
            ("task", "task"),
            ("latency_p50", "latency_p50"),
            ("p50", "latency_p50"),
            ("latency_p95", "latency_p95"),
            ("p95", "latency_p95"),
            ("rate_per_hour", "rate_per_hour"),
            ("null_rate", "null_rate"),
            ("n", "n"),
        ):
            if name in lowered and lowered[name] is not None:
                value = lowered[name]
                setattr(profile, attribute, value if attribute == "task" else float(value))
        if isinstance(profile.n, float):
            profile.n = int(profile.n)
        return profile._finish()


def _expand_feature_column(column: pd.Series) -> Optional[pd.DataFrame]:
    """Turn a column of ``{"feature": value}`` dicts into real columns."""
    values = [value if isinstance(value, Mapping) else None for value in column]
    if not any(value is not None for value in values):
        return None
    return pd.DataFrame([dict(value) if value else {} for value in values], index=column.index)


def profile_or_note(reference: Any) -> "ReferenceProfile":
    """Parse a reference, turning a bad one into a note instead of a crash.

    Monitoring must not take the host application down at start-up either, so a
    reference that cannot be read leaves an empty profile whose note shows up in
    every report.
    """
    try:
        return ReferenceProfile.from_any(reference)
    except Exception as exc:  # noqa: BLE001 - deliberate: never break the caller
        logger.warning("model-watchdog: could not read the reference: %s", exc)
        profile = ReferenceProfile()
        profile.notes.append("reference could not be read: %s" % exc)
        return profile
