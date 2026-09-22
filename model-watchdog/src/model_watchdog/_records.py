"""The shape of one logged record, and how a pile of them becomes a DataFrame.

A record on disk is one JSON object per line::

    {"ts": "2026-09-22T10:00:00.123456+00:00", "prediction": 0.81,
     "actual": 1, "latency_ms": 12.4, "features": {...}, "meta": {...}}

Only ``ts`` is always present; the rest appear when they were logged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

import pandas as pd

from ._stats import jsonable, to_datetime_utc

#: Columns that belong to the record itself rather than to the model.
CORE_COLUMNS = ("ts", "prediction", "actual", "latency_ms")


def utc_now() -> datetime:
    """The current time, timezone-aware and in UTC."""
    return datetime.now(timezone.utc)


def timestamp_to_datetime(stamp: Any) -> datetime:
    """A UTC pandas Timestamp as a plain timezone-aware ``datetime``.

    Built field by field on purpose. ``Timestamp.to_pydatetime()`` raises a
    pandas ``UserWarning`` ("Discarding nonzero nanoseconds in conversion")
    whenever the value carries sub-microsecond precision, and that warning
    escapes into whatever filters the calling application has set. Under a
    ``-W error::UserWarning`` policy - ordinary in CI - it would turn into an
    exception inside ``log()``, whose blanket ``except`` would then drop the
    record. A truncated nanosecond is not worth a lost prediction.
    """
    return datetime(
        int(stamp.year),
        int(stamp.month),
        int(stamp.day),
        int(stamp.hour),
        int(stamp.minute),
        int(stamp.second),
        int(stamp.microsecond),
        tzinfo=timezone.utc,
    )


def to_utc(value: Any) -> Optional[datetime]:
    """Coerce a datetime, pandas Timestamp or ISO string to aware UTC.

    ``None`` for anything that cannot be read as a timestamp; callers that must
    not silently widen a window check for that and raise.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    return timestamp_to_datetime(stamp)


def as_feature_dict(features: Any) -> Optional[Dict[str, Any]]:
    """Normalise whatever was passed as ``features`` into a flat dict.

    Accepts a mapping, a pandas Series, a one-row DataFrame, or a sequence
    (named ``f0``, ``f1``, ...). Raises ``ValueError`` on anything else; the
    caller is ``log()``, which turns that into a warning rather than a crash.
    """
    if features is None:
        return None
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError(
                "features DataFrame must have exactly one row, got %d" % len(features)
            )
        features = features.iloc[0]
    if isinstance(features, pd.Series):
        return {str(key): jsonable(value) for key, value in features.items()}
    if isinstance(features, Mapping):
        return {str(key): jsonable(value) for key, value in features.items()}
    if isinstance(features, (list, tuple)) or hasattr(features, "tolist"):
        values = list(features.tolist()) if hasattr(features, "tolist") else list(features)
        return {"f%d" % index: jsonable(value) for index, value in enumerate(values)}
    raise ValueError(
        "features must be a dict, Series, one-row DataFrame or sequence, got %s"
        % type(features).__name__
    )


def build_record(
    features: Any = None,
    prediction: Any = None,
    actual: Any = None,
    latency_ms: Any = None,
    meta: Optional[Mapping[str, Any]] = None,
    ts: Any = None,
) -> Dict[str, Any]:
    """One JSON-safe record. Keys that were not logged are left out."""
    stamp = to_utc(ts) or utc_now()
    record: Dict[str, Any] = {"ts": stamp.isoformat()}
    feature_values = as_feature_dict(features)
    if prediction is not None:
        record["prediction"] = jsonable(prediction)
    if actual is not None:
        record["actual"] = jsonable(actual)
    if latency_ms is not None:
        record["latency_ms"] = jsonable(latency_ms)
    if feature_values:
        record["features"] = feature_values
    extra = {str(key): jsonable(value) for key, value in (meta or {}).items()}
    if extra:
        record["meta"] = extra
    return record


class Frame(NamedTuple):
    """A DataFrame of records plus which columns came from where."""

    data: pd.DataFrame
    feature_columns: List[str]
    meta_columns: List[str]


def _numeric_if_clean(series: pd.Series) -> pd.Series:
    """Numeric dtype when every present value is a number, else left alone."""
    present = series.notna()
    if not present.any():
        return series
    converted = pd.to_numeric(series, errors="coerce")
    if bool(converted[present].notna().all()):
        return converted
    return series


def _unique(name: str, taken: Sequence[str], suffix: str) -> str:
    candidate = name if name not in taken else name + suffix
    while candidate in taken:
        candidate = candidate + "_"
    return candidate


def to_frame(records: Sequence[Mapping[str, Any]]) -> Frame:
    """Records as a tidy DataFrame: core columns, then features, then meta.

    A feature or meta key that collides with a core column name is suffixed
    (``prediction`` -> ``prediction_feature``) so no value is ever lost.
    """
    feature_names: List[str] = []
    meta_names: List[str] = []
    for record in records:
        for key in (record.get("features") or {}):
            if key not in feature_names:
                feature_names.append(str(key))
        for key in (record.get("meta") or {}):
            if key not in meta_names:
                meta_names.append(str(key))

    feature_columns: Dict[str, str] = {}
    used = list(CORE_COLUMNS)
    for name in feature_names:
        column = _unique(name, used, "_feature")
        feature_columns[name] = column
        used.append(column)
    meta_columns: Dict[str, str] = {}
    for name in meta_names:
        column = _unique(name, used, "_meta")
        meta_columns[name] = column
        used.append(column)

    rows: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {
            "ts": record.get("ts"),
            "prediction": record.get("prediction"),
            "actual": record.get("actual"),
            "latency_ms": record.get("latency_ms"),
        }
        for key, value in (record.get("features") or {}).items():
            row[feature_columns[str(key)]] = value
        for key, value in (record.get("meta") or {}).items():
            row[meta_columns[str(key)]] = value
        rows.append(row)

    columns = list(used)
    if rows:
        data = pd.DataFrame(rows).reindex(columns=columns)
    else:
        data = pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    data["ts"] = to_datetime_utc(data["ts"])
    for column in ("prediction", "actual"):
        data[column] = _numeric_if_clean(data[column])
    data["latency_ms"] = pd.to_numeric(data["latency_ms"], errors="coerce")
    data = data.reset_index(drop=True)
    return Frame(data, list(feature_columns.values()), list(meta_columns.values()))
