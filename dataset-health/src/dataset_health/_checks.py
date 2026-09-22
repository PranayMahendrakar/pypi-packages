"""The individual checks. Each ``check_*`` takes a :class:`Frame` and a :class:`Config` and
returns a list of :class:`Issue`; :mod:`._core` runs them in order and assembles the report."""

from __future__ import annotations

import datetime as _dt
import math
import numbers
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    infer_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

from ._report import Issue, json_safe

MIN_ROWS_FOR_STATS = 10  # id_like, outliers, skew and leakage statistics need this many values
CLASSIFICATION_MAX_CLASSES = 20  # a numeric target with at most this many distinct values is a label
DATE_SAMPLE = 1000  # how many strings to try parsing as dates
MAX_CORRELATION_PAIRS = 100  # cap on reported high_correlation pairs
PANDAS_CORR_MAX_COLUMNS = 100  # pairwise-complete correlation up to this many columns, then imputed
LOW_CARDINALITY_NUMERIC = 20  # numeric features with at most this many values also get the mapping rule
MIN_GROUP_SIZE = 5  # a categorical feature needs this many rows per value for the mapping rules

_ID_TOKENS = frozenset(
    {"id", "ids", "uid", "uuid", "guid", "key", "pk", "index", "idx", "identifier", "no", "num", "number",
     "code", "serial", "sku", "hash", "token"}
)
_NUMERIC_TEXT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class Config:
    """Thresholds shared by the checks (see :class:`dataset_health.HealthChecker`)."""

    missing_warning: float = 0.05
    missing_critical: float = 0.5
    near_constant: float = 0.99
    id_unique_ratio: float = 0.98
    high_cardinality: int = 50
    imbalance: float = 0.10
    imbalance_critical: float = 0.01
    leakage_corr: float = 0.95
    high_corr: float = 0.90
    outlier_share: float = 0.05
    skew: float = 3.0


# --------------------------------------------------------------------------------------
# column profiles
# --------------------------------------------------------------------------------------


def column_type(series: pd.Series) -> str:
    """``"boolean"``, ``"datetime"`` (incl. timedelta), ``"numeric"`` or ``"categorical"``."""
    if is_bool_dtype(series):
        return "boolean"
    if is_datetime64_any_dtype(series) or is_timedelta64_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def as_float(series: pd.Series) -> Optional[np.ndarray]:
    """Float64 array with NaN for missing; ``None`` for columns that are not numeric-like."""
    kind = column_type(series)
    if kind == "categorical":
        return None
    if kind == "datetime":
        if is_timedelta64_dtype(series):
            raw = series.to_numpy(dtype="timedelta64[ns]")
        else:
            raw = series.to_numpy(dtype="datetime64[ns]")
        values = raw.astype("int64").astype("float64")
        values[series.isna().to_numpy()] = np.nan
        return values
    return series.to_numpy(dtype="float64", na_value=np.nan)


class Profile:
    """Cheap per-column facts every check reads: type, missing share, value counts."""

    __slots__ = ("name", "series", "type", "dtype", "n_missing", "n", "missing_share", "hashable", "counts",
                 "n_unique", "_values")

    def __init__(self, name: str, series: pd.Series) -> None:
        self.name = name
        self.series = series
        self.type = column_type(series)
        self.dtype = str(series.dtype)
        missing = series.isna()
        self.n_missing = int(missing.sum())
        self.n = int(len(series) - self.n_missing)
        self.missing_share = float(self.n_missing / len(series)) if len(series) else 0.0
        self.hashable = True
        try:
            counts = series.value_counts(dropna=True)
            if len(counts):
                hash(counts.index[0])
        except TypeError:
            self.hashable = False
            counts = self.categories().value_counts(dropna=True)
        self.counts = counts
        self.n_unique = int(len(counts))
        self._values: Optional[np.ndarray] = None

    @property
    def unique_ratio(self) -> float:
        return self.n_unique / self.n if self.n else 0.0

    @property
    def top_value(self) -> Any:
        return self.counts.index[0] if len(self.counts) else None

    @property
    def top_share(self) -> float:
        return float(self.counts.iloc[0]) / self.n if self.n else 0.0

    def values(self) -> Optional[np.ndarray]:
        """Float view (NaN for missing) for numeric, boolean and datetime columns."""
        if self._values is None:
            self._values = as_float(self.series)
        return self._values

    def categories(self) -> pd.Series:
        """The column as hashable categories (unhashable cells become their repr), NaN kept."""
        if self.hashable:
            return self.series
        out = pd.Series(np.nan, index=self.series.index, dtype=object)
        mask = self.series.notna().to_numpy()
        out[mask] = self.series[mask].map(repr)
        return out


def detect_task(target: Profile) -> Optional[str]:
    """``"classification"`` for non-numeric or few-valued targets, ``"regression"`` otherwise."""
    if target.n == 0:
        return None
    if target.type in ("categorical", "boolean"):
        return "classification"
    if target.type == "numeric" and target.n_unique <= CLASSIFICATION_MAX_CLASSES:
        return "classification"
    return "regression"


class Frame:
    """The table under inspection plus everything the checks share."""

    def __init__(self, df: pd.DataFrame, target: Optional[str]) -> None:
        self.df = df
        self.n_rows = len(df)
        self.n_columns = len(df.columns)
        self.names: List[str] = [str(c) for c in df.columns]
        self.profiles: Dict[str, Profile] = {name: Profile(name, df[name]) for name in self.names}
        self.target = target
        self.feature_names = [n for n in self.names if n != target]
        self.task = detect_task(self.profiles[target]) if target is not None else None
        self.notes: List[str] = []
        self.id_like: set = set()  # filled by check_id_like
        self.text_like: set = set()  # filled by check_object_types (dates / numbers stored as text)
        self._isna: Optional[pd.DataFrame] = None

    def isna(self) -> pd.DataFrame:
        if self._isna is None:
            self._isna = self.df.isna()
        return self._isna

    def all_profiles(self) -> List[Profile]:
        return [self.profiles[n] for n in self.names]

    def features(self) -> List[Profile]:
        return [self.profiles[n] for n in self.feature_names]

    def target_profile(self) -> Optional[Profile]:
        return self.profiles[self.target] if self.target is not None else None

    def numeric_target_scope(self) -> List[Profile]:
        """Features plus the target when the target is a regression label."""
        profiles = self.features()
        if self.task == "regression":
            profiles = profiles + [self.profiles[self.target]]
        return profiles

    def column_profiles(self) -> Dict[str, Dict[str, Any]]:
        return {
            p.name: {
                "role": "target" if p.name == self.target else "feature",
                "type": p.type,
                "dtype": p.dtype,
                "missing_share": p.missing_share,
                "n_unique": p.n_unique,
            }
            for p in self.all_profiles()
        }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _pct(share: float) -> str:
    return f"{100.0 * share:.1f}%"


def _fmt(value: Any) -> str:
    value = json_safe(value)
    return repr(value) if isinstance(value, str) else str(value)


def _examples(index: pd.Index, limit: int = 10) -> List[Any]:
    return json_safe(list(index[:limit]))


def _tokens(name: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(name).lower())


def _name_looks_like_id(name: str) -> bool:
    return any(token in _ID_TOKENS for token in _tokens(name))


def _contains_tokens(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(list(haystack[i : i + len(needle)]) == list(needle) for i in range(len(haystack) - len(needle) + 1))


def _type_group(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return "datetime"
    return type(value).__name__


def _date_share(values: pd.Series) -> Tuple[float, List[str]]:
    """Share of ``values`` (all strings) that parse as dates, judged on a sample."""
    sample = values if len(values) <= DATE_SAMPLE else values.sample(DATE_SAMPLE, random_state=0)
    text = sample.astype(str).str.strip()
    candidates = text[~text.str.match(_NUMERIC_TEXT) & text.str.contains(r"\d", regex=True)]
    if len(candidates) < 0.9 * len(sample) or len(candidates) == 0:
        return 0.0, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(candidates, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(candidates, errors="coerce")
    ok = parsed.notna()
    examples = list(dict.fromkeys(candidates[ok.to_numpy()].tolist()))[:3]
    return float(ok.sum()) / len(sample), examples


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x, y = x[mask], y[mask]
    if x.std() == 0 or y.std() == 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(x, y)[0, 1])
    return r if math.isfinite(r) else None


def _stump(x: np.ndarray, positive: np.ndarray) -> Optional[Tuple[float, float]]:
    """Best single-threshold split of ``x`` for a binary target: (balanced accuracy, threshold)."""
    mask = np.isfinite(x)
    x, y = x[mask], positive[mask].astype("int64")
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    # Below a handful of rows in a class, one row moves balanced accuracy by more than the
    # error budget, so any extreme feature would "separate" the classes by coincidence.
    if min(n0, n1) < MIN_GROUP_SIZE or len(y) < MIN_ROWS_FOR_STATS:
        return None
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    cum1 = np.cumsum(ys)
    cum0 = np.arange(1, len(xs) + 1) - cum1
    valid = np.nonzero(xs[:-1] != xs[1:])[0]  # a split may only fall between distinct values
    if len(valid) == 0:
        return None
    left1, left0 = cum1[valid], cum0[valid]
    low_is_negative = 0.5 * (left0 / n0 + (n1 - left1) / n1)
    low_is_positive = 0.5 * (left1 / n1 + (n0 - left0) / n0)
    best = np.maximum(low_is_negative, low_is_positive)
    k = int(np.argmax(best))
    threshold = float((xs[valid[k]] + xs[valid[k] + 1]) / 2.0)
    return float(best[k]), threshold


def _mapping_purity(feature: pd.Series, target: pd.Series) -> Optional[Tuple[float, float]]:
    """(purity, baseline): purity is the share of rows whose target equals the majority target
    of their feature value; baseline is the share of the overall majority target."""
    mask = (feature.notna() & target.notna()).to_numpy()
    if mask.sum() < MIN_ROWS_FOR_STATS:
        return None
    pairs = pd.DataFrame({"f": feature.to_numpy()[mask], "t": target.to_numpy()[mask]})
    sizes = pairs.groupby(["f", "t"], observed=True, sort=False).size()
    total = float(sizes.sum())
    purity = float(sizes.groupby(level=0, observed=True).max().sum()) / total
    baseline = float(sizes.groupby(level=1, observed=True).sum().max()) / total
    return purity, baseline


def _correlation_ratio(feature: pd.Series, y: np.ndarray) -> Optional[float]:
    """eta: the multiple correlation of a numeric target with a categorical feature."""
    mask = feature.notna().to_numpy() & np.isfinite(y)
    if mask.sum() < MIN_ROWS_FOR_STATS:
        return None
    yy = y[mask]
    total = float(((yy - yy.mean()) ** 2).sum())
    if total == 0:
        return None
    groups = pd.DataFrame({"f": feature.to_numpy()[mask], "y": yy}).groupby("f", observed=True)["y"]
    stats = groups.agg(["mean", "count"])
    between = float((stats["count"] * (stats["mean"] - yy.mean()) ** 2).sum())
    return math.sqrt(max(0.0, min(1.0, between / total)))


def _corr_matrix(matrix: np.ndarray, frame: Frame) -> np.ndarray:
    has_nan = bool(np.isnan(matrix).any())
    if has_nan and matrix.shape[1] <= PANDAS_CORR_MAX_COLUMNS:
        return pd.DataFrame(matrix).corr().to_numpy()
    if has_nan:
        matrix = np.where(np.isnan(matrix), np.nanmean(matrix, axis=0), matrix)
        frame.notes.append(
            f"high_correlation: more than {PANDAS_CORR_MAX_COLUMNS} numeric columns, so missing values "
            "were mean-imputed before correlating"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(matrix, rowvar=False)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------


def check_empty(frame: Frame, cfg: Config) -> List[Issue]:
    """Entirely missing columns and rows."""
    issues: List[Issue] = []
    for p in frame.all_profiles():
        if p.n == 0:
            is_target = p.name == frame.target
            issues.append(
                Issue(
                    "critical" if is_target else "warning",
                    "empty_column",
                    [p.name],
                    f"'{p.name}' is entirely missing" + (" and it is the target" if is_target else "; drop it"),
                    {"n_rows": frame.n_rows, "role": "target" if is_target else "feature"},
                )
            )
    empty_rows = frame.isna().all(axis=1)
    n_empty = int(empty_rows.sum())
    if n_empty:
        share = n_empty / frame.n_rows
        issues.append(
            Issue(
                "warning" if share >= 0.01 else "info",
                "empty_row",
                [],
                f"{n_empty:,} rows ({_pct(share)}) are entirely missing; drop them",
                {"count": n_empty, "share": share, "example_index": _examples(frame.df.index[empty_rows.to_numpy()])},
            )
        )
    return issues


def check_missing(frame: Frame, cfg: Config) -> List[Issue]:
    """Per-column missing share, and rows with more than half of their values missing."""
    issues: List[Issue] = []
    for p in frame.all_profiles():
        if p.n_missing == 0 or p.n == 0:
            continue
        share = p.missing_share
        if share > cfg.missing_critical:
            severity = "critical"
        elif share >= cfg.missing_warning:
            severity = "warning"
        else:
            severity = "info"
        issues.append(
            Issue(
                severity,
                "missing",
                [p.name],
                f"'{p.name}' is {_pct(share)} missing ({p.n_missing:,} of {frame.n_rows:,} rows)",
                {"share": share, "count": p.n_missing},
            )
        )
    if frame.n_columns > 1:
        row_share = frame.isna().mean(axis=1)
        mostly = ((row_share > 0.5) & (row_share < 1.0)).to_numpy()
        n = int(mostly.sum())
        if n:
            share = n / frame.n_rows
            severity = "critical" if share >= 0.10 else "warning" if share >= 0.01 else "info"
            issues.append(
                Issue(
                    severity,
                    "missing",
                    [],
                    f"{n:,} rows ({_pct(share)}) have more than half of their values missing",
                    {"rows": n, "share": share, "example_index": _examples(frame.df.index[mostly])},
                )
            )
    return issues


def check_duplicates(frame: Frame, cfg: Config) -> List[Issue]:
    """Exact duplicate rows."""
    try:
        dup = frame.df.duplicated(keep="first")
    except TypeError:
        dup = frame.df.astype(str).duplicated(keep="first")
    n = int(dup.sum())
    if not n:
        return []
    share = n / frame.n_rows
    severity = "critical" if share >= 0.20 else "warning" if share >= 0.01 else "info"
    return [
        Issue(
            severity,
            "duplicates",
            [],
            f"{n:,} exact duplicate rows ({_pct(share)}); keep one copy of each",
            {"count": n, "share": share, "example_index": _examples(frame.df.index[dup.to_numpy()])},
        )
    ]


def check_object_types(frame: Frame, cfg: Config) -> List[Issue]:
    """Object columns that mix Python types, and text columns that hold numbers or dates."""
    issues: List[Issue] = []
    for p in frame.all_profiles():
        if p.type != "categorical" or p.n == 0 or isinstance(p.series.dtype, pd.CategoricalDtype):
            continue
        kind = infer_dtype(p.series, skipna=True)
        non_null = p.series.dropna().astype(object)
        if kind.startswith("mixed") and kind != "mixed-integer-float":
            groups = non_null.map(_type_group).value_counts()
            described = ", ".join(f"{int(count):,} {name}" for name, count in groups.items())
            examples = {name: _fmt(non_null[non_null.map(_type_group) == name].iloc[0]) for name in groups.index[:4]}
            issues.append(
                Issue(
                    "warning",
                    "mixed_types",
                    [p.name],
                    f"'{p.name}' mixes Python types ({described}); coerce it to one type",
                    {"types": {str(k): int(v) for k, v in groups.items()}, "examples": examples},
                )
            )
            continue
        if kind != "string":
            continue
        parsed = pd.to_numeric(non_null, errors="coerce")
        numeric_share = float(parsed.notna().mean())
        if numeric_share >= 1.0:
            frame.text_like.add(p.name)
            issues.append(
                Issue(
                    "info",
                    "numeric_as_strings",
                    [p.name],
                    f"'{p.name}' stores numbers as text (all {p.n:,} values parse); convert it with pd.to_numeric",
                    {"numeric_share": 1.0, "examples": non_null.head(3).tolist()},
                )
            )
            continue
        if numeric_share >= 0.8:
            frame.text_like.add(p.name)
            bad = non_null[parsed.isna().to_numpy()]
            bad_examples = list(dict.fromkeys(bad.astype(str).tolist()))[:5]
            issues.append(
                Issue(
                    "warning",
                    "mixed_types",
                    [p.name],
                    f"'{p.name}' mixes numbers stored as text with {len(bad):,} non-numeric values "
                    f"(e.g. {', '.join(repr(v) for v in bad_examples[:3])})",
                    {"numeric_share": numeric_share, "non_numeric_count": len(bad), "non_numeric_examples": bad_examples},
                )
            )
            continue
        date_share, examples = _date_share(non_null)
        if date_share >= 0.9:
            frame.text_like.add(p.name)
            issues.append(
                Issue(
                    "warning",
                    "dates_as_strings",
                    [p.name],
                    f"'{p.name}' holds dates stored as text (e.g. {', '.join(repr(v) for v in examples[:2])}); "
                    "parse it with pd.to_datetime",
                    {"parsed_share": date_share, "examples": examples},
                )
            )
    return issues


def check_constant(frame: Frame, cfg: Config) -> List[Issue]:
    """Columns with a single value, or with one value above the near-constant share."""
    if frame.n_rows < 2:
        # With one row every column holds exactly one value, so the finding would be
        # vacuous: say so in the notes instead of filing an issue per column.
        frame.notes.append(
            "the table has a single row, so the constant / near-constant checks were skipped"
        )
        return []
    issues: List[Issue] = []
    for p in frame.numeric_target_scope() if frame.target is not None else frame.all_profiles():
        if p.n == 0:
            continue
        if p.n_unique <= 1:
            issues.append(
                Issue(
                    "warning",
                    "constant",
                    [p.name],
                    f"'{p.name}' has a single value ({_fmt(p.top_value)}); it carries no information",
                    {"value": p.top_value, "share": 1.0},
                )
            )
        elif p.top_share > cfg.near_constant:
            issues.append(
                Issue(
                    "warning",
                    "near_constant",
                    [p.name],
                    f"'{p.name}' is {_pct(p.top_share)} one value ({_fmt(p.top_value)}); it carries almost no information",
                    {"value": p.top_value, "share": p.top_share, "n_unique": p.n_unique},
                )
            )
    return issues


def check_id_like(frame: Frame, cfg: Config) -> List[Issue]:
    """Text columns that are almost all unique, and integer columns that look like row ids."""
    issues: List[Issue] = []
    for p in frame.features():
        if p.n < MIN_ROWS_FOR_STATS or p.unique_ratio <= cfg.id_unique_ratio:
            continue
        reason = None
        if p.type == "categorical":
            reason = "unique text values"
        elif p.type == "numeric":
            values = _finite(p.values())
            if len(values) and (is_integer_dtype(p.series) or bool(np.all(np.mod(values, 1) == 0))):
                span = float(values.max() - values.min()) + 1.0
                if span <= 1.05 * len(values):
                    reason = "consecutive integers"
                elif len(values) > 1 and bool(np.all(np.diff(values) > 0)):
                    reason = "increasing integers"
                elif _name_looks_like_id(p.name):
                    reason = "integers in a column named like an identifier"
        if reason is None:
            continue
        frame.id_like.add(p.name)
        issues.append(
            Issue(
                "warning",
                "id_like",
                [p.name],
                f"'{p.name}' is {_pct(p.unique_ratio)} unique ({reason}); it identifies rows rather than "
                "describing them, so drop it before modeling",
                {"n_unique": p.n_unique, "unique_ratio": p.unique_ratio, "reason": reason},
            )
        )
    return issues


def check_high_cardinality(frame: Frame, cfg: Config) -> List[Issue]:
    """Categorical features with many distinct values (identifier-like columns are reported separately)."""
    issues: List[Issue] = []
    for p in frame.features():
        if p.type != "categorical" or p.name in frame.id_like or p.name in frame.text_like:
            continue
        if p.n_unique > cfg.high_cardinality:
            issues.append(
                Issue(
                    "warning",
                    "high_cardinality",
                    [p.name],
                    f"'{p.name}' has {p.n_unique:,} distinct values ({_pct(p.unique_ratio)} unique); "
                    "one-hot encoding will explode, so group rare values or use target/frequency encoding",
                    {"n_unique": p.n_unique, "unique_ratio": p.unique_ratio},
                )
            )
    return issues


def check_class_imbalance(frame: Frame, cfg: Config) -> List[Issue]:
    """Minority class share of a classification target."""
    if frame.task != "classification":
        return []
    p = frame.target_profile()
    assert p is not None
    if p.n == 0:
        return []
    counts = p.counts
    if len(counts) < 2:
        return [
            Issue(
                "critical",
                "class_imbalance",
                [p.name],
                f"target '{p.name}' has a single class ({_fmt(counts.index[0])}); there is nothing to learn",
                {"n_classes": 1, "classes": {str(json_safe(counts.index[0])): 1.0}},
            )
        ]
    minority, majority = counts.index[-1], counts.index[0]
    minority_count = int(counts.iloc[-1])
    share = minority_count / p.n
    if share >= cfg.imbalance:
        return []
    severity = "critical" if share < cfg.imbalance_critical else "warning"
    classes = {str(json_safe(label)): float(count) / p.n for label, count in counts.head(20).items()}
    return [
        Issue(
            severity,
            "class_imbalance",
            [p.name],
            f"minority class {_fmt(minority)} is {_pct(share)} of target '{p.name}' "
            f"({minority_count:,} of {p.n:,} rows); use class weights, resampling or a threshold sweep",
            {
                "n_classes": int(len(counts)),
                "minority": minority,
                "minority_share": share,
                "minority_count": minority_count,
                "majority": majority,
                "majority_share": float(counts.iloc[0]) / p.n,
                "classes": classes,
            },
        )
    ]


def check_target_leakage(frame: Frame, cfg: Config) -> List[Issue]:
    """Features that are nearly identical to the target, or named after it."""
    if frame.target is None or frame.task is None:
        return []
    t = frame.target_profile()
    assert t is not None
    target_tokens = _tokens(t.name)
    t_values = t.values()  # None for a text target
    t_cats = t.categories()
    binary = frame.task == "classification" and t.n_unique == 2
    positive = (t_cats == t.counts.index[-1]).to_numpy() if binary else None
    n_classes = t.n_unique
    issues: List[Issue] = []
    for p in frame.features():
        name_match = _contains_tokens(_tokens(p.name), target_tokens)
        evidence: Optional[Tuple[str, Dict[str, Any]]] = None
        if p.n >= 3:
            evidence = _leak_evidence(p, t, t_values, t_cats, positive, n_classes, frame, cfg)
        if evidence is None and not name_match:
            continue
        if evidence is not None:
            message, detail = evidence
            detail["name_match"] = name_match
            issues.append(Issue("critical", "target_leakage", [p.name], message, detail))
        else:
            issues.append(
                Issue(
                    "warning",
                    "target_leakage",
                    [p.name],
                    f"'{p.name}' is named after the target '{t.name}'; make sure it is known at prediction time",
                    {"rule": "name", "name_match": True},
                )
            )
    return issues


def _leak_evidence(
    p: Profile,
    t: Profile,
    t_values: Optional[np.ndarray],
    t_cats: pd.Series,
    positive: Optional[np.ndarray],
    n_classes: int,
    frame: Frame,
    cfg: Config,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    f_values = p.values()
    numeric_like = f_values is not None
    if numeric_like and t_values is not None and p.n_unique > 1:
        r = _pearson(f_values, t_values)
        if r is not None and abs(r) > cfg.leakage_corr:
            return (
                f"'{p.name}' is nearly identical to target '{t.name}' (|corr| = {abs(r):.3f})",
                {"rule": "correlation", "correlation": r},
            )
    # A row identifier always "separates" a target that happens to be sorted, so the
    # single-threshold rule would fire on row order rather than on a real relationship.
    if numeric_like and positive is not None and p.n_unique > 1 and p.name not in frame.id_like:
        stump = _stump(f_values, positive)
        if stump is not None and stump[0] > 0.98:
            accuracy, threshold = stump
            return (
                f"'{p.name}' separates the classes of target '{t.name}' with a single threshold "
                f"({_pct(accuracy)} balanced accuracy at {threshold:g})",
                {"rule": "threshold", "balanced_accuracy": accuracy, "threshold": threshold},
            )
    categorical_like = p.type in ("categorical", "boolean") or (
        p.type == "numeric" and p.n_unique <= LOW_CARDINALITY_NUMERIC
    )
    well_populated = (
        p.name not in frame.id_like and 2 <= p.n_unique and p.n_unique * MIN_GROUP_SIZE <= p.n
    )
    if not (categorical_like and well_populated):
        return None
    f_cats = p.categories()
    if frame.task == "classification":
        forward = _mapping_purity(f_cats, t_cats)
        if forward is not None and _nearly_determines(*forward):
            return (
                f"each value of '{p.name}' maps to one class of target '{t.name}' "
                f"({_pct(forward[0])} of rows follow the mapping)",
                {"rule": "mapping", "direction": "feature -> target", "purity": forward[0], "baseline": forward[1]},
            )
        backward = _mapping_purity(t_cats, f_cats)
        if backward is not None and _nearly_determines(*backward):
            return (
                f"'{p.name}' is a function of target '{t.name}' "
                f"({_pct(backward[0])} of rows follow the mapping)",
                {"rule": "mapping", "direction": "target -> feature", "purity": backward[0], "baseline": backward[1]},
            )
    elif t_values is not None:
        eta = _correlation_ratio(f_cats, t_values)
        if eta is not None and eta > cfg.leakage_corr:
            return (
                f"'{p.name}' explains {_pct(eta * eta)} of the variance of target '{t.name}' "
                f"(correlation ratio {eta:.3f})",
                {"rule": "correlation_ratio", "correlation_ratio": eta},
            )
    return None


def _nearly_determines(purity: float, baseline: float) -> bool:
    """True when the mapping removes at least 98% of the error a majority guess makes."""
    if baseline >= 1.0:
        return False
    return (1.0 - purity) <= 0.02 * (1.0 - baseline)


def check_high_correlation(frame: Frame, cfg: Config) -> List[Issue]:
    """Pairs of numeric features with |corr| above the threshold."""
    profiles = [
        p for p in frame.features() if p.type in ("numeric", "boolean", "datetime") and p.n_unique > 1 and p.n >= 3
    ]
    if len(profiles) < 2:
        return []
    matrix = np.column_stack([p.values() for p in profiles])
    corr = _corr_matrix(matrix, frame)
    rows, cols = np.triu_indices(len(profiles), k=1)
    r = corr[rows, cols]
    with np.errstate(invalid="ignore"):
        hits = np.nonzero(np.abs(r) > cfg.high_corr)[0]
    pairs = sorted(((abs(float(r[k])), int(rows[k]), int(cols[k])) for k in hits), key=lambda x: (-x[0], x[1], x[2]))
    if len(pairs) > MAX_CORRELATION_PAIRS:
        frame.notes.append(
            f"high_correlation: showing the {MAX_CORRELATION_PAIRS} strongest of {len(pairs):,} correlated pairs"
        )
        pairs = pairs[:MAX_CORRELATION_PAIRS]
    issues: List[Issue] = []
    for _, i, j in pairs:
        a, b = profiles[i], profiles[j]
        value = float(corr[i, j])
        issues.append(
            Issue(
                "warning",
                "high_correlation",
                [a.name, b.name],
                f"'{a.name}' and '{b.name}' are highly correlated (r = {value:+.3f}); keep one of them",
                {"correlation": value},
            )
        )
    return issues


def check_outliers(frame: Frame, cfg: Config) -> List[Issue]:
    """Share of values outside the 1.5 x IQR fences, per numeric column."""
    issues: List[Issue] = []
    for p in frame.numeric_target_scope():
        if p.type != "numeric" or p.n < MIN_ROWS_FOR_STATS or p.n_unique <= 2:
            continue
        values = p.values()[~np.isnan(p.values())]
        finite = _finite(values)
        if len(finite) < MIN_ROWS_FOR_STATS:
            continue
        q1, q3 = np.percentile(finite, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outside = int(((finite < low) | (finite > high)).sum()) + int(len(values) - len(finite))
        share = outside / len(values)
        if share <= cfg.outlier_share:
            continue
        issues.append(
            Issue(
                "warning" if share > 3 * cfg.outlier_share else "info",
                "outliers",
                [p.name],
                f"'{p.name}' has {_pct(share)} outliers by the IQR rule ({outside:,} values outside "
                f"[{low:g}, {high:g}]); check them or clip/winsorize",
                {
                    "share": share,
                    "count": outside,
                    "lower_fence": low,
                    "upper_fence": high,
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                },
            )
        )
    return issues


def check_skew(frame: Frame, cfg: Config) -> List[Issue]:
    """Heavily skewed numeric columns."""
    issues: List[Issue] = []
    for p in frame.numeric_target_scope():
        if p.type != "numeric" or p.n < MIN_ROWS_FOR_STATS or p.n_unique <= 2:
            continue
        finite = _finite(p.values())
        if len(finite) < MIN_ROWS_FOR_STATS:
            continue
        skew = float(pd.Series(finite).skew())
        if not math.isfinite(skew) or abs(skew) <= cfg.skew:
            continue
        hint = "a log or Box-Cox transform" if finite.min() > 0 else "a signed log or quantile transform"
        issues.append(
            Issue(
                "info",
                "skew",
                [p.name],
                f"'{p.name}' is heavily skewed (skew {skew:+.2f}); consider {hint}",
                {"skew": skew, "mean": float(finite.mean()), "median": float(np.median(finite))},
            )
        )
    return issues
