"""The cleaning pipeline: :func:`clean` for one call, :class:`Cleaner` for fit/transform.

Every step is a function that takes the working DataFrame, the options, a
:class:`_State` (being learned during ``fit``, applied during ``transform``)
and the action list, so ``clean``, ``fit_transform`` and ``transform`` share
one code path and always produce the same result on the same data.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import numbers
import re
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ._io import FrameLike, load_frame
from .parsing import (
    FALSE_TOKENS,
    MISSING_TOKENS,
    TRUE_TOKENS,
    infer_dayfirst,
    looks_like_date,
    parse_bool,
    parse_number,
    to_datetime_quiet,
)
from .result import Action, CleanResult

__all__ = ["clean", "Cleaner", "MISSING_MODES", "OUTLIER_MODES", "PARSE_THRESHOLD", "FORWARD_FILL"]

log = logging.getLogger(__name__)

MISSING_MODES = ("auto", "drop", "none")
OUTLIER_MODES = ("clip", "flag", "drop", "none")
#: Share of the non-missing values that must parse before a text column is converted.
PARSE_THRESHOLD = 0.9

_WS_RE = re.compile(r"\s+")
_BOOL_TOKENS = TRUE_TOKENS | FALSE_TOKENS


class _ForwardFill:
    """Marker stored as the learned imputation value of datetime columns."""

    def __repr__(self) -> str:
        return "forward-fill"


FORWARD_FILL = _ForwardFill()


# --------------------------------------------------------------------------- options / state


@dataclass(frozen=True)
class _Options:
    missing: str
    outliers: str
    iqr_factor: float
    duplicates: bool
    normalize_columns: bool
    parse_numbers: bool
    parse_dates: bool
    parse_booleans: bool

    def __post_init__(self) -> None:
        if self.missing not in MISSING_MODES:
            raise ValueError(f"missing must be one of {MISSING_MODES}, got {self.missing!r}")
        if self.outliers not in OUTLIER_MODES:
            raise ValueError(f"outliers must be one of {OUTLIER_MODES}, got {self.outliers!r}")
        factor = self.iqr_factor
        if (
            isinstance(factor, bool)
            or not isinstance(factor, numbers.Real)
            or not math.isfinite(factor)
            or factor < 0
        ):
            raise ValueError(f"iqr_factor must be a non-negative number, got {factor!r}")


@dataclass
class _State:
    """What ``fit`` learns. ``learning`` is True while it is being filled."""

    learning: bool = True
    types: Dict[Hashable, str] = field(default_factory=dict)
    datetime_params: Dict[Hashable, Dict[str, bool]] = field(default_factory=dict)
    empty_columns: List[Hashable] = field(default_factory=list)
    impute: Dict[Hashable, Tuple[str, Any]] = field(default_factory=dict)
    bounds: Dict[Hashable, Tuple[Any, Any]] = field(default_factory=dict)
    #: Columns this package parsed out of text into numbers; only these may be
    #: narrowed back to an integer dtype after imputation.
    numeric_from_text: List[Hashable] = field(default_factory=list)
    #: The dtype each column ended up with during ``fit``. ``transform`` pins
    #: every batch to these so batch N+1 has the same schema as batch N.
    final_dtypes: Dict[Hashable, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- small helpers


def _add(actions: List[Action], action: Action) -> None:
    actions.append(action)
    log.debug("%s", action)


def _colname(col: Hashable) -> str:
    return col if isinstance(col, str) else str(col)


def _check_columns(df: pd.DataFrame) -> None:
    """Reject duplicate column labels up front with a message naming them."""
    columns = df.columns
    if isinstance(columns, pd.MultiIndex) or not columns.has_duplicates:
        return
    seen: Dict[Hashable, int] = {}
    for name in columns:
        seen[name] = seen.get(name, 0) + 1
    repeated = sorted((_colname(n) for n, c in seen.items() if c > 1))
    raise ValueError(
        "duplicate column names are ambiguous: "
        + ", ".join(repr(n) for n in repeated)
        + ". Rename them before cleaning, e.g. "
        "df.columns = [f'{c}_{i}' for i, c in enumerate(df.columns)]."
    )


def _is_default_index(index: pd.Index) -> bool:
    return isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1


def _drop_rows(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Drop the rows where ``mask`` is True; the copy keeps later assignments warning-free."""
    return df[~mask].copy()


def _is_textlike(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    return ptypes.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)


def _is_numeric(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        ptypes.is_numeric_dtype(dtype)
        and not ptypes.is_bool_dtype(dtype)
        and not ptypes.is_complex_dtype(dtype)
        and not ptypes.is_timedelta64_dtype(dtype)
        and not ptypes.is_datetime64_any_dtype(dtype)
    )


def _as_bool(name: str, value: Any) -> bool:
    """Accept only a real boolean, so ``duplicates="no"`` cannot mean True."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be True or False, got {value!r}")


def _finite_values(series: pd.Series) -> np.ndarray:
    """The column's finite float values: no NaN, no inf, no warnings."""
    try:
        values = series.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        return np.empty(0, dtype="float64")
    return values[np.isfinite(values)]


def _has_infinite(series: pd.Series) -> bool:
    try:
        values = series.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        return False
    return bool(np.isinf(values).any())


def _kind_of(series: pd.Series) -> str:
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime"
    if _is_numeric(series):
        return "numeric"
    return "other"


def _is_whole(series: pd.Series) -> bool:
    """True for a float column whose values are all finite whole numbers."""
    if not ptypes.is_float_dtype(series.dtype):
        return False
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    if values.size == 0 or not np.isfinite(values).all():
        return False
    return bool((values == np.floor(values)).all() and np.abs(values).max() < 2**62)


def _to_int(series: pd.Series) -> pd.Series:
    target = "Int64" if isinstance(series.dtype, pd.Float64Dtype) else "int64"
    return series.astype(target)


def _native(value: Any) -> Any:
    """Turn numpy scalars into Python scalars so learned values stay plain."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, _ForwardFill):
        return "forward-fill"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return repr(value)


def _non_null(series: pd.Series) -> Tuple[np.ndarray, pd.Series]:
    mask = series.notna().to_numpy(dtype=bool)
    return mask, series[mask]


# --------------------------------------------------------------------------- step 1: column names


def _normalize_name(name: Hashable) -> Hashable:
    if not isinstance(name, str):
        return name
    return _WS_RE.sub("_", name.strip()).lower()


def _dedupe_names(names: List[Hashable]) -> List[Hashable]:
    seen: Dict[Hashable, int] = {}
    out: List[Hashable] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
            continue
        counter = seen[name]
        while True:
            counter += 1
            candidate = f"{name}_{counter}"
            if candidate not in seen:
                break
        seen[name] = counter
        seen[candidate] = 1
        out.append(candidate)
    return out


def _step_columns(
    df: pd.DataFrame, opts: _Options, actions: List[Action], warnings: List[str]
) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if opts.normalize_columns:
            warnings.append(
                "column names were left alone: this table has a MultiIndex column index, "
                "which cannot be normalized. Flatten it first, e.g. "
                "df.columns = ['_'.join(str(p) for p in c) for c in df.columns]"
            )
        return df
    old = list(df.columns)
    if opts.normalize_columns:
        new: List[Hashable] = []
        for position, name in enumerate(old, start=1):
            normalized = _normalize_name(name)
            if isinstance(normalized, str) and not normalized:
                # a blank or whitespace-only header is unusable as a name
                normalized = f"column_{position}"
            new.append(normalized)
    else:
        new = list(old)
    new = _dedupe_names(new)
    renamed = [(o, n) for o, n in zip(old, new) if not (type(o) is type(n) and o == n)]
    if not renamed:
        return df
    df.columns = new
    for old_name, new_name in renamed:
        _add(actions, Action(_colname(new_name), "rename_column", f"renamed from {old_name!r}", 0))
    return df


# --------------------------------------------------------------------------- step 2: text cells


def _clean_text(series: pd.Series) -> Tuple[pd.Series, int, int]:
    """Strip string cells and turn missing tokens into NaN; returns (series, n_stripped, n_missing)."""
    if isinstance(series.dtype, pd.StringDtype):
        stripped = series.str.strip()
        n_stripped = int((stripped != series).fillna(False).sum())
        missing = stripped.str.lower().isin(MISSING_TOKENS).fillna(False).astype(bool)
        n_missing = int(missing.sum())
        if n_stripped == 0 and n_missing == 0:
            return series, 0, 0
        return stripped.mask(missing.to_numpy()), n_stripped, n_missing
    values = series.to_numpy(dtype=object, copy=True)
    n_stripped = n_missing = 0
    for i, value in enumerate(values):
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text != value:
            n_stripped += 1
        if text.lower() in MISSING_TOKENS:
            values[i] = np.nan
            n_missing += 1
        elif text != value:
            values[i] = text
    if n_stripped == 0 and n_missing == 0:
        return series, 0, 0
    return pd.Series(values, index=series.index, name=series.name, dtype=object), n_stripped, n_missing


def _step_text(df: pd.DataFrame, actions: List[Action]) -> pd.DataFrame:
    for col in list(df.columns):
        series = df[col]
        if not _is_textlike(series):
            continue
        cleaned, n_stripped, n_missing = _clean_text(series)
        if n_stripped == 0 and n_missing == 0:
            continue
        df[col] = cleaned
        if n_stripped:
            _add(
                actions,
                Action(
                    _colname(col),
                    "strip_whitespace",
                    f"stripped surrounding whitespace from {n_stripped} cell(s)",
                    n_stripped,
                ),
            )
        if n_missing:
            _add(
                actions,
                Action(
                    _colname(col),
                    "missing_tokens",
                    f"replaced {n_missing} missing token(s) with NaN",
                    n_missing,
                ),
            )
    return df


# --------------------------------------------------------------------------- steps 3-5: parsing


def _all_bool_tokens(series: pd.Series) -> bool:
    _, non_null = _non_null(series)
    if len(non_null) == 0:
        return False
    for value in non_null:
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, str) and value.strip().lower() in _BOOL_TOKENS:
            continue
        return False
    return True


def _try_numeric(series: pd.Series, *, force: bool) -> Optional[Tuple[pd.Series, int]]:
    mask, non_null = _non_null(series)
    n = len(non_null)
    if n == 0:
        return None
    parsed = [parse_number(v) for v in non_null]
    n_ok = sum(p is not None for p in parsed)
    if not force and (n_ok == 0 or n_ok < PARSE_THRESHOLD * n):
        return None
    values = np.full(len(series), np.nan, dtype="float64")
    values[np.flatnonzero(mask)] = [np.nan if p is None else p for p in parsed]
    out = pd.Series(values, index=series.index, name=series.name)
    if _is_whole(out):
        out = out.astype("int64")
    return out, n - n_ok


def _try_datetime(
    series: pd.Series,
    *,
    force: bool,
    dayfirst: Optional[bool] = None,
    mixed: Optional[bool] = None,
) -> Optional[Tuple[pd.Series, int, Dict[str, bool]]]:
    mask, non_null = _non_null(series)
    n = len(non_null)
    if n == 0:
        return None
    if not force:
        hints = sum(1 for v in non_null if looks_like_date(v))
        if hints < PARSE_THRESHOLD * n:
            return None
    if dayfirst is None:
        dayfirst = infer_dayfirst(non_null)
    modes = [False, True] if mixed is None else [bool(mixed), not mixed]
    candidate = pd.Series(non_null.to_numpy(dtype=object))
    best: Optional[Tuple[pd.Series, int, bool]] = None
    for mode in modes:
        parsed = to_datetime_quiet(candidate, dayfirst=bool(dayfirst), mixed=mode)
        if parsed is None:
            continue
        n_ok = int(parsed.notna().sum())
        if best is None or n_ok > best[1]:
            best = (parsed, n_ok, mode)
        if n_ok == n:
            break
    if best is None:
        return None
    parsed, n_ok, mode = best
    if not force and (n_ok == 0 or n_ok < PARSE_THRESHOLD * n):
        return None
    out = pd.Series(pd.NaT, index=series.index, dtype=parsed.dtype, name=series.name)
    out.iloc[np.flatnonzero(mask)] = parsed.array
    return out, n - n_ok, {"dayfirst": bool(dayfirst), "mixed": bool(mode)}


def _try_boolean(series: pd.Series, *, force: bool) -> Optional[Tuple[pd.Series, int]]:
    mask, non_null = _non_null(series)
    n = len(non_null)
    if n == 0:
        return None
    parsed = [parse_bool(v) for v in non_null]
    n_ok = sum(p is not None for p in parsed)
    if not force and (n_ok == 0 or n_ok < n):
        return None
    values = np.full(len(series), None, dtype=object)
    values[np.flatnonzero(mask)] = parsed
    out = pd.Series(pd.array(values, dtype="boolean"), index=series.index, name=series.name)
    if n_ok == len(series):
        out = out.astype(bool)
    return out, n - n_ok


def _parse_column(
    series: pd.Series, col: Hashable, opts: _Options, state: _State
) -> Optional[Tuple[str, pd.Series, int, Optional[Dict[str, bool]]]]:
    """Return (kind, converted, n_coerced, params) or None when the column stays as it is."""
    if state.learning:
        if opts.parse_numbers and not (opts.parse_booleans and _all_bool_tokens(series)):
            numeric = _try_numeric(series, force=False)
            if numeric is not None:
                return "numeric", numeric[0], numeric[1], None
        if opts.parse_dates:
            dates = _try_datetime(series, force=False)
            if dates is not None:
                return "datetime", dates[0], dates[1], dates[2]
        if opts.parse_booleans:
            booleans = _try_boolean(series, force=False)
            if booleans is not None:
                return "boolean", booleans[0], booleans[1], None
        return None
    kind = state.types.get(col)
    if kind == "numeric" and opts.parse_numbers:
        numeric = _try_numeric(series, force=True)
        return None if numeric is None else ("numeric", numeric[0], numeric[1], None)
    if kind == "datetime" and opts.parse_dates:
        params = state.datetime_params.get(col, {})
        dates = _try_datetime(series, force=True, **params)
        return None if dates is None else ("datetime", dates[0], dates[1], dates[2])
    if kind == "boolean" and opts.parse_booleans:
        booleans = _try_boolean(series, force=True)
        return None if booleans is None else ("boolean", booleans[0], booleans[1], None)
    return None


def _parse_detail(kind: str, converted: pd.Series, params: Optional[Dict[str, bool]], n_coerced: int) -> str:
    if kind == "numeric":
        detail = f"parsed text as numbers ({converted.dtype})"
    elif kind == "datetime":
        order = "day-first" if params and params.get("dayfirst") else "month-first"
        extra = ", mixed formats" if params and params.get("mixed") else ""
        detail = f"parsed text as dates ({order}{extra})"
    else:
        detail = f"parsed text as booleans ({converted.dtype})"
    if n_coerced:
        detail += f"; {n_coerced} unparseable value(s) set to NaN"
    return detail


def _step_parse(df: pd.DataFrame, opts: _Options, state: _State, actions: List[Action]) -> pd.DataFrame:
    for col in list(df.columns):
        series = df[col]
        if not _is_textlike(series):
            if state.learning:
                state.types[col] = _kind_of(series)
            continue
        outcome = _parse_column(series, col, opts, state)
        if outcome is None:
            if state.learning:
                state.types[col] = "other"
            continue
        kind, converted, n_coerced, params = outcome
        df[col] = converted
        if state.learning:
            state.types[col] = kind
            if kind == "numeric" and col not in state.numeric_from_text:
                state.numeric_from_text.append(col)
            if kind == "datetime" and params is not None:
                state.datetime_params[col] = params
        n_cells = int(series.notna().sum())
        _add(actions, Action(_colname(col), f"parse_{kind}", _parse_detail(kind, converted, params, n_coerced), n_cells))
    return df


# --------------------------------------------------------------------------- step 6: duplicates and empties


def _step_drop(df: pd.DataFrame, opts: _Options, state: _State, actions: List[Action]) -> pd.DataFrame:
    if opts.duplicates and len(df) > 0 and df.shape[1] > 0:
        try:
            duplicated = df.duplicated().to_numpy(dtype=bool)
        except TypeError:  # unhashable cells such as lists: compare their text form
            duplicated = df.astype(str).duplicated().to_numpy(dtype=bool)
        n = int(duplicated.sum())
        if n:
            df = _drop_rows(df, duplicated)
            _add(actions, Action(None, "drop_duplicates", f"dropped {n} exact duplicate row(s)", n))
    if len(df) > 0 and df.shape[1] > 0:
        empty_rows = df.isna().all(axis=1).to_numpy(dtype=bool)
        n = int(empty_rows.sum())
        if n:
            df = _drop_rows(df, empty_rows)
            _add(actions, Action(None, "drop_empty_rows", f"dropped {n} row(s) with every value missing", n))
    if state.learning:
        state.empty_columns = []
        if len(df) > 0:
            for col in list(df.columns):
                if df[col].isna().all():
                    state.empty_columns.append(col)
                    df = df.drop(columns=[col])
                    _add(actions, Action(_colname(col), "drop_empty_column", "dropped column: every value missing", int(len(df))))
        return df
    for col in state.empty_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            _add(actions, Action(_colname(col), "drop_empty_column", "dropped column: every value was missing when fitted", int(len(df))))
    if len(df) > 0:
        for col in list(df.columns):
            if col not in state.impute and df[col].isna().all():
                df = df.drop(columns=[col])
                _add(
                    actions,
                    Action(
                        _colname(col),
                        "drop_empty_column",
                        "dropped column: every value missing and no imputation value was learned",
                        int(len(df)),
                    ),
                )
    return df


# --------------------------------------------------------------------------- step 7: missing values


def _mode(non_null: pd.Series) -> Any:
    """Most frequent value; ties go to the value that appears first.

    Returns None when the column holds values pandas cannot count or compare
    (lists, dicts, sets), which is the signal to leave the column alone.
    """
    try:
        counts = non_null.value_counts(dropna=True)
    except (TypeError, ValueError):
        return None
    if counts.empty:
        return None
    try:
        top = counts.iloc[0]
        if top <= 0:
            return None
        winners = list(counts.index[(counts == top).to_numpy()])
        for value in non_null:
            if any(value is w or value == w for w in winners):
                return _native(value)
        return _native(counts.index[0])
    except (TypeError, ValueError):
        # unhashable or non-comparable cells: no usable mode
        return None


def _impute_rule(series: pd.Series) -> Optional[Tuple[str, Any]]:
    """(strategy, value) for a column, learned from its non-missing values."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    dtype = series.dtype
    if ptypes.is_datetime64_any_dtype(dtype):
        return "forward-fill", FORWARD_FILL
    if ptypes.is_bool_dtype(dtype):
        value = _mode(non_null)
        return None if value is None else ("mode", value)
    if ptypes.is_timedelta64_dtype(dtype):
        return "median", non_null.median()
    if _is_numeric(series):
        median = _median(non_null)
        if median is None:
            return None
        if ptypes.is_integer_dtype(dtype) and median.is_integer():
            return "median", int(median)
        return "median", median
    value = _mode(non_null)
    return None if value is None else ("mode", value)


def _median(non_null: pd.Series) -> Optional[float]:
    """Median as a plain float, tolerating extension dtypes such as Sparse."""
    try:
        return float(non_null.median())
    except (TypeError, ValueError):
        pass
    try:  # densify: Sparse and friends answer to_numpy even when median refuses
        values = np.asarray(non_null.to_numpy(), dtype="float64")
    except (TypeError, ValueError):
        return None
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _fill(series: pd.Series, strategy: str, value: Any) -> pd.Series:
    if strategy == "forward-fill":
        return series.ffill().bfill()
    if ptypes.is_object_dtype(series.dtype):
        # Assign positionally rather than through a boolean mask: pandas'
        # putmask cannot broadcast a container (dict, set, list) and raises
        # IndexError from its internals instead of filling the cells.
        missing = np.flatnonzero(series.isna().to_numpy(dtype=bool))
        if missing.size == 0:
            return series
        values = series.to_numpy(dtype=object, copy=True)
        for position in missing:
            values[position] = value
        return pd.Series(values, index=series.index, name=series.name, dtype=object)
    return series.fillna(value)


def _restore_dtype(series: pd.Series, *, int_origin: bool) -> pd.Series:
    """Narrow a just-filled column back to the dtype it should have.

    ``int_origin`` is True only for columns that were integers to begin with or
    that this package parsed out of text into whole numbers; a float64 column
    the caller handed us stays float64 whatever its values happen to be.
    """
    if int_origin and _is_whole(series):
        return _to_int(series)
    if isinstance(series.dtype, pd.BooleanDtype) and not series.isna().any():
        return series.astype(bool)
    return series


def _step_missing(
    df: pd.DataFrame,
    opts: _Options,
    state: _State,
    actions: List[Action],
    warnings: List[str],
) -> pd.DataFrame:
    if opts.missing == "drop":
        if len(df) > 0 and df.shape[1] > 0:
            has_missing = df.isna().any(axis=1).to_numpy(dtype=bool)
            n = int(has_missing.sum())
            if n:
                df = _drop_rows(df, has_missing)
                _add(actions, Action(None, "drop_missing_rows", f"dropped {n} row(s) with at least one missing value", n))
        return df
    if opts.missing == "none":
        return df
    for col in list(df.columns):
        series = df[col]
        if state.learning:
            try:
                rule = _impute_rule(series)
            except (TypeError, ValueError):
                # an exotic extension dtype pandas cannot summarise
                rule = None
            if rule is None:
                if int(series.isna().sum()):
                    message = (
                        f"column {_colname(col)!r}: no usable fill value could be learned; "
                        "its missing values were left as they are"
                    )
                    warnings.append(message)
                continue
            state.impute[col] = rule
        else:
            rule = state.impute.get(col)
            if rule is None:
                continue
        n_missing = int(series.isna().sum())
        if n_missing == 0:
            continue
        strategy, value = rule
        try:
            filled = _fill(series, strategy, value)
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            message = (
                f"column {_colname(col)!r}: could not fill {n_missing} missing value(s) "
                f"with {strategy} {_fmt(value)} ({exc}); left as they are"
            )
            warnings.append(message)
            continue
        int_origin = ptypes.is_integer_dtype(series.dtype) or col in state.numeric_from_text
        df[col] = _restore_dtype(filled, int_origin=int_origin)
        _add(
            actions,
            Action(
                _colname(col),
                "impute",
                f"filled {n_missing} missing value(s) with {strategy}"
                + ("" if strategy == "forward-fill" else f" {_fmt(value)}"),
                n_missing,
            ),
        )
    return df


# --------------------------------------------------------------------------- step 8: outliers


def _iqr_bounds(series: pd.Series, factor: float) -> Optional[Tuple[Any, Any]]:
    # Quantiles are computed on the finite values only. Feeding inf to
    # numpy's quantile interpolation produces inf - inf and leaks a
    # RuntimeWarning out of a dependency; the infinities are still caught by
    # the comparison against the bounds below, which is where they belong.
    finite = _finite_values(series)
    if finite.size < 2:
        return None
    values = pd.Series(finite)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr <= 0:
        return None
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    if ptypes.is_integer_dtype(series.dtype):
        # keep integer columns integer: round the bounds inwards, never past the quartiles
        return int(min(math.ceil(low), math.floor(q1))), int(max(math.floor(high), math.ceil(q3)))
    return low, high


def _step_outliers(
    df: pd.DataFrame,
    opts: _Options,
    state: _State,
    actions: List[Action],
    warnings: List[str],
) -> pd.DataFrame:
    if opts.outliers == "none":
        return df
    to_drop = np.zeros(len(df), dtype=bool)
    for col in list(df.columns):
        series = df[col]
        if not _is_numeric(series):
            continue
        if state.learning:
            bounds = _iqr_bounds(series, opts.iqr_factor)
            if bounds is None:
                if _has_infinite(series):
                    warnings.append(
                        f"column {_colname(col)!r}: contains infinite value(s) and too few "
                        "finite ones to compute IQR bounds; outlier handling was skipped "
                        "for it and the infinities were left in place"
                    )
                continue
            state.bounds[col] = bounds
        else:
            bounds = state.bounds.get(col)
            if bounds is None:
                continue
        low, high = bounds
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            outside = ((series < low) | (series > high)).fillna(False).to_numpy(dtype=bool)
        n = int(outside.sum())
        if n == 0:
            continue
        span = f"[{_fmt(low)}, {_fmt(high)}]"
        if opts.outliers == "clip":
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                df[col] = series.clip(lower=low, upper=high)
            _add(actions, Action(_colname(col), "clip_outliers", f"clipped {n} value(s) to {span}", n))
        elif opts.outliers == "flag":
            _add(actions, Action(_colname(col), "flag_outliers", f"{n} value(s) outside {span} left unchanged", n))
        else:
            to_drop |= outside
            # per-column counts are values, not rows: several columns can flag
            # the same row, so the row count is reported once, below.
            _add(actions, Action(_colname(col), "drop_outliers", f"{n} value(s) outside {span}", n))
    if opts.outliers == "drop":
        n_rows = int(to_drop.sum())
        if n_rows:
            df = _drop_rows(df, to_drop)
            _add(
                actions,
                Action(None, "drop_outlier_rows", f"dropped {n_rows} row(s) holding outlier value(s)", n_rows),
            )
    return df


# --------------------------------------------------------------------------- degenerate results


def _degenerate_warnings(
    input_shape: Tuple[int, int], output_shape: Tuple[int, int]
) -> List[str]:
    """Say so loudly when cleaning left nothing behind, instead of returning a silent empty table."""
    messages: List[str] = []
    if input_shape[0] > 0 and output_shape[0] == 0:
        messages.append(
            f"all {input_shape[0]} row(s) were removed; check the actions above and consider "
            'missing="none" or outliers="flag"'
        )
    if input_shape[1] > 0 and output_shape[1] == 0:
        messages.append(
            f"all {input_shape[1]} column(s) were removed; every column was entirely missing"
        )
    return messages


# --------------------------------------------------------------------------- final dtypes


def _step_finalize(df: pd.DataFrame, state: _State, warnings: List[str]) -> pd.DataFrame:
    """Narrow the nullable dtypes this package introduced, then pin the schema.

    Only columns *we* parsed as booleans are narrowed; a nullable dtype the
    caller supplied (``Int64``, ``boolean``) is theirs and is left alone.

    While fitting, the dtype every column ends up with is recorded. While
    transforming, each column is cast back to the dtype it had at fit time, so
    a production batch cannot come back as int64 or object just because of
    which values that batch happened to contain.
    """
    for col in list(df.columns):
        if state.types.get(col) != "boolean":
            continue
        series = df[col]
        if isinstance(series.dtype, pd.BooleanDtype) and not series.isna().any():
            df[col] = series.astype(bool)
    if state.learning:
        state.final_dtypes = {col: df[col].dtype for col in df.columns}
        return df
    for col in list(df.columns):
        target = state.final_dtypes.get(col)
        if target is None or df[col].dtype == target:
            continue
        pinned = _pin_dtype(df[col], target)
        if pinned is None:
            warnings.append(
                f"column {_colname(col)!r}: this batch is {df[col].dtype}, not the fitted "
                f"{target}; casting it would have changed its values, so the values were kept"
            )
            continue
        df[col] = pinned
    return df


def _pin_dtype(series: pd.Series, target: Any) -> Optional[pd.Series]:
    """``series`` as ``target``, or None when that cast would change a value."""
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            converted = series.astype(target)
            restored = converted.astype(series.dtype)
        missing = series.isna().to_numpy(dtype=bool)
        if not np.array_equal(missing, converted.isna().to_numpy(dtype=bool)):
            return None
        same = (restored == series).to_numpy(dtype=bool) | missing
    except Exception:  # noqa: BLE001 - any dtype pandas cannot round-trip
        return None
    return converted if bool(same.all()) else None


# --------------------------------------------------------------------------- the pipeline


def _run(df_or_path: FrameLike, opts: _Options, state: _State, *, dry_run: bool) -> CleanResult:
    original = load_frame(df_or_path)
    _check_columns(original)
    input_shape = (int(original.shape[0]), int(original.shape[1]))
    default_index = _is_default_index(original.index)
    working = original.copy(deep=True) if dry_run else original
    actions: List[Action] = []
    warnings: List[str] = []
    working = _step_columns(working, opts, actions, warnings)
    if not state.learning and state.types and not (set(state.types) & set(working.columns)):
        warnings.append(
            f"none of the {len(state.types)} fitted column(s) are present in this frame; "
            "nothing that was learned could be applied and only stateless cleanup ran"
        )
    working = _step_text(working, actions)
    working = _step_parse(working, opts, state, actions)
    working = _step_drop(working, opts, state, actions)
    working = _step_missing(working, opts, state, actions, warnings)
    working = _step_outliers(working, opts, state, actions, warnings)
    working = _step_finalize(working, state, warnings)
    if default_index and len(working) != input_shape[0]:
        working = working.reset_index(drop=True)
    output_shape = (int(working.shape[0]), int(working.shape[1]))
    output_dtypes = {_colname(col): str(dtype) for col, dtype in working.dtypes.items()}
    warnings.extend(_degenerate_warnings(input_shape, output_shape))
    for message in warnings:
        log.warning("%s", message)
    log.info("smartclean-df: %d action(s), %s -> %s%s", len(actions), input_shape, output_shape, " (dry run)" if dry_run else "")
    return CleanResult(
        df=original if dry_run else working,
        actions=actions,
        input_shape=input_shape,
        output_shape=output_shape,
        dry_run=dry_run,
        warnings=warnings,
        output_dtypes=output_dtypes,
    )


# --------------------------------------------------------------------------- public API


class Cleaner:
    """Learn cleaning decisions on one table and apply the same ones to others.

    ``fit`` records which text columns parse as numbers / dates / booleans (and
    how), every column's imputation value, the IQR outlier bounds and the
    all-empty columns to drop. ``transform`` applies exactly that to new data.
    Options are the same as :func:`clean`, minus ``dry_run`` (which
    ``transform`` and ``fit_transform`` take per call).
    """

    def __init__(
        self,
        *,
        missing: str = "auto",
        outliers: str = "clip",
        iqr_factor: float = 3.0,
        duplicates: bool = True,
        normalize_columns: bool = True,
        parse_numbers: bool = True,
        parse_dates: bool = True,
        parse_booleans: bool = True,
    ) -> None:
        self._options = _Options(
            missing=missing,
            outliers=outliers,
            iqr_factor=iqr_factor,
            duplicates=_as_bool("duplicates", duplicates),
            normalize_columns=_as_bool("normalize_columns", normalize_columns),
            parse_numbers=_as_bool("parse_numbers", parse_numbers),
            parse_dates=_as_bool("parse_dates", parse_dates),
            parse_booleans=_as_bool("parse_booleans", parse_booleans),
        )
        self._state: Optional[_State] = None

    # -- fitting -------------------------------------------------------------

    def fit(self, df_or_path: FrameLike) -> "Cleaner":
        """Learn parsing, imputation and outlier decisions from ``df_or_path``."""
        state = _State(learning=True)
        _run(df_or_path, self._options, state, dry_run=False)
        state.learning = False
        self._state = state
        return self

    def transform(self, df_or_path: FrameLike, *, dry_run: bool = False) -> CleanResult:
        """Apply what ``fit`` learned to ``df_or_path``."""
        return _run(
            df_or_path, self._options, self._require_state(), dry_run=_as_bool("dry_run", dry_run)
        )

    def fit_transform(self, df_or_path: FrameLike, *, dry_run: bool = False) -> CleanResult:
        """``fit`` and ``transform`` in one pass (what :func:`clean` does)."""
        state = _State(learning=True)
        result = _run(df_or_path, self._options, state, dry_run=_as_bool("dry_run", dry_run))
        state.learning = False
        self._state = state
        return result

    # -- learned state -------------------------------------------------------

    @property
    def options(self) -> Dict[str, Any]:
        """The options this cleaner was built with."""
        return dataclasses.asdict(self._options)

    @property
    def fitted_(self) -> bool:
        """True once ``fit`` or ``fit_transform`` has run."""
        return self._state is not None

    @property
    def column_types_(self) -> Dict[str, str]:
        """Learned type per column: ``numeric``, ``datetime``, ``boolean`` or ``other``."""
        return {_colname(k): v for k, v in self._require_state().types.items()}

    @property
    def datetime_formats_(self) -> Dict[str, Dict[str, bool]]:
        """``{"dayfirst": ..., "mixed": ...}`` per column parsed as dates."""
        return {_colname(k): dict(v) for k, v in self._require_state().datetime_params.items()}

    @property
    def impute_values_(self) -> Dict[str, Any]:
        """Learned fill value per column (``FORWARD_FILL`` for datetime columns)."""
        return {_colname(k): v[1] for k, v in self._require_state().impute.items()}

    @property
    def clip_bounds_(self) -> Dict[str, Tuple[Any, Any]]:
        """Learned ``(low, high)`` outlier bounds per numeric column."""
        return {_colname(k): tuple(v) for k, v in self._require_state().bounds.items()}

    @property
    def empty_columns_(self) -> List[str]:
        """Columns that were entirely missing when fitted; ``transform`` drops them."""
        return [_colname(c) for c in self._require_state().empty_columns]

    def _require_state(self) -> _State:
        if self._state is None:
            raise RuntimeError("this Cleaner is not fitted yet; call fit(df) or fit_transform(df) first")
        return self._state

    def __repr__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in self.options.items())
        return f"Cleaner({opts}, fitted={self.fitted_})"


def clean(
    df_or_path: FrameLike,
    *,
    missing: str = "auto",
    outliers: str = "clip",
    iqr_factor: float = 3.0,
    duplicates: bool = True,
    normalize_columns: bool = True,
    parse_numbers: bool = True,
    parse_dates: bool = True,
    parse_booleans: bool = True,
    dry_run: bool = False,
) -> CleanResult:
    """Clean a DataFrame (or a .csv/.tsv/.parquet path) and report every change.

    Steps, in order: normalize column names; strip text and turn missing
    tokens into NaN; parse text columns that are numbers, dates or booleans;
    drop duplicate rows and all-empty rows/columns; fill missing values
    (``missing="auto"``: median / mode / forward-fill, ``"drop"``: drop the
    rows, ``"none"``: leave them); handle IQR outliers (``outliers="clip"``,
    ``"flag"``, ``"drop"`` or ``"none"``, bounds at ``iqr_factor`` IQRs beyond
    the quartiles). The input is never modified; with ``dry_run=True`` the
    returned ``df`` is an unchanged copy and only the actions are reported.
    """
    cleaner = Cleaner(
        missing=missing,
        outliers=outliers,
        iqr_factor=iqr_factor,
        duplicates=duplicates,
        normalize_columns=normalize_columns,
        parse_numbers=parse_numbers,
        parse_dates=parse_dates,
        parse_booleans=parse_booleans,
    )
    return cleaner.fit_transform(df_or_path, dry_run=dry_run)
