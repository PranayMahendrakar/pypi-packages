"""Checks that need no code from the user: schemas, numeric ranges, and the
rules for reading whatever a user-written check returns."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------- dtypes
_TYPE_LABELS = {int: "int", float: "float", str: "str", bool: "bool", bytes: "bytes", object: "object"}


def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


def _is_text(series: pd.Series) -> bool:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    return bool(pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series))


_FAMILY_TESTS = {
    "int": lambda s: bool(pd.api.types.is_integer_dtype(s)),
    "float": lambda s: bool(pd.api.types.is_float_dtype(s)),
    "number": _is_numeric,
    "bool": lambda s: bool(pd.api.types.is_bool_dtype(s)),
    "text": _is_text,
    "category": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
    "datetime": lambda s: bool(pd.api.types.is_datetime64_any_dtype(s)),
    "timedelta": lambda s: bool(pd.api.types.is_timedelta64_dtype(s)),
    "any": lambda s: True,
}

_FAMILY_ALIASES: Dict[str, str] = {"any": "any", "*": "any", "": "any"}
for _family, _aliases in {
    "int": ("int", "ints", "integer", "integers", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "long"),
    "float": ("float", "floats", "double", "float16", "float32", "float64", "real"),
    "number": ("number", "numeric", "num", "numbers"),
    "bool": ("bool", "boolean", "bools", "flag"),
    "text": ("str", "string", "strings", "text", "object", "o"),
    "category": ("category", "categorical", "factor"),
    "datetime": ("datetime", "date", "dates", "timestamp", "datetime64", "datetime64[ns]", "time"),
    "timedelta": ("timedelta", "duration", "timedelta64", "timedelta64[ns]"),
}.items():
    for _alias in _aliases:
        _FAMILY_ALIASES[_alias] = _family


def dtype_label(want: Any) -> str:
    """How a requested dtype is written back to the user."""
    if want is None:
        return "any"
    if isinstance(want, type):
        return _TYPE_LABELS.get(want, want.__name__)
    return str(want)


def dtype_matches(series: pd.Series, want: Any) -> bool:
    """Is `series` of the dtype `want` asks for?

    Dtype names are matched by family, so "int" and "int64" both accept any
    integer width and a schema does not break when it moves between platforms.
    ``None`` (or "any") accepts anything, which is how a schema asks only that
    a column be present.
    """
    if want is None:
        return True
    family = _FAMILY_ALIASES.get(dtype_label(want).strip().lower())
    if family is not None:
        return bool(_FAMILY_TESTS[family](series))
    try:
        return bool(series.dtype == pd.api.types.pandas_dtype(want))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------- schema
def normalize_schema(schema: Any) -> Dict[str, Any]:
    """Turn every accepted schema spelling into ``{column: dtype-or-None}``."""
    if schema is None:
        raise ValueError("expect_schema() needs a schema, got None")
    if isinstance(schema, pd.DataFrame):
        return {str(col): str(schema[col].dtype) for col in schema.columns}
    if isinstance(schema, Mapping):
        out = {str(key): value for key, value in schema.items()}
    elif isinstance(schema, str):
        out = {schema: None}
    elif isinstance(schema, (Sequence, set, frozenset)):
        out = {}
        for item in schema:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                out[str(item[0])] = item[1]
            else:
                out[str(item)] = None
    else:
        raise TypeError(
            "expect_schema() takes a dict of column to dtype, a list of column names, "
            "or a DataFrame to copy the dtypes from, got " + type(schema).__name__
        )
    if not out:
        raise ValueError("expect_schema() needs at least one column")
    return out


def describe_schema(schema: Mapping[str, Any]) -> str:
    """One short line naming the columns a schema asks for."""
    return ", ".join("{0}:{1}".format(name, dtype_label(want)) for name, want in schema.items())


def schema_problems(data: Any, schema: Mapping[str, Any]) -> List[str]:
    """Everything wrong with `data` against `schema`, in reading order."""
    if not isinstance(data, pd.DataFrame):
        return ["expected a DataFrame with columns, got " + type(data).__name__]
    counts = pd.Series(list(data.columns)).value_counts()
    duplicates = sorted(str(name) for name, count in counts.items() if count > 1)
    if duplicates:
        return ["duplicate column names: " + ", ".join(duplicates)]
    problems: List[str] = []
    for name, want in schema.items():
        if name not in data.columns:
            problems.append("column '{0}' is missing".format(name))
            continue
        series = data[name]
        if not dtype_matches(series, want):
            problems.append(
                "column '{0}' has dtype {1}, expected {2}".format(name, series.dtype, dtype_label(want))
            )
    return problems


# ---------------------------------------------------------------------- range
def _as_series(data: Any, column: Optional[str]) -> Tuple[Optional[pd.Series], Optional[str]]:
    """The column to check, or the reason there is not one."""
    if isinstance(data, pd.DataFrame):
        if column is None:
            return None, "expect_range() needs a column name when the data is a table"
        if column not in data.columns:
            known = ", ".join(str(col) for col in data.columns) or "(no columns)"
            return None, "column '{0}' is missing; the data has: {1}".format(column, known)
        selected = data[column]
        if isinstance(selected, pd.DataFrame):
            return None, "column '{0}' appears more than once in the data".format(column)
        return selected, None
    if isinstance(data, pd.Series):
        if column is None or str(data.name) == str(column):
            return data, None
        return None, "the data is a single series named '{0}', not a table with column '{1}'".format(
            data.name, column
        )
    if isinstance(data, Mapping):
        if column is None:
            return None, "expect_range() needs a column name when the data is a mapping"
        if column not in data:
            return None, "key '{0}' is missing from the data".format(column)
        return pd.Series(data[column], name=str(column)), None
    if column is not None:
        return None, "expect_range('{0}') needs a table, got {1}".format(column, type(data).__name__)
    try:
        return pd.Series(np.asarray(data).ravel()), None
    except (TypeError, ValueError):
        return None, "expect_range() cannot read numbers out of " + type(data).__name__


def _is_temporal(series: pd.Series) -> bool:
    """Is this a date or a duration?

    pandas will happily turn either into nanoseconds since the epoch, so they
    have to be caught before ``to_numeric`` quietly makes a number out of a
    value that is not one.
    """
    dtype = series.dtype
    return bool(
        pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype)
    )


def _worst(values: pd.Series, mask: pd.Series, low_side: bool) -> str:
    """' (worst X at row Y)' for the worst offender under `mask`."""
    picked = values[mask]
    if len(picked) == 0:
        return ""
    label = picked.idxmin() if low_side else picked.idxmax()
    number = picked.min() if low_side else picked.max()
    return " (worst {0:.6g} at row {1})".format(float(number), label)


def range_problems(
    data: Any,
    column: Optional[str],
    low: Optional[float],
    high: Optional[float],
) -> Tuple[List[str], Optional[str]]:
    """Range failures, plus a note about anything that could not be checked.

    The note covers both halves of "unusable": values that are present but do
    not read as numbers are counted and named, and a column with no usable
    numbers at all says so. Neither is ever passed over in silence.
    """
    series, reason = _as_series(data, column)
    if series is None:
        return [reason or "the column could not be read"], None
    name = column if column is not None else (series.name if series.name is not None else "the data")
    note: Optional[str] = None
    if _is_numeric(series):
        numbers = series
    elif _is_temporal(series):
        if bool(series.notna().any()):
            return ["column '{0}' is not numeric (dtype {1})".format(name, series.dtype)], None
        return [], "no usable numbers in column '{0}'; nothing to compare against".format(name)
    else:
        numbers = pd.to_numeric(series, errors="coerce")
        if bool(series.notna().any()) and not bool(numbers.notna().any()):
            return ["column '{0}' is not numeric (dtype {1})".format(name, series.dtype)], None
        unreadable = int((series.notna() & numbers.isna()).sum())
        if unreadable:
            note = (
                "column '{0}' has {1} of {2} values that are not numbers; they could not be "
                "range checked".format(name, unreadable, len(series))
            )
    if not bool(numbers.notna().any()):
        return [], "no usable numbers in column '{0}'; nothing to compare against".format(name)
    problems: List[str] = []
    total = len(numbers)
    if low is not None:
        below = numbers < low
        count = int(below.sum())
        if count:
            problems.append(
                "column '{0}' has {1} of {2} values below {3:.6g}".format(name, count, total, low)
                + _worst(numbers, below, True)
            )
    if high is not None:
        above = numbers > high
        count = int(above.sum())
        if count:
            problems.append(
                "column '{0}' has {1} of {2} values above {3:.6g}".format(name, count, total, high)
                + _worst(numbers, above, False)
            )
    return problems, note


# ------------------------------------------------------- user-written checks
def interpret_check_result(raw: Any) -> Tuple[bool, Optional[str]]:
    """Read whatever a check returned as ``(passed, message)``.

    Accepts ``True`` / ``False``, ``(bool, message)``, and a boolean mask of one
    value per row, which passes only when every row passes. The mask may be a
    Series, Index, ndarray, list or tuple; a two-item pair whose first element is a
    boolean and whose second is not is read as ``(passed, message)`` instead.
    """
    # A (passed, message) pair: two items whose first is a boolean and whose second
    # is not, so it cannot be confused with a two-row mask like [False, False].
    # The message is coerced, because a check returning (False, 404) used to fall
    # through every branch and land on bool(raw) -- and a non-empty tuple is truthy,
    # so a failing check silently reported a pass.
    if (
        isinstance(raw, (tuple, list))
        and len(raw) == 2
        and isinstance(raw[0], (bool, np.bool_))
        and not isinstance(raw[1], (bool, np.bool_))
    ):
        message = raw[1]
        return bool(raw[0]), (str(message) if message is not None else None)
    # Any other sequence is a per-row mask. Lists and tuples have to be included:
    # bool([False, False, False]) is True, so a mask where every row failed used to
    # pass. Strings and bytes are deliberately excluded, being sequences by accident.
    if isinstance(raw, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        flags = np.asarray(raw)
        if flags.size == 0:
            return True, "the check looked at no rows"
        try:
            flags = flags.astype(bool)
        except (TypeError, ValueError):
            return False, "the check returned values that are not true or false"
        bad = int(np.count_nonzero(~flags))
        if bad:
            return False, "{0} of {1} rows did not pass".format(bad, flags.size)
        return True, None
    if raw is None:
        return False, "the check returned None; return True, False, or (bool, message)"
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw), None
    try:
        return bool(raw), None
    except (TypeError, ValueError):
        return False, "the check returned {0}; return True, False, or (bool, message)".format(
            type(raw).__name__
        )


def run_check(func: Any, data: Any) -> Tuple[bool, Optional[str]]:
    """Call a user check and never let it escape as a raw exception."""
    try:
        raw = func(data)
    except Exception as exc:  # noqa: BLE001 - a broken check is a failed check, not a crash
        LOG.debug("check raised", exc_info=True)
        return False, "the check raised {0}: {1}".format(type(exc).__name__, exc)
    return interpret_check_result(raw)
