"""Cell-level parsers: missing tokens, numbers, dates and booleans.

Everything here is pure and stateless; the pipeline in :mod:`core` decides
which columns to convert.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import warnings
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "MISSING_TOKENS",
    "TRUE_TOKENS",
    "FALSE_TOKENS",
    "is_na_scalar",
    "is_missing_token",
    "parse_number",
    "parse_bool",
    "looks_like_date",
    "infer_dayfirst",
    "to_datetime_quiet",
    "supports_mixed_format",
]

#: Strings (compared case-insensitively after stripping) that mean "missing".
MISSING_TOKENS = frozenset({"", "na", "n/a", "null", "none", "-", "?", "nan"})

TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0"})

_NUMBER_RE = re.compile(
    r"""^\s*
    (?P<open>\()?\s*
    (?P<sign1>[+-])?\s*
    (?P<currency>[$€£¥₹])?\s*
    (?P<sign2>[+-])?\s*
    (?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)
    (?P<exp>[eE][+-]?\d+)?
    \s*(?P<pct>%)?\s*
    (?P<close>\))?\s*$""",
    re.VERBOSE,
)
_LEADING_ZERO_RE = re.compile(r"^0\d")

_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE_HINT_RE = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"  # 2024-01-05, 05/01/2024, 5.1.2024
    r"|\b" + _MONTHS + r"\s+\d{1,2}"  # Jan 5, January 5 2024
    r"|\d{1,2}\s+" + _MONTHS,  # 5 Jan 2024
    re.IGNORECASE,
)
_DMY_RE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")

_PANDAS_MAJOR = int(pd.__version__.split(".")[0])


def supports_mixed_format() -> bool:
    """True when this pandas accepts ``format="mixed"`` in ``to_datetime``."""
    return _PANDAS_MAJOR >= 2


def is_na_scalar(value: Any) -> bool:
    """True for None, NaN, NaT and pandas.NA; False for everything else."""
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return False


def is_missing_token(value: str) -> bool:
    """True when a string is one of the recognised missing tokens."""
    return value.strip().lower() in MISSING_TOKENS


def parse_number(value: Any) -> Optional[float]:
    """Return a float for a numeric-looking value, or None.

    Handles thousands separators, currency symbols, a trailing ``%`` (stripped,
    value kept as written), accounting negatives ``(300)`` and surrounding
    whitespace. Strings with leading zeros such as ``"00123"`` are treated as
    identifiers and return None.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) else float(value)
    if not isinstance(value, str):
        return None
    match = _NUMBER_RE.match(value)
    if match is None:
        return None
    if bool(match.group("open")) != bool(match.group("close")):
        return None
    if match.group("sign1") and match.group("sign2"):
        return None
    number = match.group("number")
    if _LEADING_ZERO_RE.match(number):
        return None
    try:
        result = float(number.replace(",", "") + (match.group("exp") or ""))
    except ValueError:  # pragma: no cover - the regex already guarantees a float
        return None
    if match.group("sign1") == "-" or match.group("sign2") == "-":
        result = -result
    if match.group("open"):
        result = -result
    return result


def parse_bool(value: Any) -> Optional[bool]:
    """Return True/False for a boolean-looking value, or None."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in TRUE_TOKENS:
            return True
        if token in FALSE_TOKENS:
            return False
    return None


def looks_like_date(value: Any) -> bool:
    """Cheap pre-check: could this cell plausibly be a date?"""
    if isinstance(value, (_dt.date, _dt.datetime, np.datetime64)):
        return True
    return isinstance(value, str) and _DATE_HINT_RE.search(value) is not None


def infer_dayfirst(values: Iterable[Any]) -> bool:
    """Infer day-first ordering from numeric ``a/b/y`` strings.

    Day-first when some first component exceeds 12 and no second component
    does; month-first (pandas' default) otherwise.
    """
    first_over_12 = second_over_12 = False
    for value in values:
        if not isinstance(value, str):
            continue
        match = _DMY_RE.match(value)
        if match is None:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12:
            first_over_12 = True
        if second > 12:
            second_over_12 = True
    return first_over_12 and not second_over_12


def to_datetime_quiet(
    series: pd.Series, *, dayfirst: bool = False, mixed: bool = False
) -> Optional[pd.Series]:
    """``pd.to_datetime(errors="coerce")`` with every pandas warning silenced.

    Returns None when pandas cannot produce a datetime64 column at all.
    """
    kwargs = {"errors": "coerce", "dayfirst": dayfirst}
    if mixed:
        if not supports_mixed_format():
            return None
        kwargs["format"] = "mixed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, **kwargs)
        except (ValueError, TypeError, OverflowError):
            return None
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        return None
    return parsed
