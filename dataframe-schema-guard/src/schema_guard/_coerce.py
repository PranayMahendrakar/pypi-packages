"""Column coercion used by :meth:`Schema.enforce` (internal helpers).

Every ``_to_<family>`` function takes a Series and the target :class:`ColumnSpec` and
returns a new Series in that family, or raises :class:`CoercionError` carrying a
:class:`Problem` that names the column, the target family and example bad values.
Nothing here mutates its input.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pandas.errors import OutOfBoundsDatetime

from ._types import family_of, is_object_backed, json_safe, text_value
from .result import Problem

if TYPE_CHECKING:  # pragma: no cover
    from .schema import ColumnSpec

logger = logging.getLogger(__name__)

_PANDAS_MAJOR = int(pd.__version__.split(".")[0])

_TRUE_WORDS = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE_WORDS = frozenset({"false", "f", "no", "n", "0", "0.0"})

#: A stored category that looks like an ISO date or timestamp, e.g. '2024-01-01T00:00:00'.
_ISO_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ].*)?$")

#: Sentinel for "no alias", so a stored category of ``None`` is still distinguishable.
_NO_ALIAS = object()


class CoercionError(Exception):
    """A column could not be coerced; ``problem`` explains why."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        super().__init__(str(problem))


def coerce_column(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    """Return ``series`` converted to ``spec.family`` (unchanged when it already matches)."""
    family = family_of(series)
    is_object = is_object_backed(series)
    target = spec.family
    if target == "int":
        return series if (family == "int" and not is_object) else _to_int(series, spec)
    if target == "float":
        return series if (family == "float" and not is_object) else _to_float(series, spec)
    if target == "bool":
        return series if (family == "bool" and not is_object) else _to_bool(series, spec)
    if target == "datetime":
        parsed = series if (family == "datetime" and not is_object) else _to_datetime(series, spec)
        return apply_timezone(parsed, spec)
    if target == "string":
        return series if family == "string" else _to_string(series, spec)
    if target == "category":
        return _to_category(series, spec)
    raise ValueError(f"unknown family {target!r}")  # pragma: no cover - ColumnSpec validates


def null_column(spec: "ColumnSpec", length: int) -> pd.Series:
    """An all-null Series of ``length`` rows in the right dtype for ``spec.family``."""
    family = spec.family
    if family == "int":
        out = pd.Series([pd.NA] * length, dtype="Int64")
    elif family == "float":
        out = pd.Series(np.full(length, np.nan), dtype="float64")
    elif family == "bool":
        out = pd.Series([pd.NA] * length, dtype="boolean")
    elif family == "datetime":
        # A filled-in column must carry the schema's timezone too, or a batch missing the
        # column would come out with a different dtype from one that has it.
        dtype: Any = "datetime64[ns]" if spec.tz is None else pd.DatetimeTZDtype(tz=spec.tz)
        out = pd.Series(pd.array([pd.NaT] * length, dtype=dtype))
    elif family == "category":
        out = pd.Series(pd.Categorical([np.nan] * length, categories=spec.categories or []))
    else:
        out = pd.Series([np.nan] * length, dtype=object)
    out.name = spec.name
    return out


# --------------------------------------------------------------------------- helpers


def _fail(
    spec: "ColumnSpec",
    series: pd.Series,
    reason: str,
    bad_values: Optional[List[Any]] = None,
    bad_count: Optional[int] = None,
) -> CoercionError:
    detail: dict = {
        "expected": spec.family,
        "actual_dtype": str(series.dtype),
        "actual_family": family_of(series),
    }
    message = f"cannot coerce to {spec.family}: {reason}"
    if bad_values is not None:
        examples = [json_safe(v) for v in bad_values[:10]]
        detail["bad_values"] = examples
        detail["bad_count"] = int(bad_count if bad_count is not None else len(bad_values))
        shown = ", ".join(repr(v) for v in examples[:5])
        message += f" (e.g. {shown})"
    return CoercionError(Problem("dtype_mismatch", spec.name, message, detail))


def _plain(series: pd.Series) -> pd.Series:
    """Categoricals as their underlying values; everything else untouched."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.astype(object)
    return series


def apply_timezone(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    """Put a datetime Series in ``spec.tz`` so every batch enforces to the same dtype.

    A tz-naive batch is localised to the schema's timezone, a differently-zoned one is
    converted to it (the instant is preserved), and a tz-aware batch for a naive schema is
    converted to UTC before the offset is dropped. Without this, two batches enforced by
    one schema come out with different dtypes and ``pd.concat`` of them degrades to object.
    """
    target = spec.tz
    if not pdt.is_datetime64_any_dtype(series.dtype):
        # Mixed offsets parse to an object column of Timestamps; UTC gives them one dtype.
        series = pd.to_datetime(series, errors="coerce", utc=True)
    current = getattr(series.dtype, "tz", None)
    if current is None and target is None:
        return series
    try:
        if current is None:
            return series.dt.tz_localize(target)
        if target is None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
        return series.dt.tz_convert(target)
    except Exception as exc:  # pytz raises AmbiguousTimeError/NonExistentTimeError, not ValueError
        raise _fail(spec, series, f"cannot put these datetimes in {_tz_label(target)} ({exc})") from exc


def _tz_label(tz: Optional[str]) -> str:
    return "no timezone (tz-naive)" if tz is None else f"timezone {tz!r}"


def _number_from_text(text: str) -> Optional[Any]:
    """The finite number a stored text category names, or ``None`` when it names none."""
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _category_aliases(allowed: List[Any]) -> Dict[Any, Any]:
    """Other JSON forms of each stored category, mapped back onto the stored value.

    A CSV reader decides on its own whether a zip code column is text or ``int64``, so the
    same category arrives as ``'1001'`` in one batch and as ``1001`` - or ``1001.0``, once
    a blank cell upcasts the column - in the next. Those are one category, and they have to
    land on one code: category codes go straight into a model, so two batches of identical
    data that produce different codes are exactly the break this package exists to stop.
    """
    aliases: Dict[Any, Any] = {}
    for stored in allowed:
        if isinstance(stored, bool):
            continue
        if isinstance(stored, str):
            number = _number_from_text(stored)
            # Only when the two name each other exactly. A zip code '02134' parses as 2134,
            # but the leading zero is the point of storing it as text, and 2134 is a
            # different category - so that one stays unknown rather than being absorbed.
            if number is not None and text_value(number) == stored:
                # int 1001 and float 1001.0 are the same dict key, so one entry covers both.
                aliases.setdefault(number, stored)
        elif isinstance(stored, (int, float)):
            as_text = text_value(stored)
            if _number_from_text(as_text) == stored:
                aliases.setdefault(as_text, stored)
    return aliases


def category_key(allowed: Optional[List[Any]]) -> Callable[[Any], Any]:
    """Return the function that maps an incoming value onto its stored category.

    A :class:`ColumnSpec` stores categories as JSON-safe values (a ``Timestamp`` becomes an
    ISO string, ``bytes`` become text), so raw incoming values have to be put through the
    same conversion or nothing matches. A value that names a stored category in another JSON
    type matches it too: ``1001`` and ``1001.0`` both find the stored ``'1001'``, and
    ``'1001'`` finds a stored ``1001``. When the stored set is ISO timestamps, a value that
    parses as a datetime is matched on its ISO form as well, so the same date gets the same
    category code whether it arrives as a ``Timestamp`` or as ``'2024-01-01'``.

    :meth:`Schema.validate` and :meth:`Schema.enforce` both go through this, so the two
    never disagree about which values are unknown.
    """
    known = set(allowed or [])
    datetime_like = bool(known) and all(isinstance(v, str) and _ISO_LIKE.match(v) for v in known)
    aliases = _category_aliases(list(allowed or []))

    def key(value: Any) -> Any:
        plain = json_safe(value)
        if plain in known:
            return plain
        if aliases and plain is not None and not isinstance(plain, bool):
            stored = aliases.get(plain, _NO_ALIAS)
            if stored is not _NO_ALIAS:
                return stored
        if not datetime_like:
            return plain
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return plain
        if stamp is pd.NaT:
            return plain
        iso = stamp.isoformat()
        return iso if iso in known else plain

    return key


def _real_part(base: pd.Series, spec: "ColumnSpec", series: pd.Series) -> pd.Series:
    """The real part of a complex column, or a failure when that would throw half of it away.

    numpy casts complex to float by dropping the imaginary part and warning about it from
    library code. Halving every value is exactly the silent loss the rest of this module
    refuses to do, so a column that actually carries imaginary parts is a hard failure.
    """
    values = base.to_numpy(dtype="complex128")
    imaginary = np.zeros(len(values), dtype=bool)
    known = ~np.isnan(values.imag)
    imaginary[known] = values.imag[known] != 0
    if imaginary.any():
        count = int(imaginary.sum())
        raise _fail(
            spec,
            series,
            f"{count} value(s) are complex; converting them would discard the imaginary part",
            list(pd.unique(base[imaginary])),
            count,
        )
    return pd.Series(values.real, index=base.index, name=base.name)


def _as_numeric(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    """Parse text/object values as numbers; raise naming the values that do not parse."""
    base = _plain(series)
    if pdt.is_complex_dtype(base.dtype):
        base = _real_part(base, spec, series)
    if pdt.is_numeric_dtype(base.dtype) and not is_object_backed(base):
        return base
    numeric = pd.to_numeric(base, errors="coerce")
    bad = base.notna() & numeric.isna()
    if bad.any():
        count = int(bad.sum())
        raise _fail(spec, series, f"{count} value(s) are not numeric", list(pd.unique(base[bad])), count)
    return numeric


def _whole_numbers(numeric: pd.Series, spec: "ColumnSpec", series: pd.Series) -> None:
    """Raise unless every non-null value is a finite whole number."""
    nonnull = numeric.dropna()
    if len(nonnull) == 0:
        return
    values = nonnull.to_numpy(dtype="float64")
    finite = np.isfinite(values)
    bad = ~finite | (values != np.floor(np.where(finite, values, 0.0)))
    if bad.any():
        count = int(bad.sum())
        raise _fail(spec, series, f"{count} value(s) are not whole numbers", list(pd.unique(nonnull[bad])), count)


def _bool_series(data: np.ndarray, mask: np.ndarray, like: pd.Series) -> pd.Series:
    if mask.any():
        values: Any = pd.arrays.BooleanArray(np.asarray(data, dtype=bool), np.asarray(mask, dtype=bool))
    else:
        values = np.asarray(data, dtype=bool)
    return pd.Series(values, index=like.index, name=like.name)


# --------------------------------------------------------------------------- families


def _to_int(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    family = family_of(series)
    if family == "datetime":
        raise _fail(spec, series, "datetime values cannot become integers")
    if family == "bool":
        flags = _plain(series).astype("boolean")
        return flags.astype("Int64") if flags.isna().any() else flags.astype("int64")
    numeric = _as_numeric(series, spec)
    if pdt.is_integer_dtype(numeric.dtype):
        return numeric
    _whole_numbers(numeric, spec, series)
    if numeric.isna().any():
        return numeric.astype("Int64")
    return numeric.astype("int64")


def _to_float(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    family = family_of(series)
    if family == "datetime":
        raise _fail(spec, series, "datetime values cannot become floats")
    if family == "bool":
        return _plain(series).astype("boolean").astype("float64")
    numeric = _as_numeric(series, spec)
    if pdt.is_float_dtype(numeric.dtype):
        return numeric
    return numeric.astype("float64")


def _to_bool(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    family = family_of(series)
    if family == "datetime":
        raise _fail(spec, series, "datetime values cannot become booleans")
    base = _plain(series)
    if family == "bool":
        flags = base.astype("boolean")
        return flags if flags.isna().any() else flags.astype(bool)
    if family in ("int", "float"):
        numeric = _as_numeric(base, spec)
        nonnull = numeric.dropna()
        values = nonnull.to_numpy(dtype="float64")
        bad = ~((values == 0.0) | (values == 1.0))
        if bad.any():
            count = int(bad.sum())
            raise _fail(spec, series, f"{count} value(s) are not 0/1", list(pd.unique(nonnull[bad])), count)
        mask = numeric.isna().to_numpy(dtype=bool)
        data = np.nan_to_num(numeric.to_numpy(dtype="float64", na_value=np.nan)) == 1.0
        return _bool_series(data, mask, series)
    text = base.astype(object).map(lambda v: str(v).strip().lower(), na_action="ignore")
    is_true = text.isin(_TRUE_WORDS)
    is_false = text.isin(_FALSE_WORDS)
    is_null = base.isna()
    bad = ~(is_true | is_false | is_null)
    if bad.any():
        count = int(bad.sum())
        raise _fail(
            spec,
            series,
            f"{count} value(s) are not recognised booleans (true/false, yes/no, 1/0)",
            list(pd.unique(base[bad])),
            count,
        )
    return _bool_series(is_true.to_numpy(dtype=bool), is_null.to_numpy(dtype=bool), series)


def _parse_datetimes(base: pd.Series, **options: Any) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Parse ``base`` as datetimes; return the result and the mask of values that failed.

    pandas warns - and has said it will one day raise - when one batch mixes UTC offsets.
    That batch is the shape :func:`apply_timezone` exists to normalise, so the warning is
    noise from library code the caller did nothing wrong to trigger, and the future error is
    handled by retrying the way pandas will then require, rather than letting a raw
    ``ValueError`` escape :meth:`Schema.enforce` in place of a :class:`SchemaError`.
    Returns ``None`` when this set of options cannot parse the column at all.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            parsed = pd.to_datetime(base, errors="coerce", **options)
        except (ValueError, TypeError, OverflowError):
            if options.get("utc"):
                return None
            try:
                parsed = pd.to_datetime(base, errors="coerce", utc=True, **options)
            except (ValueError, TypeError, OverflowError):
                return None
    return parsed, base.notna() & parsed.isna()


def _out_of_ns_range(value: Any) -> bool:
    """True when ``value`` is a real date a datetime64[ns] column cannot hold.

    pandas 2 parses '3000-01-01' happily on its own (at second resolution) but coerces it
    to ``NaT`` inside a nanosecond column, so the bound has to be checked rather than the
    parse, or the value looks unparseable when it is merely out of range.
    """
    try:
        stamp = pd.Timestamp(value)
    except OutOfBoundsDatetime:
        return True
    except (TypeError, ValueError, OverflowError):
        return False
    if stamp is pd.NaT:
        return False
    try:
        naive = stamp.tz_convert(None) if stamp.tzinfo is not None else stamp
        return not (pd.Timestamp.min <= naive <= pd.Timestamp.max)
    except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
        return False


def _datetime_failure(
    spec: "ColumnSpec", series: pd.Series, base: pd.Series, bad: pd.Series
) -> CoercionError:
    """The failure for datetimes that did not parse, naming the reason that is actually true."""
    values = list(pd.unique(base[bad]))
    count = int(bad.sum())
    if values and all(_out_of_ns_range(value) for value in values):
        # '3000-01-01' is a perfectly good date; calling it unparseable sends the user
        # hunting for a data-entry bug that is not there.
        reason = f"{count} value(s) are outside the dates pandas can represent (1677-09-21 .. 2262-04-11)"
    else:
        reason = f"{count} value(s) are not parseable datetimes"
    return _fail(spec, series, reason, values, count)


def _to_datetime(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    family = family_of(series)
    if family in ("int", "float", "bool"):
        raise _fail(
            spec,
            series,
            "numeric values are ambiguous as datetimes (seconds? milliseconds?); "
            "convert them with pandas.to_datetime(..., unit=...) first",
        )
    base = _plain(series)
    best = _parse_datetimes(base)
    if best is None or best[1].any():
        retries: List[Dict[str, Any]] = []
        if _PANDAS_MAJOR >= 2:
            # pandas 2 infers one format from the first value; retry allowing mixed formats.
            retries.append({"format": "mixed"})
        # A column holding both tz-aware and tz-naive values - what pd.concat of a zoned and
        # a naive frame leaves behind - coerces the aware ones to NaT unless utc=True.
        # apply_timezone puts the column in the schema's timezone either way.
        retries.append({"utc": True})
        if _PANDAS_MAJOR >= 2:
            retries.append({"format": "mixed", "utc": True})
        for options in retries:
            other = _parse_datetimes(base, **options)
            if other is None:
                continue
            if best is None or int(other[1].sum()) < int(best[1].sum()):
                best = other
                if not best[1].any():
                    break
    if best is None:  # pragma: no cover - pandas refused every parse of this column
        nonnull = base.dropna()
        raise _fail(
            spec,
            series,
            f"{len(nonnull)} value(s) are not parseable datetimes",
            list(pd.unique(nonnull)),
            len(nonnull),
        )
    parsed, bad = best
    if bad.any():
        raise _datetime_failure(spec, series, base, bad)
    return parsed


def _to_string(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    base = _plain(series)
    is_null = base.isna().to_numpy(dtype=bool)
    if pdt.is_float_dtype(base.dtype) or pdt.is_object_dtype(base.dtype):
        # astype(str) writes float64 as '1001.0'. One blank cell upcasts an id column to
        # float64, so that would rewrite every id in the batch into something that matches
        # nothing in a join, a lookup or an encoder - and two batches of the same ids would
        # come out different because one of them happened to have a gap.
        text = np.array([text_value(value) for value in base.to_numpy(dtype=object)], dtype=object)
    else:
        text = base.astype(str).to_numpy(dtype=object)
    values = np.where(is_null, np.nan, text)
    return pd.Series(values, index=series.index, name=series.name, dtype=object)


def _to_category(series: pd.Series, spec: "ColumnSpec") -> pd.Series:
    allowed = spec.categories
    if isinstance(series.dtype, pd.CategoricalDtype):
        if allowed is None:
            return series
        # Compare the RAW categories: a set that only matches once converted (Timestamps,
        # bytes) still has to go through the mapping below, or this batch keeps raw values
        # while the next one gets the stored form and the two frames disagree.
        if list(series.cat.categories) == list(allowed):
            return series
    base = _plain(series)
    if allowed is None:
        return base.astype("category")
    # The spec holds JSON-safe categories, so map the incoming values the same way before
    # matching. Comparing raw values against converted ones matches nothing, and every row
    # would silently become null.
    key = category_key(allowed)
    mapped = pd.Series(
        [key(value) for value in base], index=base.index, name=base.name, dtype=object
    )
    known = set(allowed)
    unseen = [v for v in pd.unique(mapped.dropna()) if v not in known]
    if unseen:
        logger.debug(
            "column %r: %d value(s) outside the schema categories kept as new categories", spec.name, len(unseen)
        )
    categorical = pd.Categorical(mapped, categories=list(allowed) + unseen)
    return pd.Series(categorical, index=series.index, name=series.name)
