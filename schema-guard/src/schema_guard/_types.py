"""Dtype-family detection and JSON-safe value conversion (internal helpers)."""

from __future__ import annotations

import datetime as _dt
import json
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt

#: The six dtype families a schema column can have.
FAMILIES = ("int", "float", "bool", "datetime", "string", "category")

#: Families that carry a min/max range.
NUMERIC_FAMILIES = ("int", "float")

#: Families that must live in a real (non-object) pandas dtype to count as matching.
TYPED_FAMILIES = ("int", "float", "bool", "datetime")

# What pandas.api.types.infer_dtype says about an object column -> family.
_INFERRED_TO_FAMILY = {
    "string": "string",
    "empty": "string",
    "bytes": "string",
    "integer": "int",
    "floating": "float",
    "mixed-integer-float": "float",
    "decimal": "float",
    "boolean": "bool",
    "datetime": "datetime",
    "datetime64": "datetime",
    "date": "datetime",
}


def family_of(series: pd.Series) -> str:
    """Return the dtype family of a Series.

    Real pandas dtypes are classified directly (nullable Int64/boolean/Float64 and
    tz-aware datetimes included). ``object`` columns are classified by inspecting their
    non-null values, so a column of Python ints is ``"int"`` and a column of ``str`` is
    ``"string"``. Anything unsupported (timedelta, period, interval, ...) falls back to
    ``"string"``.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if pdt.is_bool_dtype(dtype):
        return "bool"
    if pdt.is_integer_dtype(dtype):
        return "int"
    if pdt.is_float_dtype(dtype):
        return "float"
    if pdt.is_datetime64_any_dtype(dtype):
        return "datetime"
    if isinstance(dtype, pd.StringDtype):
        return "string"
    if pdt.is_object_dtype(dtype):
        inferred = pdt.infer_dtype(series, skipna=True)
        return _INFERRED_TO_FAMILY.get(inferred, "string")
    return "string"


def is_object_backed(series: pd.Series) -> bool:
    """True when the Series uses the generic ``object`` dtype."""
    return bool(pdt.is_object_dtype(series.dtype))


def dtype_is_supported(dtype: Any) -> bool:
    """True when ``dtype`` maps cleanly onto one of the six families."""
    if isinstance(dtype, (pd.CategoricalDtype, pd.StringDtype)):
        return True
    return bool(
        pdt.is_bool_dtype(dtype)
        or pdt.is_integer_dtype(dtype)
        or pdt.is_float_dtype(dtype)
        or pdt.is_datetime64_any_dtype(dtype)
        or pdt.is_object_dtype(dtype)
    )


def json_safe(value: Any) -> Any:
    """Convert a scalar to a plain, JSON-serialisable, hashable Python value.

    numpy scalars become ``int``/``float``/``bool``, timestamps become ISO-8601 strings,
    nulls (``None``, ``NaN``, ``pd.NA``, ``NaT``) become ``None``, everything else ``str``.
    ``inf``/``-inf`` also become ``None``: JSON has no literal for them, and a file with a
    bare ``Infinity`` token in it is not JSON that anything but Python will read back.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        stamp = pd.Timestamp(value)
        return None if stamp is pd.NaT else stamp.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def text_value(value: Any) -> str:
    """Render a scalar as the text a ``string`` column should hold.

    Whole numbers read as whole numbers: the float ``1001.0`` becomes ``'1001'``, not
    ``'1001.0'``. A single blank cell upcasts an integer column to ``float64``, so without
    this an id column would be rewritten into strings that join to nothing downstream
    purely because one row was empty, and two batches of identical ids would come out
    different. Everything else is ``str(value)``, unchanged.
    """
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))
    return str(value)


def json_dumps(payload: Any, *, indent: Optional[int] = 2) -> str:
    """Serialise ``payload`` as RFC 8259 JSON text: UTF-8, no ``NaN``/``Infinity`` tokens.

    ``json.dumps`` writes bare ``NaN``/``Infinity`` literals by default, which Python reads
    back but JavaScript, Go, jq and every other parser reject. Non-finite floats are written
    as ``null`` and ``allow_nan=False`` makes any that slip through a loud error, not a file
    that silently cannot be read by anything else.
    """
    return json.dumps(finite_only(payload), indent=indent, ensure_ascii=False, allow_nan=False)


def finite_only(value: Any) -> Any:
    """Deep copy of ``value`` with every non-finite float (``NaN``, ``inf``) replaced by ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_only(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_only(item) for item in value]
    return value


def sorted_values(values: Any) -> list:
    """Sort values deterministically even when their types do not compare with each other."""
    items = list(values)
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=lambda v: (type(v).__name__, str(v)))


def preview(values: Any, limit: int = 5) -> str:
    """Short human-readable preview of a list of values, e.g. 'Rome', 'Lima', ... (+3 more)."""
    items = list(values)
    shown = ", ".join(repr(v) for v in items[:limit])
    extra = len(items) - limit
    return shown + (f", ... (+{extra} more)" if extra > 0 else "")
