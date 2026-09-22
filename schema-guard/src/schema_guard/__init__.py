"""schema-guard: infer a DataFrame schema once, then validate or enforce it forever.

    schema = schema_guard.infer(train_df)      # learn the shape of good data
    schema.save("schema.json")                 # keep it next to the model

    result = schema.validate(new_df)           # report what changed (never modifies new_df)
    fixed = schema.enforce(new_df)             # coerce dtypes, reorder, drop extras, fill missing
"""

from __future__ import annotations

from typing import Any, Union

import pandas as pd

from ._types import FAMILIES
from .guard import guard
from .result import PROBLEM_KINDS, Problem, SchemaError, ValidationResult
from .schema import ColumnSpec, Schema, as_schema

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "infer",
    "load",
    "validate",
    "enforce",
    "guard",
    "Schema",
    "ColumnSpec",
    "as_schema",
    "ValidationResult",
    "Problem",
    "SchemaError",
    "FAMILIES",
    "PROBLEM_KINDS",
]


def infer(
    df: Any,
    *,
    categorical_max_unique: int = 50,
    numeric_ranges: bool = True,
    nullable: Union[str, bool] = "observed",
) -> Schema:
    """Learn a :class:`Schema` from a DataFrame or a ``.csv``/``.parquet`` path (see :meth:`Schema.infer`)."""
    return Schema.infer(
        df,
        categorical_max_unique=categorical_max_unique,
        numeric_ranges=numeric_ranges,
        nullable=nullable,
    )


def load(path: Any) -> Schema:
    """Read a schema saved with :meth:`Schema.save`."""
    return Schema.load(path)


def validate(df: Any, schema: Any, *, strict_ranges: bool = False) -> ValidationResult:
    """Validate ``df`` against ``schema`` (a :class:`Schema`, a schema dict or a saved JSON path)."""
    return as_schema(schema).validate(df, strict_ranges=strict_ranges)


def enforce(
    df: Any,
    schema: Any,
    *,
    mode: str = "coerce",
    extra: str = "drop",
    missing: str = "fill",
) -> pd.DataFrame:
    """Enforce ``schema`` on ``df`` and return the new frame (see :meth:`Schema.enforce`)."""
    return as_schema(schema).enforce(df, mode=mode, extra=extra, missing=missing)
