""":class:`Schema` and :class:`ColumnSpec`: infer, save/load, validate and enforce."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._coerce import CoercionError, category_key, coerce_column, null_column
from ._io import check_columns, check_file, is_supported_label, read_tabular
from ._types import (
    FAMILIES,
    NUMERIC_FAMILIES,
    TYPED_FAMILIES,
    dtype_is_supported,
    family_of,
    is_object_backed,
    json_dumps,
    json_safe,
    preview,
    sorted_values,
)
from .result import Problem, SchemaError, ValidationResult

logger = logging.getLogger(__name__)

#: Version of the JSON layout written by :meth:`Schema.save`.
SCHEMA_FORMAT = 1

_NULLABLE_MODES = ("observed", "always", "never")
_COLUMN_KEYS = ("name", "family", "nullable", "categories", "min", "max", "tz")


def _bound(value: Any, label: str) -> Optional[Union[int, float]]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a number or None, got a bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # NaN and +/-inf are not bounds and have no JSON literal: "no bound" is the honest record.
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    raise TypeError(f"{label} must be a number or None, got {type(value).__name__}")


def _column_name(name: Any) -> Union[str, int]:
    """Column names are stored in JSON, so only strings and integers can be kept faithfully."""
    if isinstance(name, str):
        return str(name)
    if is_supported_label(name):
        return int(name)
    raise TypeError(
        f"column name must be a string or an int, got {type(name).__name__} ({name!r}); "
        "convert the frame's labels first, e.g. df.columns = df.columns.map(str)"
    )


def _timezone(tz: Any, name: Any, family: str) -> Optional[str]:
    """Normalise a timezone to the string pandas round-trips (``'UTC'``, ``'Europe/Oslo'``)."""
    if tz is None:
        return None
    if family != "datetime":
        raise ValueError(f"column {name!r}: tz only applies to the datetime family, not {family!r}")
    label = tz if isinstance(tz, str) else str(tz)
    try:
        pd.DatetimeTZDtype(unit="ns", tz=label)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"column {name!r}: {label!r} is not a known timezone") from exc
    return label


def _series_tz(series: pd.Series) -> Optional[str]:
    """The timezone of a datetime column as a string, or None when it is naive."""
    tz = getattr(series.dtype, "tz", None)
    return None if tz is None else str(tz)


def _tz_text(tz: Optional[str]) -> str:
    """How a timezone reads in a problem message."""
    return "no timezone (tz-naive)" if tz is None else f"timezone {tz!r}"


@dataclass
class ColumnSpec:
    """What one column must look like.

    ``family`` is one of ``int``, ``float``, ``bool``, ``datetime``, ``string`` or
    ``category``. ``categories`` lists the allowed values of a text/category column
    (``None`` means "anything goes"). ``min``/``max`` bound a numeric column; values
    outside them are reported as ``out_of_range`` warnings. ``tz`` is the timezone a
    ``datetime`` column must be in (``None`` means tz-naive). Everything is stored as
    plain JSON-safe Python values, so a spec survives ``save``/``load`` unchanged.

    ``name`` must be a string or an integer - the two label types JSON keeps faithfully.
    """

    name: Any
    family: str
    nullable: bool = True
    categories: Optional[List[Any]] = None
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    tz: Optional[str] = None

    def __post_init__(self) -> None:
        self.name = _column_name(self.name)
        if self.family not in FAMILIES:
            raise ValueError(f"column {self.name!r}: unknown family {self.family!r}; expected one of {FAMILIES}")
        if not isinstance(self.nullable, (bool, np.bool_)):
            raise TypeError(f"column {self.name!r}: nullable must be True or False")
        self.nullable = bool(self.nullable)
        if self.categories is not None:
            if isinstance(self.categories, (str, bytes)) or not isinstance(self.categories, Iterable):
                raise TypeError(f"column {self.name!r}: categories must be a list of values or None")
            cleaned = [json_safe(v) for v in self.categories]
            self.categories = list(dict.fromkeys(v for v in cleaned if v is not None))
        self.min = _bound(self.min, f"column {self.name!r}: min")
        self.max = _bound(self.max, f"column {self.name!r}: max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"column {self.name!r}: min ({self.min}) is greater than max ({self.max})")
        self.tz = _timezone(self.tz, self.name, self.family)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict with ``name``, ``family``, ``nullable``, ``categories``, ``min``, ``max``.

        ``datetime`` columns carry a ``tz`` key as well; no other family can have one.
        """
        data: Dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "nullable": self.nullable,
            "categories": None if self.categories is None else list(self.categories),
            "min": self.min,
            "max": self.max,
        }
        if self.family == "datetime":
            data["tz"] = self.tz
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSpec":
        """Build a spec from a dict such as ``to_dict()`` returns (unknown keys are rejected)."""
        if not isinstance(data, dict):
            raise TypeError(f"column spec must be a dict, got {type(data).__name__}")
        unknown = sorted(set(data) - set(_COLUMN_KEYS))
        if unknown:
            raise ValueError(f"column spec has unknown keys {unknown}; expected only {list(_COLUMN_KEYS)}")
        for key in ("name", "family"):
            if key not in data:
                raise ValueError(f"column spec is missing the required key {key!r}")
        return cls(
            name=data["name"],
            family=data["family"],
            nullable=data.get("nullable", True),
            categories=data.get("categories"),
            min=data.get("min"),
            max=data.get("max"),
            tz=data.get("tz"),
        )


@dataclass(repr=False)
class Schema:
    """An ordered list of :class:`ColumnSpec` describing what a DataFrame must look like.

    Build one with :meth:`Schema.infer`, keep it with :meth:`save`/:meth:`load`, then use
    :meth:`validate` to report problems and :meth:`enforce` to fix them. Two schemas are
    equal when every column spec is equal, and ``load(save(schema)) == schema`` always holds.
    """

    columns: List[ColumnSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        specs: List[ColumnSpec] = []
        for column in self.columns:
            if isinstance(column, ColumnSpec):
                specs.append(column)
            elif isinstance(column, dict):
                specs.append(ColumnSpec.from_dict(column))
            else:
                raise TypeError(f"columns must be ColumnSpec or dict, got {type(column).__name__}")
        names = [spec.name for spec in specs]
        duplicates = sorted({str(n) for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate column names in schema: {', '.join(duplicates)}")
        self.columns = specs

    # ------------------------------------------------------------------ container protocol

    @property
    def column_names(self) -> List[Any]:
        """Column names in schema order."""
        return [spec.name for spec in self.columns]

    def __len__(self) -> int:
        return len(self.columns)

    def __iter__(self) -> Iterator[ColumnSpec]:
        return iter(self.columns)

    def __contains__(self, name: object) -> bool:
        return any(spec.name == name for spec in self.columns)

    def __getitem__(self, name: Any) -> ColumnSpec:
        for spec in self.columns:
            if spec.name == name:
                return spec
        raise KeyError(f"column {name!r} is not in the schema; columns are {self.column_names}")

    def __repr__(self) -> str:
        inner = ", ".join(f"{spec.name}:{spec.family}" for spec in self.columns)
        return f"Schema([{inner}])"

    def select(self, *names: Any) -> "Schema":
        """A new schema with only ``names``, in the order given."""
        return Schema([self[name] for name in names])

    def drop(self, *names: Any) -> "Schema":
        """A new schema without ``names`` (a name that is not in the schema raises ``KeyError``)."""
        for name in names:
            if name not in self:
                raise KeyError(f"column {name!r} is not in the schema; columns are {self.column_names}")
        return Schema([spec for spec in self.columns if spec.name not in names])

    # ------------------------------------------------------------------ inference

    @classmethod
    def infer(
        cls,
        df: Any,
        *,
        categorical_max_unique: int = 50,
        numeric_ranges: bool = True,
        nullable: Union[str, bool] = "observed",
    ) -> "Schema":
        """Learn a schema from a DataFrame (or a ``.csv``/``.parquet`` path).

        For every column it records the dtype family, whether nulls are allowed, the
        allowed categories of text/category columns with at most ``categorical_max_unique``
        distinct values (``0`` disables this; a text column with no values records none),
        the timezone of datetime columns, and the observed ``min``/``max`` of numeric
        columns when ``numeric_ranges`` is True. Column order is part of the schema.

        A text column whose values are *all distinct* records no categories: every value
        being unique is evidence of an identifier, not of a closed set, and freezing one
        would fail every later batch. The decision is logged at WARNING level.

        ``nullable`` is ``"observed"`` (nullable only if a null was seen; the default),
        ``"always"`` (every column nullable) or ``"never"`` (no column nullable).
        ``True``/``False`` are accepted as aliases for ``"always"``/``"never"``.
        """
        frame = read_tabular(df)
        check_columns(frame)
        mode = _nullable_mode(nullable)
        if not isinstance(categorical_max_unique, (int, np.integer)) or categorical_max_unique < 0:
            raise ValueError(f"categorical_max_unique must be a non-negative int, got {categorical_max_unique!r}")

        specs: List[ColumnSpec] = []
        for name in frame.columns:
            series = frame[name]
            if not dtype_is_supported(series.dtype):
                logger.warning(
                    "column %r has dtype %s, which schema-guard cannot describe; treating it as string",
                    name,
                    series.dtype,
                )
            family = family_of(series)
            if mode == "observed":
                is_nullable = bool(series.isna().any())
            else:
                is_nullable = mode == "always"

            categories: Optional[List[Any]] = None
            if family == "category":
                # The categories of a category dtype are declared by the user, not guessed.
                declared = list(series.cat.categories)
                if len(declared) <= categorical_max_unique:
                    categories = declared
            elif family == "string":
                categories = _observed_categories(series, name, categorical_max_unique)

            lower = upper = None
            if numeric_ranges and family in NUMERIC_FAMILIES:
                lower, upper = _observed_range(series)
                if family == "int" and lower is not None:
                    lower, upper = int(lower), int(upper)

            tz = _series_tz(series) if family == "datetime" else None
            specs.append(ColumnSpec(name, family, is_nullable, categories, lower, upper, tz))
        return cls(specs)

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict: ``{"format": 1, "columns": [...]}``."""
        return {"format": SCHEMA_FORMAT, "columns": [spec.to_dict() for spec in self.columns]}

    @classmethod
    def from_dict(cls, data: Any) -> "Schema":
        """Rebuild a schema from ``to_dict()`` output (a bare list of column dicts also works)."""
        if isinstance(data, cls):
            return cls(list(data.columns))
        if isinstance(data, list):
            columns = data
        elif isinstance(data, dict):
            fmt = data.get("format", SCHEMA_FORMAT)
            if not isinstance(fmt, int) or isinstance(fmt, bool) or fmt < 1:
                raise ValueError(f"schema 'format' must be a positive int, got {fmt!r}")
            if fmt > SCHEMA_FORMAT:
                raise ValueError(
                    f"schema format {fmt} is newer than this version of schema-guard understands ({SCHEMA_FORMAT})"
                )
            columns = data.get("columns")
            if not isinstance(columns, list):
                raise ValueError("schema dict needs a 'columns' list")
        else:
            raise TypeError(f"expected a schema dict or a list of column dicts, got {type(data).__name__}")
        return cls([spec if isinstance(spec, ColumnSpec) else ColumnSpec.from_dict(spec) for spec in columns])

    def to_json(self, *, indent: int = 2) -> str:
        """The schema as human-readable JSON text: UTF-8, not ASCII-escaped, standard JSON.

        The output is RFC 8259, so jq, JavaScript, Go and every other parser read it back
        (no bare ``NaN``/``Infinity`` tokens, which only Python accepts).
        """
        return json_dumps(self.to_dict(), indent=indent) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Schema":
        """Rebuild a schema from JSON text produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def save(self, path: Union[str, "os.PathLike[str]"]) -> Path:
        """Write the schema as human-readable JSON to ``path`` (parent folders are created)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "Schema":
        """Read a schema written by :meth:`save`.

        A path that is a directory raises ``ValueError`` saying so, rather than the raw
        ``PermissionError``/``IsADirectoryError`` the OS would give. A file that is not a
        schema at all raises ``ValueError`` naming the file, rather than a bare
        ``Expecting value: line 1 column 1`` that says nothing about which file was read.
        """
        file = check_file(path, "a schema JSON file")
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{str(file)!r} is not a schema JSON file (it is not UTF-8 text)") from exc
        try:
            return cls.from_json(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{str(file)!r} is not a schema JSON file: {exc}") from exc

    # ------------------------------------------------------------------ validation

    def validate(self, df: Any, *, strict_ranges: bool = False) -> ValidationResult:
        """Check a DataFrame (or ``.csv``/``.parquet`` path) against the schema without changing it.

        Reports ``missing_column``, ``extra_column``, ``wrong_order``, ``dtype_mismatch``,
        ``unexpected_null``, ``unknown_category`` and ``out_of_range`` problems. Everything
        is an error except ``out_of_range``, which is a warning unless ``strict_ranges`` is True.
        """
        frame = read_tabular(df)
        check_columns(frame)
        problems: List[Problem] = []
        frame_columns = list(frame.columns)
        present = set(frame_columns)
        schema_names = set(self.column_names)

        for position, spec in enumerate(self.columns):
            if spec.name not in present:
                problems.append(
                    Problem(
                        "missing_column",
                        spec.name,
                        f"column is missing (expected at position {position})",
                        {"expected_position": position, "family": spec.family},
                    )
                )
        for position, name in enumerate(frame_columns):
            if name not in schema_names:
                problems.append(
                    Problem(
                        "extra_column",
                        name,
                        f"column is not in the schema (dtype {frame[name].dtype})",
                        {"position": position, "actual_dtype": str(frame[name].dtype)},
                    )
                )
        expected = [name for name in self.column_names if name in present]
        actual = [name for name in frame_columns if name in schema_names]
        if expected != actual:
            problems.append(
                Problem(
                    "wrong_order",
                    None,
                    f"columns are not in schema order: expected {_names(expected)}, got {_names(actual)}",
                    {"expected": expected, "actual": actual},
                )
            )
        for spec in self.columns:
            if spec.name in present:
                problems.extend(_check_column(frame[spec.name], spec, strict_ranges))
        return ValidationResult(problems, n_rows=len(frame), n_columns=len(frame_columns))

    # ------------------------------------------------------------------ enforcement

    def enforce(
        self,
        df: Any,
        *,
        mode: str = "coerce",
        extra: str = "drop",
        missing: str = "fill",
    ) -> pd.DataFrame:
        """Return a new DataFrame shaped like the schema; the input is never modified.

        With ``mode="coerce"`` (default) every column is converted to its family (float
        with nulls -> ``Int64``, text -> numbers/datetimes/booleans, values -> categorical
        with the schema's categories, ...), columns are put in schema order, extra columns
        are dropped (``extra="drop"``), kept at the end (``"keep"``) or rejected
        (``"raise"``), and missing columns are added as all-null columns of the right dtype
        (``missing="fill"``) or rejected (``"raise"``). Values that cannot be converted
        raise :class:`SchemaError` listing every failing column.

        With ``mode="strict"`` nothing is converted: the frame is validated and a
        :class:`SchemaError` listing every problem is raised unless it already matches
        (``extra="keep"`` tolerates extra columns). A copy of the frame is returned.
        """
        if mode not in ("coerce", "strict"):
            raise ValueError(f"mode must be 'coerce' or 'strict', got {mode!r}")
        if extra not in ("drop", "keep", "raise"):
            raise ValueError(f"extra must be 'drop', 'keep' or 'raise', got {extra!r}")
        if missing not in ("fill", "raise"):
            raise ValueError(f"missing must be 'fill' or 'raise', got {missing!r}")
        frame = read_tabular(df)
        check_columns(frame)

        if mode == "strict":
            result = self.validate(frame)
            blocking = [p for p in result.errors if not (p.kind == "extra_column" and extra == "keep")]
            if blocking:
                raise SchemaError(blocking)
            return frame.copy()

        frame_columns = list(frame.columns)
        present = set(frame_columns)
        schema_names = set(self.column_names)
        missing_columns = [spec.name for spec in self.columns if spec.name not in present]
        extra_columns = [name for name in frame_columns if name not in schema_names]

        problems: List[Problem] = []
        if missing == "raise":
            for name in missing_columns:
                problems.append(
                    Problem(
                        "missing_column",
                        name,
                        f"column is missing (expected at position {self.column_names.index(name)})",
                        {"expected_position": self.column_names.index(name), "family": self[name].family},
                    )
                )
        if extra == "raise":
            for name in extra_columns:
                problems.append(
                    Problem(
                        "extra_column",
                        name,
                        f"column is not in the schema (dtype {frame[name].dtype})",
                        {"position": frame_columns.index(name), "actual_dtype": str(frame[name].dtype)},
                    )
                )
        if problems:
            raise SchemaError(problems)

        data: Dict[Any, pd.Series] = {}
        failures: List[Problem] = []
        for spec in self.columns:
            if spec.name in present:
                try:
                    data[spec.name] = coerce_column(frame[spec.name], spec).reset_index(drop=True)
                except CoercionError as exc:
                    failures.append(exc.problem)
            else:
                data[spec.name] = null_column(spec, len(frame))
        if failures:
            raise SchemaError(failures)
        if extra == "keep":
            for name in extra_columns:
                data[name] = frame[name].reset_index(drop=True)

        out = pd.DataFrame(data, index=pd.RangeIndex(len(frame)), copy=True)
        out.index = frame.index
        return out

    # ------------------------------------------------------------------ reporting

    def summary(self) -> str:
        """Human-readable table of the schema (ASCII only)."""
        count = len(self.columns)
        head = f"Schema with {count} column{'s' if count != 1 else ''}"
        if not self.columns:
            return head
        rows: List[Tuple[str, ...]] = [("#", "column", "family", "nullable", "range", "categories")]
        for position, spec in enumerate(self.columns):
            if spec.min is None and spec.max is None:
                rng = "-"
            else:
                rng = f"{spec.min} .. {spec.max}"
            if spec.categories is None:
                cats = "-"
            else:
                cats = f"{len(spec.categories)}: {preview(spec.categories)}"
            rows.append((str(position), str(spec.name), spec.family, "yes" if spec.nullable else "no", rng, cats))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines = [head]
        for row in rows:
            lines.append("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
        return "\n".join(lines)


# ---------------------------------------------------------------------- module helpers


def as_schema(obj: Any) -> Schema:
    """Accept a :class:`Schema`, a schema dict/list, or a path to a saved schema JSON."""
    if isinstance(obj, Schema):
        return obj
    if isinstance(obj, (dict, list)):
        return Schema.from_dict(obj)
    if isinstance(obj, (str, os.PathLike)):
        return Schema.load(obj)
    raise TypeError(f"expected a Schema, a schema dict or a path to a schema JSON, got {type(obj).__name__}")


def _nullable_mode(nullable: Any) -> str:
    if isinstance(nullable, (bool, np.bool_)):
        return "always" if nullable else "never"
    if nullable in _NULLABLE_MODES:
        return str(nullable)
    raise ValueError(f"nullable must be 'observed', 'always', 'never', True or False, got {nullable!r}")


def _numeric_values(series: pd.Series) -> Optional[pd.Series]:
    """A float64 view of a numeric-ish column (NaN where not a number), or None if hopeless."""
    try:
        base = series.astype(object) if isinstance(series.dtype, pd.CategoricalDtype) else series
        numeric = pd.to_numeric(base, errors="coerce") if is_object_backed(base) else base
        return numeric.astype("float64")
    except (TypeError, ValueError):
        return None


def _observed_categories(series: pd.Series, name: Any, max_unique: int) -> Optional[List[Any]]:
    """The allowed-value set of a text column, or None when recording one would be degenerate."""
    nonnull = series.dropna()
    observed = list(pd.unique(nonnull))
    # A text column with no values says nothing about what is allowed.
    if not observed or len(observed) > max_unique:
        return None
    if len(observed) == len(nonnull):
        # Every value distinct: an id, an email, a filename, free text. A closed set built
        # from values that never repeat is evidence of nothing and rejects the next batch.
        if len(nonnull) < 2:
            # With one value there is nothing that could have repeated, so all-distinct
            # carries no information at all - and a wide one-row frame would otherwise
            # print a wall of warnings that each say nothing.
            logger.debug(
                "column %r: a single value says nothing about the allowed set, so none was recorded", name
            )
        else:
            logger.warning(
                "column %r: all %d value(s) are distinct, so no allowed-category set was recorded; "
                "a set inferred from values that never repeat would fail every later batch",
                name,
                len(nonnull),
            )
        return None
    return sorted_values(json_safe(v) for v in observed)


def _observed_range(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    base = pd.to_numeric(series, errors="coerce") if is_object_backed(series) else series
    nonnull = base.dropna()
    if len(nonnull) == 0:
        return None, None
    return json_safe(nonnull.min()), json_safe(nonnull.max())


def _names(names: List[Any], limit: int = 8) -> str:
    shown = ", ".join(repr(n) for n in names[:limit])
    if len(names) > limit:
        shown += f", ... (+{len(names) - limit} more)"
    return f"[{shown}]"


def _check_column(series: pd.Series, spec: ColumnSpec, strict_ranges: bool) -> List[Problem]:
    problems: List[Problem] = []
    family = family_of(series)
    is_object = is_object_backed(series)
    # An ``object`` column with no values (an empty frame, a column of None) says nothing
    # about its type, so it is compatible with every family; typed dtypes are still checked.
    untyped_empty = is_object and not bool(series.notna().any())

    family_mismatch = not untyped_empty and family != spec.family
    # Right values, wrong container: a NUMERIC column out of SQL Server, a frame back from
    # JSON. infer() classifies an object column by its values, so calling this an error made
    # the very frame a schema was learned from fail that schema. It is still worth saying -
    # sklearn and friends want a real dtype - so it is reported as a warning enforce() fixes.
    object_backed = (
        not untyped_empty and not family_mismatch and is_object and spec.family in TYPED_FAMILIES
    )
    if family_mismatch:
        detail: Dict[str, Any] = {
            "expected": spec.family,
            "actual_dtype": str(series.dtype),
            "actual_family": family,
        }
        message = f"expected {spec.family}, got {series.dtype} ({family})"
        if spec.family == "int" and family == "float":
            values = _numeric_values(series)
            nonnull = values.dropna() if values is not None else None
            if nonnull is not None and len(nonnull) and bool(np.isfinite(nonnull).all()) and bool(
                (nonnull == np.floor(nonnull)).all()
            ):
                target = "Int64" if series.isna().any() else "int64"
                message += f"; values are whole numbers, enforce() coerces them to {target}"
                detail["whole_numbers"] = True
        problems.append(Problem("dtype_mismatch", spec.name, message, detail))
    elif object_backed:
        problems.append(
            Problem(
                "dtype_mismatch",
                spec.name,
                f"expected a real {spec.family} dtype, got object holding {family} values; "
                "enforce() converts it",
                {
                    "expected": spec.family,
                    "actual_dtype": str(series.dtype),
                    "actual_family": family,
                    "object_backed": True,
                },
                severity="warning",
            )
        )
    elif spec.family == "datetime" and not untyped_empty and not is_object:
        # Same family, different clock. A UTC column arriving tz-naive is a classic pipeline
        # break - every timestamp shifts by hours - and the dtypes alone look compatible.
        actual_tz = _series_tz(series)
        if actual_tz != spec.tz:
            problems.append(
                Problem(
                    "dtype_mismatch",
                    spec.name,
                    f"expected datetime in {_tz_text(spec.tz)}, got {_tz_text(actual_tz)}",
                    {
                        "expected": spec.family,
                        "actual_dtype": str(series.dtype),
                        "actual_family": family,
                        "expected_tz": spec.tz,
                        "actual_tz": actual_tz,
                    },
                )
            )

    if not spec.nullable:
        null_count = int(series.isna().sum())
        if null_count:
            fraction = null_count / len(series) if len(series) else 0.0
            problems.append(
                Problem(
                    "unexpected_null",
                    spec.name,
                    f"{null_count} null value(s) in a non-nullable column",
                    {"null_count": null_count, "null_fraction": round(fraction, 6)},
                )
            )

    if spec.categories is not None and spec.family in ("string", "category"):
        allowed = set(spec.categories)
        # Same normalisation enforce() uses, so validate() never calls a value unknown that
        # enforce() would map straight onto a stored category.
        key = category_key(spec.categories)
        observed = list(pd.unique(series.dropna()))
        unknown_raw = [v for v in observed if key(v) not in allowed]
        if unknown_raw:
            unknown = [key(v) for v in unknown_raw]
            count = int(series.isin(unknown_raw).sum())
            problems.append(
                Problem(
                    "unknown_category",
                    spec.name,
                    f"{count} value(s) not in the allowed categories (e.g. {preview(unknown)})",
                    {"unknown_values": unknown[:50], "n_unknown_values": len(unknown), "unknown_count": count},
                )
            )

    has_range = spec.min is not None or spec.max is not None
    if has_range and spec.family in NUMERIC_FAMILIES and family in ("int", "float", "bool"):
        values = _numeric_values(series)
        if values is not None:
            mask = pd.Series(False, index=values.index)
            if spec.min is not None:
                mask |= values < spec.min
            if spec.max is not None:
                mask |= values > spec.max
            count = int(mask.sum())
            if count:
                nonnull = values.dropna()
                lo, hi = json_safe(nonnull.min()), json_safe(nonnull.max())
                problems.append(
                    Problem(
                        "out_of_range",
                        spec.name,
                        f"{count} value(s) outside the range {spec.min} .. {spec.max} (observed {lo} .. {hi})",
                        {
                            "min": spec.min,
                            "max": spec.max,
                            "observed_min": lo,
                            "observed_max": hi,
                            "count": count,
                            "fraction": round(count / len(values), 6) if len(values) else 0.0,
                        },
                        severity="error" if strict_ranges else "warning",
                    )
                )
    return problems
