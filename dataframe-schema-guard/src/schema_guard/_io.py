"""Reading and writing tabular input (internal helpers)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_tabular(obj: Any) -> pd.DataFrame:
    """Return ``obj`` if it is a DataFrame, or read it from a ``.csv``/``.tsv``/``.parquet`` path."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (str, os.PathLike)):
        path = check_file(obj, "a data file")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix in (".parquet", ".pq"):
            try:
                return pd.read_parquet(path)
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "reading parquet needs pyarrow: pip install 'schema-guard[parquet]'"
                ) from exc
        raise ValueError(
            f"unsupported file type {suffix!r} for {str(path)!r}: expected .csv, .tsv or .parquet"
        )
    raise TypeError(
        "expected a pandas DataFrame or a path to a .csv/.tsv/.parquet file, "
        f"got {type(obj).__name__}"
    )


def write_tabular(df: pd.DataFrame, path: Any) -> Path:
    """Write ``df`` to ``path`` (``.csv``/``.tsv``/``.parquet`` by suffix) and return the path."""
    out = Path(path)
    suffix = out.suffix.lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(out, index=False, encoding="utf-8")
    elif suffix == ".tsv":
        df.to_csv(out, index=False, sep="\t", encoding="utf-8")
    elif suffix in (".parquet", ".pq"):
        try:
            df.to_parquet(out, index=False)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "writing parquet needs pyarrow: pip install 'schema-guard[parquet]'"
            ) from exc
    else:
        raise ValueError(
            f"unsupported output type {suffix!r} for {str(out)!r}: expected .csv, .tsv or .parquet"
        )
    return out


def check_file(path: Any, what: str = "a file") -> Path:
    """Return ``path`` as a :class:`~pathlib.Path`, rejecting a directory with a clear message.

    Opening a directory raises ``PermissionError`` on Windows and ``IsADirectoryError`` on
    Linux - both ``OSError``, neither of which says what was actually wrong.
    """
    out = Path(path)
    if out.is_dir():
        raise ValueError(f"{str(out)!r} is a directory, not {what}")
    return out


def is_supported_label(name: Any) -> bool:
    """True when ``name`` can be a schema column name: a string or an integer."""
    if isinstance(name, (bool, np.bool_)):
        return False
    return isinstance(name, (str, int, np.integer))


def check_columns(df: pd.DataFrame) -> None:
    """Reject frames whose columns a schema cannot describe.

    MultiIndex columns, duplicate names and labels that are not strings or integers
    (``Timestamp`` columns from a pivot, floats, tuples, ``None``) all raise ``ValueError``.
    A schema stores column names as JSON, so a label it cannot store would be silently
    renamed and would then match nothing in the frame it came from.
    """
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("MultiIndex columns are not supported; flatten the column names first")
    if df.columns.has_duplicates:
        dupes = [str(c) for c in df.columns[df.columns.duplicated()].unique()]
        raise ValueError(f"duplicate column names are not supported: {', '.join(dupes)}")
    unsupported = [name for name in df.columns if not is_supported_label(name)]
    if unsupported:
        shown = ", ".join(f"{name!r} ({type(name).__name__})" for name in unsupported[:5])
        if len(unsupported) > 5:
            shown += f", ... (+{len(unsupported) - 5} more)"
        raise ValueError(
            "column names must be strings or integers; these are not: "
            f"{shown}. Convert them first, e.g. df.columns = df.columns.map(str)"
        )
