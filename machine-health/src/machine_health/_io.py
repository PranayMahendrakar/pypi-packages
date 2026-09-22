"""Input loading: a pandas DataFrame, a Series, or a path to a .csv / .tsv / .parquet file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

import pandas as pd

TableLike = Union[pd.DataFrame, pd.Series, str, "os.PathLike[str]"]


def load_table(source: TableLike, label: str = "df") -> pd.DataFrame:
    """Return ``source`` as a DataFrame, reading .csv/.tsv/.parquet files from disk.

    DataFrames are returned as-is (never copied or mutated); a Series becomes a
    one-column frame. ``label`` names the argument in error messages.
    """
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, pd.Series):
        return source.to_frame()
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"{label}: no such file: {path}")
        suffixes = [s.lower() for s in path.suffixes]
        if ".parquet" in suffixes or ".pq" in suffixes:
            try:
                return pd.read_parquet(path)
            except ImportError as exc:
                raise ImportError(
                    "reading .parquet files needs pyarrow: "
                    'pip install "machine-health[parquet]"'
                ) from exc
        if ".csv" in suffixes:
            return _read_separated(path, label, ",", "CSV")
        if ".tsv" in suffixes:
            return _read_separated(path, label, "\t", "TSV")
        raise ValueError(
            f"{label}: unsupported file type {path.suffix!r}; expected .csv, .tsv or .parquet"
        )
    raise TypeError(
        f"{label} must be a pandas DataFrame, a Series, or a path to a .csv/.parquet file, "
        f"not {type(source).__name__}"
    )


def _read_separated(path: Path, label: str, sep: str, kind: str) -> pd.DataFrame:
    """Read a delimited text file, turning pandas' own wording into a message that
    names the argument and the file, the way every other error here does."""
    try:
        return pd.read_csv(path, sep=sep)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"{label}: {path.name} has no columns to read; the file is empty"
        ) from None
    except pd.errors.ParserError as exc:
        raise ValueError(f"{label}: {path.name} is not readable as {kind}: {exc}") from None
    except UnicodeDecodeError:
        raise ValueError(
            f"{label}: {path.name} is not UTF-8 text; save it as UTF-8 or pass a DataFrame"
        ) from None


def check_columns(df: pd.DataFrame, label: str = "df") -> List[str]:
    """Reject duplicate column names up front and return the names as text.

    Without this a duplicated name makes ``df[name]`` return a DataFrame and the
    failure surfaces much later as an AttributeError from inside pandas.
    """
    names = [str(c) for c in df.columns]
    seen, dupes = set(), []
    for name in names:
        if name in seen and name not in dupes:
            dupes.append(name)
        seen.add(name)
    if dupes:
        raise ValueError(
            f"{label} has duplicate column names: {', '.join(repr(d) for d in dupes)}; "
            "rename them so each channel is addressable"
        )
    return names
