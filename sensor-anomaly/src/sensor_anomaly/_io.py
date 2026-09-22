"""Input loading: a pandas DataFrame, a Series, or a path to a .csv / .tsv / .parquet file."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

TableLike = Union[pd.DataFrame, pd.Series, str, "os.PathLike[str]"]


def _duplicates(names: List[str]) -> List[str]:
    """Names that appear more than once, ignoring the blank ones pandas renames."""
    real = [name for name in names if name != ""]
    return sorted({name for name, count in Counter(real).items() if count > 1})


def _header_of(path: Path, separator: str) -> List[str]:
    """The first row of a delimited file, as written, before pandas renames anything."""
    try:
        with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.reader(handle, delimiter=separator):
                return list(row)
    except OSError:  # pragma: no cover - the caller already proved the file is there
        return []
    return []


def _read_delimited(path: Path, separator: str, label: str) -> pd.DataFrame:
    """Read a .csv/.tsv, refusing a duplicated header instead of silently renaming it.

    ``read_csv`` turns a second column called ``a`` into ``a.1`` before any check of
    ours can see it, so the caller would get a channel named after no physical
    sensor and no note saying why. The DataFrame path raises on duplicates; this
    makes the file path answer the same way.
    """
    duplicates = _duplicates(_header_of(path, separator))
    if duplicates:
        raise ValueError(
            "%s: %s has duplicate column names: %s; rename them so each sensor "
            "channel is addressable"
            % (label, path.name, ", ".join(repr(name) for name in duplicates))
        )
    return pd.read_csv(path, sep=separator)


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
                    'pip install "sensor-anomaly[parquet]"'
                ) from exc
        if ".csv" in suffixes:
            return _read_delimited(path, ",", label)
        if ".tsv" in suffixes:
            return _read_delimited(path, "\t", label)
        raise ValueError(
            f"{label}: unsupported file type {path.suffix!r}; expected .csv, .tsv or .parquet"
        )
    raise TypeError(
        f"{label} must be a pandas DataFrame, a Series, or a path to a .csv/.parquet file, "
        f"not {type(source).__name__}"
    )


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Give every column a text name, refusing duplicates with a message that names them.

    Returns the frame (a cheap re-labelled view, never a data copy) and any notes
    about renames. Duplicate names raise :class:`ValueError` here rather than
    failing later with an AttributeError from inside pandas.
    """
    names = [str(c) for c in df.columns]
    duplicates = _duplicates(names)
    if duplicates:
        raise ValueError(
            "df has duplicate column names: "
            + ", ".join(repr(name) for name in duplicates)
            + "; rename them so each sensor channel is addressable"
        )
    notes: List[str] = []
    if any(not isinstance(c, str) for c in df.columns):
        notes.append("Column names that were not text were converted to text.")
        return df.set_axis(names, axis="columns"), notes
    return df, notes
