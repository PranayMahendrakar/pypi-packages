"""Input loading: a pandas DataFrame, a Series, or a path to a .csv / .parquet file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

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
                    'pip install "predictive-maintenance[parquet]"'
                ) from exc
        if ".csv" in suffixes:
            return pd.read_csv(path)
        if ".tsv" in suffixes:
            return pd.read_csv(path, sep="\t")
        raise ValueError(
            f"{label}: unsupported file type {path.suffix!r}; expected .csv, .tsv or .parquet"
        )
    raise TypeError(
        f"{label} must be a pandas DataFrame, a Series, or a path to a .csv/.parquet file, "
        f"not {type(source).__name__}"
    )
