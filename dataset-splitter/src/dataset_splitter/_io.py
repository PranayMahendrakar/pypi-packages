"""Loading tabular input: a DataFrame is passed through, a path is read."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def load_table(data: Any) -> pd.DataFrame:
    """Return `data` as a DataFrame; accepts a DataFrame or a path to .csv/.tsv/.parquet."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix in (".parquet", ".pq"):
            try:
                return pd.read_parquet(path)
            except ImportError as exc:
                raise ImportError(
                    "reading parquet needs pyarrow: pip install 'dataset-splitter[parquet]'"
                ) from exc
        raise ValueError(f"unsupported file type {suffix!r}; expected .csv, .tsv or .parquet")
    raise TypeError(
        f"expected a pandas DataFrame or a path to a .csv/.parquet file, got {type(data).__name__}"
    )
