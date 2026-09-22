"""Loading tabular input: a DataFrame is passed through, a path is read."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd

FrameLike = Union[pd.DataFrame, str, "os.PathLike[str]"]


def load_frame(data: FrameLike) -> pd.DataFrame:
    """Return ``data`` as a DataFrame; ``.csv``/``.tsv``/``.parquet`` paths are read."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix not in (".csv", ".tsv", ".parquet", ".pq"):
            if not suffix:
                # a bare string is far more often a mistyped argument than a
                # suffixless file, so name what was actually expected
                raise ValueError(
                    "expected a pandas DataFrame or a path to a .csv/.tsv/.parquet file, "
                    f"got {str(data)!r}"
                )
            raise ValueError(f"unsupported file type {suffix!r}: expected .csv, .tsv or .parquet")
        # validated here rather than left to pandas, so every bad input is reported
        # by this package in the same voice
        if not path.exists():
            raise FileNotFoundError(f"no such file: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"expected a file but {path} is a directory")
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "reading parquet needs pyarrow: pip install 'ml-feature-check[parquet]'"
            ) from exc
    raise TypeError(
        "expected a pandas DataFrame or a path to a .csv/.parquet file, "
        f"got {type(data).__name__}"
    )


def write_frame(df: pd.DataFrame, path: Union[str, "os.PathLike[str]"]) -> None:
    """Write ``df`` to ``path``, picking the format from the suffix (.csv/.tsv/.parquet)."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        df.to_csv(target, index=False)
    elif suffix == ".tsv":
        df.to_csv(target, index=False, sep="\t")
    elif suffix in (".parquet", ".pq"):
        try:
            df.to_parquet(target, index=False)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            # pandas raises a four-line engine dump here; the read path already
            # translates it, and the write path must say the same thing
            raise ImportError(
                "writing parquet needs pyarrow: pip install 'ml-feature-check[parquet]'"
            ) from exc
    else:
        raise ValueError(f"unsupported output type {suffix!r}: expected .csv, .tsv or .parquet")
