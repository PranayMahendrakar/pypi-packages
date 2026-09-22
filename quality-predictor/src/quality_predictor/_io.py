"""Loading tabular input: a DataFrame is passed through, a path is read."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

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
            raise ValueError(
                f"unsupported file type {suffix!r}: expected .csv, .tsv or .parquet"
            )
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
                "reading parquet needs pyarrow: pip install 'quality-predictor[parquet]'"
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
        df.to_csv(target, index=False, encoding="utf-8")
    elif suffix == ".tsv":
        df.to_csv(target, index=False, sep="\t", encoding="utf-8")
    elif suffix in (".parquet", ".pq"):
        try:
            df.to_parquet(target, index=False)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "writing parquet needs pyarrow: pip install 'quality-predictor[parquet]'"
            ) from exc
    else:
        raise ValueError(
            f"unsupported output type {suffix!r}: expected .csv, .tsv or .parquet"
        )


def duplicate_columns(df: pd.DataFrame) -> List[str]:
    """The column labels that appear more than once, in order of first appearance."""
    seen: List[str] = []
    dupes: List[str] = []
    for name in df.columns:
        if name in seen and name not in dupes:
            dupes.append(str(name))
        seen.append(name)
    return dupes


def require_unique_columns(df: pd.DataFrame) -> None:
    """Raise a clear ``ValueError`` naming duplicate column labels, before pandas does."""
    dupes = duplicate_columns(df)
    if dupes:
        raise ValueError(
            "duplicate column names are ambiguous: "
            + ", ".join(repr(d) for d in dupes)
            + ". Rename or drop one of each pair before fitting."
        )
