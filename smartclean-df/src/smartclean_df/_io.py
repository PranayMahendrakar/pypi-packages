"""Accept a DataFrame or a path to a tabular file."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd

__all__ = ["load_frame", "write_frame", "FrameLike"]

FrameLike = Union[pd.DataFrame, str, "os.PathLike[str]"]


def _inference_is_lossless(text: pd.Series) -> bool:
    """True when letting pandas type-infer this column of text changes nothing.

    ``"123"`` becomes ``123`` and renders back as ``"123"``, so inferring it is
    free. ``"00123"`` becomes ``123`` and renders back as ``"123"`` -- a
    different identifier -- so that column must stay text and be handed to the
    cleaning pipeline, which knows to leave zero-padded codes alone.
    """
    values = text.dropna()
    if values.empty:
        return True
    try:
        numbers = pd.to_numeric(values)
    except (ValueError, TypeError, OverflowError):
        # pandas will not turn this column into numbers either; nothing to lose
        return True
    try:
        rendered = numbers.astype(str).to_numpy()
    except (ValueError, TypeError):
        return False
    return bool((rendered == values.to_numpy()).all())


def _read_text_table(path: Path, *, sep: str) -> pd.DataFrame:
    """Read a delimited text file without letting pandas destroy any values.

    Reading a file must never be a lossy, unlogged transformation. Columns
    pandas can type-infer without changing a single value are inferred as
    usual; every other column is read as text, so that the cleaning pipeline
    decides what becomes a number, a date or a boolean -- and logs an Action
    for it -- instead of pandas deciding silently at read time.
    """
    text = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=True)
    as_text = {col: str for col in text.columns if not _inference_is_lossless(text[col])}
    if not as_text:
        return pd.read_csv(path, sep=sep)
    return pd.read_csv(path, sep=sep, dtype=as_text)


def load_frame(df_or_path: FrameLike) -> pd.DataFrame:
    """Return a deep copy of a DataFrame, or read a .csv/.tsv/.parquet file."""
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy(deep=True)
    if isinstance(df_or_path, (str, os.PathLike)):
        path = Path(df_or_path)
        suffix = path.suffix.lower()
        if suffix in (".csv", ".txt"):
            return _read_text_table(path, sep=",")
        if suffix == ".tsv":
            return _read_text_table(path, sep="\t")
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        raise ValueError(
            f"Unsupported file type {suffix!r} for {path}; expected .csv, .tsv or .parquet"
        )
    raise TypeError(
        "expected a pandas DataFrame or a path to a .csv/.tsv/.parquet file, "
        f"got {type(df_or_path).__name__}"
    )


def write_frame(df: pd.DataFrame, path: "Union[str, os.PathLike[str]]") -> Path:
    """Write ``df`` to ``path`` by extension (.csv, .tsv or .parquet)."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in (".csv", ".txt"):
        df.to_csv(target, index=False)
    elif suffix == ".tsv":
        df.to_csv(target, index=False, sep="\t")
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(target, index=False)
    else:
        raise ValueError(
            f"Unsupported output type {suffix!r} for {target}; expected .csv, .tsv or .parquet"
        )
    return target
