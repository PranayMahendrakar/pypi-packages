"""Turn DataFrame rows into comparable strings."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._text import normalize_text

KeySpec = Union[None, str, int, Sequence[Union[str, int]]]
ROW_SEPARATOR = " | "


def _cell_to_str(value: object) -> str:
    """Stable string form of one cell; missing values become ``""``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return ""
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return repr(f)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if missing is True or (isinstance(missing, (bool, np.bool_)) and bool(missing)):
        return ""
    return str(value)


def select_columns(df: pd.DataFrame, key: KeySpec) -> pd.DataFrame:
    """The columns rows are compared on: all of them, or the ``key`` column(s)."""
    if key is None:
        return df
    if isinstance(key, (str, int)) and not isinstance(key, bool):
        keys: List[object] = [key]
    else:
        keys = list(key)
    if not keys:
        raise ValueError("key must name at least one column")
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"key column(s) {missing!r} not found in DataFrame columns {list(df.columns)!r}")
    return df.loc[:, keys]


def check_columns(df: pd.DataFrame) -> None:
    """Reject frames whose columns cannot be compared unambiguously (duplicate names).

    Raised at the entry point so a duplicate name never silently makes ``key``
    select two columns at once.
    """
    if df.columns.has_duplicates:
        dupes = [str(c) for c in df.columns[df.columns.duplicated()].unique()]
        raise ValueError(f"duplicate column names are not supported: {', '.join(dupes)}")


def row_keys(df: pd.DataFrame, key: KeySpec = None, *, normalize: bool = True) -> Tuple[List[str], List[bool]]:
    """Per-row comparison strings and a flag telling which rows are entirely blank.

    Each key cell is stringified and normalized (case, whitespace, punctuation,
    NaN -> empty), then the cells are joined with ``" | "`` so column boundaries
    still count. Rows whose key cells are all missing are flagged blank and are
    never treated as duplicates of anything.
    """
    sub = select_columns(df, key)
    n = int(sub.shape[0])
    if n == 0 or sub.shape[1] == 0:
        return [""] * n, [True] * n
    columns: List[List[str]] = []
    for pos in range(sub.shape[1]):
        raw = sub.iloc[:, pos].to_numpy(dtype=object)
        if normalize:
            columns.append([normalize_text(_cell_to_str(v), strip_punctuation=True) for v in raw])
        else:
            columns.append([_cell_to_str(v) for v in raw])
    keys = [ROW_SEPARATOR.join(cells) for cells in zip(*columns)]
    blank = [all(c == "" for c in cells) for cells in zip(*columns)]
    return keys, blank


def load_table(path: object) -> pd.DataFrame:
    """Read a ``.csv`` or ``.parquet`` file into a DataFrame."""
    text = str(path)
    lower = text.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(text)
    if lower.endswith(".csv") or lower.endswith(".csv.gz") or lower.endswith(".tsv"):
        sep = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(text, sep=sep)
    raise ValueError(f"Expected a .csv or .parquet path, got {text!r}")


def is_table_path(value: object) -> bool:
    """True for a str/PathLike that points at a .csv/.tsv/.parquet file name."""
    import os

    if isinstance(value, (str, os.PathLike)):
        lower = os.fspath(value).lower()
        return lower.endswith((".csv", ".csv.gz", ".tsv", ".parquet", ".pq"))
    return False


def frame_from_records(items: Sequence[dict]) -> pd.DataFrame:
    """DataFrame from a list of dicts, preserving the order of first-seen keys."""
    return pd.DataFrame.from_records(list(items))


__all__: Optional[List[str]] = [
    "row_keys",
    "select_columns",
    "check_columns",
    "load_table",
    "is_table_path",
    "frame_from_records",
]
