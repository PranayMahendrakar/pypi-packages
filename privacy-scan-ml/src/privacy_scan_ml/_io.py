"""Loading tabular input: a DataFrame is passed through, a path is read."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TABLE_SUFFIXES = (".csv", ".tsv", ".parquet", ".pq")

#: Suffixes that plainly name a data file this package cannot read. A string ending in one
#: of these is a path the caller got wrong, never a document to scan as free text.
DATA_SUFFIXES = (
    ".xlsx", ".xls", ".xlsm", ".xlsb", ".ods", ".json", ".jsonl", ".ndjson", ".feather",
    ".orc", ".arrow", ".avro", ".h5", ".hdf5", ".hdf", ".dta", ".sav", ".sas7bdat",
    ".sqlite", ".sqlite3", ".db", ".xml", ".pkl", ".pickle",
)

_WHITESPACE = re.compile(r"\s")


def load_table(data: Any) -> pd.DataFrame:
    """Return `data` as a DataFrame; accepts a DataFrame, a Series, or a path to .csv/.tsv/.parquet."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name if data.name is not None else "value")
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix not in TABLE_SUFFIXES:
            raise ValueError(f"unsupported file type {suffix!r}; expected .csv, .tsv or .parquet")
        # A file that is not there is not there, whatever optional reader it would need:
        # say so before pyarrow gets a chance to complain about being missing.
        if "://" not in str(data) and not path.exists():
            raise FileNotFoundError(f"no such file: {path}")
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix in (".parquet", ".pq"):
            try:
                return pd.read_parquet(path)
            except ImportError as exc:
                raise ImportError(
                    "reading parquet needs pyarrow: pip install 'privacy-scan-ml[parquet]'"
                ) from exc
    raise TypeError(
        f"expected a pandas DataFrame or a path to a .csv/.parquet file, got {type(data).__name__}"
    )


def looks_like_table_path(text: str) -> bool:
    """True when a string is a short single-line path ending in a table suffix."""
    if len(text) > 1024 or len(text.splitlines()) > 1:
        return False
    return text.strip().lower().endswith(TABLE_SUFFIXES)


def frame_from_dict(data: Dict[Any, Any]) -> pd.DataFrame:
    """A dict of columns as a DataFrame, with our own message when it is not one.

    pandas says "If using all scalar values, you must pass an index", which tells the
    caller nothing about what they did wrong.
    """
    if data and not any(_is_sequence(value) for value in data.values()):
        raise ValueError(
            "a dict of columns must map each name to a sequence of values, "
            "e.g. {'email': ['a@b.com']}; every value given was a single scalar"
        )
    try:
        return pd.DataFrame(data)
    except ValueError as exc:
        raise ValueError(
            "a dict of columns must map each name to a sequence of values of the same "
            "length, e.g. {'email': ['a@b.com'], 'city': ['Pune']}"
        ) from exc


def _is_sequence(value: Any) -> bool:
    """True for something a DataFrame column can be built from, not a single scalar."""
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, (list, tuple, set, range, dict, pd.Series, pd.Index, np.ndarray)):
        return True
    return hasattr(value, "__len__") or hasattr(value, "__iter__")


def unreadable_file_suffix(text: str) -> Optional[str]:
    """The suffix of a file this package cannot read, else ``None``.

    Lets ``scan("report.docx")`` say so plainly instead of quietly scanning the file name
    as though it were a sentence. That holds whether or not the file is on disk: a
    mistyped path to a spreadsheet must not come back as "no personal data found".
    """
    if len(text) > 1024 or len(text.splitlines()) > 1:
        return None
    candidate = text.strip()
    if not candidate or candidate.lower().endswith(TABLE_SUFFIXES):
        return None
    try:
        path = Path(candidate)
        suffix = path.suffix.lower()
        if not suffix:
            return None
        if path.is_file():
            return suffix
    except (OSError, ValueError):
        return None
    # Nothing on disk: only a bare file name counts, so a sentence stays free text.
    if suffix in DATA_SUFFIXES and not _WHITESPACE.search(candidate):
        return suffix
    return None
