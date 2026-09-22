"""Reading a table off disk, for the command line."""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import pandas as pd

LOG = logging.getLogger(__name__)

_CSV_SUFFIXES = (".csv", ".txt", ".tsv")
_PARQUET_SUFFIXES = (".parquet", ".pq")


def _duplicates(names: Sequence[object]) -> List[str]:
    """The names that appear more than once, sorted."""
    if not len(names):
        return []
    counts = pd.Series([str(name) for name in names]).value_counts()
    return sorted(str(name) for name, count in counts.items() if count > 1)


def _guard_duplicates(names: Sequence[object], source: Path) -> None:
    duplicates = _duplicates(names)
    if duplicates:
        raise ValueError(
            "{0} has duplicate column names: {1}. Rename them first; every check reads "
            "columns by name.".format(source.name, ", ".join(duplicates))
        )


def _csv_header(source: Path, separator: str) -> List[str]:
    """The header row as written in the file, before pandas renames anything."""
    with open(source, "r", encoding="utf-8", errors="replace", newline="") as handle:
        first_line = handle.readline()
    if not first_line.strip():
        return []
    return next(csv.reader(io.StringIO(first_line), delimiter=separator), [])


def read_table(path: Union[str, Path]) -> Tuple[pd.DataFrame, List[str]]:
    """Read a `.csv`, `.tsv` or `.parquet` file into a DataFrame, plus any notes.

    Text is read as UTF-8; a file that is not UTF-8 is read as latin-1 instead
    and says so in the notes rather than failing. Duplicate column names are
    reported by name here, before pandas quietly renames them.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError("no such file: {0}".format(source))
    suffix = source.suffix.lower()
    notes: List[str] = []
    if suffix in _PARQUET_SUFFIXES:
        try:
            frame = pd.read_parquet(source)
        except ImportError as exc:
            raise ImportError(
                "reading .parquet needs pyarrow: pip install ml-pipeline-kit[parquet] ({0})".format(exc)
            ) from exc
    elif suffix in _CSV_SUFFIXES or suffix == "":
        separator = "\t" if suffix == ".tsv" else ","
        _guard_duplicates(_csv_header(source, separator), source)
        try:
            frame = pd.read_csv(source, sep=separator, encoding="utf-8")
        except UnicodeDecodeError:
            frame = pd.read_csv(source, sep=separator, encoding="latin-1")
            notes.append("{0} is not UTF-8; it was read as latin-1".format(source.name))
    else:
        raise ValueError("cannot read '{0}': expected .csv, .tsv or .parquet".format(source.name))
    _guard_duplicates(list(frame.columns), source)
    LOG.debug("read %s rows and %s columns from %s", len(frame), len(frame.columns), source)
    return frame, notes
