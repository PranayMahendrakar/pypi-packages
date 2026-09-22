"""Loading and saving tables: a DataFrame or a path to a .csv / .tsv / .parquet file."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

import pandas as pd
from pandas.api import types as ptypes

TableLike = Union[pd.DataFrame, str, "os.PathLike[str]"]

_ISO_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")
_ISO_KWARGS = {"format": "ISO8601"} if int(pd.__version__.split(".")[0]) >= 2 else {}


def _suffixes(path: Path) -> set:
    return {suffix.lower() for suffix in path.suffixes}


def parse_iso_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns whose every value is an ISO-8601 date/time (``YYYY-MM-DD...``) into datetimes.

    Applied to CSV/TSV loads so date columns are modelled as dates instead of being
    treated as high-cardinality text. A column is only converted when every non-null
    value starts like an ISO date and every one of them parses.
    """
    for name in df.columns:
        column = df[name]
        if column.dtype != object:
            continue
        non_null = column.dropna()
        if len(non_null) == 0 or ptypes.infer_dtype(non_null, skipna=True) != "string":
            continue
        if not non_null.str.match(_ISO_DATE_PREFIX).all():
            continue
        try:
            parsed = pd.to_datetime(non_null, errors="coerce", **_ISO_KWARGS)
        except (ValueError, TypeError):
            continue
        if parsed.isna().any():
            continue
        df[name] = pd.to_datetime(column, errors="coerce", **_ISO_KWARGS)
    return df


def load_table(data: TableLike) -> pd.DataFrame:
    """Return ``data`` as a DataFrame; paths ending in .csv, .tsv or .parquet are read.

    CSV/TSV columns holding ISO-8601 dates are parsed as datetimes (see ``parse_iso_dates``).
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        suffixes = _suffixes(path)
        if suffixes & {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if ".tsv" in suffixes:
            return parse_iso_dates(pd.read_csv(path, sep="\t"))
        if ".csv" in suffixes:
            return parse_iso_dates(pd.read_csv(path))
        raise ValueError(
            f"Unsupported file type for {str(path)!r}: expected a .csv, .tsv or .parquet file"
        )
    raise TypeError(
        "Expected a pandas DataFrame or a path to a .csv/.tsv/.parquet file, "
        f"got {type(data).__name__}"
    )


def write_table(df: pd.DataFrame, path: Union[str, "os.PathLike[str]"]) -> Path:
    """Write ``df`` to ``path`` as .csv, .tsv or .parquet (chosen by the extension)."""
    target = Path(path)
    suffixes = _suffixes(target)
    if suffixes & {".parquet", ".pq"}:
        df.to_parquet(target, index=False)
    elif ".tsv" in suffixes:
        df.to_csv(target, sep="\t", index=False, encoding="utf-8")
    elif ".csv" in suffixes:
        df.to_csv(target, index=False, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported output type for {str(target)!r}: expected a .csv, .tsv or .parquet file"
        )
    return target
