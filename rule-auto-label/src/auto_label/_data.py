"""Input coercion and small helpers shared by the labeler and the CLI.

Everything the user can pass to :meth:`auto_label.Labeler.label` is turned into a
:class:`Prepared` value here, so the rest of the package only deals with one shape.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEXT = "text"
TABULAR = "tabular"

# Dict values accepted as a column of values (anything else is a scalar we cannot broadcast).
_COLUMN_TYPES = (list, tuple, range, np.ndarray, pd.Series, pd.Index)


@dataclass
class Prepared:
    """One input, normalised: a mode, per-item texts, and the original items."""

    mode: str
    texts: List[str]
    llm_texts: List[str]
    items: List[Any]
    index: List[Any]
    frame: Optional[pd.DataFrame] = field(default=None, repr=False)
    text_columns: List[str] = field(default_factory=list)
    text_fallback: bool = False

    def __len__(self) -> int:
        return len(self.texts)


def is_missing(value: Any) -> bool:
    """True for None, NaN, NaT and pandas NA; False for anything array-like."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        flag = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(flag) if isinstance(flag, (bool, np.bool_)) else False


def as_text(value: Any) -> str:
    """Render one item as the string the rules will see; missing values become ''."""
    if isinstance(value, str):
        return value
    if is_missing(value):
        return ""
    return str(value)


def decode_text(raw: bytes, encoding: Optional[str], origin: str) -> "Tuple[str, List[str]]":
    """Decode input bytes to text, the same way :func:`read_csv` decodes a CSV.

    With an explicit ``encoding`` a decode failure is an error naming the encoding.
    Without one the bytes are read as UTF-8 and anything undecodable is replaced so
    labelling can continue - but loudly: the warning is logged *and* returned as a note,
    because the replacement characters are what the rules then match against.
    """
    if encoding:
        try:
            return raw.decode(encoding), []
        except LookupError:
            raise ValueError(f"auto_label: unknown encoding {encoding!r}") from None
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"auto_label: {origin} could not be decoded as {encoding!r}: {exc}"
            ) from None
    try:
        return raw.decode("utf-8"), []
    except UnicodeDecodeError:
        message = (
            f"{origin} is not valid UTF-8; the undecodable bytes were replaced so labelling "
            "could continue. Re-save it as UTF-8, or pass the real encoding "
            "(CLI: --encoding cp1252)."
        )
        logger.warning("%s", message)
        return raw.decode("utf-8", errors="replace"), [f"input: {message}"]


def read_csv(path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV. Without an explicit ``encoding``, non-UTF-8 bytes are replaced, loudly."""
    if encoding:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"auto_label: {os.path.basename(path)} could not be decoded as {encoding!r}: {exc}"
            ) from None
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        logger.warning(
            "%s is not valid UTF-8; the undecodable bytes were replaced so labelling could "
            "continue. Re-save the file as UTF-8, or pass the real encoding "
            "(CLI: --encoding cp1252).",
            os.path.basename(path),
        )
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def read_table(path: Any, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load a ``.csv`` or ``.parquet`` file into a DataFrame."""
    path = os.fspath(path)
    if os.path.isdir(path):
        raise ValueError(f"auto_label: {path!r} is a directory, not a .csv/.parquet file")
    if not os.path.exists(path):
        raise FileNotFoundError(f"auto_label: no such file: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return read_csv(path, encoding)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "reading .parquet needs pyarrow: pip install 'auto-label[parquet]'"
            ) from exc
    raise ValueError(f"auto_label: expected a .csv or .parquet path, got {path!r}")


def prepare(data: Any) -> Prepared:
    """Coerce any supported input into a :class:`Prepared`.

    Accepted: a ``.csv``/``.parquet`` path, a DataFrame (tabular), a dict of columns
    (tabular), a Series / list / 1-D array of text (text mode).
    """
    if isinstance(data, (str, os.PathLike)):
        data = read_table(data)
    if isinstance(data, dict):
        data = frame_from_dict(data)
    if isinstance(data, (set, frozenset)):
        raise TypeError(
            "auto_label: a set has no stable iteration order, so labels could not be matched "
            "back to items; pass a list (for example sorted(data)) or a pandas Series"
        )
    if isinstance(data, pd.DataFrame):
        return _prepare_frame(data)
    if isinstance(data, pd.Series):
        return _prepare_text(list(data.values), list(data.index))
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("auto_label: a numpy array input must be 1-D (one text per element)")
        values = data.tolist()
        return _prepare_text(values, list(range(len(values))))
    if isinstance(data, (bytes, bytearray)):
        raise TypeError("auto_label: bytes input is not supported; decode it to str first")
    if isinstance(data, Iterable):
        values = list(data)
        return _prepare_text(values, list(range(len(values))))
    raise TypeError(
        "auto_label expects a list of strings, a pandas Series or DataFrame, or a "
        f".csv/.parquet path; got {type(data).__name__}"
    )


def frame_from_dict(data: Dict[Any, Any]) -> pd.DataFrame:
    """Build a DataFrame from a dict of columns, reporting problems as auto_label errors."""
    lengths: Dict[Any, int] = {}
    for key, value in data.items():
        if isinstance(value, _COLUMN_TYPES) and not isinstance(value, (str, bytes, bytearray)):
            lengths[key] = len(value)
            continue
        raise ValueError(
            "auto_label: a dict input must map column name -> list of values (column "
            f"{key!r} is a {type(value).__name__}); wrap it in a list or pass a DataFrame"
        )
    if len(set(lengths.values())) > 1:
        shown = ", ".join(f"{k!r}: {n}" for k, n in lengths.items())
        raise ValueError(
            "auto_label: every column of a dict input must have the same length; got " + shown
        )
    try:
        return pd.DataFrame(data)
    except Exception as exc:  # noqa: BLE001 - re-raised with our own context
        raise ValueError(f"auto_label: could not build a table from the dict input: {exc}") from None


def check_columns(df: pd.DataFrame) -> None:
    """Reject a DataFrame with duplicate column names, naming them."""
    mask = df.columns.duplicated()
    if not mask.any():
        return
    dupes: List[str] = []
    for name in df.columns[mask]:
        shown = repr(name)
        if shown not in dupes:
            dupes.append(shown)
    raise ValueError(
        "auto_label: the DataFrame has duplicate column names: "
        + ", ".join(dupes)
        + ". Rename them so every column is unique."
    )


def _prepare_text(values: List[Any], index: List[Any]) -> Prepared:
    texts = [as_text(v) for v in values]
    return Prepared(mode=TEXT, texts=texts, llm_texts=list(texts), items=list(texts), index=index)


def _is_text_column(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


def _prepare_frame(df: pd.DataFrame) -> Prepared:
    check_columns(df)
    frame = df.reset_index(drop=True)
    index = list(df.index)
    n = len(frame)
    if n and frame.shape[1] == 0:
        raise ValueError(
            f"auto_label: the DataFrame has {n} row(s) but no columns, so there is nothing to "
            "label. Select the columns you want (for example df[['text']]) or pass a list or "
            "Series of text."
        )
    text_cols = [c for c in frame.columns if _is_text_column(frame[c])]

    # What keyword / regex rules see: the row's string-like values joined by spaces.
    # A table with no string-like column at all would give every rule an empty string to
    # search, so fall back to every column rendered as text and record that we did.
    fallback = False
    if not text_cols and len(frame.columns):
        text_cols = list(frame.columns)
        fallback = True
    if text_cols and n:
        parts = [[as_text(v) for v in frame[c].tolist()] for c in text_cols]
        texts = [" ".join(p for p in row if p).strip() for row in zip(*parts)]
    else:
        texts = [""] * n

    # What the LLM hook sees: every non-missing cell as "column: value".
    if n:
        columns = [str(c) for c in frame.columns]
        rows = frame.itertuples(index=False, name=None)
        llm_texts = [
            "; ".join(f"{col}: {as_text(v)}" for col, v in zip(columns, row) if not is_missing(v))
            for row in rows
        ]
    else:
        llm_texts = []

    items = frame.to_dict("records") if n else []
    return Prepared(
        mode=TABULAR,
        texts=texts,
        llm_texts=llm_texts,
        items=items,
        index=index,
        frame=frame,
        text_columns=[str(c) for c in text_cols],
        text_fallback=fallback,
    )


def json_safe(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars, NaN and datetimes for ``json.dumps``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series, pd.Index)):
        return [json_safe(v) for v in list(value)]
    if is_missing(value):
        return None
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, _dt.timedelta)):
        return str(value)
    return str(value)
