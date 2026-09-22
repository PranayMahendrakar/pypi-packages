"""Row grouping: id-like column detection, composite keys, linked ids, exact duplicates.

Every function here returns a compact integer id per row (0..k-1). Rows that share an id
must land on the same side of a split.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

_ID_SUFFIXES = {"id", "ids", "uuid", "guid"}
_ID_ANYWHERE = {"uuid", "guid"}
_ENTITY_WORDS = {
    "customer", "user", "session", "subject", "patient", "account", "device", "household",
    "client", "member", "visitor", "participant", "person", "student", "employee", "player",
    "entity", "case", "tenant", "organisation", "organization", "org", "company", "merchant",
    "driver", "vehicle", "site", "store", "clinic", "hospital", "school", "family",
}
_NUMBER_WORDS = {"no", "num", "number", "nbr", "code", "key", "ref"}
_COMPOUNDS = {w + "id" for w in _ENTITY_WORDS} | {
    "orderid", "itemid", "productid", "docid", "fileid", "groupid", "recordid", "txid",
    "transactionid", "eventid", "requestid", "traceid",
}
_SPLIT_RE = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(name: Any) -> List[str]:
    text = _CAMEL_RE.sub("_", str(name))
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


def is_id_like(name: Any) -> bool:
    """True when a column name looks like an entity identifier (customer_id, userId, uuid...)."""
    tokens = _tokens(name)
    if not tokens:
        return False
    if tokens[-1] in _ID_SUFFIXES:
        return True
    if any(t in _ID_ANYWHERE for t in tokens):
        return True
    if "".join(tokens) in _COMPOUNDS:
        return True
    if len(tokens) == 1 and tokens[0] in _ENTITY_WORDS:
        return True
    if tokens[-1] in _NUMBER_WORDS and any(t in _ENTITY_WORDS for t in tokens[:-1]):
        return True
    return False


def detect_id_columns(df: pd.DataFrame, exclude: Optional[Iterable[Any]] = None) -> List[Any]:
    """Columns that look like ids by name and whose values repeat (so grouping matters).

    Constant columns and columns unique per row are skipped, as are `exclude`d columns.
    """
    skip = set(exclude or ())
    n = len(df)
    found: List[Any] = []
    for col in df.columns:
        if col in skip or not is_id_like(col):
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame):  # duplicated column name
            continue
        if is_float_dtype(series):
            values = series.dropna()
            if len(values) and not np.all(np.mod(values.to_numpy(dtype="float64"), 1) == 0):
                continue  # fractional floats are measurements, not ids
        try:
            n_unique = int(series.nunique(dropna=True))
        except TypeError:
            continue
        if n_unique < 2 or n_unique >= n:
            continue
        found.append(col)
    return found


def _codes(df: pd.DataFrame, col: Any) -> np.ndarray:
    series = df[col]
    if isinstance(series, pd.DataFrame):
        raise ValueError(f"column {col!r} appears more than once in the frame")
    try:
        codes, _ = pd.factorize(series)
    except TypeError as exc:
        raise ValueError(f"group column {col!r} contains unhashable values") from exc
    return np.asarray(codes, dtype=np.int64)


def _isolate_missing(ids: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Give rows with a missing key an id of their own, then renumber compactly."""
    ids = ids.copy()
    if missing.any():
        start = int(ids[~missing].max()) + 1 if (~missing).any() else 0
        ids[missing] = start + np.arange(int(missing.sum()))
    return np.asarray(pd.factorize(ids)[0], dtype=np.int64)


def composite_ids(df: pd.DataFrame, cols: Sequence[Any]) -> np.ndarray:
    """One id per distinct combination of `cols`; rows missing any key stand alone."""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    code_list = [_codes(df, c) for c in cols]
    missing = np.zeros(n, dtype=bool)
    for codes in code_list:
        missing |= codes < 0
    if len(code_list) == 1:
        ids = code_list[0]
    else:
        ids = np.zeros(n, dtype=np.int64)
        for codes in code_list:
            width = int(codes.max()) + 2
            ids = np.asarray(pd.factorize(ids * width + (codes + 1))[0], dtype=np.int64)
    return _isolate_missing(ids, missing)


def link_codes(code_list: Sequence[np.ndarray]) -> np.ndarray:
    """Connected components over rows: two rows are linked when they share a value in any array.

    Negative codes mean "no value" and never link anything.
    """
    n = len(code_list[0])
    labels = np.arange(n, dtype=np.int64)
    while True:
        changed = False
        for codes in code_list:
            valid = codes >= 0
            if not valid.any():
                continue
            current = labels[valid]
            lowest = (
                pd.Series(current).groupby(codes[valid], sort=False).transform("min").to_numpy()
            )
            if np.any(lowest != current):
                labels[valid] = lowest
                changed = True
        if not changed:
            break
    return np.asarray(pd.factorize(labels)[0], dtype=np.int64)


def linked_ids(df: pd.DataFrame, cols: Sequence[Any]) -> np.ndarray:
    """One id per connected component: rows sharing a value in any of `cols` stay together."""
    if len(df) == 0:
        return np.zeros(0, dtype=np.int64)
    return link_codes([_codes(df, c) for c in cols])


def duplicate_ids(df: pd.DataFrame) -> np.ndarray:
    """One id per distinct row (all columns compared); exact duplicates share an id."""
    n = len(df)
    if n == 0 or df.shape[1] == 0:
        return np.arange(n, dtype=np.int64)
    ids = np.zeros(n, dtype=np.int64)
    for i in range(df.shape[1]):
        column = df.iloc[:, i]
        try:
            codes = pd.factorize(column)[0]
        except TypeError:  # unhashable cells (lists, dicts): compare their text form
            codes = pd.factorize(column.astype(str))[0]
        codes = np.asarray(codes, dtype=np.int64) + 1  # missing (-1) becomes a value of its own
        width = int(codes.max()) + 1
        ids = np.asarray(pd.factorize(ids * width + codes)[0], dtype=np.int64)
    return ids
