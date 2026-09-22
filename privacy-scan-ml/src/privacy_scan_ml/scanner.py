"""The scan and mask entry points: one function each, plus a class for full control."""
from __future__ import annotations

import logging
import unicodedata
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ._detectors import (
    ALL_TYPES,
    COLUMN_ONLY_TYPES,
    PATTERN_TYPES,
    Match,
    covers_all_of,
    find_matches,
    resolve,
)
from ._hints import amount_like, hinted_types, id_or_amount_like
from ._io import frame_from_dict, load_table, looks_like_table_path, unreadable_file_suffix
from ._mask import (
    WARNINGS_KEY,
    attach_salt,
    check_salt,
    check_strategy,
    mask_text,
    mask_value,
    new_salt,
    preview,
)
from ._validators import only_digits
from .report import ColumnFinding, Finding, PIIReport

log = logging.getLogger(__name__)

DEFAULT_SAMPLE = 50_000
DEFAULT_MIN_SHARE = 0.2

#: Types that need the column name to agree before a column is flagged. A six digit
#: number or a date is far too common to call personal data on the values alone.
HINT_REQUIRED = frozenset({"date_of_birth", "postal_code"}) | frozenset(COLUMN_ONLY_TYPES)
_CHECKSUM_TYPES = frozenset({"aadhaar", "pan", "credit_card", "email"})
_PROBE_ROWS = 300
_MAX_CELL_CHARS = 4000
_MAX_NAME_TOKENS = 5
_MAX_CATEGORY_CHARS = 40
_TYPE_ORDER = {name: i for i, name in enumerate(ALL_TYPES)}


# --------------------------------------------------------------------------- cell text


def _cell_text(value: Any) -> str:
    """The string a cell should be scanned as (whole-number floats lose the trailing .0)."""
    if isinstance(value, float):
        if value == value and abs(value) < 1e18 and float(value).is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, datetime):
        if (value.hour, value.minute, value.second) == (0, 0, 0):
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _column_strings(series: pd.Series) -> List[str]:
    """Non-null, non-blank cell texts of a column, trimmed and length-capped."""
    values = series.dropna()
    if len(values) == 0:
        return []
    out: List[str] = []
    for value in values.to_numpy(dtype=object):
        text = _cell_text(value).strip()
        if text:
            out.append(text[:_MAX_CELL_CHARS])
    return out


def _cell_matches(text: str, types: Optional[Iterable[str]]) -> List[Match]:
    """Validated, non-overlapping hits inside one cell value."""
    return resolve(find_matches(text, types, covers_all_of(text)))


# --------------------------------------------------------------------------- value shapes


def _is_name_char(ch: str) -> bool:
    """Letters, the punctuation names carry, and combining marks (Devanagari, Tamil, Arabic)."""
    return ch.isalpha() or ch in ".'-" or unicodedata.category(ch) in ("Mn", "Mc")


def _name_like(text: str) -> bool:
    """A short run of alphabetic, mostly capitalised tokens - what a person's name looks like."""
    if not 2 <= len(text) <= 60 or any(ch.isdigit() for ch in text):
        return False
    tokens = [t for t in text.replace(",", " ").split() if t]
    if not 1 <= len(tokens) <= _MAX_NAME_TOKENS:
        return False
    capitalised = 0
    for token in tokens:
        core = token.strip(".'-")
        if not core or not all(_is_name_char(ch) for ch in token):
            return False
        first = core[0]
        # scripts without upper/lower case (Devanagari, CJK) count as capitalised
        if first.isupper() or (not first.islower() and not first.isupper()):
            capitalised += 1
    return capitalised / len(tokens) >= 0.5


def _category_like(text: str) -> bool:
    """Short enough to be a category label rather than a sentence."""
    return 1 <= len(text) <= _MAX_CATEGORY_CHARS


def _phone_shape(strings: Sequence[str], share: float) -> bool:
    """Do the digit lengths themselves say 'phone number' rather than 'id' or 'amount'?"""
    if not strings:
        return False
    sized = sum(1 for s in strings if 10 <= len(only_digits(s)) <= 13)
    return sized / len(strings) >= 0.9 and share >= 0.8


def _confidence(primary: str, share: float, hinted: bool, n_values: int) -> float:
    """How sure the column call is, from the hit share, the column name and the checksum."""
    conf = (0.60 + 0.35 * share) if hinted else (0.45 + 0.40 * share)
    if primary in _CHECKSUM_TYPES:
        conf += 0.10
    if n_values < 5:
        conf -= 0.10
    return round(min(0.99, max(0.15, conf)), 2)


# --------------------------------------------------------------------------- column scan


def _scan_column(
    name: Any,
    series: pd.Series,
    min_share: float,
    warnings: List[str],
) -> Optional[ColumnFinding]:
    """Look at one column and decide which PII types it carries, if any."""
    hints = hinted_types(name)
    strings = _column_strings(series)
    n = len(strings)
    if n == 0:
        if hints:
            warnings.append(f"column {name!r} is named like {sorted(hints)[0]} but holds no values")
        return None

    if n <= _PROBE_ROWS:
        probe = strings
    else:
        probe = [strings[i] for i in np.unique(np.linspace(0, n - 1, _PROBE_ROWS).astype(int))]

    candidates: Set[str] = set()
    for text in probe:
        candidates.update(m.type for m in _cell_matches(text, None))
    candidates |= hints & set(PATTERN_TYPES)
    candidates -= {t for t in HINT_REQUIRED if t not in hints}

    counts: Dict[str, int] = {t: 0 for t in candidates}
    samples: Dict[str, List[str]] = {t: [] for t in candidates}
    if candidates:
        for text in strings:
            seen: Set[str] = set()
            for m in _cell_matches(text, candidates):
                if m.type in seen:
                    continue
                seen.add(m.type)
                counts[m.type] += 1
                bucket = samples[m.type]
                if len(bucket) < 3 and m.text not in bucket:
                    bucket.append(m.text)

    for pii_type in sorted(hints & set(COLUMN_ONLY_TYPES)):
        check = _name_like if pii_type == "person_name" else _category_like
        hits = 0
        bucket: List[str] = []
        for text in strings:
            if check(text):
                hits += 1
                if len(bucket) < 3 and text not in bucket:
                    bucket.append(text)
        counts[pii_type] = hits
        samples[pii_type] = bucket

    kept = {t: c / n for t, c in counts.items() if c > 0 and c / n >= min_share}

    if "phone" in kept and "phone" not in hints:
        numeric = bool(pd.api.types.is_numeric_dtype(series))
        id_named = id_or_amount_like(name)
        # a column named like money is never a phone, however phone-shaped the digits are
        if amount_like(name):
            log.debug("column %r: phone dropped, the name says it holds amounts", name)
            kept.pop("phone")
        elif id_named:
            # Values under a header that says id, ref, serial, count or amount are not
            # phone numbers. Whether they are stored as int64 or as strings says nothing
            # about that - ids arrive as strings all the time (parquet, dtype=str reads,
            # zero-padded keys) and the header is the evidence either way.
            log.debug("column %r: phone dropped, the name says it holds ids or amounts", name)
            kept.pop("phone")
        elif numeric and not _phone_shape(strings, kept["phone"]):
            log.debug("column %r: phone dropped, the digit lengths do not back it up", name)
            kept.pop("phone")

    if not kept:
        best = max(counts.values()) if counts else 0
        if hints and best:
            warnings.append(
                f"column {name!r} is named like {sorted(hints)[0]} but only "
                f"{best / n:.0%} of values validated (below min_share={min_share:g})"
            )
        elif hints & set(PATTERN_TYPES):
            warnings.append(
                f"column {name!r} is named like {sorted(hints & set(PATTERN_TYPES))[0]} "
                "but no value validated"
            )
        return None

    ordered = sorted(kept.items(), key=lambda kv: (-kv[1], _TYPE_ORDER.get(kv[0], 99)))
    primary = ordered[0][0]
    return ColumnFinding(
        column=str(name),
        types=dict(ordered),
        samples=[preview(s, primary) for s in samples.get(primary, [])[:3]],
        confidence=_confidence(primary, kept[primary], primary in hints, n),
        primary=primary,
        dtype=str(series.dtype),
        rows_checked=n,
        hinted=primary in hints,
    )


# --------------------------------------------------------------------------- input plumbing


def check_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reject a frame with duplicate column names, naming them."""
    duplicated = df.columns[df.columns.duplicated()]
    if len(duplicated):
        names = ", ".join(sorted({str(c) for c in duplicated}))
        raise ValueError(f"duplicate column names are ambiguous, rename them first: {names}")
    return df


def _check_sample(sample: Optional[int]) -> Optional[int]:
    if sample is None:
        return None
    sample = int(sample)
    if sample < 1:
        raise ValueError(f"sample must be a positive number of rows or None, got {sample}")
    return sample


def _check_min_share(min_share: float) -> float:
    min_share = float(min_share)
    if not 0.0 <= min_share <= 1.0:
        raise ValueError(f"min_share must be between 0 and 1, got {min_share}")
    return min_share


def _as_documents(data: Any) -> Optional[List[str]]:
    """Free text input as a list of documents, or None when the input is tabular."""
    if isinstance(data, str):
        if looks_like_table_path(data):
            return None
        suffix = unreadable_file_suffix(data)
        if suffix is not None:
            raise ValueError(
                f"unsupported file type {suffix!r}; expected .csv, .tsv or .parquet"
            )
        return [data]
    if isinstance(data, (list, tuple)):
        if not data:
            return []
        if all(isinstance(x, str) for x in data):
            return list(data)
    return None


def _as_frame(data: Any) -> Tuple[pd.DataFrame, bool]:
    """Tabular input as a frame; the flag says whether a Series went in."""
    if isinstance(data, pd.Series):
        name = data.name if data.name is not None else "value"
        return check_frame(data.to_frame(name=name)), True
    if isinstance(data, dict):
        return check_frame(frame_from_dict(data)), False
    return check_frame(load_table(data)), False


# --------------------------------------------------------------------------- the scanner


def _resolve_salt(strategy: str, salt: Optional[str]) -> Optional[str]:
    """The salt a hash run should use: the caller's, or a fresh one recorded on the result."""
    check_salt(salt)
    if strategy != "hash":
        return None
    return salt if salt is not None else new_salt()


class PIIScanner:
    """The scanner behind :func:`scan`, for when you want to reuse settings.

    ``sample`` caps how many rows are read (spread evenly over the frame, so the result is
    the same on every run). ``min_share`` is the share of a column's non-null values that
    must validate before the column is called personal data.
    """

    def __init__(
        self,
        *,
        sample: Optional[int] = DEFAULT_SAMPLE,
        min_share: float = DEFAULT_MIN_SHARE,
    ) -> None:
        self.sample = _check_sample(sample)
        self.min_share = _check_min_share(min_share)
        #: The salt used by the most recent ``hash`` mask, generated when none was given.
        self.last_salt: Optional[str] = None
        #: Anything the most recent mask could not do, e.g. a flagged column left as it was.
        self.last_warnings: List[str] = []

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"PIIScanner(sample={self.sample}, min_share={self.min_share})"

    # -- scanning ----------------------------------------------------------
    def scan(self, data: Any) -> PIIReport:
        """Scan a DataFrame, a csv/parquet path, a string or a list of strings."""
        docs = _as_documents(data)
        if docs is not None:
            return self.scan_text(docs)
        frame, _ = _as_frame(data)
        return self.scan_frame(frame)

    def scan_frame(self, df: pd.DataFrame) -> PIIReport:
        """Scan an already-loaded DataFrame."""
        check_frame(df)
        warnings: List[str] = []
        rows = int(len(df))
        if self.sample is not None and rows > self.sample:
            idx = np.unique(np.linspace(0, rows - 1, self.sample).astype(np.int64))
            frame = df.iloc[idx]
        else:
            frame = df
        report = PIIReport(
            mode="table",
            rows=rows,
            rows_scanned=int(len(frame)),
            n_columns=int(df.shape[1]),
            min_share=self.min_share,
        )
        if rows == 0 or df.shape[1] == 0:
            report.warnings = ["input has no rows or no columns, nothing to scan"]
            return report
        for name in df.columns:
            finding = _scan_column(name, frame[name], self.min_share, warnings)
            if finding is not None:
                report.columns[str(name)] = finding
        report.warnings = warnings
        return report

    def scan_text(self, docs: Sequence[str]) -> PIIReport:
        """Scan one or more pieces of free text."""
        report = PIIReport(mode="text", documents=len(docs))
        for i, doc in enumerate(docs):
            if not isinstance(doc, str):
                raise TypeError(f"document {i} is {type(doc).__name__}, expected str")
            for m in _cell_matches(doc, None):
                report.findings.append(
                    Finding(m.type, preview(m.text, m.type), (m.start, m.end), i)
                )
        if not report.findings and any(docs):
            report.warnings.append("no personal data matched in the text provided")
        return report

    # -- masking -----------------------------------------------------------
    def mask(
        self,
        data: Any,
        *,
        strategy: str = "redact",
        columns: Optional[Sequence[Any]] = None,
        salt: Optional[str] = None,
    ) -> Any:
        """Mask the personal data in ``data``, returning the same type that went in."""
        check_strategy(strategy)
        salt = _resolve_salt(strategy, salt)
        self.last_salt = salt
        self.last_warnings = []
        docs = _as_documents(data)
        if docs is not None:
            if columns is not None:
                raise ValueError("columns is only meaningful for tabular input")
            masked = [mask_text(doc, strategy, salt) for doc in docs]
            if isinstance(data, str):
                return attach_salt(masked[0], salt)
            out_seq = tuple(masked) if isinstance(data, tuple) else masked
            return attach_salt(out_seq, salt)
        frame, was_series = _as_frame(data)
        out = self.mask_frame(frame, strategy=strategy, columns=columns, salt=salt)
        if was_series:
            series = out[out.columns[0]]
            series.name = data.name
            return attach_salt(series, salt)
        return out

    def mask_frame(
        self,
        df: pd.DataFrame,
        *,
        strategy: str = "redact",
        columns: Optional[Sequence[Any]] = None,
        salt: Optional[str] = None,
    ) -> pd.DataFrame:
        """Mask a DataFrame; columns nothing matched in keep their dtype exactly."""
        check_strategy(strategy)
        check_frame(df)
        salt = _resolve_salt(strategy, salt)
        self.last_salt = salt
        out = df.copy()
        targets: Dict[Any, List[str]] = {}
        if columns is None:
            # report keys are strings; map them back to the frame's own column objects
            by_name = {str(c): c for c in df.columns}
            for name, finding in self.scan_frame(df).columns.items():
                targets[by_name.get(name, name)] = list(finding.types)
        else:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"columns not in the frame: {', '.join(map(str, missing))}")
            for name in columns:
                found = self.scan_frame(df[[name]]).columns.get(str(name))
                if found is not None:
                    targets[name] = list(found.types)
                else:
                    hints = sorted(hinted_types(name))
                    targets[name] = [hints[0]] if hints else ["pii"]
        untouched: List[str] = []
        for name, types in targets.items():
            masked = _mask_column(df[name], types, strategy, salt)
            if masked is not None:
                out[name] = masked
            else:
                untouched.append(str(name))
        warnings = []
        if untouched:
            # The failure this package exists to prevent: a column the scan flagged that
            # comes back byte-identical, and the caller ships it believing it is clean.
            warnings.append(
                "masking changed nothing in " + ", ".join(untouched)
                + "; those columns were left exactly as they were"
            )
            for line in warnings:
                log.warning("%s", line)
        self.last_warnings = warnings
        out.attrs[WARNINGS_KEY] = list(warnings)
        return attach_salt(out, salt)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NaT:
        return True
    return isinstance(value, float) and value != value


def _mask_column(
    series: pd.Series,
    types: Sequence[str],
    strategy: str,
    salt: Optional[str],
) -> Optional[pd.Series]:
    """A masked copy of one column, or None when nothing in it changed."""
    pattern_types = [t for t in types if t in PATTERN_TYPES]
    whole_types = [t for t in types if t not in PATTERN_TYPES]
    if whole_types or not pattern_types:
        label = whole_types[0] if whole_types else (types[0] if types else "pii")

        def transform(text: str) -> str:
            return mask_value(text, label, strategy, salt)

    else:

        def transform(text: str) -> str:
            return mask_text(text, strategy, salt, pattern_types)

    changed = False
    values: List[Any] = []
    for value in series.to_numpy(dtype=object):
        if _is_missing(value):
            values.append(value)
            continue
        text = _cell_text(value)
        new = transform(text)
        if new != text:
            changed = True
        values.append(new)
    if not changed:
        return None  # nothing matched: leave the column, and its dtype, alone
    masked = pd.Series(values, index=series.index, name=series.name, dtype=object)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return masked.astype("category")
    return masked


# --------------------------------------------------------------------------- functions


def scan(
    data: Any,
    *,
    sample: Optional[int] = DEFAULT_SAMPLE,
    min_share: float = DEFAULT_MIN_SHARE,
) -> PIIReport:
    """Find personal data in a table or in free text.

    Args:
        data: a pandas DataFrame or Series, a path to a ``.csv``/``.tsv``/``.parquet`` file,
            a dict of columns, a string of free text, or a list of strings.
        sample: how many rows to read from a big table, spread evenly (``None`` reads all).
        min_share: the share of a column's non-null values that must validate before the
            column is reported.

    Returns:
        A :class:`~privacy_scan_ml.report.PIIReport`. Every value it carries is masked.
    """
    return PIIScanner(sample=sample, min_share=min_share).scan(data)


def mask(
    data: Any,
    *,
    strategy: str = "redact",
    columns: Optional[Sequence[Any]] = None,
    salt: Optional[str] = None,
) -> Any:
    """Replace the personal data in ``data`` and hand back the same type that went in.

    Args:
        data: the same inputs :func:`scan` accepts (a path gives you the masked DataFrame).
        strategy: ``"redact"`` for ``[EMAIL]`` style tokens, ``"hash"`` for
            ``sha256(salt + value)[:12]``, ``"partial"`` to keep the last four characters.
        columns: mask exactly these columns instead of the ones the scan flags.
        salt: mixed into the hash so the same value is not identifiable across datasets.

    Returns:
        The masked copy. Columns that nothing matched in are left untouched, dtype included.
    """
    return PIIScanner().mask(data, strategy=strategy, columns=columns, salt=salt)
