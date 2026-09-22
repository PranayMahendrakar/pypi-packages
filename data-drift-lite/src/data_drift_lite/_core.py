"""``detect()`` and ``DriftMonitor``: profile the reference once, compare any number of batches."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import _stats as st
from ._io import TableLike, load_table
from ._report import ColumnDrift, DriftReport, SchemaDrift

logger = logging.getLogger(__name__)

TINY_ROWS = 20  # below this many rows the tests are unreliable; the report says so
MISSING_SHIFT_NOTE = 0.1  # note when the missing share moves by more than this
HIGH_CARDINALITY = 100  # note when a categorical column has more categories than this
_PREVIEW = 5  # how many category names to list in a note

ColumnsArg = Optional[Union[Hashable, Iterable[Hashable]]]


@dataclass
class _Profile:
    """Everything about one reference column that a batch is compared against."""

    name: Hashable
    family: str
    dtype: str
    values: st.ColumnValues
    stats: Dict[str, Any]
    edges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="float64"))
    bin_counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    cat_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return st.kind_of(self.family)


def _profile_column(name: Hashable, series: pd.Series) -> _Profile:
    cv = st.extract(series)
    dtype = str(series.dtype)
    if cv.family == "category":
        counts = st.value_counts(cv.values)
        return _Profile(name, cv.family, dtype, cv, st.category_stats(cv, counts, dtype), cat_counts=counts)
    edges = st.numeric_edges(cv.values)
    return _Profile(
        name,
        cv.family,
        dtype,
        cv,
        st.numeric_stats(cv, dtype),
        edges=edges,
        bin_counts=st.bin_counts(cv.values, edges),
    )


# --------------------------------------------------------------------------- per-column comparison


def _preview(names: Iterable[str]) -> str:
    listed = list(names)
    shown = ", ".join(repr(n) for n in listed[:_PREVIEW])
    extra = len(listed) - _PREVIEW
    return shown + (f" (+{extra} more)" if extra > 0 else "")


def _skipped_note(test_name: str, sides: List[str]) -> str:
    where = " and ".join(sides) if sides else "reference or current"
    return f"{test_name} skipped: no usable values in {where}"


def _is_drifted(
    p_value: Optional[float],
    psi_value: Optional[float],
    threshold: Optional[float],
    psi_threshold: Optional[float],
) -> bool:
    if p_value is not None and threshold is not None and p_value < threshold:
        return True
    if psi_value is not None and psi_threshold is not None and psi_value > psi_threshold:
        return True
    return False


def _compare_numeric(
    profile: _Profile, cv: st.ColumnValues, notes: List[str]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    ref = profile.values
    statistic, p_value = st.ks_test(ref.values, cv.values)
    ref_counts = profile.bin_counts
    cur_counts = st.bin_counts(cv.values, profile.edges)
    if ref.n_missing or cv.n_missing:
        # missing values form their own bin, so a change in missingness shows up in PSI
        ref_counts = np.append(ref_counts, ref.n_missing)
        cur_counts = np.append(cur_counts, cv.n_missing)
    psi_value = st.psi(ref_counts, cur_counts)
    if statistic is None:
        sides = [label for label, x in (("reference", ref), ("current", cv)) if x.values.size == 0]
        notes.append(_skipped_note("KS test", sides))
    if cv.n_nonfinite:
        notes.append(f"current has {cv.n_nonfinite} non-finite value(s), counted as missing")
    return statistic, p_value, psi_value


def _compare_categorical(
    profile: _Profile, cv: st.ColumnValues, notes: List[str]
) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, int]]:
    ref = profile.values
    cur_counts = st.value_counts(cv.values)
    seen = list(profile.cat_counts)
    unseen = {cat: n for cat, n in cur_counts.items() if cat not in profile.cat_counts}
    # PSI: the reference categories plus one pooled bin for anything the reference never saw
    ref_psi = [profile.cat_counts[cat] for cat in seen] + [0]
    cur_psi = [cur_counts.get(cat, 0) for cat in seen] + [sum(unseen.values())]
    # chi-square: every category from either side
    union = seen + list(unseen)
    ref_chi = [profile.cat_counts.get(cat, 0) for cat in union]
    cur_chi = [cur_counts.get(cat, 0) for cat in union]
    if ref.n_missing or cv.n_missing:
        ref_psi.append(ref.n_missing)
        cur_psi.append(cv.n_missing)
        ref_chi.append(ref.n_missing)
        cur_chi.append(cv.n_missing)
    statistic, p_value = st.chi2_test(np.array([ref_chi, cur_chi], dtype="float64"))
    psi_value = st.psi(ref_psi, cur_psi)
    if statistic is None:
        sides = [label for label, x in (("reference", ref), ("current", cv)) if x.n_rows == 0]
        notes.append(_skipped_note("chi-square test", sides))
    if unseen:
        word = "category" if len(unseen) == 1 else "categories"
        notes.append(f"{len(unseen)} {word} unseen in reference: {_preview(unseen)}")
    if len(union) > HIGH_CARDINALITY:
        notes.append(
            f"high cardinality ({len(union)} categories); the chi-square p-value may be unreliable"
        )
    return statistic, p_value, psi_value, cur_counts


def _compare(
    profile: _Profile,
    series: pd.Series,
    threshold: Optional[float],
    psi_threshold: Optional[float],
) -> ColumnDrift:
    cv = st.extract(series)
    dtype = str(series.dtype)
    notes: List[str] = []
    if profile.kind == "numeric":
        test = "ks"
        statistic, p_value, psi_value = _compare_numeric(profile, cv, notes)
        current_stats = st.numeric_stats(cv, dtype)
    else:
        test = "chi2"
        statistic, p_value, psi_value, cur_counts = _compare_categorical(profile, cv, notes)
        current_stats = st.category_stats(cv, cur_counts, dtype)
    before, after = profile.values.missing_share, cv.missing_share
    if abs(after - before) > MISSING_SHIFT_NOTE:
        notes.append(f"missing share moved from {before:.1%} to {after:.1%}")
    return ColumnDrift(
        kind=profile.kind,
        statistic=statistic,
        p_value=p_value,
        psi=psi_value,
        drifted=_is_drifted(p_value, psi_value, threshold, psi_threshold),
        reference_stats=dict(profile.stats),
        current_stats=current_stats,
        name=str(profile.name),
        test=test,
        notes=notes,
    )


# --------------------------------------------------------------------------- inputs and options


def _check_options(threshold: Optional[float], psi_threshold: Optional[float], sample: Optional[int]) -> None:
    if threshold is not None and not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be between 0 and 1, or None to disable the p-value rule")
    if psi_threshold is not None and not (float(psi_threshold) >= 0.0):
        raise ValueError("psi_threshold must be >= 0, or None to disable the PSI rule")
    if sample is not None and (
        isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or sample < 1
    ):
        raise ValueError("sample must be a positive integer, or None to disable sampling")


def _load(source: TableLike, label: str) -> pd.DataFrame:
    frame = load_table(source, label)
    if frame.columns.has_duplicates:
        dupes = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise ValueError(f"{label} has duplicate column names: {', '.join(dupes)}")
    return frame


def _maybe_sample(
    frame: pd.DataFrame, sample: Optional[int], random_state: Any, label: str
) -> Tuple[pd.DataFrame, Optional[str]]:
    n_rows = len(frame)
    if sample is None or n_rows <= sample:
        return frame, None
    sampled = frame.sample(n=int(sample), random_state=random_state)
    note = f"{label} sampled down to {int(sample):,} of {n_rows:,} rows (random_state={random_state!r})"
    return sampled, note


def _tiny_note(n_rows: int, label: str) -> Optional[str]:
    if n_rows >= TINY_ROWS:
        return None
    rows = "row" if n_rows == 1 else "rows"
    return f"{label} has only {n_rows} {rows} (fewer than {TINY_ROWS}); test results are unreliable at this size"


def _normalize_columns(columns: ColumnsArg, frame: pd.DataFrame) -> Optional[List[Hashable]]:
    if columns is None:
        return None
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Iterable):
        requested: List[Hashable] = [columns]
    else:
        requested = list(columns)
    if not requested:
        raise ValueError("columns= must name at least one column; pass None to compare every column")
    unknown = [c for c in requested if c not in frame.columns]
    if unknown:
        raise ValueError(f"columns not found in reference: {', '.join(repr(c) for c in unknown)}")
    ordered: List[Hashable] = []
    for c in requested:
        if c not in ordered:
            ordered.append(c)
    return ordered


# --------------------------------------------------------------------------- public API


class DriftMonitor:
    """Profile a reference dataset once, then check any number of batches against it.

    Takes the same options as :func:`detect`. The reference is loaded, optionally
    sampled and profiled in ``__init__``, so each :meth:`check` only has to look at
    the batch. Use it when scoring many batches against the same training data.

    Attributes:
        columns: the reference columns that batches are compared on.
        reference_rows: rows in the (possibly sampled) reference.
    """

    def __init__(
        self,
        reference: TableLike,
        *,
        columns: ColumnsArg = None,
        threshold: Optional[float] = 0.05,
        psi_threshold: Optional[float] = 0.2,
        sample: Optional[int] = 100_000,
        random_state: Any = 0,
    ) -> None:
        _check_options(threshold, psi_threshold, sample)
        frame = _load(reference, "reference")
        frame, sample_note = _maybe_sample(frame, sample, random_state, "reference")
        self.threshold = threshold
        self.psi_threshold = psi_threshold
        self.sample = sample
        self.random_state = random_state
        self._requested = _normalize_columns(columns, frame)
        self.columns: List[Hashable] = (
            list(self._requested) if self._requested is not None else list(frame.columns)
        )
        self.reference_rows = int(len(frame))
        self._reference_columns = set(frame.columns)
        self._profiles = {name: _profile_column(name, frame[name]) for name in self.columns}
        tiny_note = _tiny_note(self.reference_rows, "reference")
        self._notes = [note for note in (sample_note, tiny_note) if note]
        if tiny_note:
            logger.warning(tiny_note)

    def check(self, batch: TableLike) -> DriftReport:
        """Compare one batch (DataFrame or .csv/.parquet path) against the reference."""
        frame = _load(batch, "current")
        frame, sample_note = _maybe_sample(frame, self.sample, self.random_state, "current")
        notes = list(self._notes)
        if sample_note:
            notes.append(sample_note)
        tiny_note = _tiny_note(len(frame), "current")
        if tiny_note:
            notes.append(tiny_note)
            logger.warning(tiny_note)

        present = set(frame.columns)
        missing = [name for name in self.columns if name not in present]
        if self._requested is None:
            new = [name for name in frame.columns if name not in self._reference_columns]
        else:
            new = []
        dtype_changed: Dict[Hashable, Tuple[str, str]] = {}
        columns: Dict[Hashable, ColumnDrift] = {}
        for name in self.columns:
            if name not in present:
                continue
            profile = self._profiles[name]
            series = frame[name]
            if st.family_of(series.dtype) != profile.family:
                dtype_changed[name] = (profile.dtype, str(series.dtype))
                continue
            columns[name] = _compare(profile, series, self.threshold, self.psi_threshold)

        return DriftReport(
            columns=columns,
            schema=SchemaDrift(missing, new, dtype_changed),
            threshold=self.threshold,
            psi_threshold=self.psi_threshold,
            reference_rows=self.reference_rows,
            current_rows=int(len(frame)),
            notes=notes,
        )


def detect(
    reference: TableLike,
    current: TableLike,
    *,
    columns: ColumnsArg = None,
    threshold: Optional[float] = 0.05,
    psi_threshold: Optional[float] = 0.2,
    sample: Optional[int] = 100_000,
    random_state: Any = 0,
) -> DriftReport:
    """Detect whether ``current`` has drifted from ``reference``, column by column.

    Args:
        reference: training-time data - a DataFrame, a Series, or a .csv/.parquet path.
        current: the data to check - same accepted types.
        columns: compare only these columns (default: every reference column).
        threshold: a column is drifted when its test p-value is below this
            (KS test for numeric columns, chi-square for categorical). ``None``
            switches the p-value rule off.
        psi_threshold: a column is drifted when its Population Stability Index is
            above this. ``None`` switches the PSI rule off.
        sample: cap each side at this many random rows to keep large checks fast;
            ``None`` uses every row.
        random_state: seed for that sampling, so results are reproducible.

    Returns:
        A :class:`DriftReport` with one :class:`ColumnDrift` per compared column
        plus schema differences (missing / new / type-changed columns).
    """
    monitor = DriftMonitor(
        reference,
        columns=columns,
        threshold=threshold,
        psi_threshold=psi_threshold,
        sample=sample,
        random_state=random_state,
    )
    return monitor.check(current)
