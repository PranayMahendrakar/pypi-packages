"""The checker: loads the table, samples it, runs every check and assembles the report."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from . import _checks
from ._checks import Config, Frame
from ._io import TableLike, load_table
from ._report import KINDS, HealthReport, Issue

logger = logging.getLogger(__name__)

# Which check function produces which kinds, so a ``checks=`` subset runs only what it needs.
_CHECKS = (
    (_checks.check_empty, ("empty_column", "empty_row")),
    (_checks.check_missing, ("missing",)),
    (_checks.check_duplicates, ("duplicates",)),
    (_checks.check_object_types, ("mixed_types", "numeric_as_strings", "dates_as_strings")),
    (_checks.check_constant, ("constant", "near_constant")),
    (_checks.check_id_like, ("id_like",)),
    (_checks.check_high_cardinality, ("high_cardinality",)),
    (_checks.check_class_imbalance, ("class_imbalance",)),
    (_checks.check_target_leakage, ("target_leakage",)),
    (_checks.check_high_correlation, ("high_correlation",)),
    (_checks.check_outliers, ("outliers",)),
    (_checks.check_skew, ("skew",)),
)


class HealthChecker:
    """Configurable version of :func:`diagnose`; every threshold has a sensible default.

    Parameters
    ----------
    sample:
        Analyze at most this many rows (a seeded random sample); ``None`` or ``0`` uses all.
    random_state:
        Seed for that sampling.
    missing_warning, missing_critical:
        Missing share of a column at/above which it is a warning, and above which it is critical.
    near_constant:
        Share of one value above which a column is near-constant.
    id_unique_ratio:
        Unique ratio above which a text or integer column looks like an identifier.
    high_cardinality:
        Distinct values above which a categorical column is high-cardinality.
    imbalance, imbalance_critical:
        Minority class share below which a classification target is imbalanced (warning / critical).
    leakage_corr:
        |correlation| with the target above which a feature is leakage.
    high_corr:
        |correlation| above which two features are reported as highly correlated.
    outlier_share:
        Share of IQR outliers above which a numeric column is reported.
    skew:
        |skewness| above which a numeric column is reported.
    checks:
        Optional subset of :data:`dataset_health.KINDS` to run; the default runs everything.
    """

    def __init__(
        self,
        *,
        sample: Optional[int] = 200_000,
        random_state: Optional[int] = 0,
        missing_warning: float = 0.05,
        missing_critical: float = 0.5,
        near_constant: float = 0.99,
        id_unique_ratio: float = 0.98,
        high_cardinality: int = 50,
        imbalance: float = 0.10,
        imbalance_critical: float = 0.01,
        leakage_corr: float = 0.95,
        high_corr: float = 0.90,
        outlier_share: float = 0.05,
        skew: float = 3.0,
        checks: Optional[Iterable[str]] = None,
    ) -> None:
        if sample is not None and (isinstance(sample, bool) or not isinstance(sample, int) or sample < 0):
            raise ValueError(f"sample must be a non-negative int or None, not {sample!r}")
        self.sample: Optional[int] = sample or None
        self.random_state = random_state
        self.config = Config(
            missing_warning=missing_warning,
            missing_critical=missing_critical,
            near_constant=near_constant,
            id_unique_ratio=id_unique_ratio,
            high_cardinality=high_cardinality,
            imbalance=imbalance,
            imbalance_critical=imbalance_critical,
            leakage_corr=leakage_corr,
            high_corr=high_corr,
            outlier_share=outlier_share,
            skew=skew,
        )
        if checks is None:
            self.checks: Optional[frozenset] = None
        else:
            wanted = frozenset(str(c) for c in checks)
            unknown = sorted(wanted - set(KINDS))
            if unknown:
                raise ValueError(f"unknown check kinds {unknown}; choose from {list(KINDS)}")
            self.checks = wanted

    def check(self, df_or_path: TableLike, target: Optional[str] = None) -> HealthReport:
        """Run the health checks on a DataFrame or a .csv/.parquet path and return the report."""
        raw = load_table(df_or_path, "df_or_path")
        df, notes = _prepare_columns(raw)
        target_name = None if target is None else str(target)
        if target_name is not None and target_name not in df.columns:
            available = ", ".join(repr(c) for c in list(df.columns)[:20])
            if len(df.columns) > 20:
                available += ", ..."
            raise ValueError(
                f"target {target!r} is not a column of the data; "
                f"available columns: {available or '(none)'}"
            )

        n_rows_total = len(df)
        sampled = False
        if self.sample is not None and n_rows_total > self.sample:
            df = df.sample(n=self.sample, random_state=self.random_state)
            try:
                df = df.sort_index()  # keep the original row order so ordered patterns survive
            except TypeError:
                pass
            sampled = True
            notes.append(
                f"analyzed a random sample of {self.sample:,} of {n_rows_total:,} rows "
                f"(random_state={self.random_state}); counts and shares refer to the sample"
            )
            logger.info("dataset-health: analyzing a sample of %d of %d rows", self.sample, n_rows_total)

        issues: List[Issue] = []
        frame = Frame(df, target_name)
        if n_rows_total == 0 or len(df.columns) == 0:
            what = "rows" if n_rows_total == 0 else "columns"
            issues.append(Issue("critical", "empty_dataset", [], f"the dataset has no {what}", {}))
        else:
            if frame.target is not None and not frame.feature_names:
                notes.append("the target is the only column, so there are no features to check it against")
            for func, kinds in _CHECKS:
                if self.checks is not None and not self.checks.intersection(kinds):
                    continue
                found = func(frame, self.config)
                if self.checks is not None:
                    found = [i for i in found if i.kind in self.checks]
                issues.extend(found)
            notes.extend(frame.notes)

        return HealthReport(
            issues=issues,
            n_rows=n_rows_total,
            n_columns=len(df.columns),
            n_rows_analyzed=len(df),
            sampled=sampled,
            target=frame.target,
            task=frame.task,
            columns=frame.column_profiles(),
            notes=notes,
        )

    __call__ = check


def diagnose(
    df_or_path: TableLike,
    target: Optional[str] = None,
    *,
    sample: Optional[int] = 200_000,
    random_state: Optional[int] = 0,
) -> HealthReport:
    """One-call health report for a DataFrame or a .csv/.parquet file.

    ``target`` names the label column (optional): it switches on the class-imbalance and
    target-leakage checks, and its task (classification or regression) is auto-detected.
    Tables with more than ``sample`` rows are analyzed on a seeded random sample, which
    the report notes. Raises ``ValueError`` if ``target`` is not a column.
    """
    return HealthChecker(sample=sample, random_state=random_state).check(df_or_path, target)


def _prepare_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Shallow copy whose column names are strings (never mutates the input).

    Duplicate names are rejected here, with a message naming them, rather than failing
    deep inside pandas once a check asks for ``df[name]``.
    """
    notes: List[str] = []
    names = [str(c) for c in df.columns]
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    duplicates = sorted({name for name, n in counts.items() if n > 1})
    if duplicates:
        raise ValueError(
            "df_or_path has duplicate column names: "
            + ", ".join(repr(d) for d in duplicates)
            + "; give every column a unique name first, e.g. "
            "df = df.loc[:, ~df.columns.duplicated()]"
        )
    if names == list(df.columns):
        return df, notes
    out = df.copy(deep=False)
    out.columns = pd.Index(names)
    notes.append("column names that were not text were converted to text for this report")
    return out, notes
