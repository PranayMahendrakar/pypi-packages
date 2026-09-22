"""The checks themselves: :class:`FeatureChecker` and the :func:`check` shortcut."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._columns import (
    BOOL,
    CATEGORICAL,
    DATETIME,
    FLOAT_KINDS,
    NUMERIC,
    ColumnProfile,
    canonical_object,
    codes_of,
    jsonable,
    looks_like_dates,
    profile_column,
    suspicious_name_token,
    to_float,
)
from ._io import FrameLike, load_frame
from ._models import forest_importance, stump_score
from ._stats import (
    cramers_v,
    loo_mapping_accuracy,
    loo_mapping_r2,
    pairwise_pearson,
    pearson_spearman,
)
from .report import DROP_SEVERITIES, FeatureReport, Finding, rank_columns, sort_findings

log = logging.getLogger(__name__)

#: every ``Finding.kind`` this package can produce
KINDS: Tuple[str, ...] = (
    "constant",
    "near_constant",
    "high_missing",
    "id_like",
    "duplicate_of",
    "highly_correlated_with",
    "leakage_suspect",
    "date_as_string",
    "suspicious_name",
    "zero_importance",
)


# --------------------------------------------------------------------------- helpers


def _unit(value: float, name: str, *, low: float = 0.0) -> float:
    """Validate a threshold that must sit in ``(low, 1]``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in ({low:g}, 1], got {value!r}") from None
    if not np.isfinite(out) or not (low < out <= 1.0):
        raise ValueError(f"{name} must be a number in ({low:g}, 1], got {value!r}")
    return out


def _seed(value: Any, name: str) -> int:
    """Validate ``random_state``, in the same voice as the other parameters."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{name} must be an integer, got {value!r}")


def _positive_int(value: int, name: str) -> int:
    """Validate a row/column budget that must be a positive integer."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if out < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return out


def _pct(fraction: float) -> str:
    """``0.9312`` -> ``'93.1%'``."""
    return f"{100.0 * float(fraction):.1f}%"


def _show(value: Any, limit: int = 40) -> str:
    """Short, quoted, single-line rendering of a cell value for a message."""
    text = repr(value)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text.replace("\n", " ").replace("\r", " ")


def _name(col: Hashable) -> str:
    return repr(str(col))


def _identical(a: pd.Series, b: pd.Series, kind_a: str, kind_b: str) -> bool:
    """True when two columns hold exactly the same values (missing matches missing)."""
    numeric_a = kind_a in FLOAT_KINDS
    numeric_b = kind_b in FLOAT_KINDS
    if numeric_a != numeric_b:
        return False
    if numeric_a:
        return bool(np.array_equal(to_float(a), to_float(b), equal_nan=True))
    miss_a = a.isna().to_numpy()
    miss_b = b.isna().to_numpy()
    if not np.array_equal(miss_a, miss_b):
        return False
    left = a.astype(object).to_numpy()[~miss_a]
    right = b.astype(object).to_numpy()[~miss_b]
    if left.shape != right.shape:
        return False
    if left.size == 0:
        return True
    try:
        return bool(np.all(left == right))
    except (ValueError, TypeError):  # pragma: no cover - exotic cell types
        return list(left) == list(right)


def _value_key(s: pd.Series, kind: str) -> str:
    """Cheap digest of a column's values, used to shortlist duplicate candidates."""
    if kind in FLOAT_KINDS:
        payload = np.ascontiguousarray(to_float(s), dtype=np.float64).tobytes()
        tag = b"num"
    else:
        missing = s.isna().to_numpy()
        text = canonical_object(s).astype(str).to_numpy()
        joined = "\x01".join(np.where(missing, "\x00", text).tolist())
        payload = joined.encode("utf-8", "surrogatepass")
        tag = b"cat"
    return hashlib.blake2b(payload, digest_size=16, person=tag).hexdigest()


# --------------------------------------------------------------------------- checker


class FeatureChecker:
    """Runs every feature check; :func:`check` is the one-line front door.

    The defaults are the ones :func:`check` documents. The extra keyword
    arguments here exist for callers who need to tune the model-backed checks.
    """

    def __init__(
        self,
        *,
        corr_threshold: float = 0.95,
        missing_threshold: float = 0.6,
        cardinality_threshold: float = 0.98,
        sample: int = 200_000,
        random_state: int = 0,
        near_constant_threshold: float = 0.99,
        leakage_score_threshold: float = 0.99,
        leakage_corr_threshold: float = 0.95,
        min_id_rows: int = 20,
        max_model_rows: int = 20_000,
        max_model_categories: int = 100,
        max_pair_columns: int = 400,
        min_corr_rows: int = 20,
        max_importance_columns: int = 200,
        importance_cell_budget: int = 600_000,
    ) -> None:
        self.corr_threshold = _unit(corr_threshold, "corr_threshold")
        self.missing_threshold = _unit(missing_threshold, "missing_threshold")
        self.cardinality_threshold = _unit(cardinality_threshold, "cardinality_threshold")
        self.sample = _positive_int(sample, "sample")
        self.random_state = _seed(random_state, "random_state")
        self.near_constant_threshold = _unit(near_constant_threshold, "near_constant_threshold")
        self.leakage_score_threshold = _unit(leakage_score_threshold, "leakage_score_threshold")
        self.leakage_corr_threshold = _unit(leakage_corr_threshold, "leakage_corr_threshold")
        self.min_id_rows = _positive_int(min_id_rows, "min_id_rows")
        self.max_model_rows = _positive_int(max_model_rows, "max_model_rows")
        self.max_model_categories = _positive_int(max_model_categories, "max_model_categories")
        self.max_pair_columns = _positive_int(max_pair_columns, "max_pair_columns")
        self.min_corr_rows = _positive_int(min_corr_rows, "min_corr_rows")
        self.max_importance_columns = _positive_int(
            max_importance_columns, "max_importance_columns"
        )
        self.importance_cell_budget = _positive_int(
            importance_cell_budget, "importance_cell_budget"
        )

    # ------------------------------------------------------------------ public
    def check(self, data: FrameLike, target: Optional[Hashable] = None) -> FeatureReport:
        """Check every column of ``data`` except ``target`` and return a report."""
        df = load_frame(data)
        _require_unique_columns(df)
        _require_distinct_labels(df)
        n_rows = int(len(df))
        if target is not None and target not in set(df.columns):
            known = ", ".join(repr(str(c)) for c in list(df.columns)[:12])
            raise ValueError(f"target {str(target)!r} is not a column; columns are: {known}")

        order: List[Hashable] = [c for c in df.columns if target is None or c != target]
        notes: List[str] = []

        work = df
        if n_rows > self.sample:
            work = df.sample(n=self.sample, random_state=self.random_state)
            notes.append(
                f"checked a random sample of {self.sample:,} of the {n_rows:,} rows "
                f"(random_state={self.random_state})"
            )
        n_checked = int(len(work))

        if not order:
            notes.append("the frame has no feature columns to check")
            return self._empty(target, notes, n_rows, n_checked)
        if n_checked == 0:
            notes.append("the frame has no rows; every check was skipped")
            return self._empty(target, notes, n_rows, n_checked, order=order)

        profiles = {c: profile_column(c, work[c]) for c in order}
        findings: Dict[Hashable, List[Finding]] = {c: [] for c in order}

        def add(col: Hashable, kind: str, severity: str, message: str, detail: Dict[str, Any]) -> None:
            findings[col].append(
                Finding(kind=kind, severity=severity, message=message, detail={k: jsonable(v) for k, v in detail.items()})
            )

        constant = self._check_basics(work, order, profiles, add)
        self._note_too_few_rows(n_checked, order, constant, notes)
        live = [c for c in order if c not in constant]
        dates = self._check_dates(work, live, profiles, add)
        id_like = self._check_id_like(live, profiles, dates, add)
        self._check_names(live, add)
        duplicates = self._check_duplicates(work, live, profiles, add)
        redundant = [c for c in live if c not in duplicates]
        self._check_numeric_correlation(work, redundant, profiles, findings, notes, add)
        self._check_categorical_correlation(work, redundant, profiles, id_like, findings, notes, add)
        self._check_target(work, target, redundant, profiles, id_like, notes, add)

        for note in notes:
            log.debug("ml-feature-check: %s", note)

        features = {c: sort_findings(findings[c]) for c in order}
        drop, keep = rank_columns(features, order)
        return FeatureReport(
            features=features,
            drop_recommended=drop,
            keep=keep,
            target=target,
            notes=notes,
            n_rows=n_rows,
            n_rows_checked=n_checked,
            params=self._params(),
        )

    # ------------------------------------------------------------------ checks
    def _check_basics(self, work, order, profiles, add) -> set:
        """constant, high_missing, near_constant. Returns the constant columns."""
        constant = set()
        for col in order:
            p: ColumnProfile = profiles[col]
            if p.nunique <= 1:
                constant.add(col)
                if p.n_nonnull == 0:
                    add(
                        col,
                        "constant",
                        "high",
                        f"every one of the {p.n:,} values is missing",
                        {"n_rows": p.n, "n_missing": p.n, "nunique": 0},
                    )
                else:
                    extra = "" if p.missing_frac == 0 else f", the other {_pct(p.missing_frac)} missing"
                    add(
                        col,
                        "constant",
                        "high",
                        f"only one distinct value ({_show(p.top_value)}){extra}",
                        {"value": p.top_value, "missing_frac": p.missing_frac, "nunique": 1},
                    )
                continue
            if p.missing_frac > self.missing_threshold:
                add(
                    col,
                    "high_missing",
                    "high" if p.missing_frac >= 0.95 else "medium",
                    f"{_pct(p.missing_frac)} of values are missing "
                    f"(threshold {_pct(self.missing_threshold)})",
                    {"missing_frac": p.missing_frac, "n_missing": p.n - p.n_nonnull, "threshold": self.missing_threshold},
                )
            if p.top_frac > self.near_constant_threshold:
                add(
                    col,
                    "near_constant",
                    "medium",
                    f"{_pct(p.top_frac)} of the non-missing values are {_show(p.top_value)}",
                    {"value": p.top_value, "top_frac": p.top_frac, "nunique": p.nunique},
                )
        return constant

    @staticmethod
    def _note_too_few_rows(n_checked, order, constant, notes) -> None:
        """Say so when a verdict is really a symptom of the frame being too short.

        One row makes every column constant by arithmetic, not by evidence, and a
        three-row frame is barely better. Left unsaid the report reads as "drop
        everything", which is exactly the silent degenerate result the family
        conventions forbid an automatic verdict to produce.
        """
        if not order or n_checked > 3 or not constant:
            return
        if n_checked == 1:
            notes.append(
                "only 1 row: every column holds a single value, so the 'constant' verdicts "
                "below are an artifact of the row count, not evidence that a column is dead"
            )
            return
        if len(constant) * 2 >= len(order):
            notes.append(
                f"only {n_checked} rows: {len(constant)} of {len(order)} column(s) look "
                "constant, which this few rows cannot tell apart from a genuinely dead "
                "column; re-run on more rows before trusting the drop list"
            )

    def _check_dates(self, work, live, profiles, add) -> set:
        """date_as_string on text columns that are really dates."""
        dates = set()
        for col in live:
            if profiles[col].kind != CATEGORICAL:
                continue
            nonnull = work[col].dropna()
            if nonnull.empty:
                continue
            example = looks_like_dates(nonnull)
            if example is None:
                continue
            dates.add(col)
            add(
                col,
                "date_as_string",
                "low",
                f"text that parses as dates (for example {_show(example)}): convert with "
                "pandas.to_datetime and feed the parts (year, month, weekday) to the model",
                {"example": example, "dtype": profiles[col].dtype},
            )
        return dates

    def _check_id_like(self, live, profiles, dates, add) -> set:
        """id_like on columns with about one distinct value per row."""
        id_like = set()
        for col in live:
            p: ColumnProfile = profiles[col]
            if col in dates or p.n_nonnull < self.min_id_rows:
                continue
            if p.unique_ratio <= self.cardinality_threshold:
                continue
            if p.kind == NUMERIC and not p.integer_like:
                continue  # a continuous measurement is expected to be all-distinct
            detail = {
                "nunique": p.nunique,
                "n_nonnull": p.n_nonnull,
                "unique_ratio": p.unique_ratio,
                "threshold": self.cardinality_threshold,
            }
            if p.kind == DATETIME:
                add(
                    col,
                    "id_like",
                    "medium",
                    f"{_pct(p.unique_ratio)} of the {p.n_nonnull:,} timestamps are distinct: "
                    "use parts (hour, weekday, age) rather than the raw value",
                    detail,
                )
            else:
                consecutive = p.span is not None and p.span <= 1.05 * max(p.nunique, 1)
                extra = " and they run consecutively" if consecutive else ""
                add(
                    col,
                    "id_like",
                    "high",
                    f"{_pct(p.unique_ratio)} of the {p.n_nonnull:,} values are distinct{extra}: "
                    "looks like a row identifier, not a feature",
                    detail,
                )
            id_like.add(col)
        return id_like

    def _check_names(self, live, add) -> None:
        """suspicious_name on identifier-ish column names."""
        for col in live:
            token = suspicious_name_token(col)
            if token is None:
                continue
            add(
                col,
                "suspicious_name",
                "low",
                f"the name contains {_show(token)}, which usually marks a row identifier "
                "or bookkeeping column rather than a feature",
                {"token": token, "name": str(col)},
            )

    def _check_duplicates(self, work, live, profiles, add) -> Dict[Hashable, Hashable]:
        """duplicate_of: columns identical to an earlier column."""
        seen: Dict[str, List[Hashable]] = {}
        duplicates: Dict[Hashable, Hashable] = {}
        for col in live:
            kind = profiles[col].kind
            key = _value_key(work[col], kind)
            match = None
            for other in seen.get(key, ()):
                if _identical(work[col], work[other], kind, profiles[other].kind):
                    match = other
                    break
            if match is None:
                seen.setdefault(key, []).append(col)
                continue
            duplicates[col] = match
            add(
                col,
                "duplicate_of",
                "high",
                f"identical to column {_name(match)}: keep one of the two",
                {"other": str(match), "dtype": profiles[col].dtype},
            )
        return duplicates

    @staticmethod
    def _representative_first(
        cols: Sequence[Hashable],
        findings: Dict[Hashable, List[Finding]],
        profiles: Dict[Hashable, ColumnProfile],
    ) -> List[Hashable]:
        """Order a redundancy group so the survivor is a column worth keeping.

        The correlation checks keep the first column of each group they meet and
        flag the rest, so whichever column comes first decides the verdict. Left
        in frame order that hands the group to whatever the loader happened to
        put on the left, which can be a column this same report already wants
        dropped - the user is then told to keep a 90%-missing column and drop
        the complete one beside it. Sorting by usefulness first (no drop-level
        finding, then fewest missing values) makes the survivor the best column
        in the group, so which of the two is dropped no longer depends on which
        one the frame happens to list first. Columns that are equally good keep
        their frame order, which is the rule ``duplicate_of`` already follows.
        """
        pos = {c: i for i, c in enumerate(cols)}

        def rank(col: Hashable) -> Tuple[int, int, int]:
            doomed = any(f.severity in DROP_SEVERITIES for f in findings.get(col, ()))
            prof = profiles[col]
            missing = int(prof.n) - int(prof.n_nonnull)
            return (1 if doomed else 0, missing, pos[col])

        return sorted(cols, key=rank)

    def _check_numeric_correlation(self, work, cols, profiles, findings, notes, add) -> None:
        """highly_correlated_with for numeric, bool and datetime columns."""
        numeric = [c for c in cols if profiles[c].kind in FLOAT_KINDS]
        if len(numeric) < 2:
            return
        if len(numeric) > self.max_pair_columns:
            notes.append(
                f"the correlation check looked at the first {self.max_pair_columns} of "
                f"{len(numeric)} numeric columns"
            )
            numeric = numeric[: self.max_pair_columns]
        numeric = self._representative_first(numeric, findings, profiles)
        rows = len(work)
        budget = max(1_000, 20_000_000 // max(len(numeric), 1))
        frame = work
        if rows > budget:
            frame = work.sample(n=budget, random_state=self.random_state)
            notes.append(
                f"the correlation check used {budget:,} of the {rows:,} rows to stay within memory"
            )
        matrix = np.column_stack([to_float(frame[c]) for c in numeric])
        # A drop recommendation has to rest on more than a handful of rows. The old
        # floor of 3-10 shared rows let two mostly-missing columns that happened to
        # overlap on five rows condemn one of the pair, so the floor is now 20 rows
        # or 5% of the frame, whichever is larger, and the count that a coefficient
        # actually rests on is reported with it.
        min_periods = max(self.min_corr_rows, int(round(0.05 * len(frame))))
        corr, shared = pairwise_pearson(matrix, min_periods=min_periods, return_counts=True)
        kept: List[int] = []
        for j, col in enumerate(numeric):
            hit = None
            value = float("nan")
            for i in kept:
                r = corr[i, j]
                if np.isfinite(r) and abs(r) > self.corr_threshold:
                    hit, value = i, float(r)
                    break
            if hit is None:
                kept.append(j)
                continue
            n_shared = int(shared[hit, j])
            over = "" if n_shared >= len(frame) else f" over {n_shared:,} shared non-missing rows"
            add(
                col,
                "highly_correlated_with",
                "medium",
                f"correlation {value:+.3f} with column {_name(numeric[hit])}{over}: "
                "the two carry the same signal",
                {
                    "other": str(numeric[hit]),
                    "corr": value,
                    "method": "pearson",
                    "n_shared": n_shared,
                    "min_shared": int(min_periods),
                    "threshold": self.corr_threshold,
                },
            )

    def _check_categorical_correlation(self, work, cols, profiles, id_like, findings, notes, add) -> None:
        """highly_correlated_with for categorical columns, via Cramer's V."""
        cats = [
            c
            for c in cols
            if profiles[c].kind == CATEGORICAL
            and c not in id_like
            and 2 <= profiles[c].nunique <= self.max_model_categories * 10
        ]
        if len(cats) < 2:
            return
        if len(cats) > self.max_pair_columns:
            notes.append(
                f"the Cramer's V check looked at the first {self.max_pair_columns} of "
                f"{len(cats)} categorical columns"
            )
            cats = cats[: self.max_pair_columns]
        cats = self._representative_first(cats, findings, profiles)
        codes = {c: codes_of(work[c]) for c in cats}
        kept: List[Hashable] = []
        for col in cats:
            a, n_a = codes[col]
            hit = None
            value = float("nan")
            for other in kept:
                b, n_b = codes[other]
                v = cramers_v(a, b, n_a, n_b)
                if np.isfinite(v) and v > self.corr_threshold:
                    hit, value = other, float(v)
                    break
            if hit is None:
                kept.append(col)
                continue
            add(
                col,
                "highly_correlated_with",
                "medium",
                f"Cramer's V {value:.3f} with column {_name(hit)}: the two carry the same signal",
                {"other": str(hit), "corr": value, "method": "cramers_v", "threshold": self.corr_threshold},
            )

    # ------------------------------------------------------------------ target
    def _check_target(self, work, target, cols, profiles, id_like, notes, add) -> None:
        """leakage_suspect and zero_importance; both need a target."""
        if target is None:
            notes.append(
                "no target given: the leakage_suspect and zero_importance checks were skipped"
            )
            return
        series = work[target]
        prepared = self._encode_target(series)
        if prepared is None:
            notes.append(
                f"target {str(target)!r} has fewer than two usable values: the leakage_suspect "
                "and zero_importance checks were skipped"
            )
            return
        mask, y, classification, n_classes = prepared
        n_target = int(mask.sum())
        if n_target < 4:
            notes.append(
                f"only {n_target} row(s) have a usable target: the leakage_suspect and "
                "zero_importance checks were skipped"
            )
            return
        if n_target != len(work):
            notes.append(
                f"{len(work) - n_target:,} row(s) with a missing target were left out of the "
                "leakage_suspect and zero_importance checks"
            )
        kind = "classification" if classification else "regression"
        notes.append(f"target {str(target)!r} was treated as {kind}")

        self._check_leakage(work, cols, profiles, mask, y, classification, n_classes, target, add)
        self._check_importance(work, cols, profiles, id_like, mask, y, classification, notes, add)

    def _encode_target(self, series: pd.Series):
        """``(row_mask, y, classification, n_classes)`` or ``None`` if unusable."""
        profile = profile_column("target", series)
        if profile.nunique < 2:
            return None
        # BOOL is in FLOAT_KINDS (it has a meaningful float view) but a True/False
        # label is binary classification, not regression: without this clause a bool
        # target was scored with R2 and a near-perfect leak went unreported.
        classification = (
            profile.kind not in FLOAT_KINDS
            or profile.kind == BOOL
            or (profile.kind == NUMERIC and profile.integer_like and profile.nunique <= 20)
        )
        if classification:
            codes, n_classes = codes_of(series)
            mask = codes >= 0
            y = codes[mask].astype(np.int64)
            if n_classes < 2 or len(np.unique(y)) < 2:
                return None
            return mask, y, True, int(n_classes)
        values = to_float(series)
        mask = np.isfinite(values)
        y = values[mask]
        if y.size < 2 or float(np.nanmax(y) - np.nanmin(y)) == 0.0:
            return None
        return mask, y, False, 0

    def _check_leakage(self, work, cols, profiles, mask, y, classification, n_classes, target, add) -> None:
        """leakage_suspect: one feature that already knows the answer."""
        for col in cols:
            p: ColumnProfile = profiles[col]
            if p.kind in FLOAT_KINDS:
                x = to_float(work[col])[mask]
                ok = np.isfinite(x)
                if int(ok.sum()) < 4:
                    continue
                xs, ys = x[ok], y[ok]
                if not classification:
                    pearson, spearman = pearson_spearman(xs, ys)
                    best = max(
                        abs(pearson) if np.isfinite(pearson) else 0.0,
                        abs(spearman) if np.isfinite(spearman) else 0.0,
                    )
                    if best > self.leakage_corr_threshold:
                        add(
                            col,
                            "leakage_suspect",
                            "high",
                            f"correlates {best:.3f} with the target {_name(target)}: "
                            "check that it is not the answer in disguise",
                            {
                                "target": str(target),
                                "pearson": pearson,
                                "spearman": spearman,
                                "threshold": self.leakage_corr_threshold,
                            },
                        )
                        continue
                if classification and len(np.unique(ys)) < 2:
                    continue
                scored = stump_score(xs, ys, classification=classification, depth=3, random_state=self.random_state)
                if scored is None:
                    continue
                score, baseline = scored
                if score > self.leakage_score_threshold and score > baseline:
                    metric = "accuracy" if classification else "R2"
                    add(
                        col,
                        "leakage_suspect",
                        "high",
                        f"a 1-feature decision tree reaches {score:.3f} {metric} on held-out rows "
                        f"for target {_name(target)} (baseline {baseline:.3f})",
                        {
                            "target": str(target),
                            "score": score,
                            "baseline": baseline,
                            "metric": metric,
                            "threshold": self.leakage_score_threshold,
                        },
                    )
                continue

            codes, n_codes = codes_of(work[col])
            codes = codes[mask]
            if n_codes < 2:
                continue
            if classification:
                result = loo_mapping_accuracy(codes, y, n_classes)
                if result is None:
                    continue
                accuracy, majority, n_used, n_groups = result
                if accuracy > self.leakage_score_threshold and accuracy > majority:
                    add(
                        col,
                        "leakage_suspect",
                        "high",
                        f"maps onto the target {_name(target)} almost one-to-one: leave-one-out "
                        f"accuracy {accuracy:.3f} from {n_groups:,} distinct values "
                        f"(majority class {majority:.3f})",
                        {
                            "target": str(target),
                            "score": accuracy,
                            "baseline": majority,
                            "metric": "accuracy",
                            "n_groups": n_groups,
                            "n_rows": n_used,
                            "threshold": self.leakage_score_threshold,
                        },
                    )
                continue
            result = loo_mapping_r2(codes, y)
            if result is None:
                continue
            r2, n_used, n_groups = result
            if r2 > self.leakage_score_threshold:
                add(
                    col,
                    "leakage_suspect",
                    "high",
                    f"its {n_groups:,} distinct values already determine the target "
                    f"{_name(target)}: leave-one-out R2 {r2:.3f}",
                    {
                        "target": str(target),
                        "score": r2,
                        "metric": "R2",
                        "n_groups": n_groups,
                        "n_rows": n_used,
                        "threshold": self.leakage_score_threshold,
                    },
                )

    def _check_importance(self, work, cols, profiles, id_like, mask, y, classification, notes, add) -> None:
        """zero_importance: a small random forest cannot tell the column from noise."""
        usable: List[Hashable] = []
        skipped_cats: List[Hashable] = []
        for col in cols:
            p: ColumnProfile = profiles[col]
            if col in id_like:
                continue
            if p.kind not in FLOAT_KINDS and p.nunique > self.max_model_categories:
                skipped_cats.append(col)
                continue
            usable.append(col)
        if not usable:
            notes.append("no column was suitable for the zero_importance check")
            return
        if skipped_cats:
            shown = ", ".join(_name(c) for c in skipped_cats[:5])
            more = "" if len(skipped_cats) <= 5 else f" and {len(skipped_cats) - 5} more"
            notes.append(
                f"the zero_importance check skipped high-cardinality text column(s) {shown}{more} "
                f"(over {self.max_model_categories} distinct values)"
            )

        # The permutation pass costs roughly (features x rows x repeats) model
        # predictions, so a wide frame is where this check stops being cheap: 300
        # features over 20,000 rows took over a minute with nothing said about it.
        # Cap the features, scale the rows down with the feature count, and say so.
        if len(usable) > self.max_importance_columns:
            notes.append(
                f"the zero_importance check looked at the first {self.max_importance_columns} "
                f"of {len(usable)} eligible columns"
            )
            usable = usable[: self.max_importance_columns]

        rows = np.flatnonzero(mask)
        target_values = y
        row_cap = min(
            self.max_model_rows,
            max(1_000, self.importance_cell_budget // max(len(usable), 1)),
        )
        if len(rows) > row_cap:
            rng = np.random.default_rng(self.random_state)
            pick = rng.choice(len(rows), size=row_cap, replace=False)
            pick.sort()
            rows = rows[pick]
            target_values = y[pick]
            notes.append(
                f"the zero_importance check fitted its forest on {row_cap:,} sampled rows"
            )
        if len(rows) < 50:
            notes.append(
                f"only {len(rows)} row(s) with a target: too few to judge importance, so the "
                "zero_importance check was skipped"
            )
            return
        if classification:
            counts = np.bincount(target_values.astype(np.int64))
            if int((counts > 0).sum()) < 2:
                notes.append("the target has a single class on the sampled rows: zero_importance was skipped")
                return

        repeats = int(max(3, min(10, 1_500 // max(len(usable), 1))))
        if len(usable) >= 50:
            notes.append(
                f"the zero_importance check permuted {len(usable)} feature(s) over "
                f"{len(rows):,} rows, the slowest part of a wide-frame run; pass a smaller "
                "sample= or FeatureChecker(max_importance_columns=...) to cut it further"
            )

        matrix = np.column_stack([self._model_column(work[c].iloc[rows], profiles[c]) for c in usable])
        result = forest_importance(
            matrix,
            target_values,
            classification=classification,
            random_state=self.random_state,
            n_repeats=repeats,
        )
        importances = result["importances"]
        if importances is None:
            notes.append(
                "the random forest did no better than a constant prediction, so the "
                "zero_importance check was skipped"
            )
            return
        ceiling = float(result["noise_ceiling"] or 0.0)
        for col, importance in zip(usable, np.asarray(importances, dtype=float)):
            if float(importance) > ceiling + 1e-12:
                continue
            add(
                col,
                "zero_importance",
                "low",
                f"a small random forest gives it no measurable importance "
                f"({float(importance):+.4f} against a {ceiling:.4f} noise level)",
                {
                    "importance": float(importance),
                    "noise_ceiling": ceiling,
                    "model_score": float(result["score"]),
                    "model_baseline": float(result["baseline"]),
                },
            )

    def _model_column(self, series: pd.Series, profile: ColumnProfile) -> np.ndarray:
        """One model-ready float column: numbers imputed by median, text as codes."""
        if profile.kind in FLOAT_KINDS:
            values = to_float(series)
            finite = values[np.isfinite(values)]
            fill = float(np.median(finite)) if finite.size else 0.0
            return np.where(np.isfinite(values), values, fill)
        codes, _ = codes_of(series)
        return codes.astype(np.float64)

    # ------------------------------------------------------------------ misc
    def _params(self) -> Dict[str, Any]:
        return {
            "corr_threshold": self.corr_threshold,
            "missing_threshold": self.missing_threshold,
            "cardinality_threshold": self.cardinality_threshold,
            "near_constant_threshold": self.near_constant_threshold,
            "leakage_score_threshold": self.leakage_score_threshold,
            "leakage_corr_threshold": self.leakage_corr_threshold,
            "sample": self.sample,
            "random_state": self.random_state,
        }

    def _empty(
        self,
        target: Optional[Hashable],
        notes: List[str],
        n_rows: int,
        n_checked: int,
        order: Sequence[Hashable] = (),
    ) -> FeatureReport:
        features: Dict[Hashable, List[Finding]] = {c: [] for c in order}
        return FeatureReport(
            features=features,
            drop_recommended=[],
            keep=list(order),
            target=target,
            notes=notes,
            n_rows=n_rows,
            n_rows_checked=n_checked,
            params=self._params(),
        )


def _require_unique_columns(df: pd.DataFrame) -> None:
    """Duplicate column labels break every per-column lookup; say so up front."""
    if df.columns.is_unique:
        return
    counts = pd.Index(df.columns).value_counts()
    repeated = [str(name) for name, n in counts.items() if n > 1]
    raise ValueError(
        "the frame has duplicate column names, so columns cannot be told apart: "
        + ", ".join(repr(r) for r in sorted(repeated))
    )


def _require_distinct_labels(df: pd.DataFrame) -> None:
    """Labels that differ in type but share their text cannot survive ``to_dict()``.

    ``{1: ..., "1": ...}`` is a legal frame, but the JSON report, ``--json`` and
    ``--output`` are all keyed by the text form, so one column's findings would
    vanish without a word. Say so at the entry point instead.
    """
    seen: Dict[str, Hashable] = {}
    for col in df.columns:
        text = str(col)
        if text in seen:
            first = seen[text]
            raise ValueError(
                f"column labels {first!r} and {col!r} share the text form {text!r}, so the "
                "report could not tell them apart; rename one of them"
            )
        seen[text] = col


def check(
    df: FrameLike,
    target: Optional[Hashable] = None,
    *,
    corr_threshold: float = 0.95,
    missing_threshold: float = 0.6,
    cardinality_threshold: float = 0.98,
    sample: int = 200_000,
    random_state: int = 0,
) -> FeatureReport:
    """Check every column of ``df`` except ``target`` and return a :class:`FeatureReport`.

    ``df`` is a pandas DataFrame or a path to a ``.csv``/``.tsv``/``.parquet`` file.
    ``target``, when given, is the label column: it is never reported as a feature
    and it unlocks the ``leakage_suspect`` and ``zero_importance`` checks.

    ``corr_threshold`` is the ``|corr|`` (or Cramer's V) above which two columns
    count as redundant, ``missing_threshold`` the missing fraction above which a
    column is flagged, and ``cardinality_threshold`` the distinct-value ratio
    above which a column looks like an identifier. Frames longer than ``sample``
    rows are checked on a random sample drawn with ``random_state``.
    """
    checker = FeatureChecker(
        corr_threshold=corr_threshold,
        missing_threshold=missing_threshold,
        cardinality_threshold=cardinality_threshold,
        sample=sample,
        random_state=random_state,
    )
    return checker.check(df, target)
