"""Leakage-safe train/validation/test splitting.

The split works on *units*: sets of rows that must stay together (a group, the exact copies
of a row, or a single row). Units are ordered (shuffled, or by time), optionally bucketed
into strata, and then cut so that each part gets its share of rows while units stay whole.
"""
from __future__ import annotations

import json
import logging
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._io import load_table
from ._util import parse_time, strata as _strata, target_kind as _target_kind, time_key
from .groups import composite_ids, detect_id_columns, duplicate_ids, link_codes, linked_ids
from .report import SplitReport, build_report

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
SizeSpec = Union[int, float]
GroupSpec = Union[str, Sequence[Any], None]


@dataclass
class _Meta:
    n_rows: int
    positions: Dict[str, np.ndarray]
    balance_codes: Optional[np.ndarray]
    balance_labels: Optional[List[str]]
    keep_index: bool
    source: pd.DataFrame


@dataclass(eq=False)
class Split:
    """The result of one split: the parts, their original index values, and how they were made.

    ``val`` is None when ``val_size`` was 0. ``indices`` maps each existing part to the list of
    original index values it holds. Row order inside every part is the input order.
    """

    train: pd.DataFrame
    val: Optional[pd.DataFrame]
    test: pd.DataFrame
    indices: Dict[str, List[Any]]
    strategy: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    _meta: Optional[_Meta] = field(default=None, repr=False)
    _report: Optional[SplitReport] = field(default=None, init=False, repr=False)

    def frames(self) -> Dict[str, pd.DataFrame]:
        """The existing parts in order: train, (val,) test."""
        out = {"train": self.train}
        if self.val is not None:
            out["val"] = self.val
        out["test"] = self.test
        return out

    def report(self) -> SplitReport:
        """Sizes, class balance and leakage checks (computed once, then cached)."""
        if self._report is None:
            frames = self.frames()
            if self._meta is None:  # a Split assembled by hand: derive positions from the frames
                offsets = np.cumsum([0] + [len(f) for f in frames.values()])
                positions = {n: np.arange(offsets[i], offsets[i + 1]) for i, n in enumerate(frames)}
                source = pd.concat(list(frames.values()), axis=0) if frames else pd.DataFrame()
                codes = labels = None
            else:
                positions = self._meta.positions
                source = self._meta.source
                codes, labels = self._meta.balance_codes, self._meta.balance_labels
            strategy = dict(self.strategy)
            strategy.setdefault("rows", int(len(source)))
            self._report = build_report(source, frames, positions, strategy, codes, labels, self.warnings)
        return self._report

    def save(self, dir: Union[str, "Path"], format: str = "csv") -> Dict[str, Path]:
        """Write train/val/test files plus report.json into ``dir``; returns the paths written."""
        fmt = str(format).lower()
        if fmt not in ("csv", "parquet"):
            raise ValueError(f"format must be 'csv' or 'parquet', got {format!r}")
        out = Path(dir)
        out.mkdir(parents=True, exist_ok=True)
        keep_index = self._meta.keep_index if self._meta is not None else True
        paths: Dict[str, Path] = {}
        for name, frame in self.frames().items():
            path = out / f"{name}.{fmt}"
            if fmt == "csv":
                label = None
                if keep_index and frame.index.name is None and frame.index.nlevels == 1:
                    label = "index"
                frame.to_csv(path, index=keep_index, index_label=label)
            else:
                try:
                    frame.to_parquet(path, index=keep_index)
                except ImportError as exc:
                    raise ImportError(
                        "parquet output needs pyarrow: pip install 'dataset-splitter[parquet]'"
                    ) from exc
            paths[name] = path
        report_path = out / "report.json"
        report_path.write_text(json.dumps(self.report().to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        paths["report"] = report_path
        return paths

    def __repr__(self) -> str:
        val = "None" if self.val is None else f"{len(self.val)} rows"
        return (
            f"Split(train={len(self.train)} rows, val={val}, test={len(self.test)} rows, "
            f"method={self.strategy.get('method')!r})"
        )


# --------------------------------------------------------------------------------------
# parameter checks


def _check_size(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a float in [0, 1) or a non-negative int, got {value!r}")
    if isinstance(value, numbers.Integral):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
    elif not (0 <= float(value) < 1):
        raise ValueError(f"{name} as a fraction must be in [0, 1), got {value!r}")


def _rows_for(value: SizeSpec, n: int, name: str) -> int:
    if isinstance(value, numbers.Integral):
        if int(value) > n:
            raise ValueError(f"{name}={value} exceeds the {n} rows available")
        return int(value)
    return int(np.floor(float(value) * n + 0.5))


def _check_no_duplicate_columns(df: pd.DataFrame) -> None:
    """Duplicate column labels make every lookup ambiguous; fail clearly instead of guessing."""
    columns = pd.Index(df.columns)
    if not columns.has_duplicates:
        return
    dupes = list(dict.fromkeys(columns[columns.duplicated()].tolist()))
    shown = ", ".join(repr(c) for c in dupes[:10])
    more = "" if len(dupes) <= 10 else f", ... ({len(dupes)} total)"
    raise ValueError(
        f"input has duplicate column name(s): {shown}{more}; "
        "rename or drop them before splitting"
    )


def _require_column(df: pd.DataFrame, col: Any, role: str) -> None:
    if col not in df.columns:
        available = ", ".join(repr(c) for c in list(df.columns)[:15])
        more = "" if len(df.columns) <= 15 else ", ..."
        raise ValueError(f"{role} column {col!r} not found; columns are: {available}{more}")
    if isinstance(df[col], pd.DataFrame):
        raise ValueError(f"{role} column {col!r} appears more than once in the frame")


# --------------------------------------------------------------------------------------
# allocation


def _apportion(rows_k: np.ndarray, total: int, cap: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Integer row targets per stratum summing to `total`, proportional to stratum size.

    Largest-remainder rounding; ties are broken at random; `cap` bounds each stratum.
    """
    out = np.zeros(len(rows_k), dtype=np.int64)
    n = int(rows_k.sum())
    if total <= 0 or n == 0 or len(rows_k) == 0:
        return out
    raw = rows_k * (float(total) / n)
    out = np.minimum(np.floor(raw).astype(np.int64), cap)
    remaining = total - int(out.sum())
    if remaining > 0:
        order = np.lexsort((rng.random(len(rows_k)), -(raw - np.floor(raw))))
        room = cap - out
        for k in order:
            if remaining == 0:
                break
            if room[k] > 0:
                out[k] += 1
                remaining -= 1
    return out


def _allocate(
    unit_size: np.ndarray,
    unit_stratum: np.ndarray,
    order: np.ndarray,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign each unit to 0=train, 1=val, 2=test.

    Within every stratum the units are walked in `order`; train takes the first rows, val the
    next, test the last, cutting at the unit boundary nearest each target. Rounding error is
    carried to the next stratum so overall sizes stay on target.
    """
    n_units = len(unit_size)
    unit_split = np.zeros(n_units, dtype=np.int8)
    if n_units == 0:
        return unit_split
    n_strata = int(unit_stratum.max()) + 1
    rows_k = np.bincount(unit_stratum, weights=unit_size, minlength=n_strata).astype(np.int64)
    test_k = _apportion(rows_k, n_test, rows_k, rng)
    val_k = _apportion(rows_k, n_val, rows_k - test_k, rng)

    rank = np.empty(n_units, dtype=np.int64)
    rank[order] = np.arange(n_units)
    sorter = np.lexsort((rank, unit_stratum))
    sorted_strata = unit_stratum[sorter]
    starts = np.searchsorted(sorted_strata, np.arange(n_strata), side="left")
    ends = np.searchsorted(sorted_strata, np.arange(n_strata), side="right")

    carry_test = carry_val = 0
    for k in np.argsort(rows_k, kind="stable"):  # smallest strata first; the largest absorbs carry
        idx = sorter[starts[k]:ends[k]]
        if len(idx) == 0:
            continue
        sizes = unit_size[idx]
        cum = np.cumsum(sizes)
        mid = cum - sizes / 2.0
        n_k = int(rows_k[k])
        want_test = min(max(int(test_k[k]) + carry_test, 0), n_k)
        want_val = min(max(int(val_k[k]) + carry_val, 0), n_k - want_test)
        b2 = n_k - want_test
        b1 = b2 - want_val
        assigned = (mid > b1).astype(np.int8) + (mid > b2).astype(np.int8)
        unit_split[idx] = assigned
        got_test = int(sizes[assigned == 2].sum())
        got_val = int(sizes[assigned == 1].sum())
        carry_test += int(test_k[k]) - got_test
        carry_val += int(val_k[k]) - got_val

    # Repair a split that was asked for but came back empty. Units are indivisible, so the
    # midpoint cut above can miss a narrow band completely: three groups of 200 rows with
    # val_size=0.1 leaves no unit whose midpoint lands inside the 60-row validation band, and
    # validation silently ends up empty.
    #
    # When units are too few to fill everything, the priority is train, then test, then val:
    # a model with no test set cannot be evaluated at all, while a missing val set only costs
    # tuning. So test may borrow from train or from val, but val only ever borrows from train,
    # and train always keeps at least one unit.
    for target, wanted, donors in ((2, n_test, (0, 1)), (1, n_val, (0,))):
        if wanted <= 0 or bool((unit_split == target).any()):
            continue
        for donor in donors:
            pool = np.flatnonzero(unit_split == donor)
            spare = len(pool) - (1 if donor == 0 else 0)  # train must survive
            if spare > 0:
                unit_split[pool[np.argmax(rank[pool])]] = target
                break
    return unit_split


def _unit_majority(unit_ids: np.ndarray, codes: np.ndarray, n_units: int) -> np.ndarray:
    """Stratum of each unit: the most common stratum among its rows."""
    out = np.zeros(n_units, dtype=np.int64)
    if n_units == len(unit_ids):
        out[unit_ids] = codes
        return out
    table = pd.DataFrame({"u": unit_ids, "c": codes})
    counts = table.groupby(["u", "c"], sort=False).size().reset_index(name="n")
    counts = counts.sort_values(["u", "n", "c"], ascending=[True, False, True], kind="stable")
    first = counts.drop_duplicates("u")
    out[first["u"].to_numpy()] = first["c"].to_numpy()
    return out


def _pool_rare(
    codes: np.ndarray, labels: List[str], n_splits: int, target: Any, kind: str
) -> Tuple[np.ndarray, bool, List[str]]:
    """Merge strata too small to stratify on; give up (random split) when most rows are rare."""
    what = "class" if kind == "categorical" else "bin"
    counts = np.bincount(codes, minlength=len(labels))
    present = counts > 0
    n_present = int(present.sum())
    n = len(codes)
    if n_present < 2:
        if kind == "categorical":
            msg = f"target {target!r} has a single class; stratification has no effect"
        else:
            msg = f"target {target!r}: too few rows to build quantile bins; the split is random"
        return codes, False, [msg]
    rare = present & (counts < n_splits)
    if not rare.any():
        return codes, True, []
    rare_rows = int(counts[rare].sum())
    n_rare = int(rare.sum())
    if rare_rows * 2 > n or n_rare == n_present:
        msg = (
            f"target {target!r}: {n_rare} of {n_present} {what}es have fewer than {n_splits} rows "
            f"({rare_rows} of {n} rows); stratification is skipped and the split is random"
        )
        return codes, False, [msg]
    pooled = codes.copy()
    pooled[rare[codes]] = len(labels)
    pooled = np.asarray(pd.factorize(pooled)[0], dtype=np.int64)
    names = [labels[i] for i in np.flatnonzero(rare)]
    shown = ", ".join(repr(x) for x in names[:5]) + (f", ... ({len(names)} total)" if len(names) > 5 else "")
    msg = (
        f"target {target!r}: {what}es with fewer than {n_splits} rows were pooled for stratification: {shown}"
    )
    return pooled, True, [msg]


# --------------------------------------------------------------------------------------
# the splitter


class Splitter:
    """Configurable splitter. ``split()`` is the one-line front door to it.

    Parameters
    ----------
    target : stratify on this column (classes, or quantile bins for numeric targets) and
        report class balance per part.
    group : column, list of columns (composite key), or "auto" to detect id-like columns;
        rows of one group never straddle two parts.
    time : chronological split on this column; oldest rows train, newest test, no shuffling.
    test_size, val_size : fraction of rows (float in [0, 1)) or absolute row count (int).
    random_state : seed for shuffling; ignored for chronological splits.
    dedupe : keep exact duplicate rows on the same side.
    stratify : "auto" (when a target is given and the split is not chronological), True or False.
    n_bins : quantile bins for numeric targets.
    max_categories : numeric targets with at most this many distinct values count as categorical.
    """

    def __init__(
        self,
        *,
        target: Any = None,
        group: GroupSpec = None,
        time: Any = None,
        test_size: SizeSpec = 0.2,
        val_size: SizeSpec = 0.1,
        random_state: Optional[int] = 0,
        dedupe: bool = True,
        stratify: Union[str, bool] = "auto",
        n_bins: int = 10,
        max_categories: int = 20,
    ) -> None:
        _check_size(test_size, "test_size")
        _check_size(val_size, "val_size")
        if (
            not isinstance(test_size, numbers.Integral)
            and not isinstance(val_size, numbers.Integral)
            and float(test_size) + float(val_size) >= 1
        ):
            raise ValueError(f"test_size + val_size must be below 1, got {test_size} + {val_size}")
        if stratify not in ("auto", True, False):
            raise ValueError("stratify must be 'auto', True or False")
        if int(n_bins) < 1:
            raise ValueError("n_bins must be at least 1")
        if group is not None and not isinstance(group, str) and not isinstance(group, (list, tuple)):
            raise TypeError("group must be a column name, a list of column names, or 'auto'")
        if isinstance(group, (list, tuple)) and len(group) == 0:
            raise ValueError("group list is empty")
        self.target = target
        self.group = group
        self.time = time
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.dedupe = bool(dedupe)
        self.stratify = stratify
        self.n_bins = int(n_bins)
        self.max_categories = int(max_categories)

    # ------------------------------------------------------------------ helpers

    def _resolve_group(
        self, df: pd.DataFrame, warns: List[str]
    ) -> Tuple[Optional[List[Any]], Optional[str], bool]:
        spec = self.group
        if spec is None:
            return None, None, False
        if isinstance(spec, str) and spec == "auto" and "auto" not in df.columns:
            exclude = [c for c in (self.target, self.time) if c is not None]
            cols = detect_id_columns(df, exclude=exclude)
            if not cols:
                warns.append("group='auto': no id-like column with repeated values found; rows are split individually")
                return None, None, True
            return cols, ("single" if len(cols) == 1 else "linked"), True
        if isinstance(spec, (list, tuple)):
            cols = list(spec)
            mode = "composite" if len(cols) > 1 else "single"
        else:
            cols, mode = [spec], "single"
        for c in cols:
            _require_column(df, c, "group")
        return cols, mode, False

    # ------------------------------------------------------------------ main entry

    def split(self, data: Any) -> Split:
        """Split a DataFrame (or a path to .csv/.parquet) into train/val/test."""
        df = load_table(data)
        _check_no_duplicate_columns(df)
        n = len(df)
        warns: List[str] = []
        target, time = self.target, self.time
        if target is not None:
            _require_column(df, target, "target")
        if time is not None:
            _require_column(df, time, "time")
        if target is not None and time is not None and target == time:
            raise ValueError("target and time must be different columns")
        group_cols, group_mode, detected = self._resolve_group(df, warns)

        n_test = _rows_for(self.test_size, n, "test_size")
        n_val = _rows_for(self.val_size, n, "val_size")
        if n_test + n_val > n:
            raise ValueError(f"test_size + val_size ask for {n_test + n_val} rows but only {n} are available")
        has_val = float(self.val_size) > 0
        n_splits = 3 if has_val else 2
        names = ["train", "val", "test"] if has_val else ["train", "test"]
        if n == 0:
            warns.append("input has no rows")

        # -- units: rows that must stay together
        id_arrays: List[np.ndarray] = []
        n_groups: Optional[int] = None
        if group_cols is not None:
            gids = linked_ids(df, group_cols) if group_mode == "linked" else composite_ids(df, group_cols)
            id_arrays.append(gids)
            n_groups = int(gids.max()) + 1 if n else 0
        n_dup_rows: Optional[int] = None
        if self.dedupe:
            dids = duplicate_ids(df)
            id_arrays.append(dids)
            if n:
                counts = np.bincount(dids)
                n_dup_rows = int((counts[dids] > 1).sum())
            else:
                n_dup_rows = 0
        if not id_arrays:
            unit_ids = np.arange(n, dtype=np.int64)
        elif len(id_arrays) == 1:
            unit_ids = np.asarray(pd.factorize(id_arrays[0])[0], dtype=np.int64)
        else:
            unit_ids = link_codes(id_arrays)
        n_units = int(unit_ids.max()) + 1 if n else 0
        unit_size = np.bincount(unit_ids, minlength=n_units).astype(np.int64)

        # -- order: by time, or shuffled
        rng = np.random.default_rng(self.random_state)
        if time is not None:
            parsed = parse_time(df[time], time)
            key = time_key(parsed)
            if n:
                unit_time = pd.Series(key).groupby(unit_ids).min().to_numpy()
                unit_first = pd.Series(np.arange(n)).groupby(unit_ids).min().to_numpy()
                order = np.lexsort((unit_first, unit_time))
            else:
                order = np.zeros(0, dtype=np.int64)
            shuffled = False
        else:
            order = rng.permutation(n_units)
            shuffled = True

        # -- strata
        kind: Optional[str] = None
        balance_codes: Optional[np.ndarray] = None
        balance_labels: Optional[List[str]] = None
        stratified = False
        unit_stratum = np.zeros(n_units, dtype=np.int64)
        n_strata = 1
        if target is not None:
            y = df[target]
            kind = _target_kind(y, self.max_categories)
            effective_bins = min(self.n_bins, max(1, n // (3 * n_splits)))
            balance_codes, balance_labels = _strata(y, kind, effective_bins)
            n_missing = int(y.isna().sum())
            if n_missing:
                warns.append(f"target {target!r} has {n_missing} missing value(s); they form their own stratum")
            wants = self.stratify is True or self.stratify == "auto"
            if wants and time is None and n > 0:
                strat_codes, stratified, messages = _pool_rare(
                    balance_codes, balance_labels, n_splits, target, kind
                )
                warns.extend(messages)
                if stratified:
                    unit_stratum = _unit_majority(unit_ids, strat_codes, n_units)
                    n_strata = int(unit_stratum.max()) + 1
            elif wants and time is not None:
                log.debug("chronological split: target %r is used for balance reporting only", target)

        # -- cut
        unit_split = _allocate(unit_size, unit_stratum, order, n_val, n_test, rng)
        row_split = unit_split[unit_ids] if n else np.zeros(0, dtype=np.int8)
        codes_for = {"train": 0, "val": 1, "test": 2}
        positions = {name: np.flatnonzero(row_split == codes_for[name]) for name in names}
        frames = {name: df.iloc[positions[name]] for name in names}
        indices = {name: df.index[positions[name]].tolist() for name in names}

        for name in names:
            if n and len(positions[name]) == 0:
                size_name = {"train": None, "val": "val_size", "test": "test_size"}[name]
                if size_name is None:
                    warns.append("train split is empty")
                else:
                    unit_note = f" in {n_units} indivisible unit(s)" if n_units < n else ""
                    warns.append(
                        f"{name} split is empty: {n} row(s){unit_note} is too few for "
                        f"{size_name}={getattr(self, size_name)}"
                    )

        method = "chronological" if time is not None else ("stratified" if stratified else "random")
        strategy: Dict[str, Any] = {
            "method": method,
            "shuffled": shuffled,
            "rows": n,
            "units": n_units,
            "target": target,
            "target_kind": kind,
            "stratified": stratified,
            "n_strata": n_strata if stratified else None,
            "group": group_cols,
            "group_mode": group_mode,
            "group_detected": detected,
            "n_groups": n_groups,
            "time": time,
            "dedupe": self.dedupe,
            "duplicate_rows": n_dup_rows,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
        }
        index = df.index
        keep_index = not (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1)
        meta = _Meta(
            n_rows=n,
            positions=positions,
            balance_codes=balance_codes,
            balance_labels=balance_labels,
            keep_index=keep_index,
            source=df,
        )
        log.debug("split %d rows into %s", n, {k: len(v) for k, v in positions.items()})
        return Split(
            train=frames["train"],
            val=frames.get("val"),
            test=frames["test"],
            indices=indices,
            strategy=strategy,
            warnings=warns,
            _meta=meta,
        )


def split(
    data: Any,
    *,
    target: Any = None,
    group: GroupSpec = None,
    time: Any = None,
    test_size: SizeSpec = 0.2,
    val_size: SizeSpec = 0.1,
    random_state: Optional[int] = 0,
    dedupe: bool = True,
) -> Split:
    """Split a DataFrame (or a .csv/.parquet path) into leakage-safe train/val/test parts.

    ``target`` stratifies (classes, or quantile bins for numeric targets); ``group`` keeps each
    group on one side ("auto" detects id-like columns); ``time`` makes the split chronological
    (oldest rows train, newest test); ``dedupe`` keeps exact duplicate rows together.
    Returns a ``Split`` with ``.train``, ``.val`` (None when ``val_size=0``), ``.test``,
    ``.indices`` and ``.report()``.
    """
    return Splitter(
        target=target,
        group=group,
        time=time,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        dedupe=dedupe,
    ).split(data)
