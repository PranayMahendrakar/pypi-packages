"""SplitReport: sizes, class balance and the leakage checks that make a split trustworthy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._util import jsonable, parse_time, time_key
from .groups import composite_ids, duplicate_ids, linked_ids


@dataclass
class SplitReport:
    """What the split looks like and whether anything leaks between its parts.

    ``ok`` is True only when the parts partition the input exactly, no group spans two
    parts, no exact duplicate spans two parts, and (for chronological splits) time order holds.
    """

    rows: int
    sizes: Dict[str, int]
    fractions: Dict[str, float]
    strategy: Dict[str, Any]
    class_balance: Optional[Dict[str, Dict[str, float]]]
    class_counts: Optional[Dict[str, Dict[str, int]]]
    balance_max_deviation: Optional[float]
    target_stats: Optional[Dict[str, Dict[str, Optional[float]]]]
    group_overlap: Dict[str, Any]
    duplicate_leakage: Dict[str, Any]
    time_ordering: Dict[str, Any]
    partition: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every leakage check passed and no row was lost or duplicated."""
        if not self.partition.get("exact", False):
            return False
        if self.group_overlap.get("checked") and self.group_overlap.get("overlapping_groups", 0):
            return False
        if self.duplicate_leakage.get("checked") and self.duplicate_leakage.get("leaked_rows", 0):
            return False
        if self.time_ordering.get("checked") and not self.time_ordering.get("ordered", True):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary with every field plus ``ok``."""
        return jsonable(
            {
                "ok": self.ok,
                "rows": self.rows,
                "sizes": self.sizes,
                "fractions": self.fractions,
                "strategy": self.strategy,
                "class_balance": self.class_balance,
                "class_counts": self.class_counts,
                "balance_max_deviation": self.balance_max_deviation,
                "target_stats": self.target_stats,
                "group_overlap": self.group_overlap,
                "duplicate_leakage": self.duplicate_leakage,
                "time_ordering": self.time_ordering,
                "partition": self.partition,
                "warnings": list(self.warnings),
            }
        )

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        parts = " | ".join(
            f"{name} {size} ({self.fractions[name] * 100:.1f}%)" for name, size in self.sizes.items()
        )
        lines = [f"dataset-splitter report: {self.rows} rows -> {parts}"]
        st = self.strategy
        method = st.get("method")
        if method == "chronological":
            desc = f"chronological split on {st.get('time')!r} (oldest rows train, newest test, no shuffling)"
        elif method == "stratified":
            desc = (
                f"stratified random split on {st.get('target')!r} "
                f"({st.get('target_kind')}, {st.get('n_strata')} strata), random_state={st.get('random_state')}"
            )
        else:
            desc = f"random split (shuffled), random_state={st.get('random_state')}"
        lines.append(f"  method      {desc}")

        g = self.group_overlap
        if g.get("checked"):
            cols = g.get("columns") or []
            shown = repr(cols[0]) if len(cols) == 1 else repr(list(cols))
            mode = f" ({g.get('mode')})" if len(cols) > 1 else ""
            status = "ok" if g.get("overlapping_groups", 0) == 0 else "LEAK"
            lines.append(
                f"  groups      {shown}{mode}: {g.get('groups')} groups, "
                f"{g.get('overlapping_groups')} in more than one split  [{status}]"
            )
        else:
            lines.append("  groups      none")

        d = self.duplicate_leakage
        status = "ok" if d.get("leaked_rows", 0) == 0 else "LEAK"
        lines.append(
            f"  duplicates  {d.get('duplicate_rows', 0)} duplicate rows, "
            f"{d.get('leaked_rows', 0)} leaked across splits  [{status}]"
        )

        t = self.time_ordering
        if t.get("checked"):
            status = "ok" if t.get("ordered") else "FAIL"
            extra = ""
            if t.get("overlap_rows"):
                extra = f", {t['overlap_rows']} rows overlap an earlier split (groups kept whole)"
            lines.append(
                f"  time        {t.get('column')!r}: {'ordered' if t.get('ordered') else 'NOT ordered'} "
                f"at {t.get('level')} level{extra}  [{status}]"
            )
        else:
            lines.append("  time        none")

        p = self.partition
        status = "ok" if p.get("exact") else "MISMATCH"
        lines.append(
            f"  partition   {p.get('input_rows')} rows in, {p.get('output_rows')} rows out  [{status}]"
        )

        if self.class_balance:
            lines.append("  class balance (fraction of rows in each split)")
            names = list(self.class_balance.keys())
            labels = list(self.class_balance["all"].keys())
            labels.sort(key=lambda lab: -self.class_balance["all"][lab])
            width = max(5, min(28, max((len(str(lab)) for lab in labels), default=5)))
            header = "    " + "label".ljust(width) + "".join(n.rjust(8) for n in names)
            lines.append(header)
            for lab in labels[:12]:
                row = "    " + str(lab)[:width].ljust(width)
                row += "".join(f"{self.class_balance[n].get(lab, 0.0):8.3f}" for n in names)
                lines.append(row)
            if len(labels) > 12:
                lines.append(f"    ... {len(labels) - 12} more")
            if self.balance_max_deviation is not None:
                lines.append(f"    max deviation from overall: {self.balance_max_deviation:.3f}")

        if self.target_stats:
            lines.append("  target statistics per split")
            for name, stats in self.target_stats.items():
                mean = stats.get("mean")
                std = stats.get("std")
                lo = stats.get("min")
                hi = stats.get("max")
                fmt = lambda v: "nan" if v is None else f"{v:.4g}"  # noqa: E731
                lines.append(
                    f"    {name:<6} mean {fmt(mean)}  std {fmt(std)}  min {fmt(lo)}  max {fmt(hi)}"
                )

        if self.warnings:
            lines.append("  warnings")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append(f"  ok: {self.ok}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# building the report


def _concat(frames: Dict[str, pd.DataFrame]) -> "tuple[pd.DataFrame, np.ndarray]":
    parts = list(frames.values())
    labels = np.concatenate([np.full(len(f), i, dtype=np.int64) for i, f in enumerate(parts)])
    if not parts:
        return pd.DataFrame(), labels
    if len(parts) == 1:
        return parts[0], labels
    return pd.concat(parts, axis=0, copy=False), labels


def _check_groups(
    all_df: pd.DataFrame, labels: np.ndarray, names: Sequence[str], cols: Optional[List[Any]], mode: Optional[str]
) -> Dict[str, Any]:
    if not cols:
        return {"checked": False, "columns": None, "mode": None, "groups": None, "overlapping_groups": 0}
    if len(all_df) == 0:
        return {"checked": True, "columns": list(cols), "mode": mode, "groups": 0, "overlapping_groups": 0,
                "groups_per_split": {name: 0 for name in names}}
    ids = linked_ids(all_df, cols) if mode == "linked" else composite_ids(all_df, cols)
    per_group = pd.DataFrame({"g": ids, "s": labels}).groupby("g")["s"].nunique()
    per_split = {
        name: int(len(np.unique(ids[labels == i]))) for i, name in enumerate(names)
    }
    return {
        "checked": True,
        "columns": list(cols),
        "mode": mode,
        "groups": int(len(per_group)),
        "overlapping_groups": int((per_group > 1).sum()),
        "groups_per_split": per_split,
    }


def _check_duplicates(all_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    n = len(all_df)
    if n == 0:
        return {"checked": True, "duplicate_rows": 0, "leaked_rows": 0, "leaked_groups": 0}
    ids = duplicate_ids(all_df)
    counts = np.bincount(ids)
    dup_mask = counts[ids] > 1
    duplicate_rows = int(dup_mask.sum())
    if duplicate_rows == 0:
        return {"checked": True, "duplicate_rows": 0, "leaked_rows": 0, "leaked_groups": 0}
    per_group = pd.DataFrame({"g": ids[dup_mask], "s": labels[dup_mask]}).groupby("g")["s"].nunique()
    leaked = per_group.index[per_group > 1].to_numpy()
    leaked_rows = int(np.isin(ids, leaked).sum())
    return {
        "checked": True,
        "duplicate_rows": duplicate_rows,
        "leaked_rows": leaked_rows,
        "leaked_groups": int(len(leaked)),
    }


def _check_time(
    all_df: pd.DataFrame,
    labels: np.ndarray,
    names: Sequence[str],
    time_col: Any,
    group_cols: Optional[List[Any]],
    group_mode: Optional[str],
) -> Dict[str, Any]:
    if time_col is None:
        return {"checked": False, "column": None, "level": None, "ordered": True, "overlap_rows": 0, "ranges": None}
    level = "group" if group_cols else "row"
    result: Dict[str, Any] = {"checked": True, "column": time_col, "level": level, "ordered": True,
                              "overlap_rows": 0, "ranges": {}}
    if len(all_df) == 0:
        return result
    parsed = parse_time(all_df[time_col], time_col)
    key = time_key(parsed)
    if level == "group":
        ids = linked_ids(all_df, group_cols) if group_mode == "linked" else composite_ids(all_df, group_cols)
        starts = pd.Series(key).groupby(ids).transform("min").to_numpy()
    else:
        starts = key
    values = parsed.to_numpy()
    prev_max_start = None
    prev_max_key = None
    overlap_rows = 0
    ordered = True
    for i, name in enumerate(names):
        mask = labels == i
        if not mask.any():
            continue
        result["ranges"][name] = {"min": values[mask].min(), "max": values[mask].max()}
        if prev_max_start is not None and starts[mask].min() < prev_max_start:
            ordered = False
        if prev_max_key is not None:
            overlap_rows += int((key[mask] < prev_max_key).sum())
        prev_max_start = max(prev_max_start, starts[mask].max()) if prev_max_start is not None else starts[mask].max()
        prev_max_key = max(prev_max_key, key[mask].max()) if prev_max_key is not None else key[mask].max()
    result["ordered"] = bool(ordered)
    result["overlap_rows"] = int(overlap_rows)
    return result


def _check_partition(positions: Dict[str, np.ndarray], n_rows: int) -> Dict[str, Any]:
    parts = [np.asarray(p) for p in positions.values()]
    joined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    unique = np.unique(joined)
    covered = int(len(unique[(unique >= 0) & (unique < n_rows)]))
    exact = len(joined) == n_rows and len(unique) == n_rows and covered == n_rows
    return {
        "input_rows": int(n_rows),
        "output_rows": int(len(joined)),
        "exact": bool(exact),
        "lost_rows": int(n_rows - covered),
        "duplicated_rows": int(len(joined) - len(unique)),
    }


def _class_balance(
    positions: Dict[str, np.ndarray], codes: np.ndarray, labels: List[str]
) -> "tuple[dict, dict, Optional[float]]":
    k = len(labels)
    overall = np.bincount(codes, minlength=k)
    counts: Dict[str, Dict[str, int]] = {"all": {lab: int(c) for lab, c in zip(labels, overall)}}
    total = int(overall.sum())
    fractions: Dict[str, Dict[str, float]] = {
        "all": {lab: (float(c) / total if total else 0.0) for lab, c in zip(labels, overall)}
    }
    max_dev = 0.0
    for name, pos in positions.items():
        c = np.bincount(codes[pos], minlength=k)
        counts[name] = {lab: int(v) for lab, v in zip(labels, c)}
        if len(pos):
            frac = c / float(len(pos))
            max_dev = max(max_dev, float(np.abs(frac - overall / float(total)).max()))
        else:
            frac = np.zeros(k)
        fractions[name] = {lab: float(v) for lab, v in zip(labels, frac)}
    return fractions, counts, (round(max_dev, 6) if total else None)


def _target_stats(y: pd.Series, positions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Optional[float]]]:
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64")

    def stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    out = {"all": stats(values)}
    for name, pos in positions.items():
        out[name] = stats(values[pos])
    return out


def build_report(
    df: pd.DataFrame,
    frames: Dict[str, pd.DataFrame],
    positions: Dict[str, np.ndarray],
    strategy: Dict[str, Any],
    balance_codes: Optional[np.ndarray],
    balance_labels: Optional[List[str]],
    split_warnings: Sequence[str],
) -> SplitReport:
    """Compute every check from the split frames themselves (independent of how they were made)."""
    names = list(frames.keys())
    n_rows = int(strategy.get("rows", len(df)))
    sizes = {name: int(len(frame)) for name, frame in frames.items()}
    fractions = {name: (size / n_rows if n_rows else 0.0) for name, size in sizes.items()}
    warnings_out = list(split_warnings)

    all_df, labels = _concat(frames)
    group_cols = strategy.get("group")
    group_mode = strategy.get("group_mode")
    group_overlap = _check_groups(all_df, labels, names, group_cols, group_mode)
    duplicate_leakage = _check_duplicates(all_df, labels)
    time_ordering = _check_time(all_df, labels, names, strategy.get("time"), group_cols, group_mode)
    partition = _check_partition(positions, n_rows)

    class_balance = class_counts = None
    max_dev = None
    target_stats = None
    target = strategy.get("target")
    if target is not None and balance_codes is not None and balance_labels:
        class_balance, class_counts, max_dev = _class_balance(positions, balance_codes, balance_labels)
        if strategy.get("target_kind") == "numeric":
            try:
                target_stats = _target_stats(df[target], positions)
            except (TypeError, ValueError):
                target_stats = None

    if group_overlap["checked"] and group_overlap["overlapping_groups"]:
        warnings_out.append(
            f"{group_overlap['overlapping_groups']} group(s) appear in more than one split"
        )
    if duplicate_leakage["leaked_rows"]:
        warnings_out.append(
            f"{duplicate_leakage['leaked_rows']} exact duplicate rows appear in more than one split"
            + (" (dedupe=False)" if not strategy.get("dedupe", True) else "")
        )
    if time_ordering["checked"]:
        if not time_ordering["ordered"]:
            warnings_out.append(f"splits are not in chronological order on {time_ordering['column']!r}")
        elif time_ordering["overlap_rows"]:
            warnings_out.append(
                f"{time_ordering['overlap_rows']} row(s) in later splits are timestamped before the end of an "
                f"earlier split because groups were kept whole"
            )
    if not partition["exact"]:
        warnings_out.append(
            f"splits do not partition the input: {partition['lost_rows']} rows lost, "
            f"{partition['duplicated_rows']} duplicated"
        )

    return SplitReport(
        rows=n_rows,
        sizes=sizes,
        fractions=fractions,
        strategy=dict(strategy),
        class_balance=class_balance,
        class_counts=class_counts,
        balance_max_deviation=max_dev,
        target_stats=target_stats,
        group_overlap=group_overlap,
        duplicate_leakage=duplicate_leakage,
        time_ordering=time_ordering,
        partition=partition,
        warnings=warnings_out,
    )
