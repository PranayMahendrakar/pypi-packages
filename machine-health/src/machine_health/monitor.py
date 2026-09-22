"""Continuous scoring: one fixed baseline, one score per incoming batch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._io import TableLike, check_columns, load_table
from ._util import column_map, describe_names, fmt, jsonable, resolve_time, select_channels, sort_key
from .result import COMPONENTS, MachineScore, grade_rank
from .rules import Rule, check_rule_channels, normalize_rules
from .scoring import (
    TREND_TOLERANCE,
    _baseline_stats,
    build_score,
    resolve_weights,
    window_values,
)

log = logging.getLogger(__name__)

#: How many past updates the monitor averages when deciding the trend.
TREND_HISTORY = 3


@dataclass(frozen=True)
class Alert:
    """Something worth waking someone up for: a grade drop or a broken limit."""

    when: Any
    severity: str  # "warning" or "critical"
    message: str

    def summary(self) -> str:
        """Plain-ASCII one-liner."""
        stamp = "" if self.when is None else f"{fmt(self.when)}: "
        return f"[{self.severity}] {stamp}{self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return jsonable(
            {"when": self.when, "severity": self.severity, "message": self.message}
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


class HealthMonitor:
    """Score batch after batch against one fixed healthy baseline.

    The baseline never moves, so today's score is comparable with last week's.
    Every :meth:`update` appends a row to :attr:`history` and may append to
    :attr:`alerts`.

    Args:
        baseline_df: the healthy period, as a DataFrame or a path to a file.
        rules: your limits, in any shape :func:`machine_health.score` accepts.
        weights: how much each component counts.
        time: name of the timestamp column, used for alert times and ordering.
        channels: which columns to score; defaults to every numeric column of
            ``baseline_df``.
    """

    def __init__(
        self,
        baseline_df: TableLike,
        rules: Any = None,
        weights: Optional[Mapping[str, float]] = None,
        *,
        time: Any = None,
        channels: Optional[Sequence[Any]] = None,
    ) -> None:
        frame = load_table(baseline_df, "baseline_df")
        check_columns(frame, "baseline_df")
        if len(frame) == 0:
            raise ValueError(
                "baseline_df has no rows; the monitor needs a healthy period to compare against"
            )
        self.rules: List[Rule] = normalize_rules(rules)
        self.weights, self._weight_notes = resolve_weights(weights)
        self.time = time

        times, time_name, time_notes = resolve_time(frame, time)
        self._time_name = time_name
        if times is not None:
            order = np.argsort(sort_key(times), kind="stable")
            frame = frame.iloc[order]
            times = times.iloc[order]

        exclude = (time_name,) if time_name else ()
        available, skipped = select_channels(frame, None, exclude=exclude)
        if channels is None:
            chosen = list(available)
        else:
            chosen, _ = select_channels(frame, channels, exclude=exclude)
            for rule in self.rules:
                if rule.channel in available and rule.channel not in chosen:
                    chosen.append(rule.channel)
        check_rule_channels(self.rules, available)
        if not chosen:
            raise ValueError(
                "no numeric channel found in baseline_df; machine-health scores numeric "
                f"sensor columns. columns present: {describe_names(list(frame.columns))}"
            )
        self.channels: List[str] = chosen
        self._skipped = skipped
        self._baseline = frame

        setup_notes = list(self._weight_notes) + list(time_notes)
        if skipped and channels is None:
            setup_notes.append(f"ignored non-numeric column(s): {describe_names(skipped)}")
        self._stats = _baseline_stats(frame, chosen, column_map(frame), setup_notes)
        self._setup_notes = setup_notes

        #: The baseline scored against itself: the starting point every batch is compared with.
        self.baseline_score: MachineScore = build_score(
            window_values(frame, chosen),
            len(frame),
            self._stats,
            chosen,
            self.rules,
            self.weights,
            times=list(times) if times is not None else None,
            notes=setup_notes + ["this is the baseline scored against itself"],
            n_baseline_rows=len(frame),
        )
        self.alerts: List[Alert] = []
        self._rows: List[Dict[str, Any]] = []
        self._last: MachineScore = self.baseline_score
        self.n_updates = 0

    # ------------------------------------------------------------------ state

    @property
    def last_score(self) -> MachineScore:
        """The most recent score; before the first update, the baseline's own score."""
        return self._last

    @property
    def history(self) -> pd.DataFrame:
        """One row per update: when, value, grade, trend, each component, counts."""
        columns = [
            "update",
            "when",
            "value",
            "grade",
            "trend",
            *COMPONENTS,
            "rows",
            "violations",
        ]
        if not self._rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self._rows, columns=columns)

    # ----------------------------------------------------------------- update

    def update(self, batch: TableLike) -> MachineScore:
        """Score one batch against the fixed baseline and record it.

        An empty batch changes nothing: the previous score comes back with an
        extra note instead of an exception.
        """
        frame = load_table(batch, "batch")
        check_columns(frame, "batch")
        if len(frame) == 0:
            note = "empty batch: the previous score is returned unchanged"
            log.info(note)
            return self._last.with_note(note)

        notes: List[str] = []
        times = None
        if self._time_name is not None and self._time_name in column_map(frame):
            times, _, time_notes = resolve_time(frame, self._time_name)
            notes.extend(time_notes)
            order = np.argsort(sort_key(times), kind="stable")
            frame = frame.iloc[order]
            times = times.iloc[order]
        elif self._time_name is not None:
            notes.append(
                f"batch has no {self._time_name!r} column; rows keep the order they arrived in"
            )

        colmap = column_map(frame)
        missing = [name for name in self.channels if name not in colmap]
        if missing:
            notes.append(
                f"batch has no column for channel(s) {describe_names(missing)}; "
                "they count as missing data"
            )
        extra = [name for name in colmap if name not in self.channels and name != self._time_name]
        if extra:
            notes.append(
                f"batch column(s) {describe_names(extra)} are not baseline channels and were ignored"
            )

        score = build_score(
            window_values(frame, self.channels),
            len(frame),
            self._stats,
            self.channels,
            self.rules,
            self.weights,
            times=list(times) if times is not None else None,
            notes=notes,
            n_baseline_rows=len(self._baseline),
        )
        self._apply_history_trend(score)
        self.n_updates += 1
        self._record(score)
        self._raise_alerts(score)
        self._last = score
        return score

    # ----------------------------------------------------------------- inside

    def _apply_history_trend(self, score: MachineScore) -> None:
        """Once there is history, the trend compares batches instead of halves of one batch."""
        previous = [float(row["value"]) for row in self._rows]
        if not previous:
            return
        recent = previous[-TREND_HISTORY:]
        reference = sum(recent) / len(recent)
        change = score.value - reference
        if change > TREND_TOLERANCE:
            score.trend = "improving"
        elif change < -TREND_TOLERANCE:
            score.trend = "degrading"
        else:
            score.trend = "stable"
        score.notes.append(
            f"trend compares this score ({score.value:.1f}) with the mean of the previous "
            f"{len(recent)} update(s) ({reference:.1f})"
        )

    def _record(self, score: MachineScore) -> None:
        row: Dict[str, Any] = {
            "update": self.n_updates,
            "when": score.when,
            "value": round(float(score.value), 4),
            "grade": score.grade,
            "trend": score.trend,
            "rows": int(score.n_rows),
            "violations": len(score.violations),
        }
        for name in COMPONENTS:
            if name in score.unmeasured:
                row[name] = float("nan")  # never measured: not a free 100
            else:
                row[name] = round(float(score.components.get(name, float("nan"))), 4)
        self._rows.append(row)

    def _raise_alerts(self, score: MachineScore) -> None:
        previous = self._last
        drop = grade_rank(score.grade) - grade_rank(previous.grade)
        if drop >= 1:
            self.alerts.append(
                Alert(
                    when=score.when,
                    severity="critical" if drop >= 2 or score.grade == "F" else "warning",
                    message=(
                        f"grade dropped {previous.grade} -> {score.grade} "
                        f"({previous.value:.1f} -> {score.value:.1f} out of 100)"
                    ),
                )
            )
        for violation in score.violations:
            self.alerts.append(
                Alert(
                    when=violation.first_time if violation.first_time is not None else score.when,
                    severity=violation.severity,
                    message=violation.message,
                )
            )

    def summary(self) -> str:
        """Plain-ASCII report of the latest score plus how the monitor is doing."""
        lines = [self._last.summary()]
        lines.append(
            f"monitor: {self.n_updates} update(s), {len(self.alerts)} alert(s), "
            f"baseline {len(self._baseline)} row(s) scoring "
            f"{self.baseline_score.value:.1f} (grade {self.baseline_score.grade})"
        )
        for alert in self.alerts[-5:]:
            lines.append(f"  {alert.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the latest score, the history and the alerts."""
        return jsonable(
            {
                "latest": self._last.to_dict(),
                "baseline_score": self.baseline_score.to_dict(),
                "updates": self.n_updates,
                "history": self.history.to_dict(orient="records"),
                "alerts": [a.to_dict() for a in self.alerts],
                "channels": list(self.channels),
                "rules": [r.to_dict() for r in self.rules],
            }
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"HealthMonitor(channels={len(self.channels)}, rules={len(self.rules)}, "
            f"updates={self.n_updates}, last={self._last.value:.1f})"
        )
