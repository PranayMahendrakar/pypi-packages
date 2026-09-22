"""Result objects: :class:`Issue`, :class:`HealthReport`, the scoring table, and the renderers."""

from __future__ import annotations

import datetime as _dt
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

SEVERITIES = ("critical", "warning", "info")

# Every issue kind, in the order they are listed within one severity block of a summary.
KINDS = (
    "empty_dataset",
    "target_leakage",
    "class_imbalance",
    "missing",
    "empty_column",
    "empty_row",
    "duplicates",
    "mixed_types",
    "id_like",
    "constant",
    "near_constant",
    "high_cardinality",
    "dates_as_strings",
    "numeric_as_strings",
    "high_correlation",
    "outliers",
    "skew",
)

# Base penalty per (kind, severity). A group of n issues sharing kind and severity costs
# base * (1 + log2(n)): the tenth slightly-missing column hurts far less than the first.
PENALTIES: Dict[str, Dict[str, float]] = {
    "empty_dataset": {"critical": 100.0},
    "target_leakage": {"critical": 30.0, "warning": 8.0},
    "class_imbalance": {"critical": 20.0, "warning": 8.0},
    "missing": {"critical": 10.0, "warning": 4.0, "info": 1.0},
    "empty_column": {"critical": 10.0, "warning": 3.0},
    "empty_row": {"warning": 3.0, "info": 1.0},
    "duplicates": {"critical": 12.0, "warning": 5.0, "info": 1.0},
    "mixed_types": {"warning": 5.0},
    "id_like": {"warning": 4.0},
    "constant": {"warning": 3.0},
    "near_constant": {"warning": 3.0},
    "high_cardinality": {"warning": 3.0},
    "dates_as_strings": {"warning": 3.0},
    "numeric_as_strings": {"info": 1.0},
    "high_correlation": {"warning": 3.0},
    "outliers": {"warning": 3.0, "info": 1.0},
    "skew": {"info": 1.0},
}
DEFAULT_PENALTY: Dict[str, float] = {"critical": 15.0, "warning": 5.0, "info": 1.0}

GRADES = ((90, "healthy"), (75, "mostly healthy"), (50, "needs attention"), (0, "unhealthy"))


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain Python; NaN/NaT/inf become ``None``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, _dt.timedelta, np.timedelta64)):
        return str(value)
    if isinstance(value, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Index, pd.Series)):
        return [json_safe(v) for v in list(value)]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def penalty_for(kind: str, severity: str) -> float:
    """Base penalty of one issue of this kind and severity."""
    return PENALTIES.get(kind, {}).get(severity, DEFAULT_PENALTY[severity])


def compute_score(issues: Iterable["Issue"]) -> int:
    """100 minus the weighted penalties of ``issues``, floored at 0 and capped at 100."""
    groups = Counter((issue.kind, issue.severity) for issue in issues)
    total = 0.0
    for (kind, severity), n in groups.items():
        total += penalty_for(kind, severity) * (1.0 + math.log2(n))
    return int(max(0, min(100, round(100.0 - total))))


def grade_for(score: int) -> str:
    """Word for a score: healthy / mostly healthy / needs attention / unhealthy."""
    for floor, word in GRADES:
        if score >= floor:
            return word
    return GRADES[-1][1]


@dataclass
class Issue:
    """One finding.

    ``severity`` is ``"critical"``, ``"warning"`` or ``"info"``; ``kind`` names the check
    (see :data:`KINDS`); ``columns`` lists the columns involved (empty for row-level
    findings); ``message`` is human text; ``detail`` holds the numbers behind it (JSON-safe).
    """

    severity: str
    kind: str
    columns: List[str]
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, not {self.severity!r}")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        self.columns = [str(c) for c in self.columns]
        self.detail = json_safe(dict(self.detail))

    @property
    def penalty(self) -> float:
        """Base penalty this issue carries on its own."""
        return penalty_for(self.kind, self.severity)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict with severity, kind, columns, message and detail."""
        return {
            "severity": self.severity,
            "kind": self.kind,
            "columns": list(self.columns),
            "message": self.message,
            "detail": json_safe(self.detail),
        }

    def __str__(self) -> str:
        return f"[{self.severity}] {self.kind}: {self.message}"


def sort_issues(issues: Iterable[Issue]) -> List[Issue]:
    """Critical first, then by kind order (as in :data:`KINDS`), then by column names."""
    kind_rank = {kind: i for i, kind in enumerate(KINDS)}
    return sorted(
        issues,
        key=lambda i: (
            SEVERITIES.index(i.severity),
            kind_rank.get(i.kind, len(KINDS)),
            i.kind,
            [str(c) for c in i.columns],
        ),
    )


def _columns_label(columns: List[str], limit: int = 3) -> str:
    if not columns:
        return "(rows)"
    shown = ", ".join(columns[:limit])
    if len(columns) > limit:
        shown += f", +{len(columns) - limit} more"
    return shown


def _pct(share: float) -> str:
    return f"{100.0 * share:.1f}%"


@dataclass
class HealthReport:
    """What :func:`dataset_health.diagnose` returns.

    ``score`` is 0-100 (100 = nothing found) and ``issues`` the sorted findings.
    ``critical``, ``warnings`` and ``info`` are filtered views; ``summary()`` is human
    text, ``to_dict()`` is JSON-safe and ``to_markdown()`` renders a report page.
    """

    issues: List[Issue]
    n_rows: int
    n_columns: int
    n_rows_analyzed: int
    sampled: bool
    target: Optional[str]
    task: Optional[str]
    columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    score: Optional[int] = None  # computed from the issues when not given

    def __post_init__(self) -> None:
        self.issues = sort_issues(self.issues)
        if self.score is None:
            self.score = compute_score(self.issues)
        self.score = int(max(0, min(100, self.score)))

    # ----- filtered views -------------------------------------------------------------
    @property
    def critical(self) -> List[Issue]:
        """Issues with severity ``"critical"``."""
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> List[Issue]:
        """Issues with severity ``"warning"``."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def info(self) -> List[Issue]:
        """Issues with severity ``"info"``."""
        return [i for i in self.issues if i.severity == "info"]

    @property
    def grade(self) -> str:
        """One-phrase reading of the score: healthy / mostly healthy / needs attention / unhealthy."""
        return grade_for(self.score or 0)

    @property
    def counts(self) -> Dict[str, int]:
        """``{"critical": n, "warning": n, "info": n}``."""
        return {s: sum(1 for i in self.issues if i.severity == s) for s in SEVERITIES}

    @property
    def kinds(self) -> List[str]:
        """Distinct issue kinds present, in report order."""
        seen: List[str] = []
        for issue in self.issues:
            if issue.kind not in seen:
                seen.append(issue.kind)
        return seen

    def by_kind(self, kind: str) -> List[Issue]:
        """Issues of one kind, e.g. ``report.by_kind("missing")``."""
        return [i for i in self.issues if i.kind == kind]

    def by_column(self, column: str) -> List[Issue]:
        """Issues that involve ``column``."""
        column = str(column)
        return [i for i in self.issues if column in i.columns]

    # ----- renderers ------------------------------------------------------------------
    def _headline(self) -> str:
        c = self.counts
        found = (
            "no issues found"
            if not self.issues
            else f"{c['critical']} critical, {c['warning']} warning, {c['info']} info"
        )
        return f"dataset-health: score {self.score}/100 ({self.grade}) - {found}"

    def _shape_line(self) -> str:
        line = f"rows: {self.n_rows:,} | columns: {self.n_columns:,}"
        if self.sampled:
            line += f" | analyzed: {self.n_rows_analyzed:,} rows (random sample)"
        if self.target is not None:
            line += f" | target: {self.target} ({self.task})"
        return line

    def summary(self) -> str:
        """Human-readable text, issues grouped by severity."""
        lines = [self._headline(), self._shape_line()]
        for severity in SEVERITIES:
            group = [i for i in self.issues if i.severity == severity]
            if not group:
                continue
            lines.append("")
            lines.append(severity.upper())
            for issue in group:
                lines.append(f"  {issue.kind:<18} {_columns_label(issue.columns):<24} {issue.message}")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict: score, grade, shape, counts, issues, per-column profile, notes."""
        return {
            "score": int(self.score or 0),
            "grade": self.grade,
            "n_rows": int(self.n_rows),
            "n_columns": int(self.n_columns),
            "n_rows_analyzed": int(self.n_rows_analyzed),
            "sampled": bool(self.sampled),
            "target": self.target,
            "task": self.task,
            "counts": self.counts,
            "issues": [issue.to_dict() for issue in self.issues],
            "columns": json_safe(self.columns),
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        """Markdown report: headline, shape table, issues table, column profile, notes."""

        def cell(text: Any) -> str:
            return str(text).replace("|", "\\|").replace("\n", " ")

        c = self.counts
        lines = ["# dataset-health report", ""]
        lines.append(f"**Score: {self.score}/100 ({self.grade})**")
        lines.append("")
        lines.append(f"{c['critical']} critical, {c['warning']} warning, {c['info']} info")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---|")
        lines.append(f"| rows | {self.n_rows:,} |")
        lines.append(f"| columns | {self.n_columns:,} |")
        if self.sampled:
            lines.append(f"| rows analyzed | {self.n_rows_analyzed:,} (random sample) |")
        if self.target is not None:
            lines.append(f"| target | `{cell(self.target)}` ({self.task}) |")
        lines.append("")
        lines.append("## Issues")
        lines.append("")
        if not self.issues:
            lines.append("No issues found.")
        else:
            lines.append("| severity | kind | columns | message |")
            lines.append("|---|---|---|---|")
            for issue in self.issues:
                cols = ", ".join(f"`{cell(col)}`" for col in issue.columns) or "(rows)"
                lines.append(f"| {issue.severity} | {issue.kind} | {cols} | {cell(issue.message)} |")
        if self.columns:
            lines.append("")
            lines.append("## Columns")
            lines.append("")
            lines.append("| column | role | type | dtype | missing | unique |")
            lines.append("|---|---|---|---|---|---|")
            for name, prof in self.columns.items():
                lines.append(
                    f"| `{cell(name)}` | {prof.get('role', '')} | {prof.get('type', '')} | "
                    f"{cell(prof.get('dtype', ''))} | {_pct(float(prof.get('missing_share') or 0.0))} | "
                    f"{prof.get('n_unique', '')} |"
                )
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            lines.append("")
            lines.extend(f"- {cell(note)}" for note in self.notes)
        lines.append("")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
