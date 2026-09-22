"""The result object: one number, and everything needed to explain it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple

from ._util import fmt, jsonable
from .rules import Violation

#: Fixed score bands, high to low. ``(grade, lowest score that earns it)``.
GRADE_BANDS: Tuple[Tuple[str, float], ...] = (
    ("A", 90.0),
    ("B", 80.0),
    ("C", 70.0),
    ("D", 60.0),
    ("F", 0.0),
)

#: The four parts every score is built from, in reporting order.
COMPONENTS: Tuple[str, ...] = ("stability", "compliance", "anomaly", "availability")

_GRADE_ORDER = [grade for grade, _ in GRADE_BANDS]

_COMPONENT_HELP = {
    "stability": "how much each channel now varies compared with its baseline",
    "compliance": "whether your own limits are respected",
    "anomaly": "share of recent points that look like outliers",
    "availability": "missing data and flatlined channels",
}


def _points(lost: float) -> str:
    """'-12.50' for points lost, plain '0.00' when nothing was lost."""
    return f"-{lost:.2f}" if lost >= 0.005 else "0.00"


def grade_for(value: float) -> str:
    """Map a 0-100 score to its fixed letter band (A >= 90, B >= 80, C >= 70, D >= 60, else F)."""
    number = float(value)
    for grade, floor in GRADE_BANDS:
        if number >= floor:
            return grade
    return GRADE_BANDS[-1][0]


def grade_rank(grade: str) -> int:
    """0 for 'A' up to 4 for 'F'; higher means worse."""
    try:
        return _GRADE_ORDER.index(str(grade).upper())
    except ValueError:
        return len(_GRADE_ORDER) - 1


@dataclass
class MachineScore:
    """A machine's health as one 0-100 number, plus the reasoning behind it.

    ``value`` is ``100 - penalty``. The penalty is split two ways that both add up
    to it exactly: by component (``component_penalties``) and by channel
    (``contributors``). That is what makes the number explainable.
    """

    value: float
    grade: str
    components: Dict[str, float] = field(default_factory=dict)
    violations: List[Violation] = field(default_factory=list)
    contributors: Dict[str, float] = field(default_factory=dict)
    trend: str = "stable"
    weights: Dict[str, float] = field(default_factory=dict)
    component_penalties: Dict[str, float] = field(default_factory=dict)
    #: Components that could not be measured at all (no rules, too short a baseline).
    #: They carry no weight, and :meth:`to_dict` reports them as null rather than 100.
    unmeasured: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_baseline_rows: int = 0
    when: Any = None

    @property
    def penalty(self) -> float:
        """The points subtracted from 100, i.e. ``100 - value``."""
        return 100.0 - self.value

    @property
    def ok(self) -> bool:
        """True when the grade is C or better and no hard limit was broken."""
        return grade_rank(self.grade) <= grade_rank("C") and not self.critical_violations

    @property
    def critical_violations(self) -> List[Violation]:
        """Only the violations of hard ``min``/``max`` limits."""
        return [v for v in self.violations if v.severity == "critical"]

    def top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """The ``n`` channels costing the most points, worst first."""
        ranked = sorted(self.contributors.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(name, points) for name, points in ranked if points > 0][:n]

    def with_note(self, note: str) -> "MachineScore":
        """A copy of this score carrying one more note."""
        return replace(self, notes=list(self.notes) + [note])

    def summary(self) -> str:
        """Human-readable plain-ASCII report of the score and why it is what it is."""
        lines = [
            f"machine health: {self.value:.1f} / 100  (grade {self.grade}, trend {self.trend})"
        ]
        head = f"window: {self.n_rows} row(s), baseline: {self.n_baseline_rows} row(s)"
        if self.when is not None:
            head += f", latest: {fmt(self.when)}"
        lines.append(head)
        lines.append(
            f"channels ({len(self.channels)}): "
            + (", ".join(self.channels) if self.channels else "<none>")
        )
        lines.append("components (score, weight, points lost):")
        for name in COMPONENTS:
            if name not in self.components:
                continue
            weight = self.weights.get(name, 0.0)
            lost = self.component_penalties.get(name, 0.0)
            missing = name in self.unmeasured
            mark = "  not scored" if missing or weight <= 0 else ""
            shown = "   n/a" if missing else f"{self.components[name]:6.1f}"
            lines.append(
                f"  {name:<13}{shown}  weight {weight:4.2f}  {_points(lost):>8}{mark}"
            )
        top = self.top_contributors()
        if top:
            lines.append("points lost by channel:")
            for name, points in top:
                lines.append(f"  {name:<20}{_points(points):>8}")
        if self.violations:
            lines.append(f"violations ({len(self.violations)}):")
            for violation in self.violations[:10]:
                lines.append(f"  {violation.summary()}")
            if len(self.violations) > 10:
                lines.append(f"  ... and {len(self.violations) - 10} more")
        else:
            lines.append("violations: none")
        if self.notes:
            lines.append("notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of everything in the score."""
        return jsonable(
            {
                "value": round(float(self.value), 4),
                "grade": self.grade,
                "trend": self.trend,
                "ok": self.ok,
                "penalty": round(float(self.penalty), 4),
                # A component that was never measured is null, not a free 100.
                "components": {
                    k: (None if k in self.unmeasured else round(float(v), 4))
                    for k, v in self.components.items()
                },
                "unmeasured": list(self.unmeasured),
                "weights": {k: round(float(v), 6) for k, v in self.weights.items()},
                "component_penalties": {
                    k: round(float(v), 6) for k, v in self.component_penalties.items()
                },
                "contributors": {k: round(float(v), 6) for k, v in self.contributors.items()},
                "violations": [v.to_dict() for v in self.violations],
                "channels": list(self.channels),
                "notes": list(self.notes),
                "n_rows": int(self.n_rows),
                "n_baseline_rows": int(self.n_baseline_rows),
                "when": self.when,
                "grade_bands": {grade: floor for grade, floor in GRADE_BANDS},
                "component_meaning": dict(_COMPONENT_HELP),
            }
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()
