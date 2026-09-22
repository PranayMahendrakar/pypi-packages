"""machine-health: one continuously updated 0-100 health score per machine.

    >>> from machine_health import score
    >>> result = score(df, rules={"temp": {"max": 80}})
    >>> print(result.value, result.grade)

The score combines four components - stability, compliance, anomaly and
availability - and reports each of them separately, so the number can always be
explained: the points lost add up per component and per channel.
"""

from ._io import load_table
from .monitor import Alert, HealthMonitor
from .result import COMPONENTS, GRADE_BANDS, MachineScore, grade_for
from .rules import Rule, Violation
from .scoring import DEFAULT_WEIGHTS, HealthScorer, score

__version__ = "0.1.0"

__all__ = [
    "Alert",
    "COMPONENTS",
    "DEFAULT_WEIGHTS",
    "GRADE_BANDS",
    "HealthMonitor",
    "HealthScorer",
    "MachineScore",
    "Rule",
    "Violation",
    "__version__",
    "grade_for",
    "load_table",
    "score",
]
