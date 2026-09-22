"""Estimate how close equipment is to failure from sensor history.

    import predictive_maintenance as pm
    health = pm.health_score(df)
    rul = pm.estimate_rul(df)
    health.score, health.trend, health.contributors, rul.remaining

With labelled failures, :class:`MaintenanceModel` learns the patterns that come
before one and scores each row for failure inside a horizon.
"""

from ._health import health_score
from ._model import MaintenanceModel
from ._results import FailureRisk, HealthResult, RULResult
from ._rul import estimate_rul

__version__ = "0.1.0"

__all__ = [
    "health_score",
    "estimate_rul",
    "MaintenanceModel",
    "HealthResult",
    "RULResult",
    "FailureRisk",
    "__version__",
]
