"""One-call health report for any CSV or Parquet dataset.

    import dataset_health
    report = dataset_health.diagnose(df, target="label")
    report.score, report.critical, report.summary(), report.to_dict()
"""

from ._core import HealthChecker, diagnose
from ._report import KINDS, SEVERITIES, HealthReport, Issue

__version__ = "0.1.0"

__all__ = [
    "diagnose",
    "HealthChecker",
    "HealthReport",
    "Issue",
    "KINDS",
    "SEVERITIES",
    "__version__",
]
