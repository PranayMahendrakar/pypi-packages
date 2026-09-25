"""production-anomaly: catch abnormal production rates, downtime and process drift.

    import production_anomaly as pa
    report = pa.analyze(df, shift_hours=(6, 22), target_rate=3600)
    print(report.summary())
"""

from __future__ import annotations

import logging

from .analyzer import ProductionAnalyzer, analyze
from .report import QUALITY_NOTE, Event, ProductionReport, Stoppage
from .shifts import Schedule, Shift, parse_shift_hours

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "ProductionAnalyzer",
    "ProductionReport",
    "Stoppage",
    "Event",
    "Schedule",
    "Shift",
    "parse_shift_hours",
    "QUALITY_NOTE",
    "__version__",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
