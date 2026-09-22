"""energy-analyzer-ai: find unusual energy consumption and what it is costing.

Point it at meter readings and it tells you what was used, what looks wrong,
what changed, and what the unusual part costs::

    import energy_analyzer_ai

    report = energy_analyzer_ai.analyze(readings, tariff=0.28)
    print(report.summary())
    for finding in report.findings:
        print(finding)

A Monday 9am is compared with other Monday 9ams, not with 3am. Cumulative meters
(the kind that only counts up) are detected and differenced instead of being read
as one huge value, gaps are reported rather than filled, and every cost stays
``None`` until you give a tariff.
"""
from ._core import EnergyAnalyzer, analyze, baseline_load, forecast
from ._model import StepChange, Trend
from ._report import Anomaly, EnergyReport
from ._tariff import Tariff

__version__ = "0.1.0"

__all__ = [
    "Anomaly",
    "EnergyAnalyzer",
    "EnergyReport",
    "StepChange",
    "Tariff",
    "Trend",
    "analyze",
    "baseline_load",
    "forecast",
    "__version__",
]
