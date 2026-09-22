"""Detect whether production data has drifted from training data, column by column.

    import data_drift_lite
    report = data_drift_lite.detect(reference_df, current_df)
    report.drifted, report.drifted_columns, report.summary(), report.to_dict()
"""

from ._core import DriftMonitor, detect
from ._report import ColumnDrift, DriftReport, SchemaDrift

__version__ = "0.1.0"

__all__ = [
    "detect",
    "DriftMonitor",
    "DriftReport",
    "ColumnDrift",
    "SchemaDrift",
    "__version__",
]
