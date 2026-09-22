"""privacy-scan-ml: find personal data in a dataset before it leaks into a model.

    >>> import pandas as pd, privacy_scan_ml as psm
    >>> report = psm.scan(pd.DataFrame({"email": ["a@example.com"]}))
    >>> report.has_pii
    True
"""
from __future__ import annotations

from ._detectors import ALL_TYPES, COLUMN_ONLY_TYPES, PATTERN_TYPES, SENSITIVE_TYPES
from ._mask import STRATEGIES
from .report import ColumnFinding, Finding, PIIReport
from .scanner import PIIScanner, mask, scan

__version__ = "0.1.0"

__all__ = [
    "scan",
    "mask",
    "PIIScanner",
    "PIIReport",
    "ColumnFinding",
    "Finding",
    "ALL_TYPES",
    "PATTERN_TYPES",
    "COLUMN_ONLY_TYPES",
    "SENSITIVE_TYPES",
    "STRATEGIES",
    "__version__",
]
