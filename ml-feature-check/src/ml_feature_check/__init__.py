"""Catch useless, redundant, leaking and suspicious features before you train on them.

    >>> from ml_feature_check import check
    >>> report = check(df, target="label")
    >>> print(report.summary())
    >>> clean = report.apply(df)
"""

from .checker import KINDS, FeatureChecker, check
from .report import FeatureReport, Finding

__version__ = "0.1.0"

__all__ = [
    "check",
    "FeatureChecker",
    "FeatureReport",
    "Finding",
    "KINDS",
    "__version__",
]
