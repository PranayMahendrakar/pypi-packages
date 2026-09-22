"""text-quality-ai: score text for readability, repetition, structure and clarity,
and say what to fix.

    >>> import text_quality_ai
    >>> report = text_quality_ai.score("The report was written by the team.")
    >>> report.grade in "ABCDF"
    True

No model is downloaded, no corpus is needed and nothing touches the network.
"""
from __future__ import annotations

from ._core import (
    TextScorer,
    available_targets,
    compare,
    readability,
    repetition,
    score,
)
from ._report import Issue, QualityReport

__version__ = "0.1.0"

__all__ = [
    "score",
    "readability",
    "repetition",
    "compare",
    "TextScorer",
    "QualityReport",
    "Issue",
    "available_targets",
    "__version__",
]
