"""synthetic-tabular: realistic synthetic tabular data from a pandas DataFrame.

A Gaussian copula learns each column's distribution and the correlations between
columns, then samples new rows that look like the original without copying it.
"""
from .evaluate import ColumnFidelity, FidelityReport, evaluate
from .synthesizer import Synthesizer, generate

__version__ = "0.1.0"

__all__ = [
    "Synthesizer",
    "generate",
    "evaluate",
    "FidelityReport",
    "ColumnFidelity",
    "__version__",
]
