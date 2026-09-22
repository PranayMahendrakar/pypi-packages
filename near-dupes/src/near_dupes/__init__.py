"""near-dupes: find near-duplicate text, records and images, then dedupe keeping the best copy.

Quick use::

    import near_dupes
    result = near_dupes.find_duplicates(texts)   # or a DataFrame, or image paths
    clean = result.dedupe(texts)
"""
from ._core import DuplicateFinder, DuplicateResult, dedupe, find_duplicates
from ._text import text_similarity

__version__ = "0.1.0"

__all__ = [
    "DuplicateFinder",
    "DuplicateResult",
    "dedupe",
    "find_duplicates",
    "text_similarity",
    "__version__",
]
