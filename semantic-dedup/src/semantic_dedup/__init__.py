"""semantic-dedup: remove passages that repeat the same meaning, not just the same words.

Quick use::

    import semantic_dedup
    result = semantic_dedup.dedupe(texts)   # or a .txt / .csv / .jsonl path
    print(result.summary())
    clean = result.texts
"""
from ._core import (
    AUTO_MINHASH_ABOVE,
    KEEP_MODES,
    METHODS,
    DedupeResult,
    Deduper,
    dedupe,
    find_duplicates,
    similarity,
)
from ._text import canonical_tokens, normalize

__version__ = "0.1.0"

__all__ = [
    "DedupeResult",
    "Deduper",
    "dedupe",
    "find_duplicates",
    "similarity",
    "canonical_tokens",
    "normalize",
    "METHODS",
    "KEEP_MODES",
    "AUTO_MINHASH_ABOVE",
    "__version__",
]
