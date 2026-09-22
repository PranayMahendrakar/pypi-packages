"""Tokenising and BM25 scoring.

The ranking function is Okapi BM25 with the two constants that the original
TREC experiments settled on and that Lucene still ships:

``k1 = 1.5``
    Term-frequency saturation.  A term appearing ten times in a document counts
    for more than once, but nowhere near ten times as much.
``b = 0.75``
    Length normalisation.  ``b = 0`` ignores document length entirely, ``b = 1``
    divides fully by it; 0.75 is the usual middle.

The inverse document frequency uses the non-negative ("Lucene") form::

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

which never turns negative for a term that appears in more than half the store,
so a very common word contributes a little rather than subtracting.  A rarer
term always carries more weight than a common one, which is the property the
whole ranking rests on.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

K1 = 1.5
"""BM25 term-frequency saturation constant."""

B = 0.75
"""BM25 length-normalisation constant."""

# One token per CJK character (those scripts do not space their words), runs of
# digits, and runs of letters otherwise.  ``[^\W\d_]`` is "word character that is
# not a digit or underscore", which keeps accents and every non-Latin alphabet.
_TOKEN_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
    r"|\d+"
    r"|[^\W\d_]+",
    re.UNICODE,
)

MAX_TOKEN_CHARS = 64
"""Longer runs are cut here so a pathological input cannot bloat the index."""


def tokenize(text: str) -> List[str]:
    """Lowercase ``text`` and split it into index terms.

    Matching is exact on these terms: there is no stemming and no stop-word
    list, so ``"cat"`` does not match ``"cats"``.  BM25's idf already discounts
    words that turn up everywhere, which is what a stop-word list is for.
    """
    if not text:
        return []
    return [token[:MAX_TOKEN_CHARS] for token in _TOKEN_RE.findall(text.lower())]


def term_frequencies(tokens: Iterable[str]) -> Dict[str, int]:
    """Count each term once per document."""
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def idf(n_docs: int, doc_freq: int) -> float:
    """Inverse document frequency of a term seen in ``doc_freq`` of ``n_docs``."""
    if n_docs <= 0:
        return 0.0
    doc_freq = max(0, min(doc_freq, n_docs))
    return math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def bm25_term_score(
    term_freq: int,
    doc_length: int,
    avg_length: float,
    term_idf: float,
    k1: float = K1,
    b: float = B,
) -> float:
    """Contribution of a single query term to one document's BM25 score."""
    if term_freq <= 0:
        return 0.0
    if avg_length <= 0:
        avg_length = 1.0
    norm = 1.0 - b + b * (doc_length / avg_length)
    return term_idf * (term_freq * (k1 + 1.0)) / (term_freq + k1 * norm)
