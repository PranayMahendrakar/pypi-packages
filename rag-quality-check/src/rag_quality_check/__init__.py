"""rag-quality-check: measure whether your retriever is retrieving the right things.

Quick use::

    import rag_quality_check

    report = rag_quality_check.evaluate([
        {"query": "how do I reset my password",
         "retrieved": ["p_reset", "p_billing"],
         "relevant": ["p_reset"]},
    ])
    print(report.summary())

Every metric is in ``0..1`` and its exact formula is in the docstring of the
function that produces it, because an undocumented variant of nDCG is worse than
no nDCG at all. When a case has no ``relevant`` ids the numbers are *estimated*
from similarity rather than measured, and the report says so every time.
"""
from ._core import (
    ANSWER_METRICS,
    METRIC_HELP,
    RETRIEVAL_METRICS,
    CaseResult,
    RagReport,
    evaluate,
    evaluate_case,
)
from ._metrics import (
    dcg,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ._text import Similarity, cosine, coverage, split_sentences, tokenize

__version__ = "0.1.0"

__all__ = [
    "ANSWER_METRICS",
    "METRIC_HELP",
    "RETRIEVAL_METRICS",
    "CaseResult",
    "RagReport",
    "Similarity",
    "cosine",
    "coverage",
    "dcg",
    "evaluate",
    "evaluate_case",
    "hit_rate",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "split_sentences",
    "tokenize",
    "__version__",
]
