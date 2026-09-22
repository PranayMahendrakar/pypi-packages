"""The five retrieval metrics, each with its exact formula in the docstring.

These numbers get compared across systems and across papers, and every one of
them has more than one definition in the wild. The variant implemented here is
written out in full below; nothing is left to the reader's assumption.

Shared notation
---------------
``gains``
    The gain of each item in the ranked list, rank 1 first, already
    de-duplicated and already truncated to the top ``k``. A gain of ``0`` means
    not relevant; any gain above ``0`` means relevant. With a plain list of
    relevant ids every relevant item has gain ``1.0``.
``ideal_gains``
    Every non-zero gain known for this query, largest first - the best ranking
    that could possibly have been returned. With a plain list of relevant ids
    this is ``[1.0] * len(relevant)``.
``k``
    The cut-off. It is always the ``k`` the caller asked for, never the length
    of the list that came back, so retrieving fewer than ``k`` items lowers
    ``precision_at_k`` instead of quietly changing the denominator.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

__all__ = [
    "dcg",
    "hit_rate",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]


def precision_at_k(gains: Sequence[float], k: int) -> float:
    """``precision@k = (relevant items among the top k) / k``.

    The denominator is ``k`` itself, not the number of items actually returned.
    A system that returns 3 passages when asked for 5 and gets all 3 right
    scores ``0.6``, not ``1.0``; the report notes every case where that
    happened so the number is never a silent surprise. Range ``0..1``.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for gain in gains[:k] if gain > 0)
    return hits / k


def recall_at_k(gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> Optional[float]:
    """``recall@k = (relevant items among the top k) / (all relevant items)``.

    The denominator is the number of relevant items known for the query, so
    this answers "how much of what I should have found did I find". Returns
    ``None`` - excluded, not zero - when the query has no relevant items at
    all, since dividing by nothing is not a score of nothing. Range ``0..1``.
    """
    total = sum(1 for gain in ideal_gains if gain > 0)
    if total == 0:
        return None
    hits = sum(1 for gain in gains[:k] if gain > 0)
    return hits / total


def mrr(gains: Sequence[float], k: int) -> float:
    """``mrr = 1 / (rank of the first relevant item)``, ranks counted from 1.

    ``0.0`` when nothing in the top ``k`` is relevant. This is the reciprocal
    rank of a single query; the report's ``mrr`` is the mean of these across
    cases, which is what "mean reciprocal rank" means. Range ``0..1``.
    """
    for position, gain in enumerate(gains[:k], start=1):
        if gain > 0:
            return 1.0 / position
    return 0.0


def dcg(gains: Sequence[float], k: int) -> float:
    """``DCG@k = sum over i=1..k of gain_i / log2(i + 1)``.

    The linear-gain form. The exponential form ``(2**gain_i - 1) / log2(i + 1)``
    is the other common variant and is deliberately *not* used here; with the
    binary gains this package produces by default the two agree exactly, and
    for graded gains the linear form is the one reported.
    """
    total = 0.0
    for position, gain in enumerate(gains[:k], start=1):
        if gain:
            total += gain / math.log2(position + 1)
    return total


def ndcg_at_k(gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> Optional[float]:
    """``ndcg@k = DCG@k / IDCG@k``, both with linear gains (see :func:`dcg`).

    ``IDCG@k`` is :func:`dcg` of the ideal ranking: every gain known for the
    query, largest first, cut at ``k``. Returns ``None`` when ``IDCG@k`` is
    ``0`` (no relevant items known), because a ratio with no ideal to divide by
    is undefined rather than bad. Range ``0..1``.
    """
    ideal = dcg(sorted(ideal_gains, reverse=True), k)
    if ideal <= 0.0:
        return None
    return min(1.0, dcg(gains, k) / ideal)


def hit_rate(gains: Sequence[float], k: int) -> float:
    """``hit_rate = 1.0 if any of the top k is relevant, else 0.0``.

    Per case this is 0 or 1; the report's ``hit_rate`` is the mean, which reads
    as "the share of queries where the answer was in there somewhere". Range
    ``0..1``.
    """
    return 1.0 if any(gain > 0 for gain in gains[:k]) else 0.0
