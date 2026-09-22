"""The five retrieval metrics, checked against their documented formulas by hand."""
from __future__ import annotations

import math

import pytest

from rag_quality_check import dcg, hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_divides_by_k_not_by_what_came_back():
    # 3 of 3 retrieved are relevant, but the caller asked for 5.
    assert precision_at_k([1.0, 1.0, 1.0], 5) == pytest.approx(0.6)
    assert precision_at_k([1.0, 1.0, 1.0], 3) == pytest.approx(1.0)


def test_precision_ignores_items_past_k():
    assert precision_at_k([0.0, 0.0, 1.0], 2) == pytest.approx(0.0)


def test_precision_of_nothing_is_zero_not_a_crash():
    assert precision_at_k([], 5) == 0.0
    assert precision_at_k([], 0) == 0.0


def test_recall_is_over_all_known_relevant_items():
    # found 1 of the 4 relevant items
    assert recall_at_k([1.0, 0.0], [1.0] * 4, 2) == pytest.approx(0.25)


def test_recall_is_none_when_nothing_is_relevant():
    assert recall_at_k([0.0, 0.0], [], 2) is None
    assert recall_at_k([0.0], [0.0, 0.0], 1) is None


def test_mrr_is_the_reciprocal_of_the_first_hit_rank():
    assert mrr([1.0, 0.0, 0.0], 3) == pytest.approx(1.0)
    assert mrr([0.0, 1.0, 0.0], 3) == pytest.approx(0.5)
    assert mrr([0.0, 0.0, 1.0], 3) == pytest.approx(1 / 3)
    assert mrr([0.0, 0.0, 1.0], 2) == 0.0
    assert mrr([], 5) == 0.0


def test_dcg_uses_the_documented_linear_gain_log2_discount():
    # 1/log2(2) + 0 + 1/log2(4)
    expected = 1 / math.log2(2) + 1 / math.log2(4)
    assert dcg([1.0, 0.0, 1.0], 3) == pytest.approx(expected)
    assert dcg([], 3) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_bounded_below_it():
    assert ndcg_at_k([1.0, 1.0], [1.0, 1.0], 2) == pytest.approx(1.0)
    # one relevant item pushed to rank 2: 1/log2(3) divided by 1/log2(2)
    assert ndcg_at_k([0.0, 1.0], [1.0], 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_none_when_there_is_no_ideal_to_divide_by():
    assert ndcg_at_k([0.0, 0.0], [], 2) is None


def test_ndcg_handles_graded_gains_and_never_exceeds_one():
    value = ndcg_at_k([3.0, 1.0], [3.0, 1.0], 2)
    assert value == pytest.approx(1.0)
    assert ndcg_at_k([1.0, 3.0], [3.0, 1.0], 2) <= 1.0


def test_hit_rate_is_binary():
    assert hit_rate([0.0, 0.0, 1.0], 3) == 1.0
    assert hit_rate([0.0, 0.0, 1.0], 2) == 0.0
    assert hit_rate([], 3) == 0.0


def test_every_metric_docstring_states_its_formula():
    for function in (precision_at_k, recall_at_k, mrr, dcg, ndcg_at_k, hit_rate):
        doc = function.__doc__ or ""
        assert "=" in doc, function.__name__
        assert len(doc.strip()) > 80, function.__name__
