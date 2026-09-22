"""Regressions for the issues an independent review found in 0.1.0.

Each test here is named for what it protects, because these are the failures
that got past the first suite.
"""
from __future__ import annotations

import time

import pytest

from semantic_dedup import (
    Deduper,
    canonical_tokens,
    dedupe,
    find_duplicates,
    similarity,
)
from semantic_dedup import _vectors


def _case_notes(n: int):
    """``n`` genuinely distinct case notes: the correct answer is zero duplicates."""
    subjects = [
        "invoice", "shipment", "password reset", "refund", "api key",
        "billing address", "tax form", "license", "webhook", "upgrade",
        "dashboard", "report",
    ]
    states = ["failed", "is delayed", "was approved", "needs review", "was cancelled"]
    texts = [
        f"Case {i}: the {subjects[i % 12]} for customer {i} {states[i % 5]} on day {i % 30}."
        for i in range(n)
    ]
    assert len(set(texts)) == n
    return texts


# -- blocker: the stemmer must never touch a number or an identifier -----------
@pytest.mark.parametrize(
    "token",
    ["1000", "1199", "4500", "2200", "5000", "2022", "100", "99", "sku12", "a1100"],
)
def test_numbers_and_identifiers_survive_canonicalization(token):
    # The English doubled-final-consonant rule must not fire on digits: it used
    # to turn "1000" into "100", so N and N-with-its-last-digit-repeated became
    # the same feature.
    assert canonical_tokens(token) == [token.casefold()]


@pytest.mark.parametrize(
    "a,b",
    [
        ("Refund issued for 1000 rupees.", "Refund issued for 100 rupees."),
        ("Order 4500 shipped today.", "Order 450 shipped today."),
        ("Ticket 1199 was escalated.", "Ticket 119 was escalated."),
        ("Payment of 2200 received.", "Payment of 220 received."),
        ("We shipped 5000 units.", "We shipped 500 units."),
    ],
)
def test_passages_differing_only_by_an_amount_are_not_duplicates(a, b):
    score = similarity(a, b)
    assert score < 1.0, f"{a!r} and {b!r} scored a perfect {score}"
    assert score < 0.82, f"{a!r} and {b!r} would merge at the default threshold"

    result = dedupe([a, b])
    assert result.removed == []
    assert result.groups == []
    assert result.texts == [a, b]


def test_an_amount_still_separates_passages_inside_a_small_corpus():
    # TF-IDF weights are relative to the corpus, so the same pair scores higher
    # surrounded by unrelated text than it does alone. It still must not group:
    # this is the shape a mixed real-world file takes.
    a, b = "Refund issued for 1000 rupees.", "Refund issued for 100 rupees."
    corpus = ["El pedido se retraso", "Lunch is at noon.", "another unrelated line", a, b]
    result = dedupe(corpus)
    assert result.groups == []
    assert result.removed == []
    assert result.texts == corpus


def test_a_matching_identifier_still_groups_a_paraphrase():
    # The numeric weighting must not go the other way and keep real duplicates
    # apart just because they quote a number.
    pairs = [
        ("Ticket 1199 was escalated.", "Ticket 1199 has been escalated."),
        ("The 2024 report was postponed.", "We moved the 2024 report to a later date."),
    ]
    for a, b in pairs:
        assert similarity(a, b) >= 0.82, f"{a!r} and {b!r} stopped matching"
        assert dedupe([a, b]).n_groups == 1


def test_sharing_a_number_does_not_make_two_notes_duplicates():
    a = "Case 7: the invoice failed on day 3."
    b = "Case 7: the license was approved on day 3."
    assert similarity(a, b) < 0.82
    assert dedupe([a, b]).groups == []


def test_similarity_agrees_with_threshold_one_on_near_identical_numbers():
    # similarity() reporting 1.00 for a pair that threshold=1.0 correctly keeps
    # apart was the secondary symptom of the same bug.
    a, b = "Refund issued for 1000 rupees.", "Refund issued for 100 rupees."
    assert similarity(a, b) < 1.0
    assert dedupe([a, b], threshold=1.0).groups == []
    assert dedupe([a, b], threshold=1.0).removed == []


def test_no_false_alarm_on_five_thousand_distinct_case_notes():
    # Every text differs from every other by its case/customer number. Anything
    # removed here is silent data loss.
    texts = _case_notes(5000)
    result = dedupe(texts)
    assert result.n_groups == 0
    assert result.n_removed == 0
    assert result.reduction == 0.0
    assert result.texts == texts

    report = find_duplicates(texts[:1000])
    assert report.n_groups == 0
    assert report.pairs == []


# -- major: method="tfidf" has to stay usable at a realistic size --------------
def test_tfidf_stays_fast_on_a_large_corpus():
    texts = _case_notes(5000)
    start = time.perf_counter()
    result = dedupe(texts, method="tfidf")
    elapsed = time.perf_counter() - start
    assert result.method == "tfidf"
    assert result.n_removed == 0
    # The quadratic scan took ~19s here; the pruned one takes well under 1s.
    assert elapsed < 12.0, f"tfidf on 5000 texts took {elapsed:.1f}s"


def test_the_prune_returns_exactly_what_an_exhaustive_scan_returns():
    # The speedup is only legitimate if it changes no answer, so hold the
    # pruned search against the reference scan on a corpus big enough to
    # trigger it (PREFIX_FILTER_MIN_ROWS rows) at several thresholds.
    texts = _case_notes(600)
    texts += ["The meeting was postponed.", "We moved the meeting to a later date."] * 3
    texts += ["Case 7: the license for customer 7 was approved on day 7."]
    matrix = _vectors.build_tfidf(texts)
    assert matrix.n_rows > _vectors.PREFIX_FILTER_MIN_ROWS
    assert _vectors._prefix_index(matrix, 0.82) is not None, "the prune never fired"

    for threshold in (0.3, 0.82, 0.95):
        pruned = sorted(
            (i, j, round(s, 9)) for i, j, s in _vectors.pairs_above(matrix, threshold)
        )
        exhaustive = sorted(
            (i, j, round(s, 9))
            for i, j, s in _vectors._pairs_above_exhaustive(matrix, threshold)
        )
        assert pruned == exhaustive, f"the prune changed the answer at {threshold}"


# -- minor: similarity(minhash) on featureless text ---------------------------
@pytest.mark.parametrize(
    "a,b", [("!", "?"), ("\U0001F680", "\U0001F389"), ("...", "???")]
)
def test_symbol_only_texts_are_not_perfectly_similar(a, b):
    assert similarity(a, b, method="minhash") == 0.0
    assert similarity(a, b) == 0.0
    # Identical symbol-only text still scores 1.0, on both methods.
    assert similarity(a, a, method="minhash") == 1.0
    assert similarity(a, a) == 1.0


# -- minor: wrong container types are rejected, not silently misread -----------
def test_a_dict_is_rejected_instead_of_being_deduped_by_its_keys():
    with pytest.raises(TypeError) as excinfo:
        dedupe({"a": "some text", "b": "other text"})
    message = str(excinfo.value)
    assert "dict" in message
    assert ".txt" in message


def test_a_set_is_rejected_because_its_order_is_not_stable():
    with pytest.raises(TypeError) as excinfo:
        dedupe({"some text", "other text"})
    assert "set" in str(excinfo.value)


def test_a_directory_path_says_it_is_a_directory(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        dedupe(str(tmp_path))
    message = str(excinfo.value)
    assert "directory" in message
    assert ".txt" in message


# -- minor: a reversed n-gram range cannot pass as "no duplicates" -------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"word_ngram": (5, 1)},
        {"char_ngram": (4, 2)},
        {"word_ngram": (5, 1), "char_ngram": (4, 2)},
    ],
)
def test_reversed_ngram_ranges_are_rejected(kwargs):
    texts = ["The meeting was postponed.", "We moved the meeting to a later date."]
    with pytest.raises(ValueError) as excinfo:
        dedupe(texts, **kwargs)
    assert "reversed" in str(excinfo.value)
    with pytest.raises(ValueError):
        Deduper(**kwargs)


def test_a_feature_free_run_is_warned_about_not_called_clean():
    # Symbols only: nothing can be compared, so the report has to say so rather
    # than confidently report "no duplicates found".
    result = dedupe(["!!!", "???", "..."])
    assert result.groups == []
    assert result.warnings, "a featureless run reported no warning"
    assert "no pair can match" in result.warnings[0]
    assert "note:" in result.summary()
