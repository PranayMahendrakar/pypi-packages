"""Edge cases: empty and tiny inputs, exact thresholds, unicode, scale, determinism."""
import time

import numpy as np
import pytest

from semantic_dedup import Deduper, dedupe, find_duplicates, similarity
from semantic_dedup import canonical_tokens, normalize


# -- nothing, or almost nothing ------------------------------------------------
@pytest.mark.parametrize("method", ["auto", "tfidf", "minhash"])
def test_empty_input_is_a_valid_empty_result(method):
    result = dedupe([], method=method)
    assert result.kept == []
    assert result.removed == []
    assert result.groups == []
    assert result.pairs == []
    assert result.texts == []
    assert result.n_texts == 0
    assert result.n_kept == 0
    assert result.n_removed == 0
    assert result.n_groups == 0
    assert result.n_duplicates == 0
    assert result.reduction == 0.0  # no division by zero
    assert "0 texts" in result.summary()
    assert result.to_dict()["n_texts"] == 0


@pytest.mark.parametrize("method", ["auto", "tfidf", "minhash"])
def test_single_item_is_kept_and_reports_nothing(method):
    result = dedupe(["the only passage there is"], method=method)
    assert result.kept == [0]
    assert result.removed == []
    assert result.groups == []
    assert result.pairs == []
    assert result.texts == ["the only passage there is"]
    assert result.reduction == 0.0
    assert "1 text," in result.summary()
    assert "no duplicates" in result.summary()


def test_empty_and_whitespace_strings_do_not_blow_up():
    result = dedupe(["", "", "   ", "real text here"])
    assert result.groups == [[0, 1, 2]]  # all three normalize to ""
    assert result.kept == [2, 3] or result.kept == [0, 3]
    assert len(result.kept) == 2
    assert similarity("", "") == 1.0
    assert similarity("", "anything") == 0.0


def test_punctuation_only_texts_have_no_features_and_no_zero_division():
    # No word characters at all: the feature vector is empty by design.
    result = dedupe(["!!!", "???", "!!!", "..."])
    assert result.groups == [[0, 2]]  # only the exact repeat groups
    assert similarity("!!!", "???") == 0.0
    assert canonical_tokens("!!!") == []


# -- very short strings --------------------------------------------------------
def test_strings_under_three_characters():
    texts = ["a", "b", "ab", "a", "ba", "c", ""]
    result = dedupe(texts, threshold=0.5)
    assert result.groups == [[0, 3]]
    assert result.kept == [0, 1, 2, 4, 5, 6]
    for value in (similarity("a", "b"), similarity("ab", "ba"), similarity("a", "ab")):
        assert 0.0 <= value <= 1.0
    assert similarity("a", "a") == 1.0
    assert similarity("ab", "ab", method="minhash") == 1.0


def test_short_strings_survive_every_method():
    short = ["a", "b", "a", "cd", "cd"]
    for method in ("tfidf", "minhash"):
        result = dedupe(short, method=method, threshold=0.9)
        assert result.groups == [[0, 2], [3, 4]]
        assert result.n_removed == 2


# -- thresholds ----------------------------------------------------------------
@pytest.mark.parametrize("threshold", [0.01, 0.3, 0.5, 0.82, 0.99, 1.0])
def test_identical_strings_group_at_any_threshold(threshold):
    texts = ["exactly the same sentence", "exactly the same sentence", "another one"]
    result = dedupe(texts, threshold=threshold)
    assert result.groups == [[0, 1]]
    assert result.n_removed == 1
    assert (0, 1, 1.0) in result.pairs


def test_identical_after_normalization_counts_as_identical():
    # Case, NFKC width and runs of whitespace do not make a new text.
    texts = ["The  Meeting Was Postponed", "the meeting was postponed", "ＴＨＥ ＭＥＥＴＩＮＧ ＷＡＳ ＰＯＳＴＰＯＮＥＤ"]
    assert normalize(texts[0]) == normalize(texts[1]) == normalize(texts[2])
    result = dedupe(texts, threshold=1.0)
    assert result.groups == [[0, 1, 2]]


def test_threshold_one_means_exact_matches_only():
    texts = [
        "the price went up",
        "costs increased",          # a paraphrase, not an exact match
        "the price went up",        # an exact match
        "something unrelated",
    ]
    loose = dedupe(texts, threshold=0.82)
    assert loose.groups == [[0, 1, 2]]

    exact = dedupe(texts, threshold=1.0)
    assert exact.groups == [[0, 2]]
    assert exact.removed == [2]
    assert all(score == 1.0 for _, _, score in exact.pairs)


def test_threshold_bounds_are_enforced():
    for bad in (0.0, -0.1, 1.01, 2.0):
        with pytest.raises(ValueError, match="threshold"):
            dedupe(["a", "b"], threshold=bad)


# -- unicode -------------------------------------------------------------------
def test_unicode_texts_group_and_report_cleanly():
    texts = [
        "会議は延期されました",
        "会議は延期されました",
        "Café fermé aujourd'hui",
        "café fermé aujourd'hui",
        "Совещание перенесено",
        "Lunch is at noon.",
    ]
    result = dedupe(texts)
    assert result.groups == [[0, 1], [2, 3]]
    assert result.n_removed == 2
    report = result.summary()
    assert "会議は延期されました" in report
    assert "Café" in report
    payload = result.to_dict()
    assert payload["groups"] == [[0, 1], [2, 3]]


def test_unicode_is_not_mangled_by_normalization():
    assert normalize("Café") == "café"
    assert "延期" in "".join(canonical_tokens("会議は延期されました"))
    assert similarity("Совещание перенесено", "совещание  перенесено") == 1.0


def test_emoji_and_symbols_are_handled():
    texts = ["ship it 🚀", "ship it 🚀", "hold off 🛑"]
    result = dedupe(texts)
    assert result.groups == [[0, 1]]
    assert "🚀" in result.summary()


# -- scale ---------------------------------------------------------------------
def test_ten_thousand_items_complete_quickly_with_minhash():
    subjects = ["invoice", "shipment", "password reset", "refund", "api key",
                "billing address", "tax form", "license", "webhook", "upgrade"]
    states = ["failed", "is delayed", "was approved", "needs review", "was cancelled"]
    texts = [
        f"Case {i}: the {subjects[i % 10]} for customer {i} {states[i % 5]} on day {i % 30}."
        for i in range(10000)
    ]
    texts += ["Case 1: the shipment for customer 1 is delayed on day 1."] * 4

    start = time.perf_counter()
    result = dedupe(texts)
    elapsed = time.perf_counter() - start

    assert result.method == "minhash"          # auto switches above 5000 texts
    assert result.n_texts == 10004
    assert result.n_groups >= 1
    assert result.n_removed >= 4               # the four planted copies at least
    assert elapsed < 10.0, f"10k items took {elapsed:.1f}s"
    # Indices still point at the caller's positions.
    assert max(result.kept) < len(texts)
    assert result.texts == [texts[i] for i in result.kept]


def test_auto_switches_to_minhash_above_the_cutoff():
    from semantic_dedup import AUTO_MINHASH_ABOVE

    small = [f"sentence number {i} about nothing much" for i in range(200)]
    assert dedupe(small).method == "tfidf"
    big = [f"sentence number {i} about nothing much" for i in range(AUTO_MINHASH_ABOVE + 10)]
    assert dedupe(big).method == "minhash"


# -- indices and determinism ---------------------------------------------------
def test_indices_always_refer_to_the_original_positions():
    # Exact duplicates are collapsed internally; the result must not leak that.
    texts = [
        "zzz filler one",
        "repeat me",
        "zzz filler two",
        "repeat me",
        "zzz filler three",
        "repeat me",
    ]
    result = find_duplicates(texts)
    assert result.groups == [[1, 3, 5]]
    for i, j, score in result.pairs:
        assert 0 <= i < j < len(texts)
        assert texts[i] == texts[j] or score < 1.0
    dropped = dedupe(texts, keep="first")
    assert dropped.kept == [0, 1, 2, 4]
    assert dropped.removed == [3, 5]
    assert dropped.texts == [texts[i] for i in dropped.kept]


@pytest.mark.parametrize("method", ["tfidf", "minhash"])
def test_same_input_gives_the_same_output_every_run(method):
    texts = [f"note {i % 40} about the quarterly plan" for i in range(300)]
    texts += ["a paraphrase: notes on the plan for the quarter", "unrelated text"]
    first = dedupe(texts, method=method).to_dict()
    for _ in range(3):
        assert dedupe(texts, method=method).to_dict() == first
    assert Deduper(method=method).run(texts).to_dict() == first


def test_embed_path_is_deterministic_too():
    def embed(batch):
        rows = []
        for text in batch:
            tokens = canonical_tokens(text)
            rows.append([len(tokens), sum(len(t) for t in tokens), 1.0])
        return np.array(rows, dtype=float)

    texts = ["alpha beta", "beta alpha", "a much longer sentence entirely"]
    first = dedupe(texts, method="embed", embed=embed, threshold=0.999).to_dict()
    assert dedupe(texts, method="embed", embed=embed, threshold=0.999).to_dict() == first


def test_group_range_and_warnings_are_plain_data():
    result = dedupe(["repeat this", "repeat this", "other"])
    assert result.group_range([0, 1]) == (1.0, 1.0)
    assert result.group_range([2]) is None
    assert result.warnings == []
