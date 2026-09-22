"""The public API: dedupe, find_duplicates, similarity and the result object."""
import json

import numpy as np
import pytest

import semantic_dedup
from semantic_dedup import DedupeResult, Deduper, dedupe, find_duplicates, similarity

PARAPHRASES = [
    "The meeting was postponed.",
    "The meeting has been pushed back.",
    "Lunch is at noon.",
]


def test_paraphrases_group_but_unrelated_text_does_not():
    result = dedupe(PARAPHRASES)
    assert result.groups == [[0, 1]]
    assert result.kept == [1, 2]
    assert result.removed == [0]
    assert result.texts == [PARAPHRASES[1], PARAPHRASES[2]]


def test_result_counts_and_reduction():
    result = dedupe(["a cat sat", "a cat sat", "a cat sat", "a dog barked"])
    assert result.n_texts == 4
    assert result.n_removed == 2  # three identical copies collapse to one
    assert result.n_kept == 2
    assert result.n_groups == 1
    assert result.n_duplicates == 2
    assert result.reduction == pytest.approx(0.5)


def test_indices_refer_to_original_positions():
    texts = ["unique one", "the price went up", "unique two", "costs increased", "unique three"]
    result = dedupe(texts)
    assert result.groups == [[1, 3]]
    # keep="longest" keeps index 1, so index 3 is the one that goes.
    assert result.removed == [3]
    assert result.kept == [0, 1, 2, 4]
    assert result.texts == [texts[i] for i in result.kept]
    for i, j, score in result.pairs:
        assert 0 <= i < j < len(texts)
        assert 0.0 <= score <= 1.0


def test_find_duplicates_reports_without_dropping():
    texts = ["the same thing", "the same thing", "something else"]
    result = find_duplicates(texts)
    assert result.kept == [0, 1, 2]
    assert result.removed == []
    assert result.n_removed == 0
    assert result.reduction == 0.0
    assert result.groups == [[0, 1]]
    assert result.n_duplicates == 1
    assert result.texts == texts
    assert "nothing removed" in result.summary()


def test_find_duplicates_accepts_the_same_keywords():
    result = find_duplicates(["x y z", "x y z", "q"], threshold=0.9, keep="first", method="tfidf")
    assert result.method == "tfidf"
    assert result.keep == "first"
    assert result.threshold == 0.9
    assert result.groups == [[0, 1]]


@pytest.mark.parametrize(
    "keep, expected_kept",
    [
        ("first", 0),
        ("last", 2),
        ("longest", 1),
        ("most_complete", 1),
    ],
)
def test_keep_modes(keep, expected_kept):
    texts = [
        "the server crashed",
        "the server crashed this morning during the backup",
        "the server crashed again",
    ]
    result = dedupe(texts, threshold=0.4, keep=keep)
    assert result.groups == [[0, 1, 2]]
    assert result.kept == [expected_kept]


def test_similarity_is_symmetric_and_bounded():
    assert similarity("Our prices went up.", "Costs increased.") == 1.0
    assert similarity("Please fix the login bug.", "Please resolve the sign in defect.") == 1.0
    low = similarity("The cat sat on the mat.", "Quarterly revenue fell sharply.")
    assert 0.0 <= low < 0.2
    a, b = "the report was published", "the document was released"
    assert similarity(a, b) == similarity(b, a)


def test_similarity_methods_and_errors():
    assert similarity("one two three", "one two three", method="minhash") == 1.0
    assert similarity("one two three", "one two three", method="auto") == 1.0
    assert 0.0 <= similarity("alpha beta", "alpha gamma", method="minhash") < 1.0
    with pytest.raises(ValueError, match="tfidf"):
        similarity("a", "b", method="nonsense")
    with pytest.raises(ValueError, match="embed"):
        similarity("a", "b", method="embed")
    with pytest.raises(TypeError):
        similarity("a", 3)


def test_method_tfidf_and_minhash_agree_on_a_small_corpus():
    texts = [f"report number {i} was published today" for i in range(40)]
    texts += ["report number 7 was published today", "a totally unrelated sentence"]
    exhaustive = dedupe(texts, method="tfidf", threshold=0.9)
    banded = dedupe(texts, method="minhash", threshold=0.9)
    assert exhaustive.groups == banded.groups
    assert exhaustive.method == "tfidf" and banded.method == "minhash"


def test_auto_picks_tfidf_for_small_inputs():
    assert dedupe(["a b c", "d e f"]).method == "tfidf"


def test_embed_hook_is_used_and_validated():
    def embed(batch):
        return np.array([[1.0, 0.0] if "cat" in text else [0.0, 1.0] for text in batch])

    result = dedupe(["a cat", "another cat", "a dog", "one more dog"], method="embed", embed=embed, threshold=0.99)
    assert result.method == "embed"
    assert result.groups == [[0, 1], [2, 3]]

    # auto picks embed as soon as one is supplied
    assert dedupe(["a cat", "a dog"], embed=embed, threshold=0.99).method == "embed"

    with pytest.raises(ValueError, match="one vector per text"):
        dedupe(["a", "b", "c"], method="embed", embed=lambda batch: np.zeros((2, 4)))
    with pytest.raises(ValueError, match="embed"):
        dedupe(["a", "b"], method="embed")
    with pytest.raises(TypeError):
        dedupe(["a", "b"], method="embed", embed="not callable")


def test_embed_rows_are_normalized_for_you():
    def embed(batch):
        return np.array([[3.0, 4.0] if "cat" in t else [-4.0, 3.0] for t in batch]) * 10.0

    result = dedupe(["a cat", "a big cat", "a dog"], method="embed", embed=embed, threshold=0.99)
    assert result.groups == [[0, 1]]


def test_invalid_arguments_are_rejected_clearly():
    with pytest.raises(ValueError, match="threshold"):
        dedupe(["a"], threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        dedupe(["a"], threshold=1.5)
    with pytest.raises(ValueError, match="method"):
        dedupe(["a"], method="magic")
    with pytest.raises(ValueError, match="keep"):
        dedupe(["a"], keep="shortest")
    with pytest.raises(ValueError, match="num_perm"):
        Deduper(num_perm=2)
    with pytest.raises(ValueError, match="texts\\[1\\]"):
        dedupe(["fine", 42])


def test_summary_and_to_dict_round_trip():
    result = dedupe(["the price went up", "costs increased", "an unrelated note"])
    text = result.summary()
    assert "semantic-dedup:" in text
    assert "group 1" in text
    assert text.isprintable() or "\n" in text
    payload = result.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert payload["n_texts"] == 3
    assert payload["groups"] == [[0, 1]]
    assert payload["reduction"] == pytest.approx(1 / 3, abs=1e-6)
    assert isinstance(result, DedupeResult)


def test_summary_uses_plain_ascii_punctuation():
    result = dedupe(["the price went up", "costs increased", "a note"])
    for char in result.summary():
        assert char == "\n" or 32 <= ord(char) <= 126, f"non-ascii char {char!r} in summary"


def test_summary_limits_the_number_of_groups_shown():
    texts = []
    for i in range(8):
        texts += [f"topic {i} was discussed", f"topic {i} was discussed"]
    result = dedupe(texts)
    assert result.n_groups == 8
    assert "and 3 more groups" in result.summary()
    assert "and 3 more groups" not in result.summary(max_groups=8)


def test_same_input_gives_the_same_output():
    texts = ["the price went up", "costs increased", "a note", "a note", "unrelated"]
    first = dedupe(texts).to_dict()
    for _ in range(3):
        assert dedupe(texts).to_dict() == first


def test_deduper_class_is_reusable():
    deduper = Deduper(threshold=0.9, keep="first")
    one = deduper.run(["x y z", "x y z", "q r s"])
    two = deduper.run(["x y z", "x y z", "q r s"])
    assert one.to_dict() == two.to_dict()
    assert one.kept == [0, 2]
    assert deduper.run(["x y z", "x y z"], drop=False).removed == []


def test_module_exports():
    for name in semantic_dedup.__all__:
        assert hasattr(semantic_dedup, name)
    assert semantic_dedup.__version__ == "0.1.0"
