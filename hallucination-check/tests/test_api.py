"""The public API: check, check_batch, GroundingChecker and the result objects."""
import json

import numpy as np
import pytest

import hallucination_check as hc

ANSWER = "The Eiffel Tower is in Paris. It was completed in 1889. It cost 42 million francs."
SOURCES = ["The Eiffel Tower stands in Paris, France, and was completed in 1889."]


def test_check_returns_a_report_with_a_flagged_claim():
    report = hc.check(ANSWER, SOURCES)
    assert isinstance(report, hc.GroundingReport)
    assert report.n_claims == 3
    assert report.n_supported == 2
    assert report.score == pytest.approx(66.7, abs=0.1)
    assert [c.text for c in report.unsupported] == ["It cost 42 million francs."]


def test_claim_fields_are_populated():
    claim = hc.check(ANSWER, SOURCES).claims[1]
    assert isinstance(claim, hc.Claim)
    assert claim.supported is True
    assert 0.0 <= claim.confidence <= 1.0
    assert claim.best_source == "s1"
    assert "1889" in claim.best_span
    assert "s1" in claim.reason
    assert claim.to_dict()["text"] == claim.text


def test_unsupported_claim_explains_itself():
    claim = hc.check(ANSWER, SOURCES).claims[2]
    assert claim.supported is False
    assert "appear nowhere in the sources" in claim.reason
    assert "42 million" in claim.reason


def test_citations_only_cover_supported_claims():
    report = hc.check(ANSWER, SOURCES)
    assert report.citations == {0: "s1", 1: "s1"}


def test_source_ids_are_kept_when_given():
    passages = [
        {"id": "wiki:eiffel#3", "text": "The Eiffel Tower stands in Paris."},
        {"id": "wiki:paris#1", "text": "Paris is the capital of France."},
    ]
    report = hc.check("The Eiffel Tower is in Paris.", passages)
    assert report.source_ids == ["wiki:eiffel#3", "wiki:paris#1"]
    assert report.claims[0].best_source == "wiki:eiffel#3"


def test_a_plain_string_source_works():
    report = hc.check("Paris is in France.", "Paris is a city in France.")
    assert report.score == 100.0
    assert report.source_ids == ["s1"]


def test_summary_is_ascii_and_mentions_the_numbers():
    text = hc.check(ANSWER, SOURCES).summary()
    text.encode("ascii")  # plain ASCII punctuation only
    assert "66.7 / 100 grounded" in text
    assert "not supported (1)" in text
    assert "lexical grounding is a signal, not proof" in text


def test_summary_when_everything_is_supported():
    text = hc.check("Paris is in France.", "Paris is a city in France.").summary()
    assert "every claim is backed by a source" in text


def test_to_dict_is_json_serialisable():
    payload = hc.check(ANSWER, SOURCES).to_dict()
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["score"] == 66.7
    assert restored["unsupported"] == [2]
    assert restored["citations"] == {"0": "s1", "1": "s1"}
    assert restored["n_claims"] == 3 and restored["n_unsupported"] == 1
    assert restored["granularity"] == "sentence"
    assert len(restored["claims"]) == 3


def test_repr_is_useful():
    assert "GroundingReport(score=" in repr(hc.check(ANSWER, SOURCES))


def test_threshold_changes_the_verdicts():
    strict = hc.check("The trial ran for six months.", ["The trial ran for six months at two sites."], threshold=0.99)
    lenient = hc.check("The trial ran for six months.", ["The trial ran for six months at two sites."], threshold=0.1)
    assert strict.score == 0.0
    assert lenient.score == 100.0


def test_granularity_clause_splits_further():
    answer = "Cats sleep a lot; dogs run a lot."
    per_sentence = hc.check(answer, ["Cats sleep a lot and dogs run a lot."])
    per_clause = hc.check(answer, ["Cats sleep a lot and dogs run a lot."], granularity="clause")
    assert per_sentence.n_claims == 1
    assert per_clause.n_claims == 2
    assert per_clause.granularity == "clause"


def test_granularity_paragraph_splits_on_blank_lines():
    answer = "First para about cats.\n\nSecond para about dogs."
    report = hc.check(answer, ["Cats and dogs."], granularity="paragraph")
    assert report.n_claims == 2


def test_contradiction_is_reported_separately():
    report = hc.check(
        "The trial enrolled 4,200 patients.",
        ["The trial enrolled 1,200 patients at six sites."],
    )
    assert report.score == 0.0
    assert [c.text for c in report.contradictions] == ["The trial enrolled 4,200 patients."]
    assert "1,200" in report.contradictions[0].reason
    assert report.to_dict()["contradictions"] == [0]
    assert "[contradiction]" in report.summary()


def test_a_missing_number_is_not_a_contradiction_when_nothing_rivals_it():
    report = hc.check(
        "The Eiffel Tower cost 42 million francs.",
        ["The Eiffel Tower stands in Paris and is made of iron."],
    )
    assert report.unsupported and not report.contradictions


def test_flag_missing_facts_can_be_turned_off():
    answer = "The trial enrolled 4,200 patients."
    sources = ["The trial enrolled 1,200 patients."]
    assert hc.check(answer, sources).score == 0.0
    loose = hc.GroundingChecker(flag_missing_facts=False).check(answer, sources)
    assert loose.score == 100.0


def test_check_batch_with_tuples_and_dicts():
    reports = hc.check_batch(
        [
            ("A cat sat on the mat.", ["A cat sat on the mat yesterday."]),
            {"answer": "Dogs bark loudly.", "sources": "Cats meow quietly."},
        ]
    )
    assert [r.score for r in reports] == [100.0, 0.0]
    assert all(isinstance(r, hc.GroundingReport) for r in reports)


def test_check_batch_passes_options_through():
    reports = hc.check_batch([("Cats sleep.", ["Cats sleep a lot."])], threshold=0.99)
    assert reports[0].threshold == 0.99
    assert reports[0].score == 0.0


def test_check_batch_rejects_a_bad_pair():
    with pytest.raises(ValueError, match="pair 0"):
        hc.check_batch(["just a string"])
    with pytest.raises(ValueError, match="pair 0"):
        hc.check_batch([{"answer": "x"}])


def test_checker_class_exposes_the_extra_knobs():
    checker = hc.GroundingChecker(threshold=0.4, window=1, top_k=5, char_n=3)
    report = checker.check(ANSWER, SOURCES)
    assert report.threshold == 0.4
    assert report.n_claims == 3
    assert [r.n_claims for r in checker.check_batch([(ANSWER, SOURCES)])] == [3]


def test_embed_hook_is_blended_in():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return np.array([[float(len(t)), float(t.count("a"))] for t in texts])

    report = hc.check("Cats sleep a lot.", ["Felines rest most of the day."], embed=embed)
    assert calls, "embed was never called"
    assert report.claims[0].confidence > hc.check(
        "Cats sleep a lot.", ["Felines rest most of the day."]
    ).claims[0].confidence


def test_embed_sees_every_source_sentence_not_just_lexical_matches():
    seen = {}

    def embed(texts):
        seen["texts"] = list(texts)
        return np.eye(len(texts), 3)

    hc.check("Cats sleep a lot.", ["Felines rest most of the day."], embed=embed)
    assert "Felines rest most of the day." in seen["texts"]


def test_a_broken_embed_hook_degrades_to_lexical():
    def bad(texts):
        raise RuntimeError("model is not loaded")

    report = hc.check(ANSWER, SOURCES, embed=bad)
    assert report.score == pytest.approx(66.7, abs=0.1)
    assert any("model is not loaded" in w for w in report.warnings)


def test_an_embed_hook_with_the_wrong_shape_degrades_to_lexical():
    report = hc.check(ANSWER, SOURCES, embed=lambda texts: [[1.0, 2.0]])
    assert report.score == pytest.approx(66.7, abs=0.1)
    assert any("semantic matching was skipped" in w for w in report.warnings)


def test_bad_arguments_raise_clear_errors():
    with pytest.raises(ValueError, match="threshold"):
        hc.check("x", "y", threshold=2.0)
    with pytest.raises(ValueError, match="granularity"):
        hc.check("x", "y", granularity="word")
    with pytest.raises(TypeError, match="answer must be a string"):
        hc.check(42, "y")
    with pytest.raises(TypeError, match="embed must be a callable"):
        hc.check("x", "y", embed="nope")
    with pytest.raises(ValueError, match="'text' key"):
        hc.check("x", [{"id": "a"}])
    with pytest.raises(TypeError, match="must be a str or a dict"):
        hc.check("x", [123])
    with pytest.raises(TypeError, match="sources must be"):
        hc.check("x", 123)


def test_version_is_exposed():
    assert hc.__version__ == "0.1.0"
