"""The edge cases that have to be settled, not accidental."""
import time
import unicodedata

import pytest

import hallucination_check as hc
from hallucination_check import _core


def test_empty_answer_scores_100_with_no_claims():
    report = hc.check("", ["Some source text."])
    assert report.score == 100.0
    assert report.claims == []
    assert report.unsupported == []
    assert report.citations == {}
    assert "nothing to check" in report.summary()


def test_whitespace_only_answer_behaves_like_an_empty_one():
    report = hc.check("   \n\t  ", ["Some source text."])
    assert report.score == 100.0 and report.n_claims == 0


def test_empty_sources_score_0_and_flag_everything():
    report = hc.check("Paris is in France. It is a city.", [])
    assert report.score == 0.0
    assert report.n_claims == 2
    assert all(not c.supported for c in report.claims)
    assert all(c.reason == "no sources were given" for c in report.claims)
    assert report.citations == {}
    assert any("no sources were given" in w for w in report.warnings)


def test_none_and_blank_sources_do_not_crash():
    assert hc.check("Paris is in France.", None).score == 0.0
    assert hc.check("Paris is in France.", "").score == 0.0
    blank = hc.check("Paris is in France.", ["", "   "])
    assert blank.score == 0.0
    assert any("held no text" in w for w in blank.warnings)


def test_empty_answer_and_empty_sources_together():
    report = hc.check("", [])
    assert report.score == 100.0 and report.n_claims == 0


def test_an_answer_identical_to_a_source_scores_100():
    text = (
        "Zoe Kowalski reported a 12 percent drop in March 2024. "
        "The team confirmed the figure on 3 April 2024."
    )
    report = hc.check(text, [text])
    assert report.score == 100.0
    assert report.n_claims == 2
    assert all(c.supported for c in report.claims)
    assert set(report.citations) == {0, 1}


def test_unicode_survives_end_to_end():
    answer = "Le rapport indique une hausse de 12 %. 李雷 a confirmé les chiffres."
    sources = ["Le rapport indique une hausse de 12 % cette année. 李雷 a confirmé les chiffres."]
    report = hc.check(answer, sources)
    assert report.n_claims == 2
    assert report.score == 100.0
    assert "李雷" in report.claims[1].text
    # CONVENTIONS.md: summary() uses plain ASCII punctuation. Anything non-ASCII in it
    # has to be a letter, digit or combining mark quoted out of the answer itself, never
    # a frame character, arrow or fancy quote the report added.
    for char in report.summary():
        if ord(char) > 127:
            assert unicodedata.category(char)[0] in "LNM", (
                "summary() must use plain ASCII punctuation, found %r" % char
            )
    assert "李雷" in report.to_dict()["claims"][1]["text"]


def test_unicode_answer_against_unrelated_sources_is_flagged_not_crashed():
    report = hc.check("Ångström Straße 北京 café naïve.", ["Something entirely different."])
    assert 0.0 <= report.score <= 100.0
    assert report.n_claims == 1


def test_numbers_in_words_match_the_same_number_in_digits():
    assert hc.check("The study enrolled twenty-three patients.", ["The study enrolled 23 patients."]).score == 100.0
    assert hc.check("The study enrolled 23 patients.", ["The study enrolled twenty-three patients."]).score == 100.0


def test_a_number_in_words_that_is_absent_is_still_flagged():
    report = hc.check(
        "The study enrolled twenty-three patients.", ["The study enrolled 40 patients."]
    )
    assert report.score == 0.0
    assert report.unsupported


def test_a_long_source_list_stays_fast():
    sources = [
        {
            "id": "doc-%d" % i,
            "text": "Passage %d covers topic %d and reports a value of %d units in trial %d."
            % (i, i % 17, i % 91, i),
        }
        for i in range(3000)
    ]
    answer = (
        "Passage 1200 covers topic 10 and reports a value of 16 units. "
        "The programme ended in 1998 after four reviews."
    )
    start = time.time()
    report = hc.check(answer, sources)
    elapsed = time.time() - start
    assert elapsed < 5.0, "took %.2fs" % elapsed
    assert report.n_claims == 2
    assert 0.0 <= report.score <= 100.0


def test_the_score_is_always_between_0_and_100():
    cases = [
        ("", []),
        ("A.", []),
        ("A. B. C.", ["A."]),
        ("Nothing here matches.", ["Totally different content."]),
        ("Same text.", ["Same text."]),
        ("12345", ["12345"]),
        ("...", ["..."]),
    ]
    for answer, sources in cases:
        score = hc.check(answer, sources).score
        assert 0.0 <= score <= 100.0, (answer, sources, score)


def test_identical_input_gives_identical_output():
    answer = "Revenue rose 12% to $4.3 billion in 2023. Margins held at 31%."
    sources = ["Revenue rose 12% to $4.3 billion in 2023, the company said."]
    first = hc.check(answer, sources).to_dict()
    for _ in range(3):
        assert hc.check(answer, sources).to_dict() == first


def test_punctuation_only_answer_is_not_a_factual_claim():
    report = hc.check("---", ["Some source."])
    assert report.n_claims == 1
    assert report.claims[0].supported is True
    assert report.claims[0].reason == "no factual content to check"
    assert any("could be checked" in w for w in report.warnings)


def test_a_single_claim_answer_works():
    report = hc.check("Paris is in France.", ["Paris is a city in France."])
    assert report.n_claims == 1 and report.score == 100.0


def test_abbreviations_do_not_split_a_sentence():
    report = hc.check(
        "Dr. Smith of Acme Inc. reported the result.", ["Dr. Smith of Acme Inc. reported the result."]
    )
    assert report.n_claims == 1


def test_the_unit_fallback_records_a_warning(monkeypatch):
    monkeypatch.setattr(_core, "split_units", lambda text, granularity: [])
    report = hc.check("Some answer with no detectable unit.", ["Some source."])
    assert report.n_claims == 1
    assert any("treated as one claim" in w for w in report.warnings)
    assert "warnings:" in report.summary()


def test_sources_can_be_any_iterable():
    report = hc.check("Paris is in France.", (s for s in ["Paris is a city in France."]))
    assert report.score == 100.0


def test_bytes_sources_are_rejected_clearly():
    with pytest.raises(TypeError, match="decode it to str"):
        hc.check("x", [b"bytes"])
