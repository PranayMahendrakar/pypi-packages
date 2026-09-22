# -*- coding: utf-8 -*-
"""Regression tests: one per finding from the independent review.

Each test names the behaviour that was wrong and pins both halves of it -- the bad
input that must now be flagged, and the good input beside it that must still pass, so
no future "fix" can buy a catch by turning the check into a false alarm.
"""
import pytest

import hallucination_check as hc
from hallucination_check._facts import collect_keys
from hallucination_check._text import split_sentences, tokenize
from hallucination_check.cli import main


# -- blocker: a fabricated figure was backed up by any passage holding the digits -----

FAB_SOURCES = [
    {"id": "A", "text": "The drug reduced symptoms in the trial cohort."},
    {"id": "B", "text": "The study enrolled 1,200 participants across 14 hospitals."},
]


def test_a_number_borrowed_from_another_passage_does_not_ground_a_claim():
    report = hc.check(
        "The drug reduced symptoms by 1,200 percent in the trial cohort.", FAB_SOURCES
    )
    assert report.score == 0.0
    assert report.claims[0].supported is False
    assert "1,200 percent" in report.claims[0].reason
    assert report.citations == {}


def test_the_same_number_in_the_same_role_is_still_grounded():
    report = hc.check("The study enrolled 1,200 participants.", FAB_SOURCES)
    assert report.score == 100.0
    assert report.citations == {0: "B"}


NAME_SOURCES = [
    {"id": "A", "text": "The vaccine was developed at Oxford."},
    {"id": "B", "text": "Separately, Pfizer reported strong quarterly earnings."},
]


def test_a_name_used_elsewhere_for_something_else_does_not_ground_a_claim():
    report = hc.check("The vaccine was developed at Pfizer.", NAME_SOURCES)
    assert report.score == 0.0
    assert "Pfizer" in report.claims[0].reason
    assert report.citations == {}


def test_the_right_name_is_still_grounded_and_cited_to_the_right_passage():
    report = hc.check("The vaccine was developed at Oxford.", NAME_SOURCES)
    assert report.score == 100.0
    assert report.citations == {0: "A"}


def test_a_claim_is_cited_to_the_passage_that_actually_carries_its_fact():
    sources = [
        {"id": "A", "text": "The vaccine was developed at Oxford."},
        {"id": "B", "text": "Pfizer developed the vaccine in 2020."},
    ]
    report = hc.check("The vaccine was developed at Pfizer.", sources)
    assert report.score == 100.0
    assert report.citations == {0: "B"}
    assert "Pfizer" in report.claims[0].best_span


# -- blocker: a possessive in the source broke name grounding ------------------------


def test_a_possessive_source_still_grounds_the_bare_name():
    # The example printed by `hallucination-check --help`, run verbatim.
    assert hc.check("Paris is the capital of France.", ["France's capital is Paris."]).score == 100.0
    assert hc.check("NASA launched the probe.", ["NASA's probe was launched last year."]).score == 100.0
    assert hc.check(
        "The company raised its dividend.", ["The company's dividend was raised."]
    ).score == 100.0


def test_the_possessive_clitic_is_stripped_off_a_token():
    assert tokenize("France's capital is Paris.") == ["france", "capital", "is", "paris"]
    assert tokenize("NASA’s budget grew.") == ["nasa", "budget", "grew"]
    assert "name:france" in collect_keys("France's capital is Paris.")
    # Stripping never leaves a one-character stub behind.
    assert tokenize("A's mark was higher.") == ["a's", "mark", "was", "higher"]


# -- major: a false measurement rode in on a digit run used for something else -------

SATURN = ["The Saturn V rocket that carried Apollo 11 stood 110 metres tall."]


def test_a_wrong_measurement_is_flagged_even_when_the_digits_appear_in_a_name():
    report = hc.check("The Saturn V rocket stood 11 metres tall.", SATURN)
    assert report.score == 0.0
    assert report.claims[0].supported is False
    assert [c.text for c in report.contradictions] == ["The Saturn V rocket stood 11 metres tall."]


def test_the_right_measurement_is_still_grounded():
    assert hc.check("The Saturn V rocket stood 110 metres tall.", SATURN).score == 100.0


# -- major: contiguous CJK and Devanagari were never split into sentences ------------

CJK_ANSWER = (
    "埃菲尔铁塔位于巴黎。"
    "它建于1889年。"
    "它由古斯塔夫·埃菲尔设计。"
    "它的门票价格是每人五百欧元。"
)
CJK_SOURCES = [
    "埃菲尔铁塔位于法国巴黎，"
    "建于1889年，由古斯塔夫·埃菲尔设计。"
]


def test_contiguous_cjk_splits_into_one_claim_per_sentence():
    assert len(split_sentences(CJK_ANSWER)) == 4
    report = hc.check(CJK_ANSWER, CJK_SOURCES)
    assert report.n_claims == 4
    assert report.score == 75.0
    assert [c.supported for c in report.claims] == [True, True, True, False]
    # The invented fourth sentence, the one about the ticket price, is the flagged one.
    assert "五百欧元" in report.unsupported[0].text


def test_cjk_matches_the_english_rendering_of_the_same_content():
    english = (
        "The Eiffel Tower is in Paris. It was built in 1889. "
        "It was designed by Gustave Eiffel. Tickets cost five hundred euros per person."
    )
    sources = ["The Eiffel Tower is in Paris, France, built in 1889, designed by Gustave Eiffel."]
    report = hc.check(english, sources)
    assert (report.n_claims, report.score) == (4, 75.0)


def test_an_ascii_space_after_the_cjk_full_stop_changes_nothing():
    spaced = hc.check(CJK_ANSWER.replace("。", "。 "), CJK_SOURCES)
    assert spaced.n_claims == 4 and spaced.score == 75.0


def test_the_devanagari_danda_ends_a_sentence():
    hindi = (
        "यह एक परीक्षण है। "
        "दूसरा वाक्य यहाँ है। "
        "तीसरा वाक्य भी है।"
    )
    assert len(split_sentences(hindi)) == 3
    double_danda = (
        "पहला वाक्य॥ "
        "दूसरा वाक्य॥"
    )
    assert len(split_sentences(double_danda)) == 2


def test_a_decimal_point_still_does_not_split_a_latin_sentence():
    assert split_sentences("The rate was 3.5 percent in May.") == [
        "The rate was 3.5 percent in May."
    ]


# -- major: a date clash was missed whenever the two dates shared any component ------

TREATY = ["The treaty was signed on 20 July 1969 in Geneva."]


@pytest.mark.parametrize(
    "answer",
    [
        "The treaty was signed on 20 June 1971 in Geneva.",   # shares the day
        "The treaty was signed on 5 May 1969 in Geneva.",     # shares the year
        "The treaty was signed on 3 June 1971 in Geneva.",    # shares nothing
    ],
)
def test_a_date_disagreement_is_reported_even_when_a_component_matches(answer):
    report = hc.check(answer, TREATY)
    assert report.score == 0.0
    assert len(report.contradictions) == 1
    assert "20 July 1969" in report.claims[0].reason


def test_the_same_date_is_still_grounded():
    assert hc.check("The treaty was signed on 20 July 1969 in Geneva.", TREATY).score == 100.0


# -- minor: the contradiction named the wrong rival number --------------------------


def test_the_contradiction_names_the_figure_the_answer_actually_disputes():
    report = hc.check("The Saturn V rocket stood 150 metres tall.", SATURN)
    assert report.claims[0].reason == "the sources say 110 where the answer says 150"


# -- minor: a mistyped file path was checked as literal text ------------------------


def test_a_mistyped_source_path_is_an_error_not_a_confident_wrong_report(tmp_path, capsys):
    answer = tmp_path / "answer.txt"
    answer.write_text("The Kyoto Protocol was adopted in 1997.", encoding="utf-8")
    assert main([str(answer), "-s", "nosuch_passages.jsonl"]) == 1
    assert "looks like a file path" in capsys.readouterr().err


def test_a_mistyped_answer_path_is_an_error(tmp_path, capsys):
    source = tmp_path / "passage.txt"
    source.write_text("The Kyoto Protocol was adopted in 1997 in Japan.", encoding="utf-8")
    assert main(["nosuchfile.txt", "-s", str(source)]) == 1
    assert "looks like a file path" in capsys.readouterr().err


def test_ordinary_literal_text_is_still_taken_literally(capsys):
    assert main(["Paris is the capital of France.", "-s", "France's capital is Paris."]) == 0
    assert "100.0 / 100 grounded" in capsys.readouterr().out
