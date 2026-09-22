"""Regressions for everything the independent review found.

Each test names the symptom a reader would have seen, so a future change that
brings one back fails here rather than in a reviewer's hands.
"""
import json

import pytest

import text_quality_ai
from text_quality_ai import TextScorer, compare, readability, repetition, score
from text_quality_ai._measures import TARGETS, clarity_measure, passive_sentence_indices
from text_quality_ai._text import parse, syllables
from text_quality_ai.cli import main

# --------------------------------------------------------------------------
# blocker: passive voice was flagging any "be + participial adjective"
# --------------------------------------------------------------------------

NOT_PASSIVE = [
    "The team was thrilled with the turnout.",
    "He was tired after the trip.",
    "I am interested in the results.",
    "She is qualified for the role.",
    "The room was crowded and loud.",
    "We are pleased with the outcome.",
    "The door is locked.",
    "They were married last June.",
    "They are done when the edges rattle.",
    "He got tired of waiting.",
    "Registration was open for six weeks, and the room was packed by nine.",
    "We are excited to do it again in spring.",
    "The office is located in Berlin.",
    "She was very surprised by now.",
    "Everyone seemed pleased with the format.",
]

TRULY_PASSIVE = [
    "The report was written by the committee.",
    "The findings were reviewed by the board.",
    "The decision was taken by the chair.",
    "However, the minutes were never circulated.",
    "It should be noted that the results were discarded.",
]

CLEAN_PASSAGE = (
    "The launch went well. The whole team was thrilled with the turnout. "
    "Registration was open for six weeks, and the room was packed by nine. "
    "Sara is qualified to run the workshop, so she took the morning session. "
    "Everyone seemed pleased with the format. The coffee ran out early, which is "
    "the only complaint we heard. Next time we order more. We are excited to do "
    "it again in spring."
)

RECIPE = (
    "Heat the pan until a drop of water skips across it. Pour in a ladle of "
    "batter and tilt the pan so it spreads. They are done when the edges rattle. "
    "Stack them under a cloth while you cook the rest."
)


@pytest.mark.parametrize("sentence", NOT_PASSIVE)
def test_predicate_adjectives_are_not_called_passive(sentence):
    assert passive_sentence_indices(parse(sentence)) == [], sentence


@pytest.mark.parametrize("sentence", TRULY_PASSIVE)
def test_real_passives_are_still_caught(sentence):
    assert passive_sentence_indices(parse(sentence)) == [0], sentence


def test_passive_precision_on_the_labelled_set():
    false_positives = [s for s in NOT_PASSIVE if passive_sentence_indices(parse(s))]
    missed = [s for s in TRULY_PASSIVE if not passive_sentence_indices(parse(s))]
    assert false_positives == []
    assert missed == []


def test_clean_active_passage_reports_no_passive_voice():
    report = score(CLEAN_PASSAGE)
    clarity = report.measures["clarity"]
    assert clarity["passive_sentences"] == 0
    assert clarity["passive_share"] == 0.0
    assert clarity["passive_examples"] == []
    assert report.components["clarity"] == 100.0
    assert "passive_voice" not in {issue.kind for issue in report.issues}
    assert not any("passive" in text.lower() for text in report.suggestions)


def test_plain_everyday_prose_keeps_its_clarity_score():
    report = score(RECIPE)
    assert report.measures["clarity"]["passive_sentences"] == 0
    assert report.components["clarity"] == 100.0
    assert "passive_voice" not in {issue.kind for issue in report.issues}


def test_an_agent_phrase_outranks_the_adjective_list():
    # "locked" normally reads as an adjective, but a named doer settles it.
    assert passive_sentence_indices(parse("The door is locked.")) == []
    assert passive_sentence_indices(parse("The door was locked by the guard.")) == [0]


def test_a_degree_adverb_rules_out_the_passive_reading():
    assert passive_sentence_indices(parse("He was very tired.")) == []
    assert passive_sentence_indices(parse("The gate was quite rusted.")) == []
    # An ordinary adverb in the same slot must not suppress a real passive.
    assert passive_sentence_indices(
        parse("The report was very carefully written by the team.")) == [0]


def test_a_by_phrase_that_is_not_an_agent_does_not_create_a_passive():
    assert passive_sentence_indices(parse("The room was packed by nine.")) == []
    assert passive_sentence_indices(parse("The form was completed by 5 today.")) == [0]
    assert passive_sentence_indices(parse("The form was completed by the applicant.")) == [0]


# --------------------------------------------------------------------------
# major: nested filler phrases were counted twice
# --------------------------------------------------------------------------

def test_a_nested_filler_phrase_is_counted_once():
    clarity = clarity_measure(
        parse("We changed it due to the fact that users complained about the slow page load.")
    )
    assert clarity["filler_count"] == 1
    assert [entry["word"] for entry in clarity["filler_words"]] == ["due to the fact that"]


def test_the_shorter_phrase_is_still_counted_on_its_own():
    clarity = clarity_measure(parse("We ignored the fact that the page was slow."))
    assert clarity["filler_count"] == 1
    assert [entry["word"] for entry in clarity["filler_words"]] == ["the fact that"]


def test_every_occurrence_is_counted_exactly_once():
    text = (
        "We shipped late due to the fact that the build broke. "
        "We shipped late again due to the fact that the tests were slow. "
        "We shipped late a third time due to the fact that nobody looked."
    )
    clarity = clarity_measure(parse(text))
    counts = {entry["word"]: entry["count"] for entry in clarity["filler_words"]}
    assert counts == {"due to the fact that": 3}
    assert clarity["filler_count"] == 3


FILLER_BODY = (
    "The team met on Monday to walk through the release plan. "
    "We agreed the migration would run in two stages, with a pause between them "
    "so the support desk could catch up. "
    "The first stage moves the smaller accounts, which gives us a week to watch "
    "the error rates before the rest follow. "
    "Nobody wanted a repeat of the spring release, where the queue backed up for "
    "two days and the desk fielded calls about it every hour. "
    "We pushed the date out by a week due to the fact that the staging cluster "
    "was still short of memory. "
    "The second stage waits for a green run of the full suite on staging. "
    "If that run is clean on Wednesday, the switch happens on Thursday morning "
    "while traffic is low. "
    "Marta will watch the dashboards and Nils will keep the status page current. "
    "Between them they cover the whole window, and either one can stop the roll "
    "out from the same console if the graphs turn the wrong way. "
    "We held the announcement back due to the fact that the pricing page still "
    "named the old plan. "
    "Everything else in the plan stayed where it was, and the rollback script "
    "has not changed since the last release. "
    "We will meet again on Friday to write the notes up while the detail is fresh."
)


def test_a_true_filler_rate_below_the_threshold_raises_no_issue():
    # Two occurrences in 223 words: a true rate of 0.9%, under the 1.0% the
    # report raises an issue at. Counting each occurrence twice used to push it
    # over and print a false alarm.
    report = score(FILLER_BODY)
    clarity = report.measures["clarity"]
    assert clarity["filler_count"] == 2
    assert clarity["filler_count"] / report.stats["words"] <= 0.010
    assert "filler_words" not in {issue.kind for issue in report.issues}


# --------------------------------------------------------------------------
# major: nominalisations flagged ordinary nouns, and ignored the target
# --------------------------------------------------------------------------

TECHNICAL = (
    "The configuration file lives in your home directory. Each version of the "
    "application reads it once at startup. If the file is missing, the tool "
    "writes a default configuration and logs the location. Check the "
    "documentation for the full list of keys. The installation script does not "
    "change any existing configuration. For example, an upgrade keeps your key "
    "and your environment settings."
)

ORDINARY_NOUNS = [
    "information", "application", "configuration", "version", "location",
    "documentation", "installation", "environment", "collection", "population",
    "edition", "organisation",
]


@pytest.mark.parametrize("word", ORDINARY_NOUNS)
def test_ordinary_nouns_are_not_called_nouns_built_out_of_verbs(word):
    text = f"Open the {word} and read it carefully before you start the work today."
    assert score(text).measures["clarity"]["nominalisation_count"] == 0, word


def test_a_well_written_technical_page_raises_no_nominalisation_issue():
    for target in text_quality_ai.available_targets():
        report = score(TECHNICAL, target=target)
        assert report.measures["clarity"]["nominalisation_count"] == 0
        assert "nominalisations" not in {issue.kind for issue in report.issues}, target


NOMINALISED = (
    "The utilisation of the toolkit begins with the specification of a schema. "
    "A run will not start until that schema is in place and the fields line up. "
    "After that comes the determination of a baseline, which takes a day or two "
    "on a quiet machine. The baseline is a plain table of counts, and the tool "
    "writes it next to the schema so the two stay together. "
    "The justification of a baseline matters more than the number itself, so "
    "write down why you picked the window you did. "
    "Verification happens once the window closes and the numbers settle."
)


def test_the_nominalisation_threshold_follows_the_target():
    report = score(NOMINALISED)
    rate = report.measures["clarity"]["nominalisation_count"] / report.stats["words"]
    # The text sits between the two limits, which is what makes the test sharp.
    assert TARGETS["general"]["nominalisation_limit"] < rate
    assert rate < TARGETS["technical"]["nominalisation_limit"]

    general = {issue.kind for issue in score(NOMINALISED, target="general").issues}
    technical = {issue.kind for issue in score(NOMINALISED, target="technical").issues}
    academic = {issue.kind for issue in score(NOMINALISED, target="academic").issues}
    assert "nominalisations" in general
    assert "nominalisations" not in technical
    assert "nominalisations" not in academic


def test_clarity_itself_relaxes_for_a_technical_audience():
    general = score(NOMINALISED, target="general").components["clarity"]
    technical = score(NOMINALISED, target="technical").components["clarity"]
    assert technical > general


def test_a_real_nominalisation_is_still_reported():
    report = score(
        "The implementation of the migration was the recommendation of the review. "
        "The cancellation of the meeting followed the notification of the delay."
    )
    assert report.measures["clarity"]["nominalisation_count"] >= 4
    assert "nominalisations" in {issue.kind for issue in report.issues}


# --------------------------------------------------------------------------
# major: --long-sentence-words was dropped by --compare
# --------------------------------------------------------------------------

SIX_MEDIUM_SENTENCES = (
    "The meeting ran long and the notes were thin so nobody remembered much. "
    "We agreed to send a summary round before the end of the working week. "
    "The summary went out on Thursday and two people replied the same day. "
    "Their replies raised a point about the budget that nobody had considered. "
    "We booked a second meeting for the Monday after the bank holiday weekend. "
    "That meeting was shorter and the notes from it were far more useful."
)


def test_compare_honours_long_sentence_words():
    strict = compare(SIX_MEDIUM_SENTENCES, SIX_MEDIUM_SENTENCES, long_sentence_words=5)
    loose = compare(SIX_MEDIUM_SENTENCES, SIX_MEDIUM_SENTENCES, long_sentence_words=30)
    assert strict["long_sentence_words"] == 5
    assert loose["long_sentence_words"] == 30
    assert strict["a"]["components"]["clarity"] < loose["a"]["components"]["clarity"]
    assert strict["a"]["components"]["clarity"] == strict["b"]["components"]["clarity"]


def test_cli_reports_one_clarity_score_for_one_file(tmp_path, capsys):
    first = tmp_path / "lsw.txt"
    second = tmp_path / "lsw2.txt"
    first.write_text(SIX_MEDIUM_SENTENCES, encoding="utf-8")
    second.write_text(SIX_MEDIUM_SENTENCES, encoding="utf-8")

    assert main([str(first), "--long-sentence-words", "5", "--compare", str(second), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    report_clarity = payload["report"]["components"]["clarity"]
    assert payload["compare"]["long_sentence_words"] == 5
    assert payload["compare"]["a"]["components"]["clarity"] == report_clarity
    assert payload["compare"]["b"]["components"]["clarity"] == report_clarity
    assert payload["compare"]["score"] == 0.0
    assert payload["report"]["measures"]["clarity"]["long_sentence_words"] == 5


# --------------------------------------------------------------------------
# minor: None was silently treated as empty text
# --------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: score(None),
    lambda: readability(None),
    lambda: repetition(None),
    lambda: compare(None, "some text here"),
    lambda: TextScorer().score_one(None),
])
def test_none_is_rejected_like_every_other_wrong_type(call):
    with pytest.raises(TypeError) as excinfo:
        call()
    assert "NoneType" in str(excinfo.value)


def test_other_wrong_types_are_unchanged():
    for value in (7, 2.5, b"bytes", {"a": 1}, object()):
        with pytest.raises(TypeError):
            score(value)


# --------------------------------------------------------------------------
# minor: .measures for list input, and the syllable hiatus cases
# --------------------------------------------------------------------------

def test_aggregate_measures_say_where_the_numbers_live():
    report = score([
        "The report was written by the team and reviewed later.",
        "Another draft here that is fine.",
    ])
    for name in report.measures:
        assert set(report.measures[name]) == {"score", "explanation", "documents"}
        assert "report.documents[i].measures" in report.measures[name]["explanation"]
    assert report.documents is not None
    assert "passive_share" in report.documents[0].measures["clarity"]


@pytest.mark.parametrize("word,expected", [
    ("idea", 3), ("ideas", 3), ("science", 2), ("area", 3), ("create", 2),
    ("being", 2), ("doing", 2), ("seeing", 2), ("playing", 2),
    # words the hiatus fix must not disturb
    ("team", 1), ("nation", 2), ("read", 1), ("each", 1), ("please", 1),
    ("king", 1), ("thing", 1), ("during", 2), ("running", 2),
])
def test_syllable_counts_for_hiatus_and_its_neighbours(word, expected):
    assert syllables(word) == expected
