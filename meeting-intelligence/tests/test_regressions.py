"""Regressions for defects found in review.

Each test here names the wrong behaviour it prevents, so a future change that
brings one back fails with the reason rather than with a bare assertion.
"""
from __future__ import annotations

import pytest

import meeting_intelligence as mi

from conftest import MEETING, UNICODE, UNLABELLED, VTT


def _owners(report):
    return [action.owner for action in report.action_items]


# ------------------------------------------------------------------ blocker 1
# A capitalised word was reported as the action owner: "Hey, can you hear me?"
# became a task assigned to a person called Hey, and "Hopefully Monday will be
# quieter." to one called Hopefully Monday. The README promises the opposite.


@pytest.mark.parametrize(
    "opener",
    ["Hey", "Hi", "Sorry", "Hmm", "Guys", "Folks", "Actually", "Wait", "Team", "Great"],
)
def test_a_capitalised_interjection_is_never_an_owner(opener):
    report = mi.analyse(
        "Alice: %s, can you check the dashboard?\nBob: Will do." % opener,
        speakers=["Alice", "Bob"],
    )
    assert opener not in _owners(report)
    assert all(owner in (None, "Alice", "Bob") for owner in _owners(report))


def test_a_capitalised_non_name_before_will_is_not_an_owner():
    report = mi.analyse("Alice: Hopefully Monday will be quieter.\nBob: Agreed.")
    assert "Hopefully Monday" not in _owners(report)
    assert all(owner in (None, "Alice", "Bob") for owner in _owners(report))


@pytest.mark.parametrize(
    "transcript, roster",
    [
        (MEETING, None),
        (MEETING, ["Alice", "Bob", "Carla"]),
        (UNICODE, None),
        (VTT, None),
        ("Alice: Hey, can you hear me?\nBob: Yes.", ["Alice", "Bob"]),
        ("Alice: Sorry, Great work. Perfect, can you resend it?\nBob: Sure.", None),
        ("Alice: Wait, Actually will not work for us.\nBob: Fine.", None),
    ],
)
def test_every_action_owner_is_somebody_the_report_knows_about(transcript, roster):
    report = mi.analyse(transcript, speakers=roster)
    for action in report.action_items:
        assert action.owner is None or action.owner in report.speakers, action.text
        assert action.to_dict()["owner"] in [None] + list(report.speakers)


def test_a_name_nobody_can_confirm_is_left_none_rather_than_guessed():
    # Priya is never a speaker and was not handed in as part of the roster.
    report = mi.analyse("Alice: Priya will write the migration.\nBob: Fine.")
    assert "Priya" not in _owners(report)
    # Naming her in the roster is exactly how a caller says she is real.
    named = mi.analyse(
        "Alice: Priya will write the migration.\nBob: Fine.", speakers=["Priya"]
    )
    assert "Priya" in _owners(named)


def test_an_indefinite_person_keeps_the_action_and_leaves_the_owner_none():
    for roster in (None, ["Alice", "Bob"]):
        report = mi.analyse(UNLABELLED, speakers=roster)
        texts = [action.text for action in report.action_items]
        assert any(text.startswith("Someone will write the migration") for text in texts)
        assert all(owner is None for owner in _owners(report))


# ------------------------------------------------------------------ blocker 2
# A bare "I'll" made an action item out of anything, so a retro full of
# "I'll be honest" produced a list of phantom tasks at 0.87 confidence.


@pytest.mark.parametrize(
    "sentence",
    [
        "I'll be honest, last sprint was painful.",
        "I will say the same thing about the backlog.",
        "I'll admit we underestimated the migration.",
        "I'll leave it there for now.",
        "I'll bet the next one is worse.",
        "I'll second that point about staffing.",
        "I'll agree with Bob on the estimate.",
        "I'll guess it takes another sprint.",
    ],
)
def test_conversational_filler_after_i_will_is_not_an_action(sentence):
    actions = mi.extract_actions(mi.parse_transcript("Alice: " + sentence), ["Alice"])
    assert actions == [], sentence


def test_a_transcript_with_nothing_to_commit_to_yields_no_action_items():
    report = mi.analyse(
        "Alice: I'll be honest, last sprint was painful.\n"
        "Bob: I will say the same.\n"
        "Carla: I'll second that.\n"
        "Alice: Right, that is all I had.\n"
    )
    assert report.action_items == []
    assert "no action items" in report.summary_text.lower()


@pytest.mark.parametrize(
    "sentence",
    [
        "I'll write the migration by Friday.",
        "I will update the runbook tomorrow.",
        "I'll send the notes after the call.",
        "I will draft the rollback plan.",
    ],
)
def test_a_real_first_person_commitment_still_fires(sentence):
    actions = mi.extract_actions(mi.parse_transcript("Alice: " + sentence), ["Alice"])
    assert len(actions) == 1, sentence
    assert actions[0].owner == "Alice"
    assert actions[0].cue == "I will"


# --------------------------------------------------------------------- major
# "Alice, can you review it?" and "Can you review it, Alice?" mean the same
# thing; the trailing form used to land on whoever spoke next instead.


def test_a_trailing_vocative_names_the_same_owner_as_the_leading_form():
    lead = (
        "Alice: Morning.\n"
        "Bob: Alice, can you review the migration?\n"
        "Carla: I have the dashboards.\n"
        "Alice: Sure.\n"
    )
    trail = (
        "Alice: Morning.\n"
        "Bob: Can you review the migration, Alice?\n"
        "Carla: I have the dashboards.\n"
        "Alice: Sure.\n"
    )
    picked = []
    for transcript in (lead, trail):
        report = mi.analyse(transcript)
        owners = [
            action.owner for action in report.action_items if "review" in action.text
        ]
        assert owners == ["Alice"], transcript
        picked.append(owners)
    assert picked[0] == picked[1]


def test_a_vocative_that_is_not_a_known_person_falls_back_to_the_addressee():
    report = mi.analyse(
        "Alice: Morning.\n"
        "Bob: Can you review the migration, everyone?\n"
        "Carla: I have the dashboards.\n"
    )
    owners = [action.owner for action in report.action_items if "review" in action.text]
    assert owners and all(owner in (None, "Alice", "Bob", "Carla") for owner in owners)


# ------------------------------------------------------------------- minor 1
# A sign-off matched the affirmation list six turns later and was recorded as
# the answer, after which the summary claimed full question closure.


_OWNERSHIP = (
    "Alice: Morning all.\n"
    "Bob: The migration landed last night.\n"
    "Frank: Who owns the rollback plan?\n"
    "Alice: The dashboards need a refresh.\n"
    "Bob: Headcount is next on the agenda.\n"
    "Carla: We agreed to hire two engineers.\n"
    "Dana: Nothing from me.\n"
    "Alice: That is everything. Thanks all.\n"
)


def test_a_closing_remark_is_not_an_answer():
    report = mi.analyse(_OWNERSHIP)
    question = report.questions[0]
    assert question.text == "Who owns the rollback plan?"
    assert question.answered is False
    assert question.answer_text is None
    assert question.answered_by is None
    assert "every question asked got an answer" not in report.summary_text
    assert "1 question left open" in report.summary_text


def test_a_who_question_is_answered_by_naming_somebody():
    report = mi.analyse(
        "Alice: Morning all.\n"
        "Frank: Who owns that?\n"
        "Alice: Assigned to Dana.\n"
        "Bob: The dashboards need a refresh.\n"
    )
    question = report.questions[0]
    assert question.answered is True
    assert question.answer_text == "Assigned to Dana."
    assert question.answered_by == "Alice"


def test_a_direct_yes_still_answers_when_it_comes_straight_back():
    report = mi.analyse("Alice: Should we archive the rows?\nBob: Yes, definitely.")
    assert report.questions[0].answered is True


# ------------------------------------------------------------------- minor 2
# Labels were built out of cue verbs and greetings rather than subject matter.


def test_a_topic_label_is_not_built_from_cue_words_or_greetings():
    report = mi.analyse(
        "Alice: Morning everyone. We decided to go with Postgres for the event store.\n"
        "Bob: I'll write the migration by Friday.\n"
        "Alice: The plan is to ship the reporting views next quarter.\n"
    )
    forbidden = {"decided", "agreed", "morning", "plan", "assigned", "friday", "go"}
    for topic in report.topics:
        assert not forbidden & {word.lower() for word in topic.keywords}, topic.label
        assert not forbidden & {word.lower() for word in topic.label.split()}


def test_an_acronym_keeps_its_capitals_in_a_topic_label():
    report = mi.analyse(
        "Alice: The PR backlog is the first item.\n"
        "Bob: We should archive older PR data from the tracker.\n"
        "Carla: The PR queue is long.\n"
    )
    labels = " ".join(topic.label for topic in report.topics)
    assert "PR" in labels
    assert "Pr " not in labels + " "
