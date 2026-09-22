"""Cue-phrase extraction of decisions, actions, due dates and questions."""
from __future__ import annotations

import pytest

import meeting_intelligence as mi

from conftest import MEETING, UNLABELLED


def _turns(text):
    return mi.parse_transcript(text)


def _texts(records):
    return " || ".join(record.text for record in records)


# ---------------------------------------------------------------- decisions


@pytest.mark.parametrize(
    "sentence, cue",
    [
        ("We decided to go with Postgres.", "we decided"),
        ("The decision is to ship on Friday.", "decision is"),
        ("Let's go with the smaller instance.", "let us go with"),
        ("We agreed to archive the old rows.", "agreed to"),
        ("The plan is to migrate in March.", "the plan is"),
        ("We settled on a weekly release.", "settled on"),
        ("The team signed off on the design.", "sign off"),
        ("Consensus is that we wait a sprint.", "consensus"),
    ],
)
def test_each_documented_decision_cue_fires(sentence, cue):
    decisions = mi.extract_decisions(_turns("Alice: " + sentence))
    assert len(decisions) == 1
    assert decisions[0].cue == cue
    assert decisions[0].confidence > 0.5
    assert decisions[0].speaker == "Alice"
    assert decisions[0].to_dict()["cue"] == cue


@pytest.mark.parametrize(
    "sentence",
    [
        "We have not decided on the database yet.",
        "We are still deciding between the two.",
        "No decision was reached on the archive.",
        "Have we decided on the database?",
        "Once we have decided we will tell you.",
    ],
)
def test_an_open_item_is_never_reported_as_settled(sentence):
    assert mi.extract_decisions(_turns("Alice: " + sentence)) == []


def test_a_hedged_decision_scores_lower():
    firm = mi.extract_decisions(_turns("Alice: We decided to go with Postgres."))[0]
    hedged = mi.extract_decisions(_turns("Alice: Maybe we decided to go with Postgres."))[0]
    assert hedged.confidence < firm.confidence


def test_decisions_are_deduplicated():
    doubled = "Alice: We decided to go with Postgres.\nBob: We decided to go with Postgres.\n"
    assert len(mi.extract_decisions(_turns(doubled))) == 1


def test_decisions_are_found_in_the_full_meeting(meeting):
    decisions = mi.extract_decisions(_turns(meeting))
    assert len(decisions) >= 3
    assert "Postgres" in _texts(decisions)
    assert {decision.cue for decision in decisions} >= {"we decided", "agreed to"}


# ------------------------------------------------------------- action items


@pytest.mark.parametrize(
    "sentence, cue",
    [
        ("I'll take the migration.", "I'll take"),
        ("I will write the runbook.", "I will"),
        ("Bob will write the migration.", "X will"),
        ("Can you review the migration?", "can you"),
        ("This is assigned to Carla.", "assigned to X"),
        ("Action item: drop the legacy table.", "action item"),
        ("We need to archive the old rows.", "we need to"),
        ("Please update the runbook today.", "please"),
        ("Make sure the backup runs first.", "make sure"),
    ],
)
def test_each_documented_action_cue_fires(sentence, cue):
    actions = mi.extract_actions(_turns("Alice: " + sentence), ["Alice", "Bob", "Carla"])
    assert actions, sentence
    assert actions[0].cue == cue


def test_action_is_constructed_positionally():
    action = mi.Action("Write the migration", "Bob", "Friday", 0.9)
    assert (action.text, action.owner, action.due, action.confidence) == (
        "Write the migration",
        "Bob",
        "Friday",
        0.9,
    )
    assert action.describe() == "Bob: Write the migration (due Friday)"
    assert action.to_dict()["owner"] == "Bob"


def test_first_person_owner_is_the_speaker():
    actions = mi.extract_actions(_turns("Bob: I'll write the migration by Friday."))
    assert actions[0].owner == "Bob"
    assert actions[0].owner_is_speaker is True
    assert actions[0].due == "Friday"


def test_can_you_lands_on_the_other_person_in_a_two_person_call():
    turns = _turns("Alice: Can you review the migration?\nBob: Sure thing, I will.")
    actions = mi.extract_actions(turns, ["Alice", "Bob"])
    assert actions[0].owner == "Bob"


def test_a_first_name_resolves_against_the_roster():
    turns = _turns("Alice: Sam will handle the archive job.")
    actions = mi.extract_actions(turns, ["Alice", "Sam Okafor"])
    assert actions[0].owner == "Sam Okafor"


def test_an_unresolvable_owner_stays_none_rather_than_guessed():
    turns = _turns("Alice: We need to archive the old rows.")
    actions = mi.extract_actions(turns, ["Alice"])
    assert actions[0].owner is None
    assert actions[0].to_dict()["owner"] is None
    assert actions[0].describe().startswith("unassigned:")


def test_actions_are_found_in_the_full_meeting(meeting):
    actions = mi.extract_actions(_turns(meeting), ["Alice", "Bob", "Carla"])
    assert len(actions) >= 4
    owners = {action.owner for action in actions}
    assert "Bob" in owners
    assert any(action.due for action in actions)
    assert all(0.0 < action.confidence <= 1.0 for action in actions)


# ----------------------------------------------------------------- due dates


@pytest.mark.parametrize(
    "sentence, due",
    [
        ("I'll ship it by Friday.", "Friday"),
        ("I'll ship it by end of week.", "end of week"),
        ("I'll ship it EOD.", "EOD"),
        ("I'll ship it by 2026-04-01.", "2026-04-01"),
        ("I'll ship it within two weeks.", "within two weeks"),
        ("I'll ship it next quarter.", "next quarter"),
        ("I'll ship it ASAP.", "ASAP"),
        ("I'll ship it by March 14.", "March 14"),
    ],
)
def test_documented_due_patterns(sentence, due):
    assert mi.find_due(sentence) == due


def test_no_date_means_none():
    assert mi.find_due("I'll ship it when it is ready.") is None
    assert mi.find_due("") is None


def test_due_patterns_are_published():
    assert isinstance(mi.DUE_PATTERNS, tuple) and mi.DUE_PATTERNS
    assert isinstance(mi.DECISION_CUES, tuple) and mi.DECISION_CUES
    assert isinstance(mi.ACTION_RULES, tuple) and mi.ACTION_RULES


# ----------------------------------------------------------------- questions


def test_questions_are_matched_to_their_answer():
    turns = _turns(
        "Alice: What do we do about the old rows?\n"
        "Bob: We archive the old rows after a year.\n"
    )
    questions = mi.extract_questions(turns)
    assert len(questions) == 1
    assert questions[0].answered is True
    assert questions[0].answered_by == "Bob"
    assert questions[0].answer_turn_index == 1
    assert mi.unanswered(questions) == []


def test_an_unanswered_question_stays_open():
    turns = _turns(
        "Alice: What do we do about the reporting views?\n"
        "Bob: We decided to go with Postgres.\n"
    )
    questions = mi.extract_questions(turns)
    assert mi.unanswered(questions) == questions
    assert questions[0].to_dict()["answered"] is False


def test_filler_is_not_counted_as_a_question():
    turns = _turns("Alice: Right? Okay? What is the migration plan for the event store?")
    questions = mi.extract_questions(turns)
    assert len(questions) == 1
    assert questions[0].text.startswith("What is the migration plan")


def test_a_direct_yes_counts_as_an_answer():
    turns = _turns("Alice: Should we archive the rows?\nBob: Yes, definitely.")
    assert mi.extract_questions(turns)[0].answered is True


def test_the_asker_answering_themselves_does_not_count():
    turns = _turns("Alice: Should we archive the rows?\nAlice: We archive the rows.")
    assert mi.extract_questions(turns)[0].answered is False


# ------------------------------------------- transcripts with no speaker labels


def test_unlabelled_text_still_yields_decisions_and_actions():
    turns = _turns(UNLABELLED)
    decisions = mi.extract_decisions(turns)
    actions = mi.extract_actions(turns, [])
    questions = mi.extract_questions(turns)
    assert decisions and "Postgres" in _texts(decisions)
    assert actions
    assert all(action.owner is None for action in actions)
    assert all(decision.speaker is None for decision in decisions)
    assert questions and questions[0].speaker is None


def test_extractors_on_an_empty_transcript():
    empty = mi.parse_transcript("")
    assert mi.extract_decisions(empty) == []
    assert mi.extract_actions(empty, []) == []
    assert mi.extract_questions(empty) == []


def test_mixed_case_and_unicode_do_not_break_the_cues():
    turns = _turns("Zoë Müller: WE DECIDED to go with Postgres. I'll email 李雷 by Friday.")
    assert mi.extract_decisions(turns)
    actions = mi.extract_actions(turns, ["Zoë Müller"])
    assert actions[0].owner == "Zoë Müller"
    assert actions[0].due == "Friday"


def test_two_people_committing_to_the_same_thing_are_two_actions():
    """Regression: actions were deduped on their words alone, so when a second speaker
    made the same commitment it was silently dropped - and the dropped person is
    precisely the one who never gets chased for it."""
    import meeting_intelligence as mi

    turns = [
        {"speaker": "Zoe", "text": "I will send the revised budget by Friday."},
        {"speaker": "Anna", "text": "I will send the revised budget by Friday."},
        {"speaker": "Zoe", "text": "I will send the revised budget by Friday."},
    ]
    report = mi.analyse(turns)
    budget = [a for a in report.action_items if "budget" in a.text]
    assert len(budget) == 2, "both speakers' commitments must survive"
    assert {a.owner for a in budget} == {"Zoe", "Anna"}
    # the same speaker saying it twice is still one action
    assert sum(1 for a in report.action_items if a.owner == "Zoe") == 1


def test_a_decision_restated_by_someone_else_is_counted_once():
    """A decision repeated by a second speaker is a confirmation, not a new decision."""
    import meeting_intelligence as mi

    turns = [
        {"speaker": "Ravi", "text": "We decided to postpone the launch to October."},
        {"speaker": "Anna", "text": "We decided to postpone the launch to October."},
    ]
    assert len(mi.analyse(turns).decisions) == 1


def test_owners_written_in_any_script_are_captured():
    import meeting_intelligence as mi

    turns = [
        {"speaker": "李雷", "text": "I will review the bearing report tonight."},
        {"speaker": "Zoë", "text": "I will circulate the notes tomorrow."},
    ]
    owners = {a.owner for a in mi.analyse(turns).action_items}
    assert "李雷" in owners and "Zoë" in owners


def test_a_caption_file_without_speakers_says_so():
    import meeting_intelligence as mi

    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nWe decided to postpone the launch.\n\n"
           "00:00:05.000 --> 00:00:09.000\nI will send the budget by Friday.\n")
    report = mi.analyse(vtt)
    assert report.decisions and report.action_items
    assert any("speaker" in w.lower() for w in report.warnings), report.warnings
