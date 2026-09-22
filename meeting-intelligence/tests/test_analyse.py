"""The one-liner, the report object, and the edge cases that must not raise."""
from __future__ import annotations

import json
import logging
import time

import pytest

import meeting_intelligence as mi

from conftest import MEETING, SRT, UNICODE, UNLABELLED, VTT


# ------------------------------------------------------------- the quickstart


def test_quickstart_from_the_readme():
    report = mi.analyse(
        """
Alice: We decided to go with Postgres for the event store.
Bob: I'll write the migration by Friday. Can you review it, Alice?
Alice: Sure. What do we do about the old rows?
"""
    )
    assert report.decisions[0].text.startswith("We decided to go with Postgres")
    assert report.action_items[0].owner == "Bob"
    assert report.action_items[0].due == "Friday"
    assert "Postgres" in report.summary()
    assert report.source_format == "labelled-text"


def test_the_whole_report_is_filled_in(meeting):
    report = mi.analyse(meeting)
    assert report.summary_text
    assert report.decisions and report.action_items and report.questions
    assert report.topics and report.participation
    assert report.n_turns == 10
    assert report.n_words > 60
    assert report.speakers == ["Alice", "Bob", "Carla"]
    assert report.is_empty is False
    assert report.llm_used is False
    assert isinstance(report.warnings, list)
    assert len(report.open_questions) + len(report.answered_questions) == len(report.questions)
    assert len(report.owned_actions) + len(report.unowned_actions) == len(report.action_items)
    assert report.actions_for("bob") == [a for a in report.action_items if a.owner == "Bob"]
    assert repr(report).startswith("MeetingReport(")


def test_analyze_is_the_same_function():
    assert mi.analyze is mi.analyse


# ---------------------------------------------------------------- edge cases


@pytest.mark.parametrize("source", ["", "   \n  ", [], None])
def test_an_empty_transcript_returns_an_empty_report_rather_than_raising(source):
    report = mi.analyse(source)
    assert report.is_empty is True
    assert report.n_turns == 0
    assert report.decisions == []
    assert report.action_items == []
    assert report.questions == []
    assert report.topics == []
    assert report.participation == []
    assert report.duration is None
    assert report.warnings, "an empty report should say why it is empty"
    assert "Empty transcript" in report.summary()
    assert "Empty transcript" in report.to_markdown()
    assert json.dumps(report.to_dict())


def test_a_one_line_transcript_works():
    report = mi.analyse("Alice: We decided to go with Postgres for the event store.")
    assert report.n_turns == 1
    assert len(report.decisions) == 1
    assert len(report.topics) == 1
    assert report.participation[0].speaker == "Alice"
    assert report.participation[0].share == pytest.approx(1.0)
    assert report.summary()
    assert report.to_markdown()


def test_a_transcript_with_no_speaker_labels_still_yields_decisions_and_actions():
    report = mi.analyse(UNLABELLED)
    assert report.source_format == "plain-text"
    assert report.speakers == []
    assert report.decisions, "cue phrases do not need a speaker label"
    assert report.action_items
    assert all(action.owner is None for action in report.action_items)
    assert any("no speaker labels" in warning for warning in report.warnings)
    assert report.participation[0].speaker == "(unlabelled)"
    assert report.to_dict()["n_action_items"] == len(report.action_items)


def test_unicode_names_and_text_survive_every_rendering():
    report = mi.analyse(UNICODE)
    assert "Zoë Müller" in report.speakers
    assert "李雷" in report.speakers
    assert "Zoë Müller" in report.summary()
    assert "李雷" in report.to_markdown()
    restored = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert "Zoë Müller" in restored["speakers"]
    assert any(action.owner == "李雷" for action in report.action_items)


def test_a_roster_supplied_for_an_unlabelled_transcript_is_reported():
    report = mi.analyse(UNLABELLED, speakers=["Alice", "Bob"])
    assert report.speakers == ["Alice", "Bob"]
    assert any("no speaker labels" in warning for warning in report.warnings)


def test_a_roster_string_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match=r"\['Alice', 'Bob'\]"):
        mi.analyse(MEETING, speakers="Alice, Bob")


def test_duplicate_and_blank_roster_names_are_cleaned():
    report = mi.analyse("Alice: We decided to ship.", speakers=["Alice", "Alice", " ", None])
    assert report.speakers == ["Alice"]


# ------------------------------------------------------------------- timings


@pytest.mark.parametrize("source, fmt", [(VTT, "vtt"), (SRT, "srt")])
def test_timed_sources_give_a_duration(source, fmt):
    report = mi.analyse(source)
    assert report.source_format == fmt
    assert report.duration is not None and report.duration > 3600
    assert report.duration_text and "h" in report.duration_text
    assert report.topics[0].start == pytest.approx(0.0)
    assert report.turns[2].start == pytest.approx(3723.125)   # hours + milliseconds


def test_an_untimed_source_has_no_duration(meeting):
    report = mi.analyse(meeting)
    assert report.duration is None
    assert report.duration_text is None
    assert report.interruptions_basis in ("none available", "cut-off markers")


def test_interruptions_are_counted_from_timing_overlap():
    turns = [
        {"speaker": "Alice", "text": "We decided to go with the event store.", "start": 0, "end": 10},
        {"speaker": "Bob", "text": "Actually I will take the migration.", "start": 6, "end": 12},
    ]
    report = mi.analyse(turns)
    assert report.interruptions_basis == "timing overlap"
    rows = {row.speaker: row for row in report.participation}
    assert rows["Bob"].interruptions == 1
    assert rows["Alice"].interrupted_by_others == 1


def test_interruptions_are_counted_from_cut_off_markers():
    report = mi.analyse("Alice: So the plan is to mig--\nBob: I'll take the migration.")
    assert report.interruptions_basis == "cut-off markers"
    rows = {row.speaker: row for row in report.participation}
    assert rows["Bob"].interruptions == 1


def test_participation_shares_add_up(meeting):
    report = mi.analyse(meeting)
    assert sum(row.share for row in report.participation) == pytest.approx(1.0)
    assert sum(row.words for row in report.participation) == report.n_words
    assert sum(row.turns for row in report.participation) == report.n_turns
    assert report.participation[0].words >= report.participation[-1].words
    assert report.participation[0].describe()


# ----------------------------------------------------------------- the topics


def test_topics_cover_every_turn_once(meeting):
    report = mi.analyse(meeting)
    covered = []
    for topic in report.topics:
        covered.extend(range(topic.start_turn, topic.end_turn + 1))
    assert covered == list(range(report.n_turns))
    assert all(topic.label for topic in report.topics)
    assert all(topic.to_dict()["n_turns"] > 0 for topic in report.topics)


def test_a_topic_label_is_not_just_a_speaker_name():
    report = mi.analyse(MEETING)
    for topic in report.topics:
        assert "Alice" not in topic.keywords
        assert "alice" not in [word.lower() for word in topic.keywords]


def test_a_short_transcript_says_it_could_not_be_split():
    report = mi.analyse("Alice: We decided to ship.\nBob: I'll write the notes.")
    assert len(report.topics) == 1
    assert any("too short" in warning for warning in report.warnings)


def test_segment_topics_is_usable_on_its_own(meeting):
    topics, warnings = mi.segment_topics(mi.parse_transcript(meeting))
    assert topics
    assert isinstance(warnings, list)
    assert mi.segment_topics([]) == ([], [])


def test_measure_participation_is_usable_on_its_own(meeting):
    rows, basis = mi.measure_participation(mi.parse_transcript(meeting))
    assert rows and isinstance(basis, str)
    assert mi.measure_participation([]) == ([], "none")


def test_format_duration_is_usable_on_its_own():
    assert mi.format_duration(None) is None
    assert mi.format_duration(1) == "1 second"
    assert mi.format_duration(45) == "45 seconds"
    assert mi.format_duration(600) == "10 minutes"
    assert mi.format_duration(3600) == "1 hour"
    assert mi.format_duration(3900) == "1 h 5 min"


# --------------------------------------------------------------- the optional llm


def test_an_llm_that_raises_is_caught_and_the_report_still_comes_back(meeting, raising_llm, caplog):
    with caplog.at_level(logging.WARNING):
        report = mi.analyse(meeting, llm=raising_llm)
    plain = mi.analyse(meeting)
    assert report.llm_used is False
    assert report.summary_text == plain.summary_text
    assert report.decisions and report.action_items
    assert any("llm call failed" in warning for warning in report.warnings)
    assert any("RuntimeError" in warning for warning in report.warnings)


def test_a_working_llm_improves_the_summary_and_the_action_wording(meeting):
    calls = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        if "NOTES:" in prompt:
            return "The team chose Postgres and split the follow-up work."
        return "\n".join(
            "%d. Do task %d" % (number, number)
            for number in range(1, prompt.count("\n%d." % 1) + 30)
            if ("%d." % number) in prompt
        )

    report = mi.analyse(meeting, llm=llm)
    assert len(calls) == 2
    assert report.llm_used is True
    assert report.summary_text == "The team chose Postgres and split the follow-up work."
    assert report.action_items[0].text.startswith("Do task 1")
    assert report.action_items[0].source_text, "the original sentence is kept"


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_an_llm_returning_nothing_keeps_the_extractive_summary(meeting, reply):
    report = mi.analyse(meeting, llm=lambda prompt: reply)
    plain = mi.analyse(meeting)
    assert report.summary_text == plain.summary_text
    assert report.llm_used is False
    assert report.warnings


def test_a_non_callable_llm_is_a_warning_not_a_crash(meeting):
    report = mi.analyse(meeting, llm="not a function")
    assert report.decisions
    assert any("callable" in warning for warning in report.warnings)


def test_the_package_is_fully_useful_with_no_llm(meeting):
    report = mi.analyse(meeting)
    assert report.summary_text and report.decisions and report.action_items
    assert report.llm_used is False


# ------------------------------------------------------- renderings and speed


def test_to_dict_is_json_safe(meeting):
    payload = mi.analyse(meeting).to_dict()
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["n_turns"] == payload["n_turns"]
    assert set(payload) >= {
        "summary_text",
        "decisions",
        "action_items",
        "questions",
        "topics",
        "participation",
        "duration",
        "warnings",
    }


def test_summary_is_plain_ascii_punctuation(meeting):
    text = mi.analyse(meeting).summary()
    for forbidden in ("→", "•", "─", "●", "–", "—"):
        assert forbidden not in text


def test_to_markdown_has_the_expected_sections(meeting):
    markdown = mi.analyse(meeting).to_markdown()
    for heading in ("# Meeting report", "## Decisions", "## Action items", "## Topics"):
        assert heading in markdown
    assert markdown.endswith("\n")


def test_a_large_transcript_analyses_quickly():
    block = (
        "Alice: We decided to go with Postgres for the event store.\n"
        "Bob: I'll write the migration by Friday.\n"
        "Carla: What do we do about the reporting views and the old rows?\n"
        "Alice: We agreed to archive anything older than a year.\n"
    )
    transcript = block * (200_000 // len(block) + 1)
    assert len(transcript.encode("utf-8")) >= 200_000
    started = time.perf_counter()
    report = mi.analyse(transcript)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, "200 KB took %.2fs" % elapsed
    assert report.n_turns > 3000
    assert report.decisions and report.action_items
    assert len(report.topics) <= 8


def test_results_are_deterministic(meeting):
    first = mi.analyse(meeting).to_dict()
    second = mi.analyse(meeting).to_dict()
    assert first == second


def test_no_audio_and_no_network_are_reachable_from_this_package():
    import meeting_intelligence

    source = ""
    from pathlib import Path

    folder = Path(meeting_intelligence.__file__).parent
    for module in sorted(folder.glob("*.py")):
        source += module.read_text(encoding="utf-8")
    for banned in (
        "import wave",
        "import audioop",
        "import sounddevice",
        "import urllib",
        "import requests",
        "import socket",
        "import torch",
        "import transformers",
        "http://",
        "https://",
    ):
        assert banned not in source, "%s must not appear in the package" % banned
