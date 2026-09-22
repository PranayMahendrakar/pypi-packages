"""Reading a transcript out of every shape people actually have."""
from __future__ import annotations

import pytest

import meeting_intelligence as mi
from meeting_intelligence._parse import parse_timestamp

from conftest import MEETING, SRT, UNICODE, UNLABELLED, VTT


def test_labelled_text_splits_into_turns(meeting):
    turns = mi.parse_transcript(meeting)
    assert turns.source_format == "labelled-text"
    assert turns.speakers == ["Alice", "Bob", "Carla"]
    assert turns[0].speaker == "Alice"
    assert turns[0].text.startswith("Morning everyone.")
    assert turns[0].index == 0
    assert all(turn.index == position for position, turn in enumerate(turns))


def test_unlabelled_prose_has_no_speakers():
    turns = mi.parse_transcript(UNLABELLED)
    assert turns.source_format == "plain-text"
    assert turns.speakers == []
    assert all(turn.speaker is None for turn in turns)
    assert " ".join(turn.text for turn in turns).startswith("We decided")


def test_empty_transcript_returns_empty_list():
    for source in ("", "   \n\t\n  ", [], None):
        turns = mi.parse_transcript(source)
        assert list(turns) == []
        assert turns.source_format == "empty"
        assert mi.detect_format(source) == "empty"


def test_one_line_transcript():
    turns = mi.parse_transcript("Alice: We decided to go with Postgres.")
    assert len(turns) == 1
    assert turns[0].speaker == "Alice"
    assert turns[0].words == 6


def test_turn_dicts_are_accepted():
    turns = mi.parse_transcript(
        [
            {"speaker": "Alice", "text": "We decided to ship on Friday.", "start": 0},
            {"speaker": "Bob", "text": "I'll do the release notes.", "start": "00:00:12.5"},
        ]
    )
    assert turns.source_format == "turns"
    assert mi.detect_format(turns) == "turns"
    assert [turn.speaker for turn in turns] == ["Alice", "Bob"]
    assert turns[1].start == pytest.approx(12.5)
    assert turns.timed is True


def test_turn_objects_and_strings_round_trip():
    original = mi.parse_transcript("Alice: We decided to go with Postgres.")
    again = mi.parse_transcript(list(original))
    assert again[0].speaker == "Alice"
    from_strings = mi.parse_transcript(["Alice: Yes we agreed to that plan."])
    assert from_strings[0].speaker == "Alice"


def test_a_bad_turn_type_is_named_in_the_error():
    with pytest.raises(TypeError, match="speaker"):
        mi.parse_transcript([object()])
    with pytest.raises(TypeError, match="transcript must be text"):
        mi.parse_transcript(3.5)
    with pytest.raises(TypeError):
        mi.detect_format(3.5)


@pytest.mark.parametrize(
    "raw, seconds",
    [
        ("00:00:04,500", 4.5),
        ("00:00:04.500", 4.5),
        ("01:02:03.125", 3723.125),
        ("1:02:03", 3723.0),
        ("02:30", 150.0),
        ("02:30.75", 150.75),
        ("nonsense", None),
        ("", None),
    ],
)
def test_timestamps_parse_with_hours_and_milliseconds(raw, seconds):
    if seconds is None:
        assert parse_timestamp(raw) is None
    else:
        assert parse_timestamp(raw) == pytest.approx(seconds)


def test_vtt_is_detected_and_timed():
    assert mi.detect_format(VTT) == "vtt"
    turns = mi.parse_transcript(VTT)
    assert turns.source_format == "vtt"
    assert turns.speakers == ["Alice", "Bob"]
    assert turns[0].start == pytest.approx(0.0)
    assert turns[0].end == pytest.approx(4.5)
    # An hour-long stamp with milliseconds survives intact.
    assert turns[2].start == pytest.approx(3723.125)
    assert "<v" not in turns[0].text


def test_srt_is_detected_and_timed():
    assert mi.detect_format(SRT) == "srt"
    turns = mi.parse_transcript(SRT)
    assert turns.source_format == "srt"
    assert turns.speakers == ["Alice", "Bob"]
    assert turns[0].end == pytest.approx(4.5)
    assert turns[-1].start == pytest.approx(3723.125)
    assert turns[-1].text == "Can you review it, Bob?"


def test_consecutive_cues_from_one_speaker_merge():
    track = """WEBVTT

00:00:00.000 --> 00:00:02.000
<v Alice>We decided to go

00:00:02.000 --> 00:00:04.000
<v Alice>with Postgres for the store.

00:00:04.000 --> 00:00:06.000
<v Bob>Understood.
"""
    turns = mi.parse_transcript(track)
    assert len(turns) == 2
    assert turns[0].text == "We decided to go with Postgres for the store."
    assert turns[0].end == pytest.approx(4.0)
    assert turns[0].duration == pytest.approx(4.0)


def test_section_headers_are_not_read_as_speakers():
    turns = mi.parse_transcript(
        "Agenda: the event store\nAlice: We decided to go with Postgres.\n"
    )
    assert turns.speakers == ["Alice"]


@pytest.mark.parametrize("suffix, body", [(".txt", MEETING), (".vtt", VTT), (".srt", SRT)])
def test_files_are_read_and_classified_by_content(tmp_path, suffix, body):
    path = tmp_path / ("transcript" + suffix)
    path.write_text(body, encoding="utf-8")
    expected = mi.detect_format(body)
    assert mi.detect_format(path) == expected
    assert mi.detect_format(str(path)) == expected
    turns = mi.parse_transcript(path)
    assert turns.source_format == expected
    assert len(turns) >= 3


def test_a_txt_file_holding_vtt_is_still_vtt(tmp_path):
    path = tmp_path / "captions.txt"
    path.write_text(VTT, encoding="utf-8")
    assert mi.parse_transcript(path).source_format == "vtt"


def test_a_missing_file_raises_naming_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        mi.parse_transcript(tmp_path / "nope.txt")


def test_a_utf8_bom_and_unicode_names_survive(tmp_path):
    path = tmp_path / "unicode.txt"
    path.write_text(UNICODE, encoding="utf-8-sig")
    turns = mi.parse_transcript(path)
    assert "Zoë Müller" in turns.speakers
    assert "李雷" in turns.speakers
    assert "café" in " ".join(turn.text for turn in turns)


def test_turn_helpers():
    turn = mi.Turn(index=0, speaker="Alice", text="One. Two.", start=1.0, end=3.5)
    assert turn.words == 2
    assert turn.duration == pytest.approx(2.5)
    assert turn.sentences() == ["One.", "Two."]
    assert turn.to_dict()["speaker"] == "Alice"
    assert mi.Turn(index=0, speaker=None, text="x").duration is None


def test_formats_constant_covers_what_detect_returns():
    seen = {mi.detect_format(source) for source in ("", MEETING, UNLABELLED, VTT, SRT)}
    assert seen <= set(mi.FORMATS)
