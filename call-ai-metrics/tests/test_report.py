"""The result object: flags, summary, explain, to_dict and to_frame."""

from __future__ import annotations

import json

import pytest

from call_ai_metrics import CallAnalyzer, SpeakerStats, analyze


def _balanced_call():
    """Ten quick, polite exchanges: nothing about it deserves a flag."""
    segments = []
    t = 0.0
    for index in range(10):
        speaker = "A" if index % 2 == 0 else "B"
        segments.append((t, t + 4.0, speaker))
        t += 4.5
    return segments


def test_balanced_call_has_no_flags():
    report = analyze(_balanced_call())
    assert report.flags == []
    assert "flags: none" in report.summary()


def test_dominance_flag_threshold_is_configurable():
    segments = [(0, 30, "A"), (30.5, 40, "B")]  # A has 76% of the talk
    assert any("A did most of the talking: 76%" in f for f in analyze(segments).flags)
    relaxed = CallAnalyzer(dominance_share=0.8).analyze(segments)
    assert not any("most of the talking" in f for f in relaxed.flags)


def test_long_monologue_and_long_silence_are_flagged():
    segments = [(0, 70, "A"), (85, 90, "B"), (90.5, 95, "A")]
    flags = analyze(segments).flags
    assert any("A spoke for 1:10 at a stretch" in f for f in flags)
    assert any("The longest silence lasted 15.0s, starting at 1:10." in f for f in flags)


def test_heavy_overlap_and_silence_share_are_flagged():
    overlapping = [(0, 10, "A"), (6, 14, "B"), (14.2, 20, "A"), (17, 22, "B")]
    assert any("talked over each other" in f for f in analyze(overlapping).flags)
    quiet = [(0, 2, "A"), (5, 7, "B"), (10, 12, "A"), (15, 17, "B")]
    assert any("of the conversation was silence" in f for f in analyze(quiet).flags)


def test_slow_replies_are_flagged_per_party():
    segments = []
    t = 0.0
    for _ in range(4):
        segments.append((t, t + 3.0, "A"))
        segments.append((t + 5.5, t + 8.0, "B"))  # B waits 2.5 s every time
        t += 8.5
    flags = analyze(segments).flags
    assert any("B typically took 2.5s to reply." == f for f in flags)
    assert not any(f.startswith("A typically") for f in flags)


def test_repeated_interruptions_are_flagged():
    segments = []
    t = 0.0
    for _ in range(4):
        segments.append((t, t + 6.0, "A"))
        segments.append((t + 4.0, t + 9.0, "B"))  # B cuts in with 2 s to go
        t += 9.5
    report = analyze(segments)
    assert report["B"].interruptions == 4
    assert any("B interrupted 4 times" in f for f in report.flags)


def test_to_dict_is_json_safe_and_complete(quickstart_segments):
    report = analyze(quickstart_segments)
    data = json.loads(json.dumps(report.to_dict()))
    assert set(data) >= {
        "source", "method", "duration_s", "overall", "speakers", "flags", "notes",
        "bleed", "settings", "turns", "interruptions", "segments",
    }
    assert set(data["speakers"]["agent"]) == set(SpeakerStats.__dataclass_fields__)
    assert data["overall"]["turn_count"] == 5
    assert data["interruptions"] == [{"time_s": 23.0, "by": "agent", "of": "customer", "overlap_s": 1.0}]
    assert data["settings"]["grace_s"] == 0.5
    assert data["bleed"] is None and data["detection"] is None


def test_to_dict_for_audio_includes_detection(call_audio):
    data = analyze(call_audio, sample_rate=8000).to_dict()
    json.dumps(data)
    assert data["bleed"]["detected"] is False
    assert data["detection"]["A"]["threshold_dbfs"] is not None


def test_to_frame(quickstart_segments):
    pytest.importorskip("pandas")
    report = analyze(quickstart_segments)
    frame = report.to_frame()
    assert list(frame.index) == ["agent", "customer"]
    assert frame.index.name == "speaker"
    assert frame.loc["agent", "interruptions"] == 1
    assert frame.loc["customer", "backchannels"] == 1
    assert frame.attrs["overall"]["turn_count"] == 5
    assert frame.attrs["flags"] == report.flags
    # The frame is a copy: editing it does not touch the report.
    frame.loc["agent", "turns"] = 99
    assert report["agent"].turns == 3


def test_to_frame_of_an_empty_call():
    pytest.importorskip("pandas")
    frame = analyze([]).to_frame()
    assert list(frame.index) == ["A", "B"]
    assert float(frame["talk_time_s"].sum()) == 0.0


def test_report_access_and_repr(quickstart_segments):
    report = analyze(quickstart_segments)
    assert report[0] is report["agent"]
    assert [stats.speaker for stats in report] == ["agent", "customer"]
    assert len(report) == 2
    text = repr(report)
    assert text.startswith("CallReport(2 parties")
    assert "agent 88%" in text and "1 interruption" in text
    assert report.speech_span_s == pytest.approx(41.0)
    assert report.turns[0].duration_s == pytest.approx(6.5)
    assert report.segments[0].duration_s == pytest.approx(6.5)
    assert "CallAnalyzer(grace_s=0.5" in repr(CallAnalyzer())


def test_explain_uses_the_actual_settings(quickstart_segments):
    text = CallAnalyzer(grace_s=0.8, max_turn_pause_s=5.0).analyze(quickstart_segments).explain()
    assert "0.80s (the grace window)" in text
    assert "pauses longer than 5.0s" in text
    assert "only as good as that segmentation" in text
    assert text.isascii()


def test_explain_for_audio_calls_it_a_heuristic(call_audio):
    text = analyze(call_audio, sample_rate=8000).explain()
    assert "energy heuristic" in text


def test_long_durations_format_as_clock_time():
    report = analyze([(0, 3700, "A"), (3700.5, 3702, "B")])
    assert "1:01:40" in report.summary()
