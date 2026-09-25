"""Diarized-segment input: validation, cleaning, and every metric on hand-checked calls."""

from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest

from call_ai_metrics import CallAnalyzer, CallReport, analyze, count_words
from conftest import assert_close


# ---------------------------------------------------------------- the quickstart


def test_quickstart_numbers_by_hand(quickstart_segments):
    report = analyze(quickstart_segments)
    agent, customer = report["agent"], report["customer"]
    assert_close(agent.talk_time_s, 6.5 + 11.3 + 18.0, 1e-6)
    assert_close(customer.talk_time_s, 2.1 + 0.4 + 2.5, 1e-6)
    assert_close(agent.talk_share, 35.8 / 40.8, 1e-6)
    # 23.0 is inside the customer's 21.5-24.0: a full second of overlap.
    assert agent.interruptions == 1
    assert customer.interrupted == 1
    assert report.interruption_events[0].time_s == 23.0
    assert_close(report.interruption_events[0].overlap_s, 1.0, 1e-6)
    # The agent's 8.7 start overlaps the customer by only 0.3 s: not an interruption.
    # The customer's "mm-hm" at 12.0-12.4 is a backchannel, not an interruption either.
    assert customer.interruptions == 0
    assert customer.backchannels == 1
    assert agent.turns == 3 and customer.turns == 2
    assert_close(agent.longest_monologue_s, 18.0, 1e-6)
    assert_close(agent.mean_turn_s, (6.5 + 11.3 + 18.0) / 3, 1e-6)
    assert_close(report.silence_share, (0.4 + 1.5) / 41.0, 1e-6)
    assert_close(report.longest_silence_s, 1.5, 1e-6)
    assert_close(report.longest_silence_at_s, 20.0, 1e-6)
    assert_close(report.overlap_share, (0.3 + 0.4 + 1.0) / 41.0, 1e-6)
    assert report.turn_count == 5
    # replies: 0.4 (customer), 0 (agent, overlapped), 1.5 (customer), 0 (agent)
    assert_close(report.response_latency_s, 0.2, 1e-6)
    assert_close(customer.response_latency_s, 0.95, 1e-6)
    assert agent.response_latency_s == 0.0
    assert any("88% of the talk time" in flag for flag in report.flags)
    assert isinstance(report, CallReport)


def test_summary_is_plain_ascii_and_mentions_segmentation(quickstart_segments):
    text = analyze(quickstart_segments).summary()
    assert text.isascii()
    assert "agent" in text and "customer" in text
    assert "only as good as the speech segmentation" in text
    assert "88%" in text


# ------------------------------------------------ interruptions and backchannels


def test_backchannel_shorter_than_grace_is_not_an_interruption():
    # A talks 0-20 s; B says "mm-hm" for 0.3 s at 5 s and 0.4 s at 11 s.
    segments = [(0.0, 20.0, "A"), (5.0, 5.3, "B"), (11.0, 11.4, "B"), (20.5, 25.0, "B")]
    report = analyze(segments)
    assert report.interruption_count == 0
    assert report["B"].interruptions == 0
    assert report["B"].backchannels == 2
    # The backchannels do not take the floor: A holds one turn, B takes one after.
    assert report["A"].turns == 1 and report["B"].turns == 1
    assert_close(report["A"].longest_monologue_s, 20.0, 1e-9)
    assert not any("interrupt" in flag for flag in report.flags)


def test_planted_interruption_is_found_and_clean_call_has_none():
    planted = [(0.0, 10.0, "A"), (8.0, 15.0, "B"), (15.4, 20.0, "A")]
    report = analyze(planted)
    assert report["B"].interruptions == 1
    assert report["A"].interrupted == 1
    event = report.interruption_events[0]
    assert (event.by, event.of, event.time_s) == ("B", "A", 8.0)
    assert_close(event.overlap_s, 2.0, 1e-9)

    clean = [(0.0, 10.0, "A"), (10.4, 18.0, "B"), (18.4, 24.0, "A")]
    quiet = analyze(clean)
    assert quiet.interruption_count == 0
    assert quiet.overlap_share == 0.0
    assert quiet.flags == []


def test_grace_window_is_the_dividing_line():
    segments = [(0.0, 10.0, "A"), (9.4, 14.0, "B")]  # 0.6 s of overlap
    assert analyze(segments).interruption_count == 1  # default grace 0.5 s
    assert CallAnalyzer(grace_s=0.8).analyze(segments).interruption_count == 0
    exact = [(0.0, 10.0, "A"), (9.5, 14.0, "B")]  # exactly the grace
    assert analyze(exact).interruption_count == 0


def test_long_talk_over_inside_the_other_turn_counts_but_takes_no_floor():
    segments = [(0.0, 20.0, "A"), (5.0, 7.0, "B")]  # B talks over A for 2 s, A carries on
    report = analyze(segments)
    assert report["B"].interruptions == 1
    assert report["B"].turns == 0
    assert report["A"].turns == 1


def test_pause_inside_the_other_speech_still_counts_as_speaking():
    # A pauses 0.2 s at 5.0; B starts in that pause and talks over A's restart.
    segments = [(0.0, 5.0, "A"), (5.2, 9.0, "A"), (5.1, 7.0, "B")]
    report = analyze(segments)
    assert report["B"].interruptions == 1


def test_simultaneous_start_is_nobody_interrupting():
    report = analyze([(0.0, 3.0, "A"), (0.0, 1.0, "B"), (4.0, 6.0, "B")])
    assert report.interruption_count == 0


# ---------------------------------------------------------------- input cleaning


def test_out_of_order_segments_are_sorted(quickstart_segments):
    shuffled = [quickstart_segments[i] for i in (5, 2, 0, 4, 1, 3)]
    ordered = analyze(quickstart_segments)
    report = analyze(shuffled)
    assert any("out of time order" in note for note in report.notes)
    assert not any("out of time order" in note for note in ordered.notes)
    assert report.to_dict()["speakers"] == ordered.to_dict()["speakers"]
    assert report.to_dict()["overall"] == ordered.to_dict()["overall"]
    starts = [segment.start_s for segment in report.segments]
    assert starts == sorted(starts)


def test_overlapping_same_speaker_segments_are_merged():
    segments = [(0.0, 5.0, "A"), (4.0, 8.0, "A"), (7.5, 9.0, "A"), (9.0, 10.0, "A"), (11.0, 13.0, "B")]
    report = analyze(segments)
    assert_close(report["A"].talk_time_s, 10.0, 1e-9)  # not 5 + 4 + 1.5 + 1
    assert [(s.start_s, s.end_s) for s in report.segments if s.speaker == "A"] == [(0.0, 10.0)]
    assert any("2 overlapping segments from the same speaker were merged" in n for n in report.notes)
    # Merging must not invent overlap between parties.
    assert report.overlap_share == 0.0


def test_touching_segments_merge_silently():
    report = analyze([(0.0, 2.0, "A"), (2.0, 4.0, "A"), (5.0, 6.0, "B")])
    assert not any("overlapping" in note for note in report.notes)
    assert report["A"].turns == 1


def test_zero_length_segments_are_ignored_with_a_note():
    report = analyze([(0.0, 2.0, "A"), (3.0, 3.0, "B"), (4.0, 5.0, "B")])
    assert_close(report["B"].talk_time_s, 1.0, 1e-9)
    assert any("zero-length" in note for note in report.notes)


def test_caller_segments_are_never_modified(quickstart_segments):
    as_lists = [list(item) for item in quickstart_segments][::-1]
    as_dicts = [{"start": s, "end": e, "speaker": p, "text": "hello there"} for s, e, p in quickstart_segments]
    before_lists = copy.deepcopy(as_lists)
    before_dicts = copy.deepcopy(as_dicts)
    analyze(as_lists)
    analyze(as_dicts, speakers={"agent": "Agent", "customer": "Customer"})
    assert as_lists == before_lists
    assert as_dicts == before_dicts


def test_deterministic(quickstart_segments):
    first = analyze(quickstart_segments).to_dict()
    for _ in range(3):
        assert analyze(list(quickstart_segments)).to_dict() == first


# --------------------------------------------------------------- edge-case calls


def test_empty_call_is_a_report_of_zeros():
    report = analyze([])
    assert list(report.speakers) == ["A", "B"]
    for stats in report:
        assert stats.talk_time_s == 0.0 and stats.talk_share == 0.0
        assert stats.turns == 0 and stats.interruptions == 0
        assert stats.mean_turn_s == 0.0 and stats.longest_monologue_s == 0.0
        assert stats.words_per_minute is None
    assert report.duration_s == 0.0
    assert report.silence_share == 0.0 and report.overlap_share == 0.0
    assert report.longest_silence_s == 0.0 and report.turn_count == 0
    assert report.response_latency_s is None
    assert report.interruption_count == 0
    assert report.flags == ["No speech was found, so there is nothing to measure."]
    json.dumps(report.to_dict())
    assert "No speech" in report.summary()


def test_empty_call_with_named_parties():
    report = analyze([], speakers=["agent", "customer"])
    assert list(report.speakers) == ["agent", "customer"]
    assert report.talk_time_s == 0.0


def test_one_sided_call_reports_silent_party_with_zero_talk():
    segments = [(0.0, 4.0, "agent"), (5.0, 9.0, "agent")]
    report = analyze(segments)
    assert list(report.speakers) == ["agent", "other"]
    assert report["other"].talk_time_s == 0.0
    assert report["other"].turns == 0
    assert report["agent"].talk_share == 1.0
    assert report.response_latency_s is None
    assert "other never spoke; the call is one-sided." in report.flags
    # The one-sided call is not also reported as "did most of the talking".
    assert not any("most of the talking" in flag for flag in report.flags)

    named = analyze(segments, speakers=["agent", "customer"])
    assert named["customer"].talk_time_s == 0.0
    assert "customer never spoke; the call is one-sided." in named.flags


def test_single_segment():
    report = analyze([(1.0, 2.5, "A")])
    assert_close(report["A"].talk_time_s, 1.5, 1e-9)
    assert report.turn_count == 1
    assert report.silence_share == 0.0
    assert_close(report.speech_start_s, 1.0, 1e-9)
    assert_close(report.duration_s, 2.5, 1e-9)


def test_three_parties_work():
    segments = [(0, 5, "A"), (5.5, 9, "B"), (9.2, 12, "C"), (11.0, 14, "A")]
    report = analyze(segments)
    assert list(report.speakers) == ["A", "B", "C"]
    assert report["A"].interruptions == 1 and report["C"].interrupted == 1
    assert any("3 parties" in note for note in report.notes)


# ---------------------------------------------------------------- turns and pace


def test_long_pause_splits_a_turn():
    segments = [(0.0, 5.0, "A"), (20.0, 25.0, "A"), (26.0, 28.0, "B")]
    report = analyze(segments)
    assert report["A"].turns == 2
    assert_close(report["A"].longest_monologue_s, 5.0, 1e-9)
    assert report.response_latency_s == pytest.approx(1.0)
    lenient = CallAnalyzer(max_turn_pause_s=30.0).analyze(segments)
    assert lenient["A"].turns == 1
    assert_close(lenient["A"].longest_monologue_s, 25.0, 1e-9)


def test_words_per_minute_from_transcript():
    words = " ".join(["word"] * 50)
    segments = [
        (0.0, 20.0, "A", words),
        (20.5, 30.5, "B", "yes I see what you mean"),
        (31.0, 41.0, "A", words),
    ]
    report = analyze(segments)
    assert report["A"].words == 100
    assert_close(report["A"].words_per_minute, 100 / (30.0 / 60.0), 1e-6)
    assert report["B"].words == 6
    assert_close(report["B"].words_per_minute, 36.0, 1e-6)
    assert any("A spoke fast: 200 words per minute" in flag for flag in report.flags)
    # B has only 10 s of speech: too little to judge pace, so no slow flag.
    assert not any("B spoke slowly" in flag for flag in report.flags)


def test_no_transcript_means_no_words_per_minute(quickstart_segments):
    for stats in analyze(quickstart_segments):
        assert stats.words is None and stats.words_per_minute is None


def test_partial_transcript_uses_only_transcribed_time():
    segments = [(0.0, 30.0, "A", " ".join(["w"] * 60)), (30.5, 60.0, "A"), (61.0, 62.0, "B")]
    report = analyze(segments)
    assert_close(report["A"].words_per_minute, 120.0, 1e-6)


def test_unicode_transcripts_and_names():
    segments = [
        (0.0, 3.0, "Дмитрий", "Добрый день, чем могу помочь?"),
        (3.5, 6.0, "山田", "予約を変更したいです"),
        (6.5, 9.0, "Zoë", "Ça marche — très bien !"),
    ]
    report = analyze(segments)
    assert report["Дмитрий"].words == 5
    assert report["山田"].words == 10
    assert report["Zoë"].words == 4  # the dash and "!" are not words
    text = report.summary()
    assert "Дмитрий" in text and "山田" in text
    json.dumps(report.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello world", 2),
        ("  well -- I don't know ...  ", 4),
        ("", 0),
        ("こんにちは世界", 7),
        ("mixed 日本 words", 4),
        ("e-mail co-op 42", 3),
    ],
)
def test_count_words(text, expected):
    assert count_words(text) == expected


# ---------------------------------------------------------------- input shapes


def test_dicts_objects_and_frames_agree(quickstart_segments):
    expected = analyze(quickstart_segments).to_dict()["speakers"]
    dicts = [{"Start": s, "END": e, "Speaker": p} for s, e, p in quickstart_segments]
    objects = [SimpleNamespace(start_s=s, end_s=e, speaker=p) for s, e, p in quickstart_segments]
    wrapped = SimpleNamespace(segments=objects)
    assert analyze(dicts).to_dict()["speakers"] == expected
    assert analyze(objects).to_dict()["speakers"] == expected
    assert analyze(wrapped).to_dict()["speakers"] == expected
    assert analyze({"segments": dicts}).to_dict()["speakers"] == expected
    assert analyze(tuple(quickstart_segments)).to_dict()["speakers"] == expected


def test_pyannote_style_annotation(quickstart_segments):
    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            for index, (start, end, label) in enumerate(quickstart_segments):
                yield SimpleNamespace(start=start, end=end), index, label

    report = analyze(FakeAnnotation())
    assert report.to_dict()["speakers"] == analyze(quickstart_segments).to_dict()["speakers"]


def test_dataframe_input_is_read_not_written(quickstart_segments):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(quickstart_segments, columns=["start", "end", "speaker"])
    frame["text"] = ["hi there", None, "ok then so", float("nan"), "right", "thanks bye"]
    before = frame.copy(deep=True)
    report = analyze(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert report["agent"].words == 7
    assert report["customer"].words == 1
    assert_close(report["agent"].talk_share, 35.8 / 40.8, 1e-6)


def test_dataframe_with_duplicate_columns_names_them():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([[0.0, 1.0, "A", "B"]], columns=["start", "end", "speaker", "speaker"])
    with pytest.raises(ValueError, match="duplicate column names: speaker"):
        analyze(frame)


def test_dataframe_mixed_dtypes_and_integer_labels():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"start_s": ["0", "5.5"], "end_s": [5.0, 9], "label": [0, 1], "extra": [None, object()]}
    )
    report = analyze(frame, speakers={0: "agent", 1: "customer"})
    assert list(report.speakers) == ["agent", "customer"]
    assert_close(report["customer"].talk_time_s, 3.5, 1e-9)


def test_csv_and_json_files(tmp_path, quickstart_segments):
    expected = analyze(quickstart_segments).to_dict()["speakers"]
    with_header = tmp_path / "segs.csv"
    with_header.write_text(
        "start,end,speaker\n" + "".join("{},{},{}\n".format(*row) for row in quickstart_segments),
        encoding="utf-8",
    )
    headerless = tmp_path / "segs.tsv"
    headerless.write_text(
        "".join("{}\t{}\t{}\n".format(*row) for row in quickstart_segments), encoding="utf-8"
    )
    as_json = tmp_path / "segs.json"
    as_json.write_text(
        json.dumps({"segments": [{"start": s, "end": e, "speaker": p} for s, e, p in quickstart_segments]}),
        encoding="utf-8",
    )
    for path in (with_header, headerless, as_json):
        report = analyze(str(path))
        assert report.to_dict()["speakers"] == expected
        assert report.source == path.name
    assert analyze(with_header).to_dict()["speakers"] == expected  # pathlib.Path too


def test_csv_with_unicode_text_and_bom(tmp_path):
    path = tmp_path / "звонок.csv"
    path.write_text(
        "﻿start;end;speaker;text\n0;4;Анна;Здравствуйте, это банк\n4.5;7;山田;はい\n",
        encoding="utf-8",
    )
    report = analyze(str(path))
    assert list(report.speakers) == ["Анна", "山田"]
    assert report["Анна"].words == 3


def test_empty_csv_is_an_empty_call(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert analyze(str(path)).talk_time_s == 0.0


# ------------------------------------------------------------------ names


def test_rename_dict_can_merge_split_clusters():
    segments = [(0, 2, "SPK_0"), (2.5, 4, "SPK_1"), (4.5, 6, "SPK_2")]
    report = analyze(segments, speakers={"SPK_0": "agent", "SPK_1": "customer", "SPK_2": "agent"})
    assert list(report.speakers) == ["agent", "customer"]
    assert_close(report["agent"].talk_time_s, 3.5, 1e-9)
    assert any("treated as one speaker" in note for note in report.notes)


def test_speaker_list_must_cover_the_labels():
    with pytest.raises(ValueError, match="does not"):
        analyze([(0, 1, "SPK_0"), (1, 2, "SPK_1")], speakers=["agent", "customer"])
    with pytest.raises(ValueError, match="single string"):
        analyze([(0, 1, "A")], speakers="agent")


# ------------------------------------------------------------------ bad input


@pytest.mark.parametrize(
    "segments, message",
    [
        ([(2.0, 1.0, "A")], "ends"),
        ([(-1.0, 1.0, "A")], "negative"),
        ([(float("nan"), 1.0, "A")], "no usable start"),
        ([(0.0, math.inf, "A")], "end time of inf"),
        ([(0.0, 1.0, None)], "no speaker"),
        ([(0.0, 1.0)], "2 fields"),
        (["0 1 A"], "string"),
        ([{"begin": 0, "speaker": "A"}], "needs start, end and speaker"),
        ([(0.0, "soon", "A")], "not a number"),
        ([42], "expected"),
    ],
)
def test_bad_segments_raise_clear_errors(segments, message):
    with pytest.raises(ValueError, match=message):
        analyze(segments)


def test_bad_settings_raise():
    with pytest.raises(ValueError, match="grace_s"):
        CallAnalyzer(grace_s=-1)
    with pytest.raises(ValueError, match="dominance_share"):
        CallAnalyzer(dominance_share=0.2)


def test_unsupported_sources_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze(str(tmp_path / "missing.wav"))
    parquet = tmp_path / "segs.parquet"
    parquet.write_bytes(b"PAR1")
    with pytest.raises(ValueError, match="pandas"):
        analyze(str(parquet))
    with pytest.raises(ValueError, match="cannot read"):
        analyze(12.5)
    with pytest.raises(ValueError, match="segments"):
        analyze({"rows": []})


def test_sample_rate_with_segments_is_ignored_with_a_note(quickstart_segments):
    report = analyze(quickstart_segments, sample_rate=8000)
    assert any("sample_rate is only used for audio" in note for note in report.notes)


def test_empty_and_single_row_dataframes():
    pd = pytest.importorskip("pandas")
    empty = pd.DataFrame({"start": [], "end": [], "speaker": []})
    report = analyze(empty)
    assert report.talk_time_s == 0.0 and list(report.speakers) == ["A", "B"]
    single = pd.DataFrame({"start": [0.5], "end": [3.0], "speaker": ["agent"]})
    one = analyze(single)
    assert_close(one["agent"].talk_time_s, 2.5, 1e-9)
    assert one["other"].talk_time_s == 0.0


def test_all_nan_columns():
    pd = pytest.importorskip("pandas")
    no_text = pd.DataFrame(
        {"start": [0.0, 4.0], "end": [3.0, 6.0], "speaker": ["A", "B"], "text": [float("nan")] * 2}
    )
    report = analyze(no_text)
    assert report["A"].words is None and report["A"].words_per_minute is None
    no_times = pd.DataFrame({"start": [float("nan")] * 2, "end": [1.0, 2.0], "speaker": ["A", "B"]})
    with pytest.raises(ValueError, match="segment 0 has no usable start time"):
        analyze(no_times)


def test_space_separated_cjk_counts_words_not_characters():
    from call_ai_metrics._segments import count_words
    assert count_words("你好 谢谢 再见") == 3
    assert count_words("你好谢谢再见") == 6
    assert count_words("hello world") == 2
