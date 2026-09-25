"""Audio input: stereo WAV files and arrays, one party per channel, and crosstalk."""

from __future__ import annotations

import numpy as np
import pytest

from call_ai_metrics import analyze
from call_ai_metrics._audio import iter_wav_blocks, read_wav_format
from conftest import (
    CALL_DURATION,
    CALL_SCRIPT,
    CLEAN_SCRIPT,
    SR,
    add_bleed,
    assert_close,
    make_call,
    true_talk,
    write_wav,
)


def test_stereo_wav_call_is_measured(tmp_path, call_audio):
    path = tmp_path / "call.wav"
    write_wav(str(path), call_audio)
    report = analyze(str(path), speakers=["agent", "customer"])
    assert report.method == "stereo-wav"
    assert report.source == "call.wav"
    assert_close(report.duration_s, CALL_DURATION, 1e-6)
    assert_close(report["agent"].talk_time_s, true_talk(CALL_SCRIPT, 0), 0.3)
    assert_close(report["customer"].talk_time_s, true_talk(CALL_SCRIPT, 1), 0.3)
    # The planted interruption is found, at the right moment...
    assert report["customer"].interruptions == 1
    event = report.interruption_events[0]
    assert (event.by, event.of) == ("customer", "agent")
    assert_close(event.time_s, 20.5, 0.05)
    assert_close(event.overlap_s, 1.5, 0.1)
    # ...and the planted 0.3 s "mm-hm" is a backchannel, not an interruption.
    assert report["customer"].backchannels == 1
    assert report["agent"].interruptions == 0
    # Replies in the script come after 0.5-0.6 s of silence.
    assert_close(report.response_latency_s, 0.55, 0.1)
    assert report.bleed is not None and not report.bleed.detected
    assert set(report.detection) == {"agent", "customer"}


def test_clean_stereo_call_raises_no_false_alarms(clean_audio):
    report = analyze(clean_audio, sample_rate=SR)
    assert report.interruption_count == 0
    assert report["A"].backchannels == 0 and report["B"].backchannels == 0
    assert report.overlap_share == 0.0
    assert not report.bleed.detected
    assert report.flags == []
    assert_close(report["A"].talk_time_s, true_talk(CLEAN_SCRIPT, 0), 0.3)
    assert_close(report["B"].talk_time_s, true_talk(CLEAN_SCRIPT, 1), 0.3)


@pytest.mark.parametrize("level_db", [-10.0, -20.0, -28.0])
def test_crosstalk_is_detected_reported_and_removed(level_db):
    leaky = add_bleed(make_call(CALL_SCRIPT, CALL_DURATION, seed=1), source=0, target=1, level_db=level_db)
    report = analyze(leaky, sample_rate=SR, speakers=["agent", "customer"])
    assert report.bleed.detected and not report.bleed.same_audio
    path = report.bleed.paths[0]
    assert (path.source, path.target, path.affected) == ("agent", "customer", True)
    assert_close(path.level_db, level_db, 2.0)
    assert abs(path.lag_ms) <= 20.0
    assert any("picks up agent" in flag for flag in report.flags)
    # Without separation the customer would be credited with all the agent's speech.
    assert_close(report["customer"].talk_time_s, true_talk(CALL_SCRIPT, 1), 0.6)
    assert report["customer"].interruptions == 1


def test_crosstalk_both_ways():
    leaky = add_bleed(make_call(CALL_SCRIPT, CALL_DURATION, seed=1), 0, 1, -18.0)
    leaky = add_bleed(leaky, 1, 0, -22.0)
    report = analyze(leaky, sample_rate=SR)
    directions = {(path.source, path.target) for path in report.bleed.paths}
    assert directions == {("A", "B"), ("B", "A")}
    assert_close(report["A"].talk_time_s, true_talk(CALL_SCRIPT, 0), 0.6)
    assert_close(report["B"].talk_time_s, true_talk(CALL_SCRIPT, 1), 0.6)


def test_crosstalk_check_raises_no_false_alarm_on_clean_calls():
    for seed in range(4):
        report = analyze(make_call(CALL_SCRIPT, CALL_DURATION, seed=seed), sample_rate=SR)
        assert not report.bleed.detected, seed
        assert not any("crosstalk" in flag for flag in report.flags)


def test_faint_crosstalk_below_threshold_is_only_a_note():
    leaky = add_bleed(make_call(CALL_SCRIPT, CALL_DURATION, seed=1, noise_db=-50.0), 0, 1, -42.0)
    report = analyze(leaky, sample_rate=SR)
    assert not any("crosstalk" in flag for flag in report.flags)
    assert_close(report["B"].talk_time_s, true_talk(CALL_SCRIPT, 1), 0.3)


def test_same_audio_on_both_channels_is_flagged():
    mono = make_call(CALL_SCRIPT, CALL_DURATION, seed=1)
    mono[:, 1] = mono[:, 0]
    report = analyze(mono, sample_rate=SR)
    assert report.bleed.same_audio
    assert report.bleed.same_audio_pairs == (("A", "B"),)
    assert len(report.flags) == 1 and "same audio" in report.flags[0]


def test_one_sided_stereo_call_does_not_raise():
    script = [(1.0, 6.0, 0), (7.0, 12.0, 0), (13.0, 19.0, 0)]
    for noise_db in (None, -70.0):
        audio = make_call(script, 20.0, noise_db=noise_db, seed=4)
        report = analyze(audio, sample_rate=SR, speakers=["agent", "customer"])
        assert report["customer"].talk_time_s == 0.0
        assert report["customer"].turns == 0
        assert report["agent"].talk_share == 1.0
        assert "customer never spoke; the call is one-sided." in report.flags
        assert any("no speech found on channel 2" in note for note in report.notes)


def test_one_sided_call_with_crosstalk_still_reports_the_silent_party():
    script = [(1.0, 6.0, 0), (7.0, 12.0, 0), (13.0, 19.0, 0)]
    audio = add_bleed(make_call(script, 20.0, seed=5), 0, 1, -15.0)
    report = analyze(audio, sample_rate=SR)
    assert report["B"].talk_time_s == 0.0
    assert report.bleed.detected
    assert "B never spoke; the call is one-sided." in report.flags


def test_silent_and_empty_audio_are_reports_of_zeros():
    for audio in (np.zeros((16000, 2)), np.zeros((0, 2)), np.zeros((2, 0))):
        report = analyze(audio if audio.shape[0] != 2 else [audio[0], audio[1]], sample_rate=SR)
        assert report.talk_time_s == 0.0
        assert report.turn_count == 0 and report.interruption_count == 0
        assert report.silence_share == 0.0 and report.overlap_share == 0.0
        assert report.flags == ["No speech was found, so there is nothing to measure."]


def test_input_shapes_agree(tmp_path, call_audio):
    reference = analyze(call_audio, sample_rate=SR).to_dict()["speakers"]
    pair = [call_audio[:, 0].copy(), call_audio[:, 1].copy()]
    assert analyze(pair, sample_rate=SR).to_dict()["speakers"] == reference
    assert analyze(tuple(pair), sample_rate=SR).to_dict()["speakers"] == reference
    assert analyze(call_audio.T.copy(), sample_rate=SR).to_dict()["speakers"] == reference
    assert analyze((call_audio, SR)).to_dict()["speakers"] == reference
    as_int16 = [np.round(ch * 32767).astype(np.int16) for ch in pair]
    int_report = analyze(as_int16, sample_rate=SR).to_dict()["speakers"]
    for name in ("A", "B"):
        assert_close(int_report[name]["talk_time_s"], reference[name]["talk_time_s"], 0.05)
    path = tmp_path / "call.wav"
    write_wav(str(path), call_audio)
    wav = analyze(str(path)).to_dict()["speakers"]
    for name in ("A", "B"):
        assert_close(wav[name]["talk_time_s"], reference[name]["talk_time_s"], 0.05)


def test_caller_arrays_are_never_modified(call_audio):
    left = call_audio[:, 0].copy()
    right = call_audio[:, 1].copy()
    left.setflags(write=False)  # the pandas 3 .to_numpy() situation
    right.setflags(write=False)
    before = (left.copy(), right.copy())
    with_nan = call_audio.copy()
    with_nan[100:110, 0] = np.nan
    nan_before = with_nan.copy()
    analyze([left, right], sample_rate=SR)
    report = analyze(with_nan, sample_rate=SR)
    assert np.array_equal(left, before[0]) and np.array_equal(right, before[1])
    assert np.array_equal(with_nan, nan_before, equal_nan=True)
    assert any("non-finite" in note for note in report.notes)


def test_audio_is_deterministic(call_audio):
    first = analyze(call_audio, sample_rate=SR).to_dict()
    assert analyze(call_audio.copy(), sample_rate=SR).to_dict() == first


def test_channels_of_different_length(call_audio):
    report = analyze([call_audio[:, 0], call_audio[: SR * 20, 1]], sample_rate=SR)
    assert any("different lengths" in note for note in report.notes)
    assert_close(report.duration_s, CALL_DURATION, 1e-6)


@pytest.mark.parametrize(
    "encoding, extensible",
    [
        ("pcm16", False),
        ("pcm8", False),
        ("pcm24", False),
        ("pcm32", True),
        ("float32", False),
        ("float32", True),
        ("ulaw", False),
        ("alaw", False),
    ],
)
def test_wav_encodings_decode(tmp_path, encoding, extensible):
    rng = np.random.default_rng(7)
    audio = np.clip(0.3 * rng.standard_normal((800, 2)), -0.95, 0.95)
    path = tmp_path / "t.wav"
    write_wav(str(path), audio, encoding=encoding, extensible=extensible, extra_chunk=True)
    fmt = read_wav_format(str(path))
    decoded = np.concatenate(list(iter_wav_blocks(str(path), fmt, 100)))
    assert decoded.shape == audio.shape
    tolerance = {"pcm8": 0.01, "ulaw": 0.03, "alaw": 0.03}.get(encoding, 1e-4)
    error = np.abs(decoded - audio)
    if encoding in ("ulaw", "alaw"):  # companded: error grows with the sample
        assert np.all(error <= tolerance * np.abs(audio) + 1e-3)
    else:
        assert np.max(error) <= tolerance


def test_eight_bit_rails_are_asymmetric(tmp_path):
    path = tmp_path / "rails.wav"
    raw = bytes([0, 255, 128, 128])  # two stereo frames: both rails, then silence
    import struct

    fmt = struct.pack("<HHIIHH", 1, 2, SR, SR * 2, 2, 8)
    body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    decoded = np.concatenate(list(iter_wav_blocks(str(path), read_wav_format(str(path)), 10)))
    assert decoded[0, 0] == -1.0
    assert decoded[0, 1] == 127.0 / 128.0
    assert np.all(decoded[1] == 0.0)


def test_mulaw_call_recording_is_measured(tmp_path, call_audio):
    path = tmp_path / "call_ulaw.wav"
    write_wav(str(path), call_audio, encoding="ulaw")
    report = analyze(str(path))
    assert_close(report["A"].talk_time_s, true_talk(CALL_SCRIPT, 0), 0.3)
    assert report["B"].interruptions == 1


def test_unicode_wav_file_name(tmp_path, call_audio):
    path = tmp_path / "звонок_山田.wav"
    write_wav(str(path), call_audio)
    report = analyze(path)
    assert report.source == "звонок_山田.wav"
    assert "звонок_山田.wav" in report.summary()


def test_mono_recordings_are_refused_with_a_way_forward(tmp_path, call_audio):
    path = tmp_path / "mono.wav"
    write_wav(str(path), call_audio[:, :1])
    with pytest.raises(ValueError, match="diarizer"):
        analyze(str(path))
    with pytest.raises(ValueError, match="diarizer"):
        analyze(call_audio[:, 0].reshape(-1, 1), sample_rate=SR)


def test_audio_input_errors(tmp_path, call_audio):
    with pytest.raises(ValueError, match="sample_rate is required"):
        analyze([call_audio[:, 0], call_audio[:, 1]])
    path = tmp_path / "call.wav"
    write_wav(str(path), call_audio)
    with pytest.raises(ValueError, match="recorded at 8000 Hz"):
        analyze(str(path), sample_rate=16000)
    with pytest.raises(ValueError, match="one name per channel"):
        analyze(str(path), speakers=["only-one"])
    with pytest.raises(ValueError, match="same name"):
        analyze(str(path), speakers=["x", "x"])
    not_wav = tmp_path / "notes.wav"
    not_wav.write_bytes(b"hello, not audio")
    with pytest.raises(ValueError, match="RIFF"):
        analyze(str(not_wav))
    with pytest.raises(ValueError, match="too low"):
        analyze(call_audio, sample_rate=100)
    with pytest.raises(ValueError, match="numeric"):
        analyze([np.array(["a"] * 10), np.array(["b"] * 10)], sample_rate=SR)


def test_speaker_names_by_channel_index(call_audio):
    report = analyze(call_audio, sample_rate=SR, speakers={1: "customer"})
    assert list(report.speakers) == ["A", "customer"]


@pytest.mark.parametrize("kind", ["noise", "sine"])
def test_speech_over_exact_digital_silence_is_found(kind):
    sr = 8000
    rng = np.random.default_rng(3)
    a = np.zeros(20 * sr)
    b = np.zeros(20 * sr)
    if kind == "noise":
        a[: 8 * sr] = 0.3 * rng.standard_normal(8 * sr)
        b[9 * sr : 18 * sr] = 0.3 * rng.standard_normal(9 * sr)
    else:
        t = np.arange(20 * sr) / sr
        tone = 0.3 * np.sin(2 * np.pi * 300 * t)
        a[: 8 * sr] = tone[: 8 * sr]
        b[9 * sr : 18 * sr] = tone[9 * sr : 18 * sr]
    report = analyze((a, b), sample_rate=sr)
    talk = [s.talk_time_s for s in report.speakers.values()] if isinstance(report.speakers, dict) else [s.talk_time_s for s in report.speakers]
    assert abs(talk[0] - 8.0) < 0.5 and abs(talk[1] - 9.0) < 0.5
    assert not any("No speech" in f for f in report.flags)
    # clean control: all-zero input still reports nothing
    empty = analyze((np.zeros(20 * sr), np.zeros(20 * sr)), sample_rate=sr)
    assert all(
        s.talk_time_s == 0
        for s in (empty.speakers.values() if isinstance(empty.speakers, dict) else empty.speakers)
    )


def test_strong_bleed_over_digital_silence_is_detected():
    sr = 8000
    rng = np.random.default_rng(4)
    a = np.zeros(20 * sr)
    b = np.zeros(20 * sr)
    a[: 8 * sr] = 0.3 * rng.standard_normal(8 * sr)
    b[9 * sr : 18 * sr] = 0.3 * rng.standard_normal(9 * sr)
    a2, b2 = a + 0.7 * b, b + 0.7 * a
    report = analyze((a2, b2), sample_rate=sr)
    assert not any("No speech" in f for f in report.flags)
    assert any("bleed" in f.lower() or "crosstalk" in f.lower() or "leak" in f.lower() for f in report.flags)
