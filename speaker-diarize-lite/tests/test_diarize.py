"""The main path: who spoke when, on synthetic voices with known turns."""

from __future__ import annotations

import json

import numpy as np
import pytest

from _voices import ONE_VOICE, SR, TWO_VOICES, conversation
from speaker_diarize_lite import Diarization, Diarizer, Segment, diarize

TOLERANCE_S = 0.25


def truth_changes(truth):
    """Times where the talker changes: the middle of the pause, or the seam."""
    out = []
    for (start_a, end_a, who_a), (start_b, _end_b, who_b) in zip(truth[:-1], truth[1:]):
        if who_a != who_b:
            out.append((end_a + start_b) / 2.0)
    return out


def reported_changes(result):
    segs = result.segments
    return [
        (a.end_s + b.start_s) / 2.0
        for a, b in zip(segs[:-1], segs[1:])
        if a.speaker != b.speaker
    ]


def frame_agreement(result, truth):
    """Share of truly-voiced 10 ms steps given the right speaker, under the best label mapping."""
    counts = {}
    total = 0
    for start, end, who in truth:
        for t in np.arange(start + 0.05, end - 0.05, 0.01):
            label = result.speaker_at(float(t))
            total += 1
            if label is not None:
                counts[(who, label)] = counts.get((who, label), 0) + 1
    used, hit = set(), 0
    for (who, label), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if who in used or label in used:
            continue
        used.update((who, label))
        hit += n
    return hit / float(total)


def assert_well_formed(result):
    assert isinstance(result, Diarization)
    previous_end = -1.0
    for seg in result.segments:
        assert isinstance(seg, Segment)
        assert seg.end_s > seg.start_s
        assert seg.start_s >= previous_end - 1e-9, "segments overlap or are out of order"
        assert 0.0 <= seg.confidence <= 1.0
        assert 0.0 <= seg.start_s and seg.end_s <= result.duration_s + 1e-6
        previous_end = seg.end_s
    assert result.num_speakers == len(result.speakers)
    assert set(result.speaking_time) == set(result.speakers)


# --------------------------------------------------------------------- finding two voices
def test_two_voices_are_found_and_turn_boundaries_located(two_voices):
    signal, truth = two_voices
    result = diarize(signal, sample_rate=SR)
    assert_well_formed(result)
    assert result.num_speakers == 2
    assert result.estimated_speakers is True
    assert result.evidence_speakers == 2
    assert result.separation >= result.separation_threshold

    expected = truth_changes(truth)
    found = reported_changes(result)
    assert len(found) == len(expected), (found, expected)
    for when, got in zip(expected, found):
        assert abs(when - got) <= TOLERANCE_S, (when, got)
    assert frame_agreement(result, truth) > 0.97


def test_handover_without_a_pause_is_located(two_voices):
    signal, truth = two_voices
    result = diarize(signal, sample_rate=SR)
    seam = truth[3][1]  # the woman's third turn runs straight into the man's
    assert truth[4][0] == pytest.approx(seam)
    nearest = min(reported_changes(result), key=lambda t: abs(t - seam))
    assert abs(nearest - seam) <= TOLERANCE_S


def test_same_voice_gets_same_label_across_turns(two_voices):
    signal, truth = two_voices
    result = diarize(signal, sample_rate=SR)
    label_of = {}
    for start, end, who in truth:
        label = result.speaker_at((start + end) / 2.0)
        assert label is not None
        label_of.setdefault(who, set()).add(label)
    assert all(len(labels) == 1 for labels in label_of.values())
    assert label_of["man"] != label_of["woman"]


@pytest.mark.parametrize("seed", [5, 8])
def test_two_voices_other_seeds(seed):
    signal, truth = conversation(TWO_VOICES, seed=seed)
    result = diarize(signal, sample_rate=SR)
    assert result.num_speakers == 2
    assert frame_agreement(result, truth) > 0.95


def test_three_voices():
    turns = [("man", 2.5, 0.5), ("woman", 2.5, 0.5), ("child", 2.5, 0.5),
             ("man", 2.0, 0.4), ("child", 2.0, 0.5), ("woman", 2.5, 0.4)]
    signal, truth = conversation(turns, seed=2)
    result = diarize(signal, sample_rate=SR)
    assert result.num_speakers == 3
    assert frame_agreement(result, truth) > 0.95


def test_speaker_labels_follow_first_appearance(two_voices):
    signal, _ = two_voices
    result = diarize(signal, sample_rate=SR)
    assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert result.segments[0].speaker == "SPEAKER_00"


def test_speaking_time_adds_up(two_voices):
    signal, truth = two_voices
    result = diarize(signal, sample_rate=SR)
    total = sum(result.speaking_time.values())
    assert total == pytest.approx(sum(seg.duration_s for seg in result.segments), abs=0.01)
    true_total = sum(end - start for start, end, _ in truth)
    assert abs(total - true_total) < 0.1 * true_total
    assert abs(result.speech_s - true_total) < 0.1 * true_total


# ------------------------------------------------------------------ one voice stays one
@pytest.mark.parametrize("seed", [1, 3, 6, 9])
def test_one_voice_returns_one_speaker_not_two(seed):
    signal, _ = conversation(ONE_VOICE, seed=seed)
    result = diarize(signal, sample_rate=SR)
    assert_well_formed(result)
    assert result.num_speakers == 1
    assert result.evidence_speakers == 1
    assert result.separation < result.separation_threshold
    assert all(seg.confidence == 1.0 for seg in result.segments)
    assert any("similar" in note for note in result.notes)


def test_one_voice_at_different_loudness_is_still_one():
    from _voices import voice

    pieces = []
    for index, level in enumerate([0.1, 0.02, 0.2, 0.05]):
        pieces += [voice(2.5, "man", seed=index, level=level), np.zeros(SR // 2)]
    result = diarize(np.concatenate(pieces), sample_rate=SR)
    assert result.num_speakers == 1


def test_similar_voices_are_merged_and_the_result_says_why():
    turns = [("man", 3.0, 0.5), ("man_twin", 3.0, 0.5), ("man", 3.0, 0.5), ("man_twin", 3.0, 0.5)]
    signal, _ = conversation(turns, seed=1)
    result = diarize(signal, sample_rate=SR)
    # The honest failure mode of hand-built features: near-identical voices look like one.
    assert result.num_speakers == 1
    assert any("embed" in note for note in result.notes)


# ------------------------------------------------------------------------------ silence
def test_digital_silence_returns_no_segments():
    result = diarize(np.zeros(SR * 3), sample_rate=SR)
    assert result.segments == []
    assert result.num_speakers == 0
    assert result.speakers == [] and result.speaking_time == {}
    assert any("no speech" in w for w in result.warnings)
    assert result.to_rttm() == ""
    assert "no speech segments found" in result.summary()


def test_faint_hiss_returns_no_segments():
    rng = np.random.default_rng(0)
    result = diarize(1e-4 * rng.standard_normal(SR * 3), sample_rate=SR)
    assert result.segments == []
    assert any("no speech" in w for w in result.warnings)


def test_speech_after_silence_is_not_missed(one_voice):
    signal, truth = one_voice
    padded = np.concatenate([np.zeros(SR * 4), signal, np.zeros(SR * 4)])
    result = diarize(padded, sample_rate=SR)
    assert result.num_speakers == 1
    assert result.segments[0].start_s == pytest.approx(truth[0][0] + 4.0, abs=0.1)


# ----------------------------------------------------------------- requested speakers
def test_num_speakers_larger_than_evidence_is_honoured_but_flagged(one_voice):
    signal, _ = one_voice
    result = diarize(signal, sample_rate=SR, num_speakers=3)
    assert_well_formed(result)
    assert result.num_speakers == 3
    assert result.estimated_speakers is False
    assert result.requested_speakers == 3
    assert result.evidence_speakers == 1
    assert any("honoured" in w and "supports only 1" in w for w in result.warnings)
    assert "supports 1 speaker" in result.summary()


def test_num_speakers_matching_evidence_raises_no_flag(two_voices):
    signal, truth = two_voices
    result = diarize(signal, sample_rate=SR, num_speakers=2)
    assert result.num_speakers == 2
    assert result.estimated_speakers is False
    assert not any("honoured" in w for w in result.warnings)
    assert frame_agreement(result, truth) > 0.97


def test_num_speakers_smaller_than_evidence_is_flagged(two_voices):
    signal, _ = two_voices
    result = diarize(signal, sample_rate=SR, num_speakers=1)
    assert result.num_speakers == 1
    assert any("suggests 2" in w for w in result.warnings)


def test_more_speakers_than_windows_is_flagged():
    from _voices import voice

    signal = np.concatenate([np.zeros(SR // 4), voice(1.2, "man", seed=1), np.zeros(SR // 4)])
    result = diarize(signal, sample_rate=SR, num_speakers=5)
    assert_well_formed(result)
    assert result.num_speakers < 5
    assert any("only" in w and "could be separated" in w for w in result.warnings)


# ------------------------------------------------------------------ short recordings
def test_shorter_than_min_segment_returns_empty_result():
    signal = 0.1 * np.sin(2 * np.pi * 150 * np.arange(int(0.3 * SR)) / SR)
    result = diarize(signal, sample_rate=SR, min_segment_s=0.5)
    assert result.segments == []
    assert result.duration_s == pytest.approx(0.3)
    assert any("shorter than min_segment_s" in w for w in result.warnings)


def test_shorter_than_one_frame_returns_empty_result():
    result = diarize(np.ones(100) * 0.1, sample_rate=SR, min_segment_s=0.0)
    assert result.segments == []
    assert any("analysis frame" in w for w in result.warnings)


def test_no_samples_returns_empty_result():
    result = diarize(np.zeros(0), sample_rate=SR)
    assert result.segments == [] and result.duration_s == 0.0


def test_bursts_shorter_than_min_segment_are_ignored():
    from _voices import voice

    signal = np.concatenate([np.zeros(SR), voice(0.3, "man", seed=2), np.zeros(SR)])
    result = diarize(signal, sample_rate=SR, min_segment_s=0.5)
    assert result.segments == []
    assert any("lasts min_segment_s" in w for w in result.warnings)
    kept = diarize(signal, sample_rate=SR, min_segment_s=0.2)
    assert len(kept.segments) == 1


def test_short_two_voice_exchange():
    signal, truth = conversation([("man", 1.5, 0.4), ("woman", 1.5, 0.0)], seed=2)
    result = diarize(signal, sample_rate=SR)
    assert result.num_speakers == 2
    assert frame_agreement(result, truth) > 0.9


def test_a_single_short_utterance_is_one_speaker():
    signal, _ = conversation([("man", 2.0, 0.0)], seed=2)
    result = diarize(signal, sample_rate=SR)
    assert result.num_speakers == 1


# ------------------------------------------------------------------------- channels
def test_stereo_is_mixed_to_mono(two_voices):
    signal, _ = two_voices
    mono = diarize(signal, sample_rate=SR)
    for stereo in (np.stack([signal, 0.5 * signal], axis=1), np.stack([signal, 0.5 * signal], axis=0)):
        result = diarize(stereo, sample_rate=SR)
        assert result.num_speakers == 2
        assert [s.speaker for s in result.segments] == [s.speaker for s in mono.segments]
        assert any("mixed 2 channels to mono" in n for n in result.notes)
    assert not any("channels" in n for n in mono.notes)


def test_out_of_phase_channels_fall_back_to_one_channel(two_voices):
    signal, _ = two_voices
    result = diarize(np.stack([signal, -signal], axis=1), sample_rate=SR)
    assert result.num_speakers == 2
    assert any("cancel" in w for w in result.warnings)
    clean = diarize(np.stack([signal, signal], axis=1), sample_rate=SR)
    assert not any("cancel" in w for w in clean.warnings)


# ---------------------------------------------------------------------- determinism
def test_deterministic(two_voices):
    signal, _ = two_voices
    first = diarize(signal, sample_rate=SR).to_dict()
    second = diarize(signal, sample_rate=SR).to_dict()
    assert first == second


def test_deterministic_under_a_fixed_seed_on_a_long_recording(two_voices):
    signal, _ = two_voices
    long_signal = np.concatenate([signal] * 9)  # enough windows to take the k-means summary path
    first = Diarizer(random_state=7).diarize(long_signal, sample_rate=SR)
    second = Diarizer(random_state=7).diarize(long_signal, sample_rate=SR)
    assert first.windows > 400
    assert first.to_dict() == second.to_dict()
    assert first.num_speakers == 2


# ----------------------------------------------------------------------- inputs
def test_input_array_is_not_modified_and_read_only_input_works(two_voices):
    signal, _ = two_voices
    frozen = signal.copy()
    frozen.flags.writeable = False
    before = frozen.copy()
    result = diarize(frozen, sample_rate=SR)
    assert result.num_speakers == 2
    assert np.array_equal(frozen, before)


def test_tuple_int16_and_list_inputs(two_voices):
    signal, _ = two_voices
    as_int16 = np.round(signal * 32767).astype(np.int16)
    reference = diarize(signal, sample_rate=SR)
    from_tuple = diarize((as_int16, SR))
    assert from_tuple.num_speakers == 2
    assert len(from_tuple.segments) == len(reference.segments)
    short = [0.0] * 100
    assert diarize(short, sample_rate=SR).segments == []


@pytest.mark.parametrize(
    "call",
    [
        lambda s: diarize(s),  # no sample rate
        lambda s: diarize((s, SR), sample_rate=8000),  # contradictory rates
        lambda s: diarize(s, sample_rate=0),
        lambda s: diarize(s, sample_rate=100),
        lambda s: diarize(s, sample_rate=16000.5),
        lambda s: diarize(s, sample_rate=SR, min_segment_s=-1),
        lambda s: diarize(s, sample_rate=SR, num_speakers=0),
        lambda s: diarize(s, sample_rate=SR, num_speakers=True),
        lambda s: diarize(s, sample_rate=SR, num_speakers=1.5),
        lambda s: diarize(s, sample_rate=SR, embed="not callable"),
        lambda s: diarize(np.full(4000, np.nan), sample_rate=SR),
        lambda s: diarize(np.zeros((2, 2, 2)), sample_rate=SR),
        lambda s: diarize(np.array(["a", "b"]), sample_rate=SR),
        lambda s: diarize(np.zeros(SR, dtype=bool), sample_rate=SR),
    ],
)
def test_bad_input_raises_a_clear_value_error(call):
    with pytest.raises(ValueError):
        call(np.zeros(SR))


def test_diarizer_rejects_bad_settings():
    for kwargs in ({"hop_s": 0}, {"hop_s": 2.0, "window_s": 1.0}, {"metric": "manhattan"},
                   {"max_speakers": 0}, {"separation_threshold": 0}, {"window_s": 0.01}):
        with pytest.raises(ValueError):
            Diarizer(**kwargs)


# ---------------------------------------------------------------------- the result
def test_to_dict_is_json_safe_and_complete(two_voices):
    signal, _ = two_voices
    result = diarize(signal, sample_rate=SR)
    data = result.to_dict()
    text = json.dumps(data, allow_nan=False)
    back = json.loads(text)
    for key in ("segments", "speakers", "speaking_time", "num_speakers", "estimated_speakers",
                "separation", "warnings", "notes", "settings", "method"):
        assert key in back
    assert back["num_speakers"] == 2
    assert back["segments"][0].keys() >= {"start_s", "end_s", "speaker", "confidence"}


def test_to_rttm_is_standard(two_voices):
    signal, _ = two_voices
    result = diarize(signal, sample_rate=SR)
    lines = result.to_rttm().splitlines()
    assert len(lines) == len(result.segments)
    for line, seg in zip(lines, result.segments):
        fields = line.split(" ")
        assert len(fields) == 10
        assert fields[0] == "SPEAKER" and fields[1] == "audio" and fields[2] == "1"
        assert float(fields[3]) == pytest.approx(seg.start_s, abs=1e-3)
        assert float(fields[4]) == pytest.approx(seg.duration_s, abs=1e-3)
        assert fields[5] == fields[6] == fields[8] == fields[9] == "<NA>"
        assert fields[7] == seg.speaker
    renamed = result.to_rttm(file_id="team meeting")
    assert renamed.splitlines()[0].split(" ")[1] == "team_meeting"


def test_summary_explains_itself_in_ascii(two_voices, one_voice):
    for signal in (two_voices[0], one_voice[0]):
        result = diarize(signal, sample_rate=SR)
        text = result.summary()
        assert all(ord(ch) < 128 for ch in text)
        assert "speaker" in text and "separation" in text and "segments:" in text
        assert str(result) == text
        assert "Diarization(" in repr(result)


def test_long_summary_is_truncated(two_voices):
    signal, _ = two_voices
    result = diarize(np.concatenate([signal] * 5), sample_rate=SR)
    assert len(result.segments) > 25
    assert "more (see to_dict() or to_rttm())" in result.summary()


def test_a_single_sample_returns_an_empty_result():
    result = diarize(np.array([0.5]), sample_rate=SR, min_segment_s=0.0)
    assert result.segments == [] and result.warnings


@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32])
def test_other_sample_types_are_scaled(two_voices, dtype):
    signal, _ = two_voices
    if dtype == np.uint8:
        data = np.clip(np.round(signal * 128 + 128), 0, 255).astype(np.uint8)
    elif dtype == np.int32:
        data = np.round(signal * 2 ** 31).astype(np.int32)
    else:
        data = signal.astype(np.float32)
    result = diarize(data, sample_rate=SR)
    assert result.num_speakers == 2
    assert not any("clipped" in w for w in result.warnings)


def _same_voice_block(dur_s=4.0, sr=16000):
    t = np.arange(int(dur_s * sr)) / sr
    src = sum(np.sin(2 * np.pi * 220.0 * h * t) / h for h in range(1, 20))
    f = np.fft.rfftfreq(src.size, 1 / sr)
    shape = sum(1 / (1 + ((f - F) / 120.0) ** 2) for F in (900.0, 2000.0, 3200.0))
    y = np.fft.irfft(np.fft.rfft(src) * shape, src.size) * 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    return 0.3 * y / np.abs(y).max()


@pytest.mark.parametrize("seed", range(8))
def test_one_voice_at_varying_loudness_is_one_speaker(seed):
    # Regression: the same voice scaled per block by [1, 0.3, 1, 0.5, 1] over a
    # noise floor was estimated as 3-4 speakers.
    rng = np.random.default_rng(seed)
    block = _same_voice_block()
    x = np.concatenate([block * s for s in (1, 0.3, 1, 0.5, 1)])
    x = x + 0.003 * rng.standard_normal(x.size)
    d = diarize(x, sample_rate=16000)
    assert d.num_speakers == 1 and d.estimated_speakers


def test_quiet_voice_block_still_separates_two_voices():
    # Clean control for the regression above: a genuine second voice, played
    # at 0.3 of the level, is still found as a second speaker.
    x, truth = conversation(TWO_VOICES, seed=1, noise_rms=0.003)
    x = x.copy()
    for start, end, who in truth:
        if who == "woman":
            x[int(start * SR):int(end * SR)] *= 0.3
    d = diarize(x, sample_rate=SR)
    assert d.num_speakers == 2


def test_steady_noise_warns_no_clear_speech():
    noise = 0.003 * np.random.default_rng(0).standard_normal(10 * SR)
    d = diarize(noise, sample_rate=SR)
    assert any("no clear speech" in w for w in d.to_dict()["warnings"])


def test_real_speech_has_no_flat_warning():
    x, _ = conversation(TWO_VOICES, seed=0)
    d = diarize(x, sample_rate=SR)
    assert not any("no clear speech" in w for w in d.to_dict()["warnings"])
