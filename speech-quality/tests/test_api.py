"""The public API: assess, assess_batch, signal_to_noise, estimate_noise_floor.

Including the edge cases that decide whether this package is safe to point at a
folder nobody has listened to: silence, one sample, a wholly clipped take, a
missing file and a file that is not audio at all.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from conftest import RATE, clipped_sine, silence, sine, speech_like, write_wav
from speech_quality import (
    AudioReport,
    BatchReport,
    __version__,
    assess,
    assess_batch,
    estimate_noise_floor,
    signal_to_noise,
)


def test_the_readme_quickstart_runs_and_passes():
    """The exact block in the README, which is what a new user copies."""
    t = np.arange(3 * 16000) / 16000.0
    hiss = np.random.default_rng(0).standard_normal(t.size + 4)
    voice = np.convolve(hiss, np.ones(5) / 5, "valid")
    voice *= 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t))

    report = assess(voice, sample_rate=16000)

    assert report.usable
    assert report.grade == "A"
    assert report.score > 85.0
    assert "usable" in report.summary()


def test_version_is_the_first_release():
    """The package announces a version, and it is the one pyproject declares."""
    assert __version__ == "0.1.0"


def test_a_clean_recording_is_usable_and_has_no_issues(clean_speech):
    """Nothing wrong means an empty issue list, not a list of near-misses."""
    report = assess(clean_speech, sample_rate=RATE)

    assert isinstance(report, AudioReport)
    assert report.usable
    assert report.issues == []
    assert report.coverage == 1.0


def test_digital_silence_says_so_instead_of_dividing_by_zero():
    """All zeros is a report, not a ZeroDivisionError and not a made-up score."""
    report = assess(silence(seconds=1.0), sample_rate=RATE)

    assert report.digital_silence
    assert not report.usable
    assert report.score == 0.0
    assert report.grade == "F"
    assert "digital silence" in report.issues[0]
    assert np.isfinite(report.metric("level").value)
    json.dumps(report.to_dict())  # nothing in there is an inf or a NaN


def test_a_single_sample_is_reported_not_crashed():
    """One sample has a level and nothing else, and the report refuses a verdict."""
    report = assess(np.array([0.5]), sample_rate=RATE)

    assert report.duration == pytest.approx(1.0 / RATE)
    assert not report.usable
    assert report.coverage < 0.5
    assert "not enough recording here to judge" in report.issues[0]
    assert report.metric("level").measured
    assert not report.metric("bandwidth").measured


def test_a_wholly_clipped_recording_fails_loudly():
    """Every sample at full scale scores near zero and says why in one line."""
    report = assess(np.ones(RATE), sample_rate=RATE)

    assert not report.usable
    assert report.score < 40.0
    assert report.metric("clipping").value == pytest.approx(1.0)
    assert any("clipping" in issue for issue in report.issues)


def test_a_badly_clipped_take_is_not_usable():
    """A driven sine is the everyday version of the same fault."""
    report = assess(clipped_sine(seconds=1.0), sample_rate=RATE)

    assert not report.usable
    assert report.metric("clipping").value > 0.5


def test_stereo_input_is_mixed_down_and_the_note_survives_to_the_report():
    """The mixdown is a decision taken for the caller, so the report carries it."""
    stereo = np.stack([speech_like(seconds=1.0, seed=1), speech_like(seconds=1.0, seed=2)], axis=1)

    report = assess(stereo, sample_rate=RATE)

    assert report.channels == 2
    assert any("mixed down to mono" in note for note in report.notes)
    assert "mixed down to mono" in report.summary()


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_every_sample_rate_from_8k_to_48k_is_assessed(rate):
    """The frequency measures follow the rate rather than assuming one."""
    report = assess(speech_like(seconds=1.0, rate=rate), sample_rate=rate)

    assert report.sample_rate == rate
    assert report.duration == pytest.approx(1.0, abs=0.01)
    assert report.metric("bandwidth").details["nyquist_hz"] == pytest.approx(rate / 2.0)


@pytest.mark.parametrize("width,bits", [(2, 16), (3, 24)])
def test_wav_files_are_assessed_straight_from_disk(wav_dir, width, bits):
    """A path is all the package needs; 16 and 24 bit both come back the same."""
    path = write_wav(wav_dir / "take{}.wav".format(bits), speech_like(seconds=1.0), width=width)

    report = assess(path)

    assert report.usable
    assert report.source.endswith(".wav")
    assert any("{}-bit".format(bits) in note for note in report.notes)


def test_wav_and_array_agree(wav_dir):
    """The same audio through a file and through an array reaches the same verdict."""
    signal = speech_like(seconds=1.0)
    path = write_wav(wav_dir / "same.wav", signal, width=4)

    from_file = assess(path)
    from_array = assess(signal, sample_rate=RATE)

    assert from_file.score == pytest.approx(from_array.score, abs=0.5)


def test_a_missing_path_raises_file_not_found(wav_dir):
    """Nothing there is a FileNotFoundError the caller can catch by type."""
    with pytest.raises(FileNotFoundError):
        assess(wav_dir / "not-recorded-yet.wav")


def test_a_non_wav_file_raises_a_clear_value_error(tmp_path):
    """An m4a gets told what is supported and what to do about it."""
    path = tmp_path / "voice-memo.m4a"
    path.write_bytes(b"\x00\x00\x00\x20ftypM4A ")

    with pytest.raises(ValueError) as caught:
        assess(path)

    assert "WAV" in str(caught.value)
    assert "ffmpeg" in str(caught.value)


def test_the_caller_array_is_never_modified(clean_speech):
    """Assessing is a read; the caller's samples are untouched afterwards."""
    before = clean_speech.copy()

    assess(clean_speech, sample_rate=RATE)

    assert np.array_equal(before, clean_speech)


def test_a_read_only_array_can_be_assessed(clean_speech):
    """Nothing writes into the input, so a locked array works as well as any other."""
    clean_speech.setflags(write=False)

    report = assess(clean_speech, sample_rate=RATE)

    assert report.usable


def test_the_answer_is_deterministic(clean_speech):
    """Two runs over the same samples give byte-identical reports."""
    first = assess(clean_speech, sample_rate=RATE).to_dict()
    second = assess(clean_speech, sample_rate=RATE).to_dict()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_sixty_seconds_of_48k_audio_is_assessed_in_well_under_a_second():
    """The framing is linear and the spectra are capped, so length is not a problem."""
    long_take = speech_like(seconds=60.0, rate=48000)

    start = time.perf_counter()
    report = assess(long_take, sample_rate=48000)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert report.duration == pytest.approx(60.0, abs=0.01)


def test_signal_to_noise_returns_one_number(clean_speech):
    """The convenience function agrees with the measure inside the report."""
    value = signal_to_noise(clean_speech, sample_rate=RATE)

    assert value == pytest.approx(assess(clean_speech, sample_rate=RATE).metric("noise").value)
    assert value > 15.0


def test_estimate_noise_floor_returns_one_number(clean_speech):
    """The floor is in dBFS and sits below the signal it was taken from."""
    floor = estimate_noise_floor(clean_speech, sample_rate=RATE)

    assert -90.0 < floor < -20.0


def test_neither_helper_blows_up_on_digital_silence():
    """Zero divided by zero is the classic crash here; there is a floor instead."""
    quiet = silence(seconds=0.5)

    assert signal_to_noise(quiet, sample_rate=RATE) == pytest.approx(0.0)
    assert estimate_noise_floor(quiet, sample_rate=RATE) == pytest.approx(-120.0)


def test_helpers_take_a_wav_path_too(wav_dir):
    """Anything assess accepts, these accept."""
    path = write_wav(wav_dir / "helper.wav", speech_like(seconds=1.0))

    assert signal_to_noise(path) > 10.0
    assert estimate_noise_floor(path) < 0.0


def test_assess_batch_keeps_going_past_an_unreadable_file(wav_dir):
    """One bad file must not lose the other ninety-nine."""
    good = write_wav(wav_dir / "good.wav", speech_like(seconds=1.0))
    missing = str(wav_dir / "gone.wav")

    batch = assess_batch([good, missing, ("quiet", silence(seconds=0.5))], sample_rate=RATE)

    assert isinstance(batch, BatchReport)
    assert len(batch) == 2
    assert len(batch.failures) == 1
    assert batch.failures[0][0] == "gone.wav"
    assert "no such file" in batch.failures[0][1]


def test_assess_batch_labels_arrays_and_ranks_the_worst():
    """A (label, recording) pair is how an array gets a name in the summary."""
    batch = assess_batch(
        [("good", speech_like(seconds=1.0)), ("dead", silence(seconds=1.0))],
        sample_rate=RATE,
    )

    worst = batch.worst(1)

    assert [report.source for report in batch.results] == ["good", "dead"]
    assert worst[0].source == "dead"
    assert batch.usable and batch.unusable
    assert 0.0 < batch.mean_score < 100.0
    assert "dead" in batch.summary()


def test_assess_batch_refuses_a_bare_path(wav_dir):
    """One string would otherwise be assessed one character at a time."""
    with pytest.raises(TypeError) as caught:
        assess_batch(str(wav_dir / "one.wav"))

    assert "list of recordings" in str(caught.value)


def test_an_empty_batch_is_an_empty_report():
    """Nothing in, nothing out, and a summary that says so."""
    batch = assess_batch([])

    assert len(batch) == 0
    assert batch.mean_score == 0.0
    assert batch.worst(3) == []
    assert "no recordings" in batch.summary()


def test_a_sine_is_not_mistaken_for_a_voice():
    """The headline case for the speech measure: a clean tone is not usable speech."""
    report = assess(sine(seconds=2.0, freq=440.0, amplitude=0.1), sample_rate=RATE)

    assert not report.metric("speech").ok
    assert any("does not behave like speech" in issue for issue in report.issues)


def test_missing_sample_rate_is_assumed_and_carried_into_the_notes():
    """Frequency numbers mean nothing without a rate, so the report says one was guessed."""
    report = assess(speech_like(seconds=1.0))

    assert report.sample_rate == 16000
    assert any("no sample rate was given" in note for note in report.notes)
