"""Each of the seven measures, against a signal built to have exactly that fault."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    RATE,
    band_limited_noise,
    clipped_sine,
    silence,
    sine,
    speech_like,
)
from speech_quality import Thresholds, assess
from speech_quality.thresholds import MEASURES


def report_for(signal, rate: int = RATE, **kwargs):
    """Assess a signal at ``rate`` and hand back the report."""
    return assess(signal, sample_rate=rate, **kwargs)


def test_every_measure_is_present_and_explains_itself(clean_speech):
    """A report always carries all seven, each with a value, a score and a sentence."""
    report = report_for(clean_speech)

    assert set(report.metrics) == set(MEASURES)
    for name in MEASURES:
        metric = report.metric(name)
        assert 0.0 <= metric.score <= 100.0
        assert metric.message
        assert metric.name == name
        assert metric.unit


def test_asking_for_a_measure_that_is_not_there_says_which_are(clean_speech):
    """A typo in a measure name gets the list of real ones back."""
    with pytest.raises(KeyError) as caught:
        report_for(clean_speech).metric("loudness")

    assert "bandwidth" in str(caught.value)


def test_level_is_healthy_at_the_target(clean_speech):
    """Speech recorded at -20 dBFS RMS passes the level measure."""
    metric = report_for(clean_speech).metric("level")

    assert metric.value == pytest.approx(-20.0, abs=1.0)
    assert metric.ok
    assert "healthy" in metric.message


def test_a_quiet_recording_is_called_quiet():
    """20 dB under target fails, and the message says to raise the gain."""
    metric = report_for(speech_like(seconds=1.0, rms=0.01)).metric("level")

    assert metric.value == pytest.approx(-40.0, abs=1.0)
    assert not metric.ok
    assert "quiet" in metric.message
    assert metric.details["offset_db"] < 0


def test_a_hot_recording_is_called_hot():
    """A recording pushed up against the ceiling fails the other way."""
    metric = report_for(speech_like(seconds=1.0, rms=0.5)).metric("level")

    assert not metric.ok
    assert "hot" in metric.message
    assert metric.details["headroom_db"] >= 0.0


def test_clean_audio_reports_no_clipping(clean_speech):
    """Nothing near full scale means a clipping share of exactly zero."""
    metric = report_for(clean_speech).metric("clipping")

    assert metric.value == 0.0
    assert metric.score == 100.0
    assert metric.ok


def test_clipping_counts_samples_and_runs():
    """A driven sine clips in as many runs as it has half-cycles."""
    metric = report_for(clipped_sine(seconds=1.0, freq=300.0, drive=3.0)).metric("clipping")

    assert metric.value > 0.5
    assert not metric.ok
    assert metric.details["runs"] == 600  # two per cycle, 300 cycles a second
    assert metric.details["longest_run_ms"] > 0.0
    assert "flat-topped" in metric.message


def test_a_wholly_clipped_recording_is_one_long_run():
    """Every sample at full scale is a single run the length of the file."""
    metric = report_for(np.ones(RATE)).metric("clipping")

    assert metric.value == pytest.approx(1.0)
    assert metric.score == 0.0
    assert metric.details["runs"] == 1
    assert metric.details["longest_run_samples"] == RATE


def test_noise_finds_the_floor_and_the_ratio(clean_speech):
    """A clean recording shows a wide gap between its quiet and loud frames."""
    metric = report_for(clean_speech).metric("noise")

    assert metric.value > 15.0
    assert metric.ok
    assert metric.details["noise_floor_dbfs"] < metric.details["signal_dbfs"]


def test_added_hiss_lowers_the_ratio_and_raises_the_floor():
    """Filling the gaps with noise is visible in both numbers, in opposite directions."""
    clean = speech_like(seconds=2.0, rms=0.1)
    hiss = np.random.default_rng(1).standard_normal(clean.size) * 0.02

    quiet = report_for(clean).metric("noise")
    noisy = report_for(clean + hiss).metric("noise")

    assert noisy.value < quiet.value - 5.0
    assert noisy.details["noise_floor_dbfs"] > quiet.details["noise_floor_dbfs"]


def test_a_signal_whose_level_never_drops_refuses_to_guess():
    """An unbroken tone has no gaps, so no floor can be told from the signal."""
    metric = report_for(sine(seconds=1.0)).metric("noise")

    assert not metric.measured
    assert "cannot be told apart" in metric.message
    assert metric.details["separable"] is False


def test_silence_at_the_edges_is_measured_in_seconds():
    """Two seconds of lead-in and one of run-out are reported as such."""
    padded = np.concatenate(
        [silence(seconds=2.0), speech_like(seconds=2.0), silence(seconds=1.0)]
    )

    metric = report_for(padded).metric("silence")

    assert metric.details["leading_s"] == pytest.approx(2.0, abs=0.1)
    assert metric.details["trailing_s"] == pytest.approx(1.0, abs=0.1)
    assert not metric.ok
    assert "dead air" in metric.message or "mostly silence" in metric.message


def test_continuous_speech_has_almost_no_silence(clean_speech):
    """Speech with no gaps reports a silent share near zero."""
    metric = report_for(clean_speech).metric("silence")

    assert metric.value < 0.2
    assert metric.ok


def test_an_all_silent_recording_says_so():
    """Nothing above the floor anywhere is reported as exactly that."""
    metric = report_for(silence(seconds=1.0)).metric("silence")

    assert metric.value == pytest.approx(1.0)
    assert not metric.ok
    assert "reads as silence" in metric.message


def test_speech_like_audio_reads_as_speech(clean_speech):
    """Band energy, centroid and syllable-rate swing together make a speech verdict."""
    metric = report_for(clean_speech).metric("speech")

    assert metric.value > 0.8
    assert metric.ok
    assert metric.details["modulation_db"] > 2.0
    assert 150.0 < metric.details["mean_centroid_hz"] < 3500.0


def test_a_steady_tone_is_not_speech_however_well_it_sits_in_the_band():
    """A 440 Hz sine has speech-band energy and a speech-band centroid; it still fails."""
    metric = report_for(sine(seconds=1.0, freq=440.0)).metric("speech")

    assert not metric.ok
    assert metric.details["steady_level"] is True
    assert "does not behave like speech" in metric.message


def test_dynamics_reports_the_crest_factor_of_speech(clean_speech):
    """Unprocessed speech keeps a comfortable gap between peak and average."""
    metric = report_for(clean_speech).metric("dynamics")

    assert metric.value > 10.0
    assert metric.ok
    assert metric.details["compressed"] is False


def test_a_sine_is_squashed_by_definition():
    """A sine's crest factor is 3 dB, which is below anything a voice does."""
    metric = report_for(sine(seconds=1.0)).metric("dynamics")

    assert metric.value == pytest.approx(3.0, abs=0.5)
    assert not metric.ok
    assert "squashed" in metric.message


def test_heavy_compression_is_named_without_being_failed():
    """A crest factor between the two limits is flagged but not called a failure."""
    squashed = np.sign(speech_like(seconds=1.0)) * np.abs(speech_like(seconds=1.0)) ** 0.25
    metric = report_for(squashed * 0.3).metric("dynamics")

    assert metric.details["compressed"] is True
    assert "compressed" in metric.message or "squashed" in metric.message


def test_bandwidth_finds_the_edge_it_was_given():
    """Noise brick-walled at 3.4 kHz is reported as stopping around there."""
    metric = report_for(band_limited_noise(seconds=1.0, top_hz=3400.0)).metric("bandwidth")

    assert metric.value == pytest.approx(3400.0, abs=350.0)
    assert not metric.ok


def test_full_band_audio_reaches_nyquist(clean_speech):
    """Wideband speech carries energy right up to the ceiling."""
    metric = report_for(clean_speech).metric("bandwidth")

    assert metric.value > 6000.0
    assert metric.ok
    assert metric.details["upsampled"] is False


def test_telephone_band_audio_presented_as_wideband_is_caught():
    """A 3.4 kHz signal in a 48 kHz file is upsampled, and the report says which."""
    narrow = band_limited_noise(seconds=1.0, rate=48000, top_hz=3400.0)

    metric = report_for(narrow, rate=48000).metric("bandwidth")

    assert metric.details["upsampled"] is True
    assert metric.details["nyquist_share"] < 0.55
    assert not metric.ok
    assert "upsampled" in metric.message


def test_narrowband_audio_in_an_8k_file_is_not_called_upsampled():
    """At 8 kHz there is no room above 4 kHz, so nothing was thrown away."""
    narrow = band_limited_noise(seconds=1.0, rate=8000, top_hz=3400.0)

    metric = report_for(narrow, rate=8000).metric("bandwidth")

    assert metric.details["upsampled"] is False
    assert metric.details["telephone_band"] is True


def test_thresholds_move_the_verdict(clean_speech):
    """Demanding 40 dB of signal-to-noise fails a recording that has 25."""
    relaxed = report_for(clean_speech).metric("noise")
    strict = report_for(clean_speech, thresholds={"min_snr_db": 40.0}).metric("noise")

    assert relaxed.ok
    assert not strict.ok
    assert relaxed.value == pytest.approx(strict.value)


def test_a_thresholds_object_works_as_well_as_a_dict(clean_speech):
    """Both ways of passing limits reach the same answer."""
    limits = Thresholds().replace(min_snr_db=40.0)

    by_object = report_for(clean_speech, thresholds=limits)
    by_dict = report_for(clean_speech, thresholds={"min_snr_db": 40.0})

    assert by_object.score == pytest.approx(by_dict.score)


def test_an_unknown_threshold_is_refused_by_name():
    """A misspelt limit is caught at the entry point, not silently ignored."""
    with pytest.raises(ValueError) as caught:
        Thresholds().replace(min_snr=40.0)

    assert "min_snr" in str(caught.value)
    assert "min_snr_db" in str(caught.value)


def test_a_non_numeric_threshold_is_refused():
    """A limit wants a number; a string would fail somewhere far away otherwise."""
    with pytest.raises(ValueError) as caught:
        Thresholds().replace(min_snr_db="loud")

    assert "wants a number" in str(caught.value)


# --- regressions: two faults an independent reviewer found and this suite missed ---

def _voice(rate, top_harmonic):
    import numpy as np

    t = np.arange(rate * 4) / rate
    v = sum((0.6 / k) * np.sin(2 * np.pi * 140 * k * t) for k in range(1, top_harmonic))
    v = v * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)) * 0.25
    return v + np.random.default_rng(0).normal(0, 0.0002, len(t))


def test_an_ordinary_48k_voice_recording_is_usable():
    """Bandwidth used to be judged as a share of Nyquist, and anything under 55% was
    'upsampled'. Speech lives below about 8 kHz whatever the format, so a clean studio
    voice at 48 kHz fills a third of the band at most - and every such recording was
    condemned as unusable while the identical audio at 16 kHz passed."""
    import speech_quality as sq

    report = sq.assess(_voice(48000, 40), sample_rate=48000)
    assert report.usable, report.issues
    assert report.metrics["bandwidth"].ok


def test_a_phone_call_saved_as_wideband_is_still_caught():
    """The check was right to exist; only its trigger was wrong. Telephone-band content
    inside a file that could hold far more really has lost its consonants."""
    import speech_quality as sq

    for rate in (16000, 48000):
        voice = _voice(48000, 24)[:: 48000 // rate]
        report = sq.assess(voice, sample_rate=rate)
        assert not report.usable, rate
        assert report.metrics["bandwidth"].details["upsampled"], rate


def test_native_narrowband_is_judged_on_its_own_merits():
    """An 8 kHz file cannot hold more than telephone band, so it is not 'upsampled'."""
    import speech_quality as sq

    report = sq.assess(_voice(48000, 24)[::6], sample_rate=8000)
    assert not report.metrics["bandwidth"].details["upsampled"]


def test_8_bit_clipping_is_counted_on_both_rails(tmp_path):
    """8-bit PCM is asymmetric: the positive side stops at 127/128, about 0.992, while
    the negative reaches -1.0. With one fixed clip level of 0.995 the positive rail
    could never register, so a file clipped on both sides reported exactly half."""
    import wave

    import numpy as np

    import speech_quality as sq

    n = np.arange(8000 * 2)
    signal = np.clip(1.6 * np.sin(2 * np.pi * 300 * n / 8000), -1, 1)
    path = tmp_path / "clipped8.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(np.clip(np.round(signal * 128 + 128), 0, 255).astype(np.uint8).tobytes())

    truth = float(np.mean(np.abs(signal) >= 0.999))
    measured = sq.assess(str(path)).metrics["clipping"].value
    assert measured == pytest.approx(truth, abs=0.01), (measured, truth)
