"""The detector's core promises: catch the fault, stay quiet on healthy sound."""

import itertools

import numpy as np
import pytest

import audio_anomaly
from _signals import SR, add_knock, add_tone, hum, scale_span


def kinds(report):
    return [e.kind for e in report.anomalies]


def test_readme_quickstart_runs_and_finds_the_knock():
    t = np.arange(4 * 16000) / 16000
    x = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    x[32000:32480] += np.random.default_rng(1).standard_normal(480) * np.exp(-np.arange(480) / 60)
    report = audio_anomaly.detect(x, sample_rate=16000)
    assert kinds(report) == ["burst"]
    assert report.anomalies[0].start_s == pytest.approx(2.0, abs=0.005)
    assert "burst" in report.summary()


def test_planted_knock_in_steady_hum_is_caught_and_clean_hum_stays_quiet():
    clean = hum(seed=3)
    quiet = audio_anomaly.detect(clean, sample_rate=SR)
    assert quiet.anomalies == []
    assert quiet.max_score < 3.0
    assert quiet.anomaly_ratio == 0.0

    report = audio_anomaly.detect(add_knock(clean, at=2.0), sample_rate=SR)
    assert kinds(report) == ["burst"]
    knock = report.anomalies[0]
    assert knock.start_s == pytest.approx(2.0, abs=0.01)
    assert knock.duration_s < 0.1
    assert knock.score > 6.0
    assert "knock" in knock.message


@pytest.mark.parametrize(
    "sr,f0,noise,frame_ms,wobble",
    list(
        itertools.product(
            (8000, 16000, 44100), (60.0, 900.0), (0.0005, 0.05), (25.0, 50.0, 100.0), (0.0, 0.05)
        )
    )[::3],
)
def test_healthy_machine_hum_is_never_flagged(sr, f0, noise, frame_ms, wobble):
    x = hum(sr=sr, seconds=3.0, f0=f0, noise=noise, seed=1, wobble=wobble)
    report = audio_anomaly.detect(x, sample_rate=sr, frame_ms=frame_ms)
    assert report.anomalies == [], report.summary()
    assert report.max_score < 3.0


def test_small_knock_is_still_caught():
    report = audio_anomaly.detect(add_knock(hum(seed=4), amp=0.1), sample_rate=SR)
    assert "burst" in kinds(report)


def test_tone_appearing_halfway_is_caught_without_a_reference():
    report = audio_anomaly.detect(add_tone(hum(seed=5), start=2.0), sample_rate=SR)
    assert kinds(report) == ["tonal"]
    tone = report.anomalies[0]
    assert tone.start_s == pytest.approx(2.0, abs=0.1)
    assert tone.end_s == pytest.approx(4.0, abs=0.01)
    assert "3.10 kHz" in tone.message
    assert 0.4 < report.anomaly_ratio < 0.6


def test_tone_appearing_halfway_is_caught_against_a_reference():
    reference = hum(seconds=3.0, seed=50)
    report = audio_anomaly.detect(add_tone(hum(seed=5), start=2.0), sample_rate=SR, reference=reference)
    assert report.mode == "reference"
    assert kinds(report) == ["tonal"]


def test_subtle_tone_is_caught():
    report = audio_anomaly.detect(add_tone(hum(seed=6), start=2.0, amp=0.005), sample_rate=SR)
    assert "tonal" in kinds(report)


def test_knock_during_a_whine_reports_both():
    x = add_knock(add_tone(hum(seconds=6.0, seed=7), start=2.0), at=4.0)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert sorted(kinds(report)) == ["burst", "tonal"]
    burst = report.of_kind("burst")[0]
    assert burst.start_s == pytest.approx(4.0, abs=0.01)


def test_repeated_knocks_are_each_reported():
    x = hum(seconds=6.0, seed=8)
    for k in range(10):
        x = add_knock(x, at=0.5 + 0.5 * k, amp=0.3, seed=k)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert kinds(report) == ["burst"] * 10


def test_level_drop_is_caught():
    x = scale_span(hum(seed=9), at=2.0, seconds=1.0, factor=0.25)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert kinds(report) == ["level_drop"]
    drop = report.anomalies[0]
    assert drop.start_s == pytest.approx(2.0, abs=0.1)
    assert drop.end_s == pytest.approx(3.0, abs=0.1)
    assert "12" in drop.message


def test_dropout_to_silence_is_caught():
    x = scale_span(hum(seed=10), at=2.0, seconds=0.3, factor=0.0)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert kinds(report) == ["dropout"]
    assert report.anomalies[0].start_s == pytest.approx(2.0, abs=0.05)


def test_short_hole_of_exact_zeros_is_a_dropout():
    x = scale_span(hum(seed=11), at=1.5, seconds=0.02, factor=0.0)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert kinds(report) == ["dropout"]
    hole = report.anomalies[0]
    assert hole.start_s == pytest.approx(1.5, abs=0.001)
    assert hole.duration_s == pytest.approx(0.02, abs=0.001)
    assert "20 ms" in hole.message


def test_sustained_broadband_rise_is_a_level_rise():
    x = hum(seed=12, noise=0.004)
    rng = np.random.default_rng(1)
    x[SR * 2 : SR * 3] += 0.05 * rng.standard_normal(SR)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert "level_rise" in kinds(report)


def test_clipped_array_is_reported():
    x = np.clip(hum(seed=13) * 6.0, -1.0, 1.0)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert "clipping" in kinds(report)
    clip = report.of_kind("clipping")[0]
    assert "full scale" in clip.message


def test_loud_but_unclipped_audio_is_not_clipping():
    x = hum(seed=13)
    x = 0.98 * x / np.max(np.abs(x))
    assert "clipping" not in kinds(audio_anomaly.detect(x, sample_rate=SR))


def test_deterministic():
    x = add_tone(add_knock(hum(seed=14)), start=2.5)
    a = audio_anomaly.detect(x, sample_rate=SR)
    b = audio_anomaly.detect(x.copy(), sample_rate=SR)
    assert a.to_dict() == b.to_dict()


def test_caller_array_is_never_modified():
    x = add_knock(hum(seed=15))
    x[100:110] = np.nan
    stereo = np.stack([x, x * 0.5], axis=1)
    ref = hum(seed=16)
    before = (x.copy(), stereo.copy(), ref.copy())
    audio_anomaly.detect(x, sample_rate=SR)
    audio_anomaly.detect(stereo, sample_rate=SR, reference=ref)
    audio_anomaly.spectral_profile(stereo, sample_rate=SR)
    audio_anomaly.Monitor(ref, sample_rate=SR).check(stereo)
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(stereo, before[1])
    np.testing.assert_array_equal(ref, before[2])


def test_higher_sensitivity_flags_less():
    x = add_knock(hum(seed=17), amp=0.1)
    loose = audio_anomaly.detect(x, sample_rate=SR, sensitivity=2.0)
    strict = audio_anomaly.detect(x, sample_rate=SR, sensitivity=50.0)
    assert len(loose.anomalies) >= 1
    assert strict.anomalies == []


def test_scores_and_frame_times_line_up():
    report = audio_anomaly.detect(add_knock(hum(seed=18)), sample_rate=SR)
    assert report.scores.shape == report.frame_times.shape
    assert np.all(np.diff(report.frame_times) > 0)
    peak_time = report.frame_times[int(np.argmax(report.scores))]
    assert peak_time == pytest.approx(2.0, abs=0.05)
    flagged = report.scores >= report.sensitivity
    assert flagged.any()


def test_whine_covering_most_of_the_recording_needs_a_reference():
    x = add_tone(hum(seed=19), start=0.3)
    self_report = audio_anomaly.detect(x, sample_rate=SR)
    assert "tonal" not in kinds(self_report)
    ref_report = audio_anomaly.detect(x, sample_rate=SR, reference=hum(seconds=3.0, seed=20))
    assert "tonal" in kinds(ref_report)
