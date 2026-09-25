"""spectral_profile and Monitor: save a reference once, check many recordings against it."""

import numpy as np
import pytest

import audio_anomaly
from _signals import SR, add_knock, add_tone, hum, write_pcm


def test_spectral_profile_shape_and_columns():
    profile = audio_anomaly.spectral_profile(hum(seconds=3.0, seed=1), sample_rate=SR)
    assert profile.ndim == 2 and profile.shape[1] == 3
    assert profile.shape[0] == 28  # 16 kHz keeps the bands below 0.95 x Nyquist
    freqs, level, spread = profile.T
    assert np.all(np.diff(freqs) > 0) and freqs[0] > 20 and freqs[-1] < 8000
    assert np.all(spread >= 0)
    hum_band = int(np.argmin(np.abs(freqs - 120.0)))
    assert level[hum_band] == pytest.approx(level.max())  # the 120 Hz hum is the loudest band


def test_profile_of_a_tone_reads_its_level_in_dbfs():
    t = np.arange(SR * 2) / SR
    profile = audio_anomaly.spectral_profile(np.sin(2 * np.pi * 1000 * t), sample_rate=SR)
    # A full-scale sine is 0 dBFS in total; the band holding it gets nearly all of that.
    total = 10 * np.log10(np.sum(10 ** (profile[:, 1] / 10)))
    assert total == pytest.approx(0.0, abs=0.05)
    assert profile[:, 1].max() > -1.5
    assert profile[int(np.argmax(profile[:, 1])), 0] == pytest.approx(1000, rel=0.25)


def test_profile_is_the_same_at_any_sample_rate():
    a = audio_anomaly.spectral_profile(hum(sr=16000, seconds=2.0, noise=0.0), sample_rate=16000)
    b = audio_anomaly.spectral_profile(hum(sr=44100, seconds=2.0, noise=0.0), sample_rate=44100)
    np.testing.assert_allclose(a[:, 0], b[: a.shape[0], 0])
    loud = a[:, 1] > -40
    np.testing.assert_allclose(a[loud, 1], b[: a.shape[0]][loud, 1], atol=0.5)


def test_saved_profile_round_trips_as_a_reference(tmp_path):
    profile = audio_anomaly.spectral_profile(hum(seconds=3.0, seed=2), sample_rate=SR)
    np.save(tmp_path / "pump.npy", profile)
    reference = np.load(tmp_path / "pump.npy")
    healthy = audio_anomaly.detect(hum(seed=3), sample_rate=SR, reference=reference)
    assert healthy.anomalies == []
    assert healthy.mode == "reference"
    assert "saved spectral profile" in healthy.compared_against
    faulty = audio_anomaly.detect(add_tone(hum(seed=3), start=2.0), sample_rate=SR, reference=reference)
    assert [e.kind for e in faulty.anomalies] == ["tonal"]


def test_profile_made_with_another_frame_ms_is_recognised():
    profile = audio_anomaly.spectral_profile(hum(seconds=3.0, seed=4), sample_rate=SR, frame_ms=100)
    report = audio_anomaly.detect(hum(seed=5), sample_rate=SR, reference=profile, frame_ms=50)
    assert report.frame_ms == 100
    assert any("frame_ms=100" in n for n in report.notes)
    assert report.anomalies == []


def test_hand_made_profile_is_interpolated_with_a_note():
    profile = audio_anomaly.spectral_profile(hum(seconds=3.0, seed=6), sample_rate=SR)
    coarse = profile[::2].copy()
    coarse[:, 0] *= 1.07  # centres that match no grid
    report = audio_anomaly.detect(hum(seed=7), sample_rate=SR, reference=coarse)
    assert any("interpolated" in n for n in report.notes)


def test_spectral_profile_refuses_silence_and_too_short():
    with pytest.raises(ValueError, match="silence"):
        audio_anomaly.spectral_profile(np.zeros(SR), sample_rate=SR)
    with pytest.raises(ValueError, match="shorter than one"):
        audio_anomaly.spectral_profile(np.ones(10), sample_rate=SR)


def test_monitor_checks_many_recordings_against_one_reference(tmp_path):
    ref_path = write_pcm(tmp_path / "healthy.wav", hum(seconds=3.0, seed=8))
    monitor = audio_anomaly.Monitor(ref_path)
    assert "healthy.wav" in repr(monitor)
    assert monitor.profile.shape == (28, 3)
    quiet = monitor.check(hum(seed=9), sample_rate=SR)
    assert quiet.anomalies == [] and quiet.mode == "reference"
    knock = monitor.check(add_knock(hum(seed=9)), sample_rate=SR)
    assert [e.kind for e in knock.anomalies] == ["burst"]
    whine = monitor.check((add_tone(hum(seed=9), start=2.0), SR))
    assert [e.kind for e in whine.anomalies] == ["tonal"]


def test_monitor_matches_detect_with_the_same_reference():
    ref = hum(seconds=3.0, seed=10)
    x = add_knock(add_tone(hum(seed=11), start=2.5), at=1.0)
    via_monitor = audio_anomaly.Monitor(ref, sample_rate=SR, sensitivity=3.5).check(x)
    via_detect = audio_anomaly.detect(x, sample_rate=SR, reference=ref, sensitivity=3.5)
    assert via_monitor.to_dict()["anomalies"] == via_detect.to_dict()["anomalies"]


def test_monitor_from_a_profile_and_options():
    profile = audio_anomaly.spectral_profile(hum(seconds=3.0, seed=12), sample_rate=SR, frame_ms=25)
    monitor = audio_anomaly.Monitor(profile, sensitivity=4.0)
    assert monitor.frame_ms == 25 and monitor.sensitivity == 4.0
    report = monitor.check((hum(seed=13), SR))
    assert report.anomalies == [] and report.frame_ms == 25


def test_monitor_refuses_an_unusable_reference():
    with pytest.raises(ValueError, match="silence"):
        audio_anomaly.Monitor(np.zeros(SR), sample_rate=SR)
    with pytest.raises(ValueError, match="needs a reference"):
        audio_anomaly.Monitor(None)
    with pytest.raises(ValueError, match="sample rate is unknown"):
        audio_anomaly.Monitor(hum(seconds=1.0))
    with pytest.raises(ValueError, match="sensitivity"):
        audio_anomaly.Monitor(hum(seconds=1.0), sample_rate=SR, sensitivity=-1)
