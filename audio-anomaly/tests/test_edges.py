"""Awkward inputs: silence, nothing, too short, odd dtypes and shapes, bad arguments."""

import numpy as np
import pytest

import audio_anomaly
from _signals import SR, add_knock, hum


def test_pure_silence_gives_no_anomalies():
    report = audio_anomaly.detect(np.zeros(SR * 3), sample_rate=SR)
    assert report.anomalies == []
    assert report.anomaly_ratio == 0.0
    assert report.n_frames > 0
    assert np.all(report.scores == 0.0)
    assert any("silence" in n for n in report.notes)


def test_near_silent_dither_is_still_silence():
    x = 1e-6 * np.random.default_rng(0).standard_normal(SR * 2)
    assert audio_anomaly.detect(x, sample_rate=SR).anomalies == []


def test_shorter_than_one_frame_returns_an_empty_report():
    report = audio_anomaly.detect(np.ones(100) * 0.1, sample_rate=SR)  # 6 ms < 50 ms
    assert report.anomalies == []
    assert report.scores.size == 0 and report.frame_times.size == 0
    assert report.anomaly_ratio == 0.0
    assert any("shorter than one" in n for n in report.notes)
    assert "Nothing was analysed" in report.summary()


def test_empty_and_single_sample_arrays():
    for x in (np.zeros(0), np.array([0.5])):
        report = audio_anomaly.detect(x, sample_rate=SR)
        assert report.anomalies == [] and report.scores.size == 0


def test_all_nan_audio_is_read_as_silence():
    report = audio_anomaly.detect(np.full(SR * 2, np.nan), sample_rate=SR)
    assert report.anomalies == []
    assert any("not a finite number" in n for n in report.notes)


def test_a_few_nans_do_not_poison_the_result():
    x = hum(seed=1)
    x[5000:5003] = np.nan
    x[9000] = np.inf
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert all(np.isfinite(report.scores))
    assert any("4 samples" in n for n in report.notes)


def test_stereo_is_mixed_to_mono_in_either_orientation():
    x = add_knock(hum(seed=2))
    stereo = np.stack([x, x], axis=1)
    a = audio_anomaly.detect(stereo, sample_rate=SR)
    b = audio_anomaly.detect(stereo.T, sample_rate=SR)
    assert [e.kind for e in a.anomalies] == ["burst"]
    assert a.to_dict()["anomalies"] == b.to_dict()["anomalies"]
    assert any("mixed down to mono" in n for n in a.notes)
    assert any("longer axis was taken as time" in n for n in b.notes)


def test_clipping_in_one_stereo_channel_is_still_found():
    x = hum(seed=3)
    left = np.clip(x * 8.0, -1.0, 1.0)
    report = audio_anomaly.detect(np.stack([left, x * 0.1], axis=1), sample_rate=SR)
    assert "clipping" in [e.kind for e in report.anomalies]


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.uint8, np.float32, np.float64])
def test_integer_and_float_dtypes(dtype):
    x = add_knock(hum(seed=4))
    if np.dtype(dtype).kind == "i":
        data = np.round(x * np.iinfo(dtype).max * 0.9).astype(dtype)
    elif np.dtype(dtype).kind == "u":
        data = np.round(128 + x * 100).astype(dtype)
    else:
        data = x.astype(dtype)
    report = audio_anomaly.detect(data, sample_rate=SR)
    assert "burst" in [e.kind for e in report.anomalies]
    assert "clipping" not in [e.kind for e in report.anomalies]


def test_int16_clipping_is_judged_against_its_rail():
    x = np.clip(np.round(hum(seed=5) * 32768 * 5), -32768, 32767).astype(np.int16)
    report = audio_anomaly.detect(x, sample_rate=SR)
    clip = report.of_kind("clipping")
    assert clip and "+32767/32768" in clip[0].message


def test_python_list_and_pair_inputs():
    x = hum(seconds=1.0, seed=6)
    a = audio_anomaly.detect(list(x), sample_rate=SR)
    b = audio_anomaly.detect((x, SR))
    assert a.sample_rate == b.sample_rate == SR
    assert a.to_dict()["scores"] == b.to_dict()["scores"]


def test_pair_rate_wins_over_sample_rate_with_a_note():
    report = audio_anomaly.detect((hum(seconds=1.0), SR), sample_rate=8000)
    assert report.sample_rate == SR
    assert any("carried its own sample rate" in n for n in report.notes)


def test_reference_of_different_length_and_sample_rate_is_resampled_with_a_note():
    x = hum(sr=16000, seconds=4.0, seed=7, noise=0.004)
    ref = hum(sr=44100, seconds=1.5, seed=8, noise=0.004 * np.sqrt(44100 / 16000))
    report = audio_anomaly.detect(x, sample_rate=16000, reference=(ref, 44100))
    assert report.anomalies == []
    assert any("resampled" in n for n in report.notes)
    assert any("lengths need not match" in n for n in report.notes)


def test_reference_at_a_lower_rate_falls_back_for_the_bands_it_cannot_reach():
    ref = hum(sr=8000, seconds=3.0, seed=9, noise=0.004 * np.sqrt(0.5))
    x = hum(sr=16000, seconds=6.0, seed=10)
    t = np.arange(x.size) / 16000
    x = x + 0.05 * np.clip((t - 3.0) / 0.02, 0, 1) * np.sin(2 * np.pi * 5000 * t)
    report = audio_anomaly.detect(x, sample_rate=16000, reference=(ref, 8000))
    assert "tonal" in [e.kind for e in report.anomalies]
    assert any("could not be compared" in n for n in report.notes)
    assert any("own typical sound instead" in n for n in report.notes)


def test_bare_array_reference_takes_the_recordings_rate_with_a_note():
    report = audio_anomaly.detect((hum(seed=11), SR), reference=hum(seconds=2.0, seed=12))
    assert report.mode == "reference"
    assert any("bare array" in n for n in report.notes)


def test_silent_reference_falls_back_with_a_warning():
    report = audio_anomaly.detect(add_knock(hum(seed=13)), sample_rate=SR, reference=np.zeros(SR))
    assert report.mode == "self"
    assert report.warnings and "silence" in report.warnings[0]
    assert [e.kind for e in report.anomalies] == ["burst"]


def test_quieter_than_reference_says_why():
    report = audio_anomaly.detect(hum(seed=14) * 0.3, sample_rate=SR, reference=hum(seconds=3.0, seed=15))
    assert [e.kind for e in report.anomalies] == ["level_drop"]
    assert any("quieter than the reference" in n for n in report.notes)


def test_leading_and_trailing_silence_is_noted_not_flagged():
    x = np.concatenate([np.zeros(SR), hum(seconds=3.0, seed=16), np.zeros(SR // 2)])
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert report.anomalies == []
    assert any("the first 1.00 s" in n and "the last" in n for n in report.notes)


def test_mostly_silent_recording_flags_the_sounds_not_the_gaps():
    x = np.zeros(SR * 6)
    for at in (1.0, 3.5):
        i = int(at * SR)
        x[i : i + 480] += np.random.default_rng(int(at)).standard_normal(480) * np.exp(-np.arange(480) / 60)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert [e.kind for e in report.anomalies] == ["burst", "burst"]
    assert any("digital silence, which was taken as its normal state" in n for n in report.notes)


def test_very_short_recording_skips_sustained_checks_with_a_note():
    report = audio_anomaly.detect(hum(seconds=0.2, seed=17), sample_rate=SR)
    assert report.anomalies == []
    assert any("too little to learn a typical sound" in n for n in report.notes)


def test_huge_float_values_warn_instead_of_calling_everything_clipped():
    report = audio_anomaly.detect(hum(seed=18) * 20000.0, sample_rate=SR)
    assert "clipping" not in [e.kind for e in report.anomalies]
    assert any("integer samples stored as floats" in w for w in report.warnings)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), "high", None])
def test_bad_sensitivity_is_refused(bad):
    with pytest.raises(ValueError, match="sensitivity"):
        audio_anomaly.detect(hum(seconds=1.0), sample_rate=SR, sensitivity=bad)


@pytest.mark.parametrize("bad", [0, -5, float("inf")])
def test_bad_frame_ms_is_refused(bad):
    with pytest.raises(ValueError, match="frame_ms"):
        audio_anomaly.detect(hum(seconds=1.0), sample_rate=SR, frame_ms=bad)


def test_frame_too_short_for_the_rate_is_refused():
    with pytest.raises(ValueError, match="at least 16"):
        audio_anomaly.detect(hum(seconds=1.0, sr=8000), sample_rate=8000, frame_ms=1)


def test_bare_array_without_a_rate_is_refused():
    with pytest.raises(ValueError, match="sample rate is unknown"):
        audio_anomaly.detect(hum(seconds=1.0))


@pytest.mark.parametrize("rate", [0, -8000, float("nan"), "fast"])
def test_impossible_rates_are_refused(rate):
    with pytest.raises(ValueError, match="sample rate"):
        audio_anomaly.detect(hum(seconds=1.0), sample_rate=rate)


def test_rate_too_low_for_any_band_is_refused():
    with pytest.raises(ValueError, match="too low"):
        audio_anomaly.detect(np.ones(400), sample_rate=40, frame_ms=500)


@pytest.mark.parametrize("bad", [{"a": 1}, 3.5, np.array(["x", "y"]), np.array([True, False])])
def test_non_audio_objects_are_refused(bad):
    with pytest.raises(TypeError):
        audio_anomaly.detect(bad, sample_rate=SR)


def test_three_dimensional_array_is_refused():
    with pytest.raises(ValueError, match="dimensions"):
        audio_anomaly.detect(np.zeros((10, 10, 10)), sample_rate=SR)


def test_profile_passed_as_audio_is_refused_with_advice():
    profile = audio_anomaly.spectral_profile(hum(seconds=2.0), sample_rate=SR)
    with pytest.raises(ValueError, match="reference="):
        audio_anomaly.detect(profile, sample_rate=SR)


def test_int64_and_integer_lists_get_their_bit_depth_inferred():
    x = np.round(add_knock(hum(seed=19)) * 20000).astype(np.int64)
    for data in (x, [int(v) for v in x]):
        report = audio_anomaly.detect(data, sample_rate=SR)
        assert [e.kind for e in report.anomalies] == ["burst"]
        assert any("fit 16-bit signed PCM" in n for n in report.notes)


# --- regression: beating twin-motor hum raised a false alarm every beat cycle ------

def test_beating_hum_is_not_a_stream_of_level_drop_alarms():
    """Regression: two motors at 120 and 120.5 Hz beat once every 2 s, and every one
    of those natural amplitude nulls looked like an independent level_drop fault -
    even against a clean reference of the same machine, since the drop detector
    compares each frame to its own local neighbours regardless of any reference."""
    import numpy as np

    import audio_anomaly as aa

    sr = 16000
    t = np.arange(10 * sr) / sr
    rng = np.random.default_rng(0)
    x = (
        0.2 * np.sin(2 * np.pi * 120 * t)
        + 0.14 * np.sin(2 * np.pi * 120.5 * t)
        + 0.05 * np.sin(2 * np.pi * 240 * t)
        + 0.01 * rng.normal(size=len(t))
    )
    ref_rng = np.random.default_rng(1)
    ref = (
        0.2 * np.sin(2 * np.pi * 120 * t + 1.0)
        + 0.14 * np.sin(2 * np.pi * 120.5 * t + 2.0)
        + 0.05 * np.sin(2 * np.pi * 240 * t)
        + 0.01 * ref_rng.normal(size=len(t))
    )
    report = aa.detect(x, sample_rate=sr, reference=(ref, sr))
    drops = [e for e in report.anomalies if "drop" in e.kind]
    assert not drops, f"beating hum raised {len(drops)} false level_drop events"
    assert any("beating" in note or "recur" in note for note in report.notes)


def test_a_single_real_dropout_is_still_caught_after_the_periodicity_fix():
    import numpy as np

    import audio_anomaly as aa

    sr = 16000
    t = np.arange(10 * sr) / sr
    rng = np.random.default_rng(0)
    x = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.01 * rng.normal(size=len(t))
    x[3 * sr : int(3.5 * sr)] *= 0.05
    report = aa.detect(x, sample_rate=sr)
    drops = [e for e in report.anomalies if "drop" in e.kind]
    assert drops, "a single genuine 0.5s dropout must still be reported"


def test_two_irregularly_spaced_real_drops_are_not_mistaken_for_periodicity():
    import numpy as np

    import audio_anomaly as aa

    sr = 16000
    t = np.arange(10 * sr) / sr
    rng = np.random.default_rng(0)
    x = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.01 * rng.normal(size=len(t))
    x[int(1 * sr) : int(1.2 * sr)] *= 0.05
    x[int(7 * sr) : int(7.3 * sr)] *= 0.05
    report = aa.detect(x, sample_rate=sr)
    drops = [e for e in report.anomalies if "drop" in e.kind]
    assert len(drops) == 2, "two unrelated, irregularly-spaced real drops must both be found"
