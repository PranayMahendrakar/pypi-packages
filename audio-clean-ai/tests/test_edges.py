"""The edges: silence, no pause, clipping, odd shapes, bad input."""

from __future__ import annotations

import os

import numpy as np
import pytest

from audio_clean_ai import clean, noise_profile
from conftest import SR, true_snr, voice_like, white, write_pcm


# ---------------------------------------------------------------- silence


def test_pure_silence_stays_exactly_silent():
    silence = np.zeros(2 * SR)
    result = clean(silence, sample_rate=SR, strength=1.0)
    assert np.array_equal(result.audio, silence)
    assert result.noise_reduction_db == 0.0
    assert result.snr_before is None and result.snr_after is None
    assert result.channels[0].silent
    assert any("digital silence" in w for w in result.warnings)
    assert "digital silence" in result.summary()


def test_near_silence_is_never_amplified():
    whisper = white(2 * SR, 1e-4, seed=5)  # -80 dBFS room tone, nothing else
    result = clean(whisper, sample_rate=SR, strength=1.0)
    assert np.max(np.abs(result.audio)) <= np.max(np.abs(whisper))
    assert np.sum(result.audio ** 2) <= np.sum(whisper ** 2)


def test_silent_channel_beside_a_live_one_stays_silent():
    n = 3 * SR
    voice = voice_like(n)
    stereo = np.stack([voice + white(n, 0.05), np.zeros(n)], axis=1)
    result = clean(stereo, sample_rate=SR)
    assert np.array_equal(result.audio[:, 1], np.zeros(n))
    assert result.channels[1].silent and not result.channels[0].silent
    assert true_snr(voice, result.audio[:, 0]) > true_snr(voice, stereo[:, 0]) + 5
    assert not any("digital silence" in w for w in result.warnings)


def test_empty_recording():
    result = clean(np.zeros(0), sample_rate=SR)
    assert result.audio.shape == (0,)
    assert any("empty" in w for w in result.warnings)
    assert "empty" in result.summary()
    stereo = clean(np.zeros((0, 2)), sample_rate=SR)
    assert stereo.audio.shape == (0, 2)


def test_leading_digital_silence_is_not_learned_as_the_noise():
    n = 3 * SR
    voice = voice_like(n)
    noisy = voice + white(n, 0.05)
    noisy[: SR // 2] = 0.0  # a zero-filled lead-in before the noise starts
    voice[: SR // 2] = 0.0
    result = clean(noisy, sample_rate=SR)
    floor = result.channels[0].noise_floor_dbfs
    assert abs(floor - 20 * np.log10(0.05)) < 1.5  # learned the hiss, not the zeros
    assert result.channels[0].quiet_stretch[0] >= 0.45
    assert true_snr(voice, result.audio) > true_snr(voice, noisy) + 5
    assert np.max(np.abs(result.audio[: SR // 4])) < 1e-12  # the zeros stay zeros


# ------------------------------------------------------- no quiet stretch


def test_no_quiet_stretch_is_flagged_and_cleaned_gently():
    n = 3 * SR
    voice = voice_like(n, gaps=False)  # talks from end to end, never pauses
    noisy = voice + white(n, 0.01)
    result = clean(noisy, sample_rate=SR, strength=1.0)
    assert not result.profile_reliable
    assert not result.channels[0].profile_reliable
    assert any("no quiet stretch" in w and "unreliable" in w for w in result.warnings)
    assert "UNRELIABLE" in result.summary()
    # Gentle: never cut by more than the 6 dB cap, even at strength 1.
    assert result.channels[0].max_cut_db == 6.0
    kept = np.sum(result.audio ** 2) / np.sum(noisy ** 2)
    assert kept > 10 ** (-6.5 / 10)
    # Hard guessing would have cut the voice itself far deeper.
    from audio_clean_ai import SpectralGate

    reckless = SpectralGate(strength=1.0, gentle_cut_db=30.0).clean(noisy, sample_rate=SR)
    assert np.sum(reckless.audio ** 2) < 0.5 * np.sum(result.audio ** 2)


def test_steady_noise_from_end_to_end_is_flagged_but_a_clip_profile_cleans_it():
    n = 3 * SR
    fan = white(n, 0.05, seed=7)
    gentle = clean(fan, sample_rate=SR)
    assert not gentle.profile_reliable
    assert gentle.snr_before is None  # no signal is claimed inside pure noise
    learned = noise_profile(fan, sample_rate=SR)
    hard = clean(fan, sample_rate=SR, noise_profile=learned)
    assert hard.profile_reliable
    assert hard.channels[0].profile_source == "given"
    assert np.sum(hard.audio ** 2) < 0.1 * np.sum(gentle.audio ** 2)


def test_a_given_profile_cleans_a_recording_with_no_pause():
    n = 3 * SR
    voice = voice_like(n, gaps=False)
    noisy = voice + white(n, 0.03, seed=11)
    room = white(2 * SR, 0.03, seed=12)  # a separate noise-only clip
    result = clean(noisy, sample_rate=SR, noise_profile=noise_profile(room, sample_rate=SR))
    assert result.profile_reliable and result.warnings == []
    assert true_snr(voice, result.audio) > true_snr(voice, noisy) + 3.0


def test_very_short_recordings_report_an_unreliable_profile():
    for n in (1, 10, 100, 600):
        result = clean(white(n, 0.1, seed=n), sample_rate=SR)
        assert result.audio.shape == (n,)
        assert not result.profile_reliable
        assert result.warnings and "unreliable" in result.warnings[0]
        assert result.channels[0].max_cut_db <= 6.0


# --------------------------------------------------------------- clipping


def test_clipping_is_never_introduced_on_a_loud_clipped_input():
    n = 3 * SR
    loud = np.clip(3.0 * voice_like(n, level=0.6) + white(n, 0.05), -1.0, 1.0)
    result = clean(loud, sample_rate=SR, strength=1.0)
    assert result.clipping_before["positive"] > 0 and result.clipping_before["negative"] > 0
    assert np.max(result.audio) <= 1.0 and np.min(result.audio) >= -1.0
    assert result.clipping_after == {"positive": 0, "negative": 0}
    assert any("clipped before" in w for w in result.warnings)


def test_no_false_clipping_alarm_on_a_normal_recording():
    n = 3 * SR
    result = clean(voice_like(n) + white(n, 0.05), sample_rate=SR)
    assert result.clipping_before == {"positive": 0, "negative": 0}
    assert result.output_gain_db == 0.0
    assert not any("clip" in w for w in result.warnings)


def test_input_above_full_scale_is_turned_down_not_clipped():
    n = 2 * SR
    hot = 2.5 * (voice_like(n) + white(n, 0.05)) / 0.35  # peaks well above 1
    result = clean(hot, sample_rate=SR)
    assert np.max(np.abs(result.audio)) < 1.0
    assert result.output_gain_db < 0.0
    assert any("turned down" in note for note in result.notes)
    assert any("above full scale" in note for note in result.notes)
    assert not any("clipped before" in w for w in result.warnings)  # loud is not clipped
    assert "turned down" in result.summary()
    shape = result.audio / np.max(np.abs(result.audio))
    reference = clean(hot / np.max(np.abs(hot)), sample_rate=SR).audio
    assert np.corrcoef(shape, reference)[0, 1] > 0.9999  # scaled, not distorted


def test_eight_bit_wav_is_measured_per_rail(tmp_path):
    """8-bit WAV reaches -1.0 but its positive rail is +127/128."""
    n = 2 * SR
    signal = 1.4 * voice_like(n, level=1.0) + white(n, 0.03)
    codes = np.clip(np.round(signal * 128.0 + 128.0), 0, 255).astype(np.uint8)
    assert codes.max() == 255 and codes.min() == 0
    path = os.path.join(str(tmp_path), "eight.wav")
    write_pcm(path, codes, SR, 1)

    result = clean(path, strength=1.0)
    assert result.positive_rail == pytest.approx(127.0 / 128.0)
    assert result.negative_rail == -1.0
    assert result.clipping_before["positive"] > 0
    assert result.clipping_before["negative"] > 0
    assert np.max(result.audio) < 127.0 / 128.0  # checked against the real rail
    assert np.min(result.audio) > -1.0
    assert result.clipping_after == {"positive": 0, "negative": 0}

    out = os.path.join(str(tmp_path), "eight-clean.wav")
    result.save(out)  # keeps 8-bit
    import wave

    with wave.open(out, "rb") as handle:
        assert handle.getsampwidth() == 1
        back = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.uint8)
    decoded = (back.astype(np.float64) - 128.0) / 127.0
    assert np.corrcoef(decoded, result.audio)[0, 1] > 0.999  # no wrap-round


def test_rail_check_uses_the_format_rail_not_one():
    """An output at 0.995 is fine for float but over the top for 8-bit WAV."""
    from audio_clean_ai._audio import Loaded
    from audio_clean_ai.core import SpectralGate

    frames = np.array([[0.995], [-0.5], [0.2]])
    eight = Loaded(frames, SR, "x", [], None, 127.0 / 128.0, -1.0, 8)
    fitted, gain_db = SpectralGate()._fit_inside_rails(frames.copy(), eight)
    assert gain_db < 0.0 and np.max(fitted) < 127.0 / 128.0
    floaty = Loaded(frames, SR, "x", [], None, 1.0, -1.0, None)
    same, zero = SpectralGate()._fit_inside_rails(frames.copy(), floaty)
    assert zero == 0.0 and np.array_equal(same, frames)


# ----------------------------------------------------------------- stereo


def test_stereo_is_processed_per_channel():
    n = 3 * SR
    voice = voice_like(n)
    left = voice + white(n, 0.05, seed=1)  # noisy
    right = 0.8 * voice + white(n, 1e-5, seed=2)  # clean
    stereo = np.stack([left, right], axis=1)
    result = clean(stereo, sample_rate=SR)
    assert result.audio.shape == stereo.shape
    assert result.noise_profile.shape == (2, result.frequencies.size)
    floors = [c.noise_floor_dbfs for c in result.channels]
    assert floors[0] > floors[1] + 40  # each channel learned its own noise
    assert true_snr(voice, result.audio[:, 0]) > true_snr(voice, left) + 5
    assert true_snr(right, result.audio[:, 1]) > 30  # the clean side is untouched
    # Each channel comes out exactly as it would on its own.
    alone = clean(left, sample_rate=SR)
    assert np.allclose(alone.audio, result.audio[:, 0], atol=1e-12)
    # Channels-first gives the same answer, transposed.
    flipped = clean(stereo.T, sample_rate=SR)
    assert np.allclose(flipped.audio.T, result.audio, atol=1e-12)
    assert "channel 2" in result.summary()


# -------------------------------------------------------------- bad input


def test_bad_inputs_raise_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="sample_rate is required"):
        clean(np.zeros(100))
    with pytest.raises(ValueError, match="dimensions"):
        clean(np.zeros((2, 3, 4)), sample_rate=SR)
    with pytest.raises(ValueError, match="complex"):
        clean(np.zeros(100, dtype=complex), sample_rate=SR)
    with pytest.raises(ValueError, match="numbers"):
        clean(np.array(["a", "b"]), sample_rate=SR)
    with pytest.raises(ValueError, match="too low"):
        clean(np.zeros(100), sample_rate=500)
    with pytest.raises(ValueError, match="pair says"):
        clean((np.zeros(100), 16000), sample_rate=8000)
    with pytest.raises(ValueError, match="channel count"):
        clean(np.zeros((40, 40)), sample_rate=SR)
    with pytest.raises(FileNotFoundError):
        clean(os.path.join(str(tmp_path), "missing.wav"))
    junk = os.path.join(str(tmp_path), "junk.wav")
    with open(junk, "wb") as handle:
        handle.write(b"this is not audio at all")
    with pytest.raises(ValueError, match="RIFF"):
        clean(junk)


def test_non_finite_samples_are_treated_as_silence():
    n = 2 * SR
    noisy = voice_like(n) + white(n, 0.05)
    noisy[1000:1010] = np.nan
    noisy[2000] = np.inf
    original = noisy.copy()
    result = clean(noisy, sample_rate=SR)
    assert np.all(np.isfinite(result.audio))
    assert any("non-finite" in note for note in result.notes)
    assert np.array_equal(np.isnan(noisy), np.isnan(original))  # untouched


def test_pair_input_and_unusual_rate_note():
    n = SR
    result = clean((white(n, 0.05), SR))
    assert result.sample_rate == SR
    odd = clean(white(6000, 0.05), sample_rate=6000)
    assert any("outside the tested" in note for note in odd.notes)


def _clean_voice(sr=16000):
    t = np.arange(4 * sr) / sr
    v = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 15))
    v = 0.1 * v * (np.sin(2 * np.pi * 3 * t) > 0)
    v[:sr] = 0.0
    return v


def test_noise_free_voice_with_digital_zero_pauses_is_left_intact():
    v = _clean_voice()
    result = clean(v, sample_rate=16000)
    assert result.audio.shape == v.shape
    error = result.audio - v
    err_snr = 10 * np.log10(np.sum(v ** 2) / max(np.sum(error ** 2), 1e-30))
    assert err_snr > 30.0
    assert abs(result.noise_reduction_db) < 0.5


def test_digital_zero_lead_in_does_not_stop_real_noise_being_cleaned():
    # Control: the same voice with real noise and a real noisy pause still gets cleaned.
    rng = np.random.default_rng(3)
    v = _clean_voice()
    noisy = v + 0.01 * rng.standard_normal(v.size)
    result = clean(noisy, sample_rate=16000)
    assert result.noise_reduction_db > 3.0


def test_nan_input_is_reported_as_a_warning():
    result = clean(np.array([np.nan, 1.0] * 8000), sample_rate=16000)
    assert any("non-finite" in w for w in result.warnings)
    assert np.all(np.isfinite(result.audio))


def test_stereo_threads_match_per_channel_mono():
    rng = np.random.default_rng(5)
    x = 0.05 * rng.standard_normal((16000 * 2, 2))
    both = clean(x, sample_rate=16000).audio
    left = clean(x[:, 0], sample_rate=16000).audio
    right = clean(x[:, 1], sample_rate=16000).audio
    assert np.array_equal(both[:, 0], left) and np.array_equal(both[:, 1], right)
