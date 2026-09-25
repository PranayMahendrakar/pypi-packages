"""The core promise: steady noise goes down, the voice stays, nothing is invented."""

from __future__ import annotations

import json

import numpy as np
import pytest

from audio_clean_ai import CleanResult, SpectralGate, clean
from conftest import SR, harmonic_ratio_db, hum, true_snr, voice_like, white


def _noisy(sr: int = SR, seconds: float = 3.0, noise: float = 0.05):
    n = int(seconds * sr)
    voice = voice_like(n, sr)
    return voice, voice + white(n, noise, seed=1)


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000])
def test_white_noise_snr_improves_and_voice_band_is_kept(sr):
    voice, noisy = _noisy(sr)
    result = clean(noisy, sample_rate=sr)

    before, after = true_snr(voice, noisy), true_snr(voice, result.audio)
    assert after > before + 5.0, (sr, before, after)
    # The voice's own harmonics come through within a couple of dB.
    assert -2.0 < harmonic_ratio_db(voice, result.audio, sr) < 0.5
    # The estimates point the same way as the truth.
    assert result.snr_after > result.snr_before + 5.0
    assert result.noise_reduction_db > 8.0
    assert result.profile_reliable
    assert result.warnings == []
    # The learned floor matches the planted noise (0.05 RMS = -26 dBFS).
    assert abs(result.channels[0].noise_floor_dbfs - 20 * np.log10(0.05)) < 1.5


def test_clean_input_is_left_alone():
    """No false alarm: speech with pauses and no real noise passes through."""
    n = 3 * SR
    voice = voice_like(n) + white(n, 1e-5, seed=2)
    result = clean(voice, sample_rate=SR)
    assert result.profile_reliable
    assert result.warnings == []
    assert true_snr(voice, result.audio) > 30.0  # the change is inaudible
    assert any("already very low" in note for note in result.notes)


def _tone_drop(planted: np.ndarray, residual: np.ndarray, tone: float) -> float:
    """How far the residual sits below the planted noise within 6 Hz of a tone."""
    frequency = np.fft.rfftfreq(planted.size, 1.0 / SR)
    near = np.abs(frequency - tone) < 6.0
    before = np.sum(np.abs(np.fft.rfft(planted))[near] ** 2)
    after = np.sum(np.abs(np.fft.rfft(residual))[near] ** 2)
    return float(10 * np.log10(before / after))


def test_steady_hum_is_removed_and_voice_kept():
    n = 3 * SR
    voice = voice_like(n)
    mains = hum(n, level=0.08)
    result = clean(voice + mains, sample_rate=SR)
    residual = result.audio - voice
    pause = slice(0, int(0.35 * SR))  # the lead-in, before the voice starts
    for tone in (50.0, 100.0, 150.0):
        assert _tone_drop(mains[pause], residual[pause], tone) > 18.0, tone
    # Over the whole file 50 Hz sits clear of the voice and stays well down.
    # (150 Hz is 10 Hz from the 140 Hz voice, so while the voice speaks it
    # cannot be told apart at 32 ms resolution; that is the honest limit.)
    assert _tone_drop(mains, residual, 50.0) > 15.0
    assert -2.0 < harmonic_ratio_db(voice, result.audio, SR) < 0.5
    bands = result.channels[0].noise_bands_dbfs
    assert max(bands.items(), key=lambda kv: kv[1])[0] in ("31.5 Hz", "63 Hz")


def test_smoothing_stops_the_residual_from_warbling():
    """Unsmoothed gating leaves isolated bins flickering: musical noise."""
    from audio_clean_ai._stft import Stft

    voice, noisy = _noisy()
    lead = int(0.45 * SR)

    def peakiness(samples: np.ndarray) -> float:
        # E[P^2] / E[P]^2 over the noise-only lead-in: 2 for untouched noise,
        # large when a few bins poke out of an otherwise gated background.
        stft = Stft(lead, 512, SR)
        power = stft.power(stft.analyse(stft.pad(samples[:lead]), 0, stft.n_frames))
        power = power[4:-4, 10:240]
        return float(np.mean(power ** 2) / np.mean(power) ** 2)

    smoothed = clean(noisy, sample_rate=SR)
    raw = SpectralGate(time_smoothing_ms=0, freq_smoothing_hz=0).clean(noisy, sample_rate=SR)
    assert peakiness(noisy) == pytest.approx(2.0, abs=0.3)
    assert peakiness(smoothed.audio) < 0.5 * peakiness(raw.audio)
    assert peakiness(smoothed.audio) < 3.0 * peakiness(noisy)


def test_output_length_always_equals_input_length():
    for sr in (8000, 11025, 16000, 44100, 48000):
        for n in (1, 2, 3, 17, 255, 256, 257, 511, 512, 513, 1000, sr, sr + 1, 2 * sr - 7):
            audio = white(n, 0.1, seed=n)
            result = clean(audio, sample_rate=sr)
            assert result.audio.shape == audio.shape, (sr, n)
            assert result.n_samples == n
            assert np.all(np.isfinite(result.audio))


def test_output_layout_matches_input_layout():
    voice, noisy = _noisy()
    stereo = np.stack([noisy, noisy * 0.5], axis=1)
    assert clean(stereo, sample_rate=SR).audio.shape == stereo.shape
    assert clean(stereo.T, sample_rate=SR).audio.shape == stereo.T.shape
    column = noisy[:, None]
    assert clean(column, sample_rate=SR).audio.shape == column.shape


def test_caller_array_is_never_modified():
    _voice, noisy = _noisy()
    original = noisy.copy()
    clean(noisy, sample_rate=SR, strength=1.0)
    assert np.array_equal(noisy, original)

    frozen = noisy.copy()
    frozen.flags.writeable = False  # what pandas 3 hands out from .to_numpy()
    result = clean(frozen, sample_rate=SR)
    assert result.audio.flags.writeable
    result.audio[0] = 0.5  # the result is ours to write into
    assert np.array_equal(frozen, original)

    as_int = np.round(noisy * 20000).astype(np.int16)
    kept = as_int.copy()
    clean(as_int, sample_rate=SR)
    assert np.array_equal(as_int, kept)


def test_deterministic():
    _voice, noisy = _noisy()
    first = clean(noisy, sample_rate=SR)
    second = clean(noisy, sample_rate=SR)
    assert np.array_equal(first.audio, second.audio)
    assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())


def test_strength_zero_returns_the_input_and_more_strength_cuts_more():
    _voice, noisy = _noisy()
    untouched = clean(noisy, sample_rate=SR, strength=0.0)
    assert np.array_equal(untouched.audio, noisy)
    assert untouched.noise_reduction_db == pytest.approx(0.0, abs=1e-9)
    reductions = [
        clean(noisy, sample_rate=SR, strength=s).noise_reduction_db
        for s in (0.25, 0.5, 0.75, 1.0)
    ]
    assert reductions == sorted(reductions)
    assert reductions[-1] > reductions[0] + 5.0


def test_preserve_speech_keeps_a_floor_in_the_speech_band():
    voice, noisy = _noisy()
    lead = slice(0, int(0.4 * SR))  # the lead-in is noise only

    def band_drop(result: CleanResult, low: float, high: float) -> float:
        spectrum_in = np.abs(np.fft.rfft(noisy[lead])) ** 2
        spectrum_out = np.abs(np.fft.rfft(result.audio[lead])) ** 2
        frequency = np.fft.rfftfreq(spectrum_in.size * 2 - 2, 1.0 / SR)[: spectrum_in.size]
        band = (frequency >= low) & (frequency <= high)
        return float(10 * np.log10(spectrum_in[band].sum() / spectrum_out[band].sum()))

    kept = clean(noisy, sample_rate=SR, strength=1.0, preserve_speech=True)
    gated = clean(noisy, sample_rate=SR, strength=1.0, preserve_speech=False)
    # Inside the band the floor holds: never cut past 15 dB.
    assert band_drop(kept, 500.0, 3000.0) <= 15.5
    # Without it the same band is cut much deeper.
    assert band_drop(gated, 500.0, 3000.0) > band_drop(kept, 500.0, 3000.0) + 8.0
    # Outside the band both cut to the full depth.
    assert band_drop(kept, 4500.0, 7500.0) > 25.0
    assert kept.channels[0].speech_cut_db == 15.0
    assert gated.channels[0].speech_cut_db == 30.0


def test_class_blocks_do_not_change_the_answer():
    _voice, noisy = _noisy()
    whole = SpectralGate(block_frames=100000).clean(noisy, sample_rate=SR)
    blocked = SpectralGate(block_frames=7).clean(noisy, sample_rate=SR)
    assert np.allclose(whole.audio, blocked.audio, atol=1e-12, rtol=0)
    assert whole.noise_reduction_db == pytest.approx(blocked.noise_reduction_db, abs=1e-9)


def test_class_settings_are_validated():
    for bad in (
        dict(strength=-0.1),
        dict(strength=1.5),
        dict(strength=float("nan")),
        dict(frame_ms=2),
        dict(profile_seconds=0),
        dict(max_cut_db=-3),
        dict(speech_band=(3400, 300)),
        dict(threshold_db=10, pass_db=5),
        dict(block_frames=0),
    ):
        with pytest.raises(ValueError):
            SpectralGate(**bad)
    with pytest.raises(ValueError, match="strength"):
        clean(np.zeros(100), sample_rate=SR, strength=2.0)


def test_integer_input_is_scaled_by_its_dtype():
    _voice, noisy = _noisy()
    pcm = np.round(noisy * 32767).astype(np.int16)
    result = clean(pcm, sample_rate=SR)
    assert result.audio.dtype == np.float64
    assert np.max(np.abs(result.audio)) <= 1.0
    assert result.positive_rail == pytest.approx(32767 / 32768)
    assert result.negative_rail == -1.0
    as_float = clean(pcm / 32768.0, sample_rate=SR)
    assert np.allclose(result.audio, as_float.audio, atol=1e-12)
