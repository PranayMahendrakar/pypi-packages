"""noise_profile(), WAV in and out, and the result object explaining itself."""

from __future__ import annotations

import json
import logging
import os
import struct
import wave

import numpy as np
import pytest

from audio_clean_ai import ChannelReport, CleanResult, SpectralGate, clean, noise_profile
from conftest import SR, true_snr, voice_like, white, write_pcm


# ---------------------------------------------------------- noise_profile


def test_noise_profile_measures_the_noise_level():
    fan = white(2 * SR, 0.05, seed=3)
    profile = noise_profile(fan, sample_rate=SR)
    assert profile.shape == (257,)
    assert np.all(profile >= 0)
    level = 10 * np.log10(profile.sum())
    assert abs(level - 20 * np.log10(0.05)) < 0.5  # white noise at -26 dBFS
    # White noise is flat: no octave-sized region holds far more per bin.
    assert np.std(10 * np.log10(profile[5:-5])) < 1.5


def test_noise_profile_shapes_and_reuse():
    fan = white(2 * SR, 0.05, seed=3)
    stereo = np.stack([fan, 0.1 * fan], axis=1)
    both = noise_profile(stereo, sample_rate=SR)
    assert both.shape == (2, 257)
    assert 10 * np.log10(both[0].sum() / both[1].sum()) == pytest.approx(20.0, abs=0.1)
    # A result's own profile can be handed straight back in.
    n = 3 * SR
    voice = voice_like(n)
    noisy = voice + white(n, 0.05)
    first = clean(noisy, sample_rate=SR)
    again = clean(noisy, sample_rate=SR, noise_profile=first.noise_profile)
    assert again.channels[0].profile_source == "given"
    assert true_snr(voice, again.audio) > true_snr(voice, noisy) + 5
    # One 1-D profile applies to every channel.
    shared = clean(np.stack([noisy, noisy], axis=1), sample_rate=SR, noise_profile=first.noise_profile)
    assert np.allclose(shared.audio[:, 0], again.audio, atol=1e-12)


def test_noise_profile_mismatches_are_explained():
    profile = noise_profile(white(SR, 0.05), sample_rate=SR)
    with pytest.raises(ValueError, match="sample rate"):
        clean(white(48000, 0.05), sample_rate=48000, noise_profile=profile)
    with pytest.raises(ValueError, match="one row per channel"):
        clean(
            np.zeros((SR, 2)),
            sample_rate=SR,
            noise_profile=np.stack([profile, profile, profile]),
        )
    with pytest.raises(ValueError, match="non-negative"):
        clean(white(SR, 0.05), sample_rate=SR, noise_profile=-profile)
    with pytest.raises(ValueError, match="empty"):
        noise_profile(np.zeros(0), sample_rate=SR)


def test_silent_noise_clip_gives_a_zero_profile_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger="audio_clean_ai.core"):
        profile = noise_profile(np.zeros(SR), sample_rate=SR)
    assert np.array_equal(profile, np.zeros(257))
    assert any("digital silence" in record.getMessage() for record in caplog.records)
    # A zero profile removes nothing rather than everything.
    n = 2 * SR
    audio = voice_like(n) + white(n, 0.05)
    result = clean(audio, sample_rate=SR, noise_profile=profile)
    assert true_snr(audio, result.audio) > 40


def test_gate_class_learns_profiles_at_its_own_frame_length():
    gate = SpectralGate(frame_ms=64)
    fan = white(2 * SR, 0.05)
    profile = gate.learn_profile(fan, sample_rate=SR)
    assert profile.shape == (513,)
    result = gate.clean(fan + voice_like(2 * SR), sample_rate=SR, noise_profile=profile)
    assert result.frequencies.size == 513
    with pytest.raises(ValueError, match="bins"):
        clean(fan, sample_rate=SR, noise_profile=profile)  # 32 ms frames: 257 bins


# ------------------------------------------------------------------- WAV


def _write_float_wav(path: str, samples: np.ndarray, sr: int, extensible: bool) -> None:
    """A 32-bit IEEE float WAV, optionally in WAVE_FORMAT_EXTENSIBLE dress."""
    data = samples.astype("<f4").tobytes()
    if extensible:
        guid = struct.pack("<H", 3) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt = struct.pack("<HHIIHH", 0xFFFE, 1, sr, sr * 4, 4, 32)
        fmt += struct.pack("<HHI", 22, 32, 4) + guid
    else:
        fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    with open(path, "wb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", len(body)) + body)


@pytest.mark.parametrize("extensible", [False, True])
def test_float_wav_is_read(tmp_path, extensible):
    n = 2 * SR
    audio = (voice_like(n) + white(n, 0.05)).astype(np.float32)
    path = os.path.join(str(tmp_path), "float.wav")
    _write_float_wav(path, audio, SR, extensible)
    result = clean(path)
    assert result.sample_rate == SR and result.audio.shape == (n,)
    direct = clean(audio.astype(np.float64), sample_rate=SR)
    assert np.allclose(result.audio, direct.audio, atol=1e-9)
    assert result.source == "float.wav"


def test_pcm_wav_round_trip_keeps_shape_rate_and_depth(tmp_path):
    n = 2 * SR
    voice = voice_like(n)
    stereo = np.stack([voice + white(n, 0.05), voice + white(n, 0.05, seed=9)], axis=1)
    path = os.path.join(str(tmp_path), "stereo.wav")
    write_pcm(path, np.round(stereo * 32767).astype(np.int16), SR, 2)
    result = clean(path)
    assert result.audio.shape == (n, 2)
    assert result.source_bits == 16
    written = result.save(os.path.join(str(tmp_path), "stereo-clean.wav"))
    with wave.open(written, "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getframerate() == SR
        assert handle.getnframes() == n
        assert handle.getsampwidth() == 2
    again = clean(written, strength=0.0)
    assert np.max(np.abs(again.audio - result.audio)) < 2.0 / 32767


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_save_writes_every_supported_depth(tmp_path, bits):
    n = SR // 2
    result = clean(voice_like(n, lead_s=0.1) + white(n, 0.02), sample_rate=SR)
    path = result.save(os.path.join(str(tmp_path), "out{}.wav".format(bits)), bits=bits)
    with wave.open(path, "rb") as handle:
        assert handle.getsampwidth() == bits // 8
        assert handle.getnframes() == n
    back = clean(path, strength=0.0).audio
    step = 2.0 / (2 ** (bits - 1) - 1)
    assert np.max(np.abs(back - result.audio)) <= step


def test_save_channels_first_and_bad_depth(tmp_path):
    n = SR
    stereo = np.stack([white(n, 0.05), white(n, 0.05, seed=4)])  # (2, n)
    result = clean(stereo, sample_rate=SR)
    assert result.audio.shape == (2, n)
    path = result.save(os.path.join(str(tmp_path), "cf.wav"))
    with wave.open(path, "rb") as handle:
        assert handle.getnchannels() == 2 and handle.getnframes() == n
    with pytest.raises(ValueError, match="bits"):
        result.save(os.path.join(str(tmp_path), "bad.wav"), bits=12)


def test_unicode_file_name(tmp_path):
    n = SR
    path = os.path.join(str(tmp_path), "r\u00e9union_\u4f1a\u8b70.wav")
    write_pcm(path, np.round((voice_like(n, lead_s=0.2) + white(n, 0.05)) * 30000).astype(np.int16), SR, 2)
    result = clean(path)
    assert result.source == "r\u00e9union_\u4f1a\u8b70.wav"
    assert "r\u00e9union" in result.summary()


# --------------------------------------------------------------- reporting


def test_result_explains_itself():
    n = 3 * SR
    result = clean(voice_like(n) + white(n, 0.05), sample_rate=SR)
    assert isinstance(result, CleanResult)
    assert isinstance(result.channels[0], ChannelReport)
    text = result.summary()
    text.encode("ascii")  # plain ASCII for an array input
    for phrase in ("noise down", "SNR", "quietest stretch", "reliable", "strength 0.80", "300-3400 Hz"):
        assert phrase in text
    assert result.snr_improvement_db == pytest.approx(result.snr_after - result.snr_before)
    assert result.duration_s == pytest.approx(3.0)


def test_to_dict_is_json_safe_and_complete():
    n = 3 * SR
    result = clean(np.stack([voice_like(n) + white(n, 0.05), np.zeros(n)], axis=1), sample_rate=SR)
    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    back = json.loads(text)
    for key in (
        "noise_reduction_db",
        "snr_before",
        "snr_after",
        "profile_reliable",
        "clipping",
        "channels",
        "warnings",
        "notes",
        "sample_rate",
        "n_samples",
    ):
        assert key in back
    assert "audio" not in back
    assert back["channels"][1]["silent"] is True
    assert back["channels"][1]["snr_before"] is None
    assert back["channels"][0]["quiet_stretch"]["end_s"] > back["channels"][0]["quiet_stretch"]["start_s"]
    assert back["clipping"]["before"] == {"positive": 0, "negative": 0}
