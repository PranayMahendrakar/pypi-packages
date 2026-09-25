"""Reading WAV files: every width, float, extensible, stereo, and clipping at the true rail."""

import struct

import numpy as np
import pytest

import audio_anomaly
from audio_anomaly._io import load_audio
from audio_anomaly._wav import read_wav
from _signals import SR, add_knock, hum, write_float_wav, write_pcm


def test_8bit_clipping_is_measured_against_the_8bit_rail(tmp_path):
    # Driven hard, 8-bit PCM flattens at 255 (+127/128 = 0.9922) and 0 (-1.0).
    path = write_pcm(tmp_path / "clipped8.wav", hum(seed=1) * 6.0, bits=8)
    samples = read_wav(str(path)).samples
    assert samples.max() == pytest.approx(127 / 128)
    assert samples.max() < 0.995  # a fixed 0.995 threshold could never see this
    report = audio_anomaly.detect(path)
    clip = report.of_kind("clipping")
    assert clip, report.summary()
    assert "8-bit PCM rail (+127/128, -1)" in clip[0].message


def test_8bit_audio_that_stays_inside_the_rail_is_not_clipping(tmp_path):
    x = hum(seed=2)
    path = write_pcm(tmp_path / "clean8.wav", 0.9 * x / np.max(np.abs(x)), bits=8)
    report = audio_anomaly.detect(path)
    assert "clipping" not in [e.kind for e in report.anomalies]


def test_8bit_positive_clipping_alone_is_found(tmp_path):
    # Only the positive half touches its rail, which sits below 1.0.
    x = hum(seed=3)
    x = np.where(x > 0, x * 8.0, x * 0.5)
    path = write_pcm(tmp_path / "pos8.wav", x, bits=8)
    loaded = load_audio(str(path))
    assert loaded.rail.positive == pytest.approx(127 / 128)
    assert loaded.clipped is not None and loaded.clipped.any()


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_every_pcm_width_round_trips_and_finds_the_knock(tmp_path, bits):
    x = add_knock(hum(seed=4), amp=0.3)
    x = 0.5 * x / np.max(np.abs(x))
    path = write_pcm(tmp_path / ("w%d.wav" % bits), x, bits=bits)
    data = read_wav(str(path))
    assert data.sample_rate == SR and data.bits == bits and not data.is_float
    assert data.samples.shape == (x.size, 1)
    tolerance = 1.0 / 2 ** (bits - 1)
    np.testing.assert_allclose(data.samples[:, 0], x, atol=tolerance)
    report = audio_anomaly.detect(path)
    assert [e.kind for e in report.anomalies] == ["burst"]
    assert report.source == str(path)


@pytest.mark.parametrize("extensible", [False, True])
def test_float_wav_is_read_even_though_wave_refuses_it(tmp_path, extensible):
    x = add_knock(hum(seed=5), amp=0.3) * 0.5
    path = write_float_wav(tmp_path / "float.wav", x, extensible=extensible)
    data = read_wav(str(path))
    assert data.is_float and data.bits == 32
    np.testing.assert_allclose(data.samples[:, 0], x, atol=1e-6)
    assert [e.kind for e in audio_anomaly.detect(path).anomalies] == ["burst"]


def test_float_wav_overs_count_as_clipping(tmp_path):
    path = write_float_wav(tmp_path / "overs.wav", np.clip(hum(seed=6) * 6.0, -1.0, 1.0))
    report = audio_anomaly.detect(path)
    assert "clipping" in [e.kind for e in report.anomalies]


def test_stereo_wav_is_mixed_and_one_clipped_channel_is_caught(tmp_path):
    x = hum(seed=7)
    stereo = np.stack([x * 8.0, x * 0.2], axis=1)
    path = write_pcm(tmp_path / "stereo.wav", stereo, bits=16)
    report = audio_anomaly.detect(path)
    assert "clipping" in [e.kind for e in report.anomalies]
    assert any("2-channel" in n for n in report.notes)


def test_sample_rate_argument_is_overridden_by_the_file_with_a_note(tmp_path):
    path = write_pcm(tmp_path / "rate.wav", hum(seconds=1.0, seed=8), sr=SR)
    report = audio_anomaly.detect(path, sample_rate=44100)
    assert report.sample_rate == SR
    assert any("recorded at 16000 Hz" in n for n in report.notes)


def test_pathlib_and_unicode_paths(tmp_path):
    path = write_pcm(tmp_path / "Pumpe_Prüfung_測試.wav", hum(seconds=1.0, seed=9))
    report = audio_anomaly.detect(path)
    assert report.source.endswith("Pumpe_Prüfung_測試.wav")
    assert "Pumpe_Prüfung_測試.wav" in report.summary()


def test_not_a_wav_file_is_a_clear_error(tmp_path):
    path = tmp_path / "notes.wav"
    path.write_text("this is not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="RIFF"):
        audio_anomaly.detect(path)


def test_truncated_header_is_a_clear_error(tmp_path):
    path = tmp_path / "cut.wav"
    path.write_bytes(b"RIFF\x10\x00\x00\x00WAVE")
    with pytest.raises(ValueError):
        audio_anomaly.detect(path)


def test_compressed_wav_is_refused_with_advice(tmp_path):
    fmt = struct.pack("<HHIIHH", 0x0055, 1, SR, SR, 1, 0)  # MP3-in-WAV
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 4) + b"\x00" * 4
    path = tmp_path / "mp3.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    with pytest.raises(ValueError, match="16-bit PCM"):
        audio_anomaly.detect(path)


def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        audio_anomaly.detect(tmp_path / "nope.wav")


CORRUPT_WAV = b"RIFF\x10\x00\x00\x00WAVEjunkjunkjunk"


def test_a_corrupt_chunk_is_a_value_error_naming_the_file(tmp_path):
    # The chunk reader inside ``wave`` raises a bare, empty RuntimeError here.
    path = tmp_path / "corrupt.wav"
    path.write_bytes(CORRUPT_WAV)
    with pytest.raises(ValueError, match=r"corrupt\.wav.*damaged"):
        audio_anomaly.detect(path)


@pytest.mark.parametrize("riff_size", [36, 100])
def test_a_wrong_riff_size_is_read_chunk_by_chunk(tmp_path, riff_size):
    # A RIFF size smaller than the file makes ``wave`` fail (36) or stop early
    # without a word (100); every chunk is intact, so the audio is all there.
    path = write_pcm(tmp_path / "streamed.wav", add_knock(hum(seed=3)))
    raw = bytearray(path.read_bytes())
    raw[4:8] = struct.pack("<I", riff_size)
    path.write_bytes(bytes(raw))
    report = audio_anomaly.detect(path)
    assert report.duration_s == pytest.approx(4.0)
    assert [e.kind for e in report.anomalies] == ["burst"]
    assert any("read chunk by chunk" in n for n in report.notes)
    assert report.warnings == []


@pytest.mark.parametrize("writer", ["pcm", "float"])
def test_a_file_cut_short_is_analysed_with_a_warning(tmp_path, writer):
    whole = tmp_path / "whole.wav"
    if writer == "pcm":
        write_pcm(whole, hum(seconds=10.0, seed=1))
    else:
        write_float_wav(whole, hum(seconds=10.0, seed=1))
    raw = whole.read_bytes()
    cut = tmp_path / "cut.wav"
    cut.write_bytes(raw[: len(raw) // 2])
    report = audio_anomaly.detect(cut)
    assert report.duration_s == pytest.approx(5.0, abs=0.01)
    assert any("declares 10.00 s" in w and "only 5.00 s" in w for w in report.warnings), report.summary()
    assert "Warnings:" in report.summary()


def test_a_complete_file_has_no_truncation_warning(tmp_path):
    path = write_pcm(tmp_path / "ok.wav", hum(seed=4))
    assert audio_anomaly.detect(path).warnings == []


def test_a_wav_holding_zero_frames_gives_an_empty_report(tmp_path):
    path = write_pcm(tmp_path / "empty.wav", np.zeros(0))
    report = audio_anomaly.detect(path)
    assert report.n_frames == 0 and report.anomalies == []
