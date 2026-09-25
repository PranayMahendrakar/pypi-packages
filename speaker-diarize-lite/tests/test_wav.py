"""Reading WAV files, including the awkward ones, and clipping measured per rail."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from _voices import SR
from speaker_diarize_lite import diarize, read_wav, write_wav
from speaker_diarize_lite._wav import rails


@pytest.mark.parametrize("bits, tolerance", [(8, 1 / 64), (16, 1e-4), (24, 1e-6), (32, 1e-8), (-32, 1e-6), (-64, 0.0)])
def test_round_trip_every_encoding(tmp_path, bits, tolerance):
    t = np.arange(4000) / 8000.0
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone_{}.wav".format(bits)
    write_wav(path, signal, 8000, bits=bits)
    wav = read_wav(path)
    assert wav.sample_rate == 8000
    assert wav.samples.shape == (4000, 1)
    assert wav.is_float == (bits < 0)
    assert wav.bits == abs(bits)
    assert np.max(np.abs(wav.samples[:, 0] - signal)) <= tolerance + 1e-12
    assert wav.warnings == []


def test_rails_are_asymmetric_for_integer_pcm():
    assert rails(8, False) == (-1.0, 127.0 / 128.0)
    assert rails(16, False) == (-1.0, 32767.0 / 32768.0)
    assert rails(32, True) == (-1.0, 1.0)


@pytest.mark.parametrize("bits", [8, 16])
def test_clipping_is_measured_on_each_rail(tmp_path, bits, two_voices):
    signal, _ = two_voices
    loud = np.clip(signal * 40.0, -1.0, 1.0)  # slammed into both rails
    path = tmp_path / "clipped.wav"
    write_wav(path, loud, SR, bits=bits)
    result = diarize(path)
    clip = [w for w in result.warnings if "clipped" in w]
    assert clip, result.warnings
    assert "negative rail" in clip[0] and "positive rail" in clip[0]

    # The same format with headroom must not raise a false alarm, even though
    # 8-bit's positive rail stops short of 1.0.
    path_clean = tmp_path / "clean.wav"
    write_wav(path_clean, signal, SR, bits=bits)
    clean = diarize(path_clean)
    assert not any("clipped" in w for w in clean.warnings)
    assert clean.num_speakers == 2


def test_positive_rail_only_clipping_on_8_bit(tmp_path, two_voices):
    signal, _ = two_voices
    lopsided = np.where(signal > 0, np.minimum(signal * 40.0, 1.0), signal)
    path = tmp_path / "top.wav"
    write_wav(path, lopsided, SR, bits=8)
    warning = [w for w in diarize(path).warnings if "clipped" in w]
    assert warning and "positive rail" in warning[0] and "negative rail" not in warning[0]


def test_wav_path_gives_file_id_and_stereo_is_mixed(tmp_path, two_voices):
    signal, _ = two_voices
    path = tmp_path / "board meeting.wav"
    write_wav(path, np.stack([signal, signal * 0.8], axis=1), SR, bits=16)
    result = diarize(str(path))
    assert result.file_id == "board_meeting"
    assert result.num_speakers == 2
    assert any("mixed 2 channels" in n for n in result.notes)
    assert result.to_rttm().split(" ")[1] == "board_meeting"


def test_wav_with_extra_chunks_and_extensible_header(tmp_path):
    t = np.arange(8000) / 8000.0
    ints = np.round(0.3 * np.sin(2 * np.pi * 200 * t) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 0xFFFE, 1, 8000, 16000, 2, 16)
    fmt += struct.pack("<HHI", 22, 16, 4) + struct.pack("<H", 1) + b"\x00" * 14
    listing = b"LIST" + struct.pack("<I", 5) + b"INFOx" + b"\x00"  # odd size, padded
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + listing
    body += b"data" + struct.pack("<I", len(ints)) + ints
    path = tmp_path / "ext.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    wav = read_wav(path)
    assert wav.sample_rate == 8000 and wav.samples.shape == (8000, 1)
    assert abs(wav.samples.max() - 0.3) < 1e-3


def test_truncated_wav_is_read_with_a_warning(tmp_path):
    path = tmp_path / "cut.wav"
    write_wav(path, np.zeros(16000), 16000, bits=16)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    wav = read_wav(path)
    assert wav.samples.shape[0] < 16000
    assert any("cut" in w for w in wav.warnings)


def test_not_a_wav_and_missing_file(tmp_path):
    bogus = tmp_path / "notes.wav"
    bogus.write_bytes(b"ID3\x03\x00 this is an mp3")
    with pytest.raises(ValueError, match="not a WAV"):
        diarize(bogus)
    with pytest.raises(FileNotFoundError):
        diarize(tmp_path / "nowhere.wav")


def test_compressed_wav_is_refused_clearly(tmp_path):
    fmt = struct.pack("<HHIIHH", 0x0002, 1, 8000, 4000, 256, 4)  # MS ADPCM
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 4) + b"\x00" * 4
    path = tmp_path / "adpcm.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    with pytest.raises(ValueError, match="0x0002"):
        read_wav(path)


def test_sample_rate_must_agree_with_the_file(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, np.zeros(8000), 8000)
    with pytest.raises(ValueError, match="does\\s+not resample"):
        diarize(path, sample_rate=16000)
    assert diarize(path, sample_rate=8000).segments == []


def test_write_wav_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        write_wav(tmp_path / "x.wav", np.zeros(10), 8000, bits=12)
    with pytest.raises(ValueError):
        write_wav(tmp_path / "x.wav", np.zeros((2, 2, 2)), 8000)
