"""WAV decoding, per-rail clipping, resampling and format sniffing."""

from __future__ import annotations

import struct

import pytest

from offline_stt_router._audio import (
    AudioInput,
    read_wav,
    resample,
    sniff_format,
    to_pcm16,
    write_wav,
)

from conftest import tone, write_wav_bytes, zero_crossing_frequency


@pytest.mark.parametrize(
    "layout",
    [
        dict(bits=8), dict(bits=16), dict(bits=24), dict(bits=32),
        dict(bits=32, float_format=True), dict(bits=64, float_format=True),
        dict(bits=16, extensible=True), dict(bits=32, float_format=True, extensible=True),
        dict(bits=16, channels=2), dict(bits=24, channels=6), dict(bits=16, extra_chunk=True),
    ],
)
def test_every_wav_layout_decodes_to_the_same_tone(layout):
    source = tone(440, 0.25, 16000, level=0.5)
    samples, rate, info = read_wav(write_wav_bytes(source, rate=16000, **layout))
    assert rate == 16000
    assert info.channels == layout.get("channels", 1)
    assert len(samples) == len(source)
    tolerance = 1.0 / 128 + 1e-6 if layout["bits"] == 8 else 1e-3
    assert max(abs(a - b) for a, b in zip(samples, source)) <= tolerance
    assert abs(zero_crossing_frequency(samples, rate) - 440) < 10


def test_8_bit_wav_rails_are_asymmetric():
    # unsigned 8-bit: byte 0 is -1.0, byte 255 is +127/128 (there is no +1.0)
    data = write_wav_bytes([0.0], bits=8)
    raw = data[:-1] + bytes([255])
    samples, _, _ = read_wav(raw)
    assert samples[0] == pytest.approx(127 / 128)
    samples, _, _ = read_wav(data[:-1] + bytes([0]))
    assert samples[0] == -1.0
    samples, _, _ = read_wav(data[:-1] + bytes([128]))
    assert samples[0] == 0.0


def test_pcm16_clipping_is_measured_per_rail():
    pcm = to_pcm16([1.0, -1.0, 127 / 128, 1.7, -1.7, float("nan"), 0.0])
    values = struct.unpack("<7h", pcm)
    assert values[0] == 32767            # positive rail clamps at 32767
    assert values[1] == -32768           # negative rail is exact, no clamp needed
    assert values[2] == 32512            # the 8-bit positive rail is not a clip
    assert values[3] == 32767 and values[4] == -32768
    assert values[5] == 0 and values[6] == 0


def test_clean_tone_never_touches_a_rail():
    pcm = to_pcm16(tone(300, 0.5, 16000, level=0.5))
    values = struct.unpack("<{0}h".format(len(pcm) // 2), pcm)
    assert max(values) < 32767 and min(values) > -32768
    assert 16000 < max(values) < 16500


def test_resample_keeps_pitch_and_duration():
    source = tone(440, 1.0, 44100)
    out = resample(source, 44100, 16000)
    assert len(out) == 16000
    assert abs(zero_crossing_frequency(out, 16000) - 440) < 5
    assert list(resample(source[:10], 44100, 44100)) == pytest.approx(source[:10])
    assert len(resample([], 44100, 16000)) == 0


def test_resample_pure_python_path_matches(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("numpy hidden for this test")
        return real_import(name, *args, **kwargs)

    source = tone(200, 0.1, 48000)
    monkeypatch.setattr(builtins, "__import__", no_numpy)
    out = resample(source, 48000, 16000)
    assert len(out) == 1600
    assert abs(zero_crossing_frequency(out, 16000) - 200) < 15


@pytest.mark.parametrize(
    "head, name",
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"), (b"fLaC\x00\x00", "flac"), (b"OggS\x00", "ogg"),
        (b"ID3\x04\x00", "mp3"), (b"\xff\xfb\x90\x00", "mp3"), (b"\x00\x00\x00\x20ftypM4A ", "mp4"),
        (b"\x1a\x45\xdf\xa3\x00", "webm"), (b"FORM\x00\x00\x00\x00AIFF", "aiff"),
        (b"name,city\nZo\xc3\xab,Krak\xc3\xb3w\n", None), (b"", None),
    ],
)
def test_sniff_format(head, name):
    assert sniff_format(head) == name


def test_bad_wavs_raise_clear_errors():
    good = write_wav_bytes(tone(100, 0.01))
    tag_at = good.index(b"fmt ") + 8
    compressed = good[:tag_at] + struct.pack("<H", 0x55) + good[tag_at + 2:]
    with pytest.raises(ValueError, match="compressed WAV"):
        read_wav(compressed)
    with pytest.raises(ValueError, match="no data chunk"):
        read_wav(good[:36])
    with pytest.raises(ValueError, match="not a RIFF"):
        read_wav(b"hello")


def test_streaming_wav_with_unset_data_size():
    data = bytearray(write_wav_bytes(tone(440, 0.1)))
    index = data.index(b"data")
    data[index + 4:index + 8] = b"\xff\xff\xff\xff"
    samples, rate, _ = read_wav(bytes(data))
    assert len(samples) == 1600 and rate == 16000


def test_write_wav_round_trip(tmp_path):
    path = write_wav(str(tmp_path / "out.wav"), tone(440, 0.2))
    samples, rate, info = read_wav(path)
    assert info.is_engine_ready() and len(samples) == 3200


def test_audio_input_from_path_bytes_and_samples(tmp_path):
    wav = tmp_path / "Zoë 李雷.wav"
    wav.write_bytes(write_wav_bytes(tone(440, 0.5, 44100), rate=44100, channels=2))
    with AudioInput.from_user(str(wav)) as item:
        assert item.format == "wav" and item.decodable
        assert item.duration == pytest.approx(0.5)
        converted = item.wav16k_path()
        assert converted != item.path  # 44.1 kHz stereo had to be converted
        samples, rate, info = read_wav(converted)
        assert info.is_engine_ready() and len(samples) == 8000
    already = tmp_path / "ready.wav"
    already.write_bytes(write_wav_bytes(tone(440, 0.1)))
    with AudioInput.from_user(already) as item:
        assert item.wav16k_path() == str(already.resolve()) or item.wav16k_path() == item.path
    with AudioInput.from_user(already.read_bytes()) as item:
        assert item.format == "wav" and len(item.samples_16k()) == 1600
    with AudioInput.from_user(tone(440, 0.1, 8000), sample_rate=8000) as item:
        assert item.format == "samples" and len(item.samples_16k()) == 1600
        assert len(item.pcm16_16k()) == 3200
    with AudioInput.from_user([[0.5, -0.5], [0.2, 0.4]], sample_rate=16000) as item:
        assert list(item.samples_16k()) == pytest.approx([0.0, 0.3])


def test_audio_input_rejects_bad_input(tmp_path):
    csv = tmp_path / "u.csv"
    csv.write_text("name,city\nZoë,Kraków\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like an audio file"):
        AudioInput.from_user(str(csv))
    with pytest.raises(FileNotFoundError):
        AudioInput.from_user(str(tmp_path / "missing.wav"))
    with pytest.raises(ValueError, match="folder"):
        AudioInput.from_user(str(tmp_path))
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        AudioInput.from_user(str(empty))
    silent = tmp_path / "silent.wav"
    silent.write_bytes(write_wav_bytes([]))
    with pytest.raises(ValueError, match="no audio samples"):
        AudioInput.from_user(str(silent))
    with pytest.raises(ValueError, match="no samples"):
        AudioInput.from_user([])
    with pytest.raises(ValueError, match="empty"):
        AudioInput.from_user(b"")
    with pytest.raises(ValueError, match="not a recognised"):
        AudioInput.from_user(b"plain text")
    with pytest.raises(TypeError):
        AudioInput.from_user(12345)
    with pytest.raises(ValueError, match="positive"):
        AudioInput.from_user([0.1, 0.2], sample_rate=-5)


def test_non_wav_needs_ffmpeg_for_decoding(tmp_path, world):
    mp3 = tmp_path / "talk.mp3"
    mp3.write_bytes(b"ID3\x04\x00" + b"\0" * 100)
    with AudioInput.from_user(str(mp3)) as item:
        assert item.format == "mp3" and not item.decodable
        with pytest.raises(RuntimeError, match="ffmpeg"):
            item.samples_16k()


def test_non_wav_is_converted_with_ffmpeg_when_present(tmp_path, world):
    target_writer = (
        "import sys, struct, math\n"
        "out = sys.argv[-1]\n"
        "n = 8000\n"
        "pcm = b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * 300 * i / 16000))) for i in range(n))\n"
        "hdr = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 1, 16000, 32000, 2, 16)\n"
        "open(out, 'wb').write(hdr + b'data' + struct.pack('<I', len(pcm)) + pcm)\n"
    )
    world.script("ffmpeg", target_writer)
    mp3 = tmp_path / "talk.mp3"
    mp3.write_bytes(b"ID3\x04\x00" + b"\0" * 100)
    with AudioInput.from_user(str(mp3)) as item:
        samples = item.samples_16k()
        assert len(samples) == 8000
        assert abs(zero_crossing_frequency(samples, 16000) - 300) < 5


LAYOUTS = [
    dict(bits=8), dict(bits=16), dict(bits=24), dict(bits=32), dict(bits=32, float_format=True),
    dict(bits=64, float_format=True), dict(bits=16, channels=2), dict(bits=24, channels=3),
]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_numpy_and_pure_python_decoders_agree(layout, monkeypatch):
    import offline_stt_router._audio as audio

    data = write_wav_bytes(tone(523, 0.05, 16000, level=0.9) + [-1.0, 0.999], **layout)
    fast, _, _ = read_wav(data)
    monkeypatch.setattr(audio, "_numpy", lambda: None)
    slow, _, _ = read_wav(data)
    assert len(fast) == len(slow)
    assert max(abs(a - b) for a, b in zip(fast, slow)) < 1e-6
    samples = list(slow) + [1.0, -1.0, 1.5, float("nan")]
    slow_pcm = to_pcm16(samples)
    monkeypatch.undo()
    assert to_pcm16(samples) == slow_pcm


def test_wav_path_is_only_checked_until_samples_are_needed(tmp_path, monkeypatch):
    import offline_stt_router._audio as audio

    path = tmp_path / "long.wav"
    path.write_bytes(write_wav_bytes(tone(440, 0.5, 44100), rate=44100, channels=2))
    calls = []
    real = audio.decode_wav
    monkeypatch.setattr(audio, "decode_wav", lambda info, payload: calls.append(1) or real(info, payload))
    item = AudioInput.from_user(str(path))
    assert item.decodable and item.duration == pytest.approx(0.5) and item.sample_rate == 44100
    assert calls == []  # nothing decoded yet: faster-whisper would read the file itself
    assert len(item.samples_16k()) == 8000
    assert len(item.samples_16k()) == 8000
    assert calls == [1]  # decoded once, then kept
    item.close()


def test_compressed_wav_is_passed_on_not_refused(tmp_path, world):
    good = write_wav_bytes(tone(100, 0.05))
    tag_at = good.index(b"fmt ") + 8
    path = tmp_path / "adpcm.wav"
    path.write_bytes(good[:tag_at] + struct.pack("<H", 0x11) + good[tag_at + 2:])
    with AudioInput.from_user(str(path)) as item:
        assert item.format == "wav" and not item.decodable
        assert "compressed WAV" in item.note
        with pytest.raises(RuntimeError, match="ffmpeg") as caught:
            item.samples_16k()
        assert "compressed WAV" in str(caught.value)
