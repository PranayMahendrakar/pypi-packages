"""Audio input for the engines: WAV decoding, 16 kHz mono conversion, sniffing.

Only the standard library is required. WAV files (8, 16, 24 and 32-bit
integer, 32 and 64-bit float, and WAVE_FORMAT_EXTENSIBLE) are decoded here,
which is what lets Vosk and whisper.cpp read any WAV and openai-whisper read a
WAV without ffmpeg. Other formats are handed to engines that decode them
themselves, or converted with ffmpeg when it is on PATH.

Decoding is lazy: a WAV path is only checked (its header read) until an engine
actually needs samples, so faster-whisper, which reads files itself, never
waits for this module. numpy is used for speed when it can be imported; the
pure Python path gives the same numbers.

8-bit WAV is unsigned and asymmetric: 0 is the negative rail (-1.0) and 255 is
the positive rail, which is 127/128, not 1.0. The decoder keeps that exactly.
Resampling is linear interpolation: adequate for speech recognition, not for
music.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from array import array
from dataclasses import dataclass
from typing import Any, BinaryIO, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

TARGET_RATE = 16000

_PCM = 0x0001
_FLOAT = 0x0003
_EXTENSIBLE = 0xFFFE

AUDIO_EXTENSIONS = (
    ".wav", ".wave", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".mp4",
    ".aac", ".webm", ".wma", ".aiff", ".aif", ".mka", ".mkv", ".mov", ".amr",
)


def sniff_format(head: bytes) -> Optional[str]:
    """Name the container from its first bytes, or ``None`` if unrecognised."""
    if len(head) >= 12 and head[:4] in (b"RIFF", b"RIFX", b"RF64") and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:16] == b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c":
        return "wma"
    return None


@dataclass
class WavInfo:
    """What a WAV header says, and where its samples are."""

    sample_rate: int
    channels: int
    bits: int
    format_tag: int
    frames: int
    data_offset: int = 0
    data_length: int = 0

    @property
    def is_float(self) -> bool:
        return self.format_tag == _FLOAT

    def is_engine_ready(self) -> bool:
        """16 kHz, mono, 16-bit PCM: what Vosk and whisper.cpp read directly."""
        return (
            self.sample_rate == TARGET_RATE
            and self.channels == 1
            and self.bits == 16
            and self.format_tag == _PCM
        )


def scan_wav(handle: BinaryIO) -> WavInfo:
    """Read a WAV's chunk headers without reading its samples."""
    handle.seek(0, os.SEEK_END)
    file_size = handle.tell()
    handle.seek(0)
    head = handle.read(12)
    if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    fmt: Optional[Tuple[int, int, int, int]] = None
    data: Optional[Tuple[int, int]] = None
    while True:
        header = handle.read(8)
        if len(header) < 8:
            break
        chunk_id = header[:4]
        size = struct.unpack("<I", header[4:8])[0]
        start = handle.tell()
        if chunk_id == b"data":
            if size in (0, 0xFFFFFFFF) or start + size > file_size:
                size = file_size - start  # streaming writers leave the size unset
            data = (start, size)
            if fmt is not None:
                break
        elif chunk_id == b"fmt ":
            body = handle.read(size)
            if len(body) < 16:
                raise ValueError("the WAV fmt chunk is too short")
            tag, channels, rate, _byte_rate, _align, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == _EXTENSIBLE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
            if data is not None:
                break
        handle.seek(start + size + (size & 1))
    if fmt is None:
        raise ValueError("the WAV file has no fmt chunk")
    if data is None:
        raise ValueError("the WAV file has no data chunk")
    tag, channels, rate, bits = fmt
    if channels < 1 or rate < 1:
        raise ValueError("the WAV header is invalid ({0} channels at {1} Hz)".format(channels, rate))
    if tag not in (_PCM, _FLOAT):
        raise ValueError(
            "WAV format tag 0x{0:04x} (compressed WAV) is not supported here; "
            "convert it to PCM".format(tag)
        )
    if tag == _PCM and bits not in (8, 16, 24, 32):
        raise ValueError("{0}-bit integer WAV is not supported".format(bits))
    if tag == _FLOAT and bits not in (32, 64):
        raise ValueError("{0}-bit float WAV is not supported".format(bits))
    frame_bytes = channels * bits // 8
    usable = data[1] - (data[1] % frame_bytes)
    return WavInfo(rate, channels, bits, tag, usable // frame_bytes, data[0], usable)


def _numpy() -> Any:
    try:
        import numpy

        numpy.frombuffer(b"\0\0\0\0", dtype="<f4").astype("float64").mean()
        return numpy
    except Exception:  # absent, or a stand-in that cannot do the work
        return None


def _decode_numpy(np: Any, info: WavInfo, payload: bytes) -> array:
    if info.format_tag == _FLOAT:
        values = np.frombuffer(payload, dtype="<f4" if info.bits == 32 else "<f8").astype(np.float64)
    elif info.bits == 8:
        values = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif info.bits == 16:
        values = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif info.bits == 24:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = np.where(values >= 1 << 23, values - (1 << 24), values).astype(np.float64) / 8388608.0
    else:
        values = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    if info.channels > 1:
        values = values.reshape(-1, info.channels).mean(axis=1)
    return array("f", values.astype(np.float32).tobytes())


def _decode_python(info: WavInfo, payload: bytes) -> array:
    big = sys.byteorder == "big"
    if info.format_tag == _FLOAT:
        values = array("f" if info.bits == 32 else "d")
        values.frombytes(payload)
        if big:
            values.byteswap()
        samples = array("f", values) if info.bits == 64 else values
    elif info.bits == 8:
        samples = array("f", ((b - 128) / 128.0 for b in payload))
    elif info.bits == 16:
        values = array("h")
        values.frombytes(payload)
        if big:
            values.byteswap()
        samples = array("f", (v / 32768.0 for v in values))
    else:
        if info.bits == 24:
            count = len(payload) // 3
            widened = bytearray(count * 4)
            widened[1::4] = payload[0::3]
            widened[2::4] = payload[1::3]
            widened[3::4] = payload[2::3]
            payload = bytes(widened)
        values = array("i")
        values.frombytes(payload)
        if big:
            values.byteswap()
        samples = array("f", (v / 2147483648.0 for v in values))
    if info.channels == 1:
        return samples
    channels = info.channels
    frames = len(samples) // channels
    lanes = [samples[c::channels] for c in range(channels)]
    return array("f", (sum(lane[i] for lane in lanes) / channels for i in range(frames)))


def decode_wav(info: WavInfo, payload: bytes) -> array:
    """Interleaved WAV payload to mono floats in [-1, 1)."""
    np = _numpy()
    if np is not None:
        return _decode_numpy(np, info, payload)
    return _decode_python(info, payload)


def resample(samples: Sequence[float], rate: int, target: int = TARGET_RATE) -> array:
    """Linear-interpolation resampling to ``target`` Hz."""
    if rate == target or len(samples) == 0:
        return array("f", samples)
    count = max(1, int(round(len(samples) * float(target) / rate)))
    np = _numpy()
    if np is not None:
        src = np.asarray(samples, dtype=np.float64)
        positions = np.arange(count, dtype=np.float64) * (float(rate) / target)
        values = np.interp(positions, np.arange(src.size, dtype=np.float64), src)
        return array("f", values.astype(np.float32).tobytes())
    step = float(rate) / target
    last = len(samples) - 1
    out = array("f", bytes(4 * count))
    for i in range(count):
        position = i * step
        left = int(position)
        if left >= last:
            out[i] = samples[last]
            continue
        frac = position - left
        out[i] = samples[left] + (samples[left + 1] - samples[left]) * frac
    return out


def read_wav(path_or_bytes: Union[str, bytes, "os.PathLike[str]"]) -> Tuple[array, int, WavInfo]:
    """Decode a WAV file to mono float samples: ``(samples, rate, info)``."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        handle: BinaryIO = io.BytesIO(bytes(path_or_bytes))
    else:
        handle = open(path_or_bytes, "rb")
    with handle:
        info = scan_wav(handle)
        handle.seek(info.data_offset)
        payload = handle.read(info.data_length)
    return decode_wav(info, payload), info.sample_rate, info


def to_pcm16(samples: Sequence[float]) -> bytes:
    """Floats in [-1, 1] to 16-bit little-endian PCM, clamped at each rail.

    +1.0 clamps to 32767 (the positive rail is one step short), -1.0 is exactly
    -32768, and NaN becomes silence.
    """
    np = _numpy()
    if np is not None:
        values = np.nan_to_num(np.asarray(samples, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(np.round(values * 32768.0), -32768, 32767).astype("<i2").tobytes()
    out = array("h", bytes(2 * len(samples)))
    for i, value in enumerate(samples):
        if value != value:  # NaN
            continue
        scaled = int(round(max(-2.0, min(2.0, value)) * 32768.0))
        out[i] = 32767 if scaled > 32767 else (-32768 if scaled < -32768 else scaled)
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def write_wav(path: str, samples: Sequence[float], rate: int = TARGET_RATE) -> str:
    """Write mono 16-bit PCM. Returns ``path``."""
    pcm = to_pcm16(samples)
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, _PCM, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(pcm)
    return path


class AudioInput:
    """What the caller passed to ``transcribe()``, checked and made convertible.

    Accepts a path, the bytes of an audio file, or a sequence of float samples
    (a list or a numpy array) together with ``sample_rate``.
    """

    def __init__(self) -> None:
        self.path: Optional[str] = None
        self.format: Optional[str] = None
        self.original: Any = None
        self.note: Optional[str] = None
        self._samples: Optional[array] = None
        self._rate: Optional[int] = None
        self._info: Optional[WavInfo] = None
        self._tmp: Optional[tempfile.TemporaryDirectory] = None

    @classmethod
    def from_user(cls, audio: Any, sample_rate: Optional[int] = None) -> "AudioInput":
        item = cls()
        item.original = audio
        if isinstance(audio, (str, os.PathLike)):
            path = os.fspath(audio)
            if not os.path.exists(path):
                raise FileNotFoundError("audio file not found: {0}".format(path))
            if os.path.isdir(path):
                raise ValueError("{0} is a folder, not an audio file".format(path))
            with open(path, "rb") as handle:
                head = handle.read(64)
            if not head:
                raise ValueError("{0} is empty".format(path))
            item.path = os.path.abspath(path)
            item.format = sniff_format(head)
            if item.format is None:
                extension = os.path.splitext(path)[1].lower()
                if extension not in AUDIO_EXTENSIONS:
                    raise ValueError(
                        "{0} does not look like an audio file (expected WAV, MP3, FLAC, "
                        "OGG, M4A/MP4, WEBM or similar)".format(path)
                    )
                item.format = extension.lstrip(".")
            if item.format == "wav":
                item._check_wav()
            return item
        if isinstance(audio, (bytes, bytearray, memoryview)):
            data = bytes(audio)
            if not data:
                raise ValueError("the audio bytes are empty")
            item.format = sniff_format(data[:64])
            if item.format is None:
                raise ValueError("the audio bytes are not a recognised audio file")
            item.path = os.path.join(item._tempdir(), "input." + item.format)
            with open(item.path, "wb") as handle:
                handle.write(data)
            if item.format == "wav":
                item._check_wav()
            return item
        samples = _as_samples(audio)
        if samples is None:
            raise TypeError(
                "audio must be a file path, the bytes of an audio file, or a sequence "
                "of float samples; got {0}".format(type(audio).__name__)
            )
        if len(samples) == 0:
            raise ValueError("the audio has no samples")
        rate = int(sample_rate or TARGET_RATE)
        if rate <= 0:
            raise ValueError("sample_rate must be positive")
        if sample_rate is None:
            logger.info("no sample_rate given for raw samples; assuming %d Hz", TARGET_RATE)
        item.format = "samples"
        item._samples = samples
        item._rate = rate
        return item

    def _tempdir(self) -> str:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="offline-stt-router-")
        return self._tmp.name

    def _check_wav(self) -> None:
        """Read the header only. An empty WAV is an error; an exotic one is a note."""
        assert self.path is not None
        try:
            with open(self.path, "rb") as handle:
                info = scan_wav(handle)
        except ValueError as exc:
            # engines that decode files themselves (faster-whisper, ffmpeg) may still cope
            self.note = "{0}: {1}".format(self.path, exc)
            return
        if info.frames == 0:
            raise ValueError("{0} contains no audio samples".format(self.path))
        self._info = info

    def _decoded(self) -> Tuple[array, int]:
        if self._samples is None:
            if self._info is not None and self.path is not None:
                self._samples, self._rate, _info = read_wav(self.path)
            else:
                self._convert_with_ffmpeg()
        assert self._samples is not None and self._rate is not None
        return self._samples, self._rate

    @property
    def decodable(self) -> bool:
        """True when this package can produce samples without ffmpeg."""
        return self._samples is not None or self._info is not None

    @property
    def sample_rate(self) -> Optional[int]:
        if self._rate is not None:
            return self._rate
        return self._info.sample_rate if self._info is not None else None

    @property
    def duration(self) -> Optional[float]:
        if self._samples is not None and self._rate:
            return len(self._samples) / float(self._rate)
        if self._info is not None:
            return self._info.frames / float(self._info.sample_rate)
        return None

    def samples_16k(self) -> array:
        """Mono float samples at 16 kHz (converting with ffmpeg if needed)."""
        samples, rate = self._decoded()
        return resample(samples, rate, TARGET_RATE)

    def pcm16_16k(self) -> bytes:
        return to_pcm16(self.samples_16k())

    def wav16k_path(self) -> str:
        """A 16 kHz mono 16-bit WAV of this audio (the original if it already is one)."""
        if self.path and self._info is not None and self._info.is_engine_ready():
            return self.path
        target = os.path.join(self._tempdir(), "input-16k.wav")
        if not os.path.exists(target):
            write_wav(target, self.samples_16k(), TARGET_RATE)
        return target

    def _convert_with_ffmpeg(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None or self.path is None:
            why = " ({0})".format(self.note) if self.note else ""
            raise RuntimeError(
                "this engine needs decoded audio and {0} is {1}{2}; convert it to 16-bit "
                "WAV or put ffmpeg on PATH".format(self.path or "the input", (self.format or "?").upper(), why)
            )
        target = os.path.join(self._tempdir(), "ffmpeg-16k.wav")
        command = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", self.path,
                   "-ac", "1", "-ar", str(TARGET_RATE), "-c:a", "pcm_s16le", target]
        done = subprocess.run(command, capture_output=True)
        if done.returncode != 0 or not os.path.exists(target):
            message = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError("ffmpeg could not decode {0}: {1}".format(
                self.path, message[-1] if message else "exit code {0}".format(done.returncode)))
        self._samples, self._rate, _info = read_wav(target)
        if len(self._samples) == 0:
            raise ValueError("{0} contains no audio samples".format(self.path))

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __enter__(self) -> "AudioInput":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _as_samples(audio: Any) -> Optional[array]:
    """A list, tuple, array or numpy array of numbers as mono floats."""
    if hasattr(audio, "tolist") and not isinstance(audio, (list, tuple, array)):
        try:
            audio = audio.tolist()
        except Exception:
            return None
    if isinstance(audio, array):
        return array("f", audio)
    if not isinstance(audio, (list, tuple)):
        return None
    if audio and isinstance(audio[0], (list, tuple)):
        frames: List[float] = []
        for frame in audio:
            if not frame:
                continue
            frames.append(sum(float(v) for v in frame) / len(frame))
        return array("f", frames)
    try:
        return array("f", (float(v) for v in audio))
    except (TypeError, ValueError):
        return None
