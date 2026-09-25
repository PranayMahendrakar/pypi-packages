"""Read a ``.wav`` file with the standard library and numpy only.

Integer PCM - by far the common case - is read with the standard library's
:mod:`wave` module. ``wave`` refuses IEEE float files (and, before Python 3.12,
files tagged ``WAVE_FORMAT_EXTENSIBLE``), which is what many recorders and
editors write today, so for those files alone a small RIFF chunk reader takes
over rather than pushing an install of ``soundfile`` onto the caller.

Samples come back as ``float64`` shaped ``(n_samples, n_channels)``, together
with what is needed to judge clipping honestly: the bit depth, and whether the
samples were integers. 8-bit WAV is unsigned with 128 as silence, so its
positive rail is 127/128, not 1.0.
"""

from __future__ import annotations

import os
import struct
import wave
from dataclasses import dataclass, field
from typing import BinaryIO, List

import numpy as np

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_FORMAT_NAMES = {
    WAVE_FORMAT_PCM: "integer PCM",
    WAVE_FORMAT_IEEE_FLOAT: "IEEE float",
    WAVE_FORMAT_EXTENSIBLE: "extensible",
}

_MAX_HEADER_CHUNKS = 64
"""Give up after this many chunks rather than loop forever on a damaged file."""

_UNKNOWN_SIZE = 0xFFFFFFFF
"""The data size a streaming recorder writes when it does not know the length yet."""


@dataclass
class WavData:
    """What :func:`read_wav` found in a file.

    Attributes:
        samples: float64 ``(n_samples, n_channels)``. Integer PCM is scaled so
            the negative rail is exactly -1.0; float WAV is passed through.
        sample_rate: samples per second.
        bits: bits per sample as stored in the file.
        is_float: True for IEEE float WAV, False for integer PCM.
        notes: anything about how the file was read that the report should say.
        warnings: damage found in the file, such as a recording cut short.
    """

    samples: np.ndarray
    sample_rate: int
    bits: int
    is_float: bool
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: How many frames the header promises; -1 when it does not say.
    declared_frames: int = -1


def _read_exactly(stream: BinaryIO, count: int, what: str) -> bytes:
    """Read ``count`` bytes or raise a message naming what was being read."""
    data = stream.read(count)
    if len(data) != count:
        raise ValueError(
            "the WAV file ends in the middle of its {}; it is truncated".format(what)
        )
    return data


def decode_pcm(raw: bytes, width: int) -> np.ndarray:
    """Turn packed little-endian integer PCM bytes into float64.

    The scale puts the negative rail at exactly -1.0, so the positive rail is
    ``(2**(bits-1) - 1) / 2**(bits-1)``: 127/128 for 8-bit, 32767/32768 for
    16-bit.
    """
    if width == 1:
        # 8-bit WAV is unsigned with 128 as silence, unlike every wider width.
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (samples - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if width == 3:
        # No numpy dtype is 3 bytes wide, so widen each sample to 4 bytes and
        # carry the sign in the byte that is added at the top.
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        widened = np.zeros((packed.shape[0], 4), dtype=np.uint8)
        widened[:, 1:] = packed
        return widened.view("<i4").reshape(-1).astype(np.float64) / 2147483648.0
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(
        "this WAV file stores {}-byte integer samples, which is not a width the "
        "format defines; 1, 2, 3 and 4 are supported".format(width)
    )


def _decode_float(raw: bytes, width: int) -> np.ndarray:
    """Turn packed little-endian IEEE float bytes into float64."""
    if width == 4:
        return np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if width == 8:
        return np.frombuffer(raw, dtype="<f8").astype(np.float64)
    raise ValueError(
        "this WAV file stores {}-byte float samples; only 32-bit and 64-bit "
        "float WAV is defined".format(width)
    )


def _truncation_warning(path: str, declared_frames: int, present_frames: int, rate: int) -> List[str]:
    """A warning when the header promises more audio than the file holds."""
    if rate < 1 or present_frames >= declared_frames:
        return []
    return [
        "{!r} declares {:.2f} s of audio in its header but holds only {:.2f} s of "
        "samples; the file was cut short (an interrupted recording or copy), and only "
        "the part that is there was analysed".format(
            path, declared_frames / float(rate), present_frames / float(rate)
        )
    ]


def _shape(samples: np.ndarray, channels: int) -> np.ndarray:
    """Reshape interleaved samples to ``(n_samples, channels)``."""
    usable = samples.shape[0] - samples.shape[0] % channels
    return samples[:usable].reshape(-1, channels)


def _read_with_wave(path: str) -> WavData:
    """The standard-library path: integer PCM through :mod:`wave`."""
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        declared = handle.getnframes()
        raw = handle.readframes(declared)
    if channels < 1:
        raise ValueError("{!r} claims {} channels".format(path, channels))
    if rate < 1:
        raise ValueError("{!r} claims a sample rate of {} Hz".format(path, rate))
    if width < 1:
        raise ValueError("{!r} claims {}-byte samples".format(path, width))
    samples = decode_pcm(raw[: len(raw) - len(raw) % width], width)
    shaped = _shape(samples, channels)
    return WavData(
        shaped,
        int(rate),
        8 * width,
        False,
        warnings=_truncation_warning(path, int(declared), int(shaped.shape[0]), int(rate)),
        declared_frames=int(declared),
    )


def _read_riff(path: str) -> WavData:
    """The fallback path: walk the RIFF chunks for the files ``wave`` refuses."""
    with open(path, "rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        header = _read_exactly(stream, 12, "RIFF header")
        riff, _size, wave_tag = struct.unpack("<4sI4s", header)
        if riff == b"RIFX":
            raise ValueError(
                "{!r} is big-endian RIFX audio, which this reader does not "
                "decode; convert it to little-endian WAV".format(path)
            )
        if riff != b"RIFF":
            raise ValueError(
                "{!r} does not start with a RIFF header, so it is not a WAV "
                "file".format(path)
            )
        if wave_tag != b"WAVE":
            raise ValueError(
                "{!r} is a RIFF file but not a WAVE one (it is tagged "
                "{!r})".format(path, wave_tag.decode("ascii", "replace"))
            )

        fmt_tag = None
        channels = 0
        sample_rate = 0
        bits = 0
        data = b""
        declared_bytes = 0
        seen_data = False

        for _ in range(_MAX_HEADER_CHUNKS):
            head = stream.read(8)
            if len(head) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", head)
            if chunk_id == b"fmt ":
                body = _read_exactly(stream, chunk_size, "format chunk")
                if len(body) < 16:
                    raise ValueError(
                        "{!r} has a format chunk of only {} bytes; 16 is the "
                        "minimum".format(path, len(body))
                    )
                fmt_tag, channels, sample_rate, _bps, _align, bits = struct.unpack(
                    "<HHIIHH", body[:16]
                )
                if fmt_tag == WAVE_FORMAT_EXTENSIBLE and len(body) >= 40:
                    # The real encoding hides in the first two bytes of the
                    # sub-format GUID.
                    (fmt_tag,) = struct.unpack("<H", body[24:26])
            elif chunk_id == b"data":
                # A short read keeps what exists; never ask for more than the
                # file holds, since a damaged size field can claim gigabytes.
                data = stream.read(min(chunk_size, max(0, file_size - stream.tell())))
                declared_bytes = len(data) if chunk_size == _UNKNOWN_SIZE else chunk_size
                seen_data = True
            else:
                stream.seek(chunk_size, 1)
            if chunk_size % 2:  # RIFF pads odd-sized chunks to an even boundary.
                stream.seek(1, 1)
            if seen_data and fmt_tag is not None:
                break

    if fmt_tag is None:
        raise ValueError("{!r} has no format chunk, so it is unreadable".format(path))
    if not seen_data:
        raise ValueError("{!r} has no data chunk, so it holds no audio".format(path))
    if channels < 1:
        raise ValueError("{!r} claims {} channels".format(path, channels))
    if sample_rate < 1:
        raise ValueError("{!r} claims a sample rate of {} Hz".format(path, sample_rate))
    if fmt_tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        # Checked before the width: compressed formats often declare 0 bits.
        known = _FORMAT_NAMES.get(fmt_tag, "format 0x{:04X}".format(fmt_tag))
        raise ValueError(
            "{!r} stores its audio as {}, which is compressed or otherwise not "
            "plain PCM; save it as 16-bit PCM or 32-bit float WAV "
            "first".format(path, known)
        )
    width = bits // 8
    if width < 1:
        raise ValueError("{!r} claims {} bits per sample".format(path, bits))

    usable = data[: len(data) - len(data) % width]
    if fmt_tag == WAVE_FORMAT_PCM:
        samples = decode_pcm(usable, width)
        is_float = False
    else:
        samples = _decode_float(usable, width)
        is_float = True
    shaped = _shape(samples, channels)
    frame_bytes = width * channels
    return WavData(
        shaped,
        int(sample_rate),
        int(bits),
        is_float,
        warnings=_truncation_warning(
            path, declared_bytes // frame_bytes, int(shaped.shape[0]), int(sample_rate)
        ),
    )


def read_wav(path: str) -> WavData:
    """Read ``path`` and return a :class:`WavData`.

    Args:
        path: a ``.wav`` file. Integer PCM at 8, 16, 24 or 32 bits is read by
            the standard library's :mod:`wave`; IEEE float at 32 or 64 bits and
            ``WAVE_FORMAT_EXTENSIBLE`` files, which ``wave`` refuses, are read by
            a small RIFF reader.

    Raises:
        ValueError: the file is not RIFF/WAVE, is damaged, or stores its
            samples in a compressed encoding.
        OSError: the file cannot be opened.
    """
    try:
        data = _read_with_wave(path)
    except wave.Error:
        # "unknown format: 3" (float) or 65534 (extensible) - read it directly.
        # A file that is not WAV at all gets its own message from _read_riff.
        return _read_riff(path)
    except EOFError:
        raise ValueError(
            "{!r} ends before its WAV header is complete; it is truncated or not "
            "a WAV file".format(path)
        ) from None
    except (RuntimeError, struct.error, OverflowError):
        # The chunk reader inside ``wave`` raises a bare RuntimeError when a
        # chunk claims more bytes than the RIFF container around it, which is
        # either real damage or just a wrong RIFF size field (some streaming
        # recorders write one). The RIFF reader ignores that field, so it can
        # still read the second kind; the first becomes a ValueError here.
        try:
            data = _read_riff(path)
        except (ValueError, struct.error, OverflowError, MemoryError):
            raise ValueError(
                "{!r} is damaged: its chunks do not fit inside the file, so it "
                "cannot be read as WAV".format(path)
            ) from None
        data.notes.append(_riff_size_note(path))
        return data
    if 0 <= data.samples.shape[0] < data.declared_frames:
        # ``wave`` also stops, silently, where a too-small RIFF size says the
        # file ends. Only a file that really is cut short keeps the warning.
        try:
            whole = _read_riff(path)
        except (ValueError, struct.error, OverflowError, MemoryError):
            return data
        if whole.samples.shape[0] > data.samples.shape[0]:
            whole.notes.append(_riff_size_note(path))
            return whole
    return data


def _riff_size_note(path: str) -> str:
    return (
        "{!r} has a RIFF header whose size does not match its contents (some "
        "streaming recorders leave it so), so it was read chunk by chunk "
        "instead".format(path)
    )
