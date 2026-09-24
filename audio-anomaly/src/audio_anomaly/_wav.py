"""Read a ``.wav`` file with nothing but the standard library and numpy.

The standard library's :mod:`wave` module refuses anything that is not integer
PCM, which rules out the 32-bit float files most recorders and editors write
today. Rather than push that problem onto the caller as an install of
``soundfile``, this module walks the RIFF chunks itself and decodes the four
integer widths plus IEEE float, including files tagged
``WAVE_FORMAT_EXTENSIBLE``.

Samples come back as ``float64`` in roughly ``[-1.0, 1.0]``, shaped
``(n_samples,)`` for mono or ``(n_samples, n_channels)`` otherwise.
"""

from __future__ import annotations

import struct
from typing import BinaryIO, Tuple

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


def _read_exactly(stream: BinaryIO, count: int, what: str) -> bytes:
    """Read ``count`` bytes or raise a message naming what was being read."""
    data = stream.read(count)
    if len(data) != count:
        raise ValueError(
            "the WAV file ends in the middle of its {}; it is truncated".format(what)
        )
    return data


def _decode_pcm(raw: bytes, width: int) -> np.ndarray:
    """Turn packed integer PCM bytes into float64 in ``[-1.0, 1.0]``."""
    if width == 1:
        # 8-bit WAV is unsigned with 128 as silence, unlike every wider width.
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (samples - 128.0) / 128.0
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        return samples / 32768.0
    if width == 3:
        # No numpy dtype is 3 bytes wide, so widen each sample to 4 bytes and
        # carry the sign in the byte we add.
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        widened = np.zeros((packed.shape[0], 4), dtype=np.uint8)
        widened[:, 1:] = packed
        samples = widened.view("<i4").reshape(-1).astype(np.float64)
        return samples / 2147483648.0
    if width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        return samples / 2147483648.0
    raise ValueError(
        "this WAV file stores {}-byte integer samples, which is not a width the "
        "format defines; 1, 2, 3 and 4 are supported".format(width)
    )


def _decode_float(raw: bytes, width: int) -> np.ndarray:
    """Turn packed IEEE float bytes into float64."""
    if width == 4:
        return np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if width == 8:
        return np.frombuffer(raw, dtype="<f8").astype(np.float64)
    raise ValueError(
        "this WAV file stores {}-byte float samples; only 32-bit and 64-bit "
        "float WAV is defined".format(width)
    )


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read ``path`` and return ``(samples, sample_rate)``.

    Args:
        path: a ``.wav`` file. Integer PCM at 8, 16, 24 or 32 bits and IEEE
            float at 32 or 64 bits are all read, including files tagged
            ``WAVE_FORMAT_EXTENSIBLE``.

    Returns:
        ``(samples, sample_rate)`` where samples is float64 shaped
        ``(n_samples,)`` for mono or ``(n_samples, n_channels)`` otherwise.

    Raises:
        ValueError: the file is not RIFF/WAVE, is truncated, or stores its
            samples in a compressed encoding this reader cannot decode.
    """
    with open(path, "rb") as stream:
        header = _read_exactly(stream, 12, "RIFF header")
        riff, _size, wave_tag = struct.unpack("<4sI4s", header)
        if riff not in (b"RIFF", b"RIFX"):
            raise ValueError(
                "{!r} does not start with a RIFF header, so it is not a WAV "
                "file".format(path)
            )
        if riff == b"RIFX":
            raise ValueError(
                "{!r} is big-endian RIFX audio, which this reader does not "
                "decode; convert it to little-endian WAV".format(path)
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
                data = _read_exactly(stream, chunk_size, "sample data")
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
        raise ValueError(
            "{!r} claims a sample rate of {} Hz".format(path, sample_rate)
        )

    width = bits // 8
    if width < 1:
        raise ValueError("{!r} claims {} bits per sample".format(path, bits))

    if fmt_tag == WAVE_FORMAT_PCM:
        samples = _decode_pcm(data[: len(data) - len(data) % width], width)
    elif fmt_tag == WAVE_FORMAT_IEEE_FLOAT:
        samples = _decode_float(data[: len(data) - len(data) % width], width)
    else:
        known = _FORMAT_NAMES.get(fmt_tag, "format 0x{:04X}".format(fmt_tag))
        raise ValueError(
            "{!r} stores its audio as {}, which is compressed or otherwise not "
            "plain PCM; save it as 16-bit PCM or 32-bit float WAV "
            "first".format(path, known)
        )

    if channels > 1:
        usable = samples.shape[0] - samples.shape[0] % channels
        samples = samples[:usable].reshape(-1, channels)
    return samples, int(sample_rate)
