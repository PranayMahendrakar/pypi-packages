"""Read and write ``.wav`` files with the standard library and numpy only.

A small RIFF chunk reader handles every layout that recorders and editors
commonly write: integer PCM at 8, 16, 24 and 32 bits, IEEE float at 32 and 64
bits, and the ``WAVE_FORMAT_EXTENSIBLE`` wrapper around either. Samples come
back as ``float64`` shaped ``(n_samples, n_channels)``.

Integer PCM is scaled so the negative rail is exactly -1.0. That makes the
positive rail ``(2**(bits-1) - 1) / 2**(bits-1)``: 127/128 for 8-bit, which is
unsigned with 128 as silence, and 32767/32768 for 16-bit. Clipping has to be
measured against each rail separately for that reason; see :func:`rails`.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, List, Tuple, Union

import numpy as np

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_MAX_CHUNKS = 256
"""Give up after this many chunks rather than loop forever on a damaged file."""

_UNKNOWN_SIZE = 0xFFFFFFFF
"""The data size a streaming recorder writes before it knows the length."""

PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class WavData:
    """What :func:`read_wav` found in a file.

    Attributes:
        samples: float64 ``(n_samples, n_channels)``.
        sample_rate: samples per second.
        bits: bits per sample as stored in the file.
        is_float: True for IEEE float WAV, False for integer PCM.
        warnings: damage found in the file, such as a recording cut short.
    """

    samples: np.ndarray
    sample_rate: int
    bits: int
    is_float: bool
    warnings: List[str] = field(default_factory=list)


def rails(bits: int, is_float: bool) -> Tuple[float, float]:
    """The (negative, positive) full-scale values a decoded sample can reach.

    Float audio is nominally bounded by -1.0 and 1.0. Integer PCM decoded by
    this module reaches exactly -1.0 on the negative side but stops one step
    short of 1.0 on the positive side: 127/128 for 8-bit.
    """
    if is_float or bits <= 0:
        return -1.0, 1.0
    top = float(2 ** (bits - 1))
    return -1.0, (top - 1.0) / top


def _read_exactly(stream: BinaryIO, count: int, what: str) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("the WAV file ends in the middle of its {}; it is truncated".format(what))
    return data


def decode_pcm(raw: bytes, width: int) -> np.ndarray:
    """Turn packed little-endian integer PCM bytes into float64 (negative rail -1.0)."""
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        widened = np.zeros((packed.shape[0], 4), dtype=np.uint8)
        widened[:, 1:] = packed
        return widened.view("<i4").reshape(-1).astype(np.float64) / 2147483648.0
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(
        "this WAV file stores {}-byte integer samples; 1, 2, 3 and 4 bytes are "
        "supported".format(width)
    )


def _decode_float(raw: bytes, width: int) -> np.ndarray:
    if width == 4:
        return np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if width == 8:
        return np.frombuffer(raw, dtype="<f8").astype(np.float64)
    raise ValueError(
        "this WAV file stores {}-byte float samples; only 32-bit and 64-bit float "
        "WAV is defined".format(width)
    )


def read_wav(path: PathLike) -> WavData:
    """Read a ``.wav`` file into float64 samples shaped ``(n_samples, n_channels)``.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file is not a WAV file, or uses an encoding (such as
            compressed ADPCM or MP3-in-WAV) that this reader does not decode.
    """
    name = os.fspath(path)
    if not os.path.isfile(name):
        raise FileNotFoundError("no such file: {!r}".format(name))
    warnings: List[str] = []
    with open(name, "rb") as stream:
        header = stream.read(12)
        if len(header) < 12 or header[0:4] not in (b"RIFF", b"RIFX", b"RF64") or header[8:12] != b"WAVE":
            raise ValueError(
                "{!r} is not a WAV file (no RIFF/WAVE header). Only .wav can be read "
                "directly; decode other formats yourself and pass (samples, sample_rate)".format(name)
            )
        if header[0:4] != b"RIFF":
            raise ValueError(
                "{!r} is a {} file; only little-endian RIFF WAV is supported".format(
                    name, header[0:4].decode("ascii")
                )
            )
        fmt = None
        data = None
        declared = 0
        for _ in range(_MAX_CHUNKS):
            chunk_header = stream.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[0:4]
            (size,) = struct.unpack("<I", chunk_header[4:8])
            if chunk_id == b"fmt ":
                fmt = _read_exactly(stream, size, "format chunk")
                if size % 2:
                    stream.read(1)
            elif chunk_id == b"data":
                declared = size
                data = stream.read() if size == _UNKNOWN_SIZE else stream.read(size)
                break
            else:
                stream.seek(size + (size % 2), os.SEEK_CUR)
        if fmt is None or len(fmt) < 16:
            raise ValueError("{!r} has no usable format chunk; it is damaged".format(name))
        if data is None:
            raise ValueError("{!r} has no data chunk; it holds no audio".format(name))

    tag, channels, rate, _byte_rate, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 26:
            raise ValueError("{!r} is an extensible WAV with a short format chunk".format(name))
        tag = struct.unpack("<H", fmt[24:26])[0]
    if tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise ValueError(
            "{!r} uses WAV encoding 0x{:04X}; only integer PCM and IEEE float are "
            "supported. Convert it to PCM WAV first".format(name, tag)
        )
    if channels < 1:
        raise ValueError("{!r} claims {} channels".format(name, channels))
    if rate < 1:
        raise ValueError("{!r} claims a sample rate of {} Hz".format(name, rate))
    width = block_align // channels if block_align else 0
    if width < 1:
        width = (bits + 7) // 8
    is_float = tag == WAVE_FORMAT_IEEE_FLOAT

    frame_bytes = width * channels
    usable = len(data) - len(data) % frame_bytes
    if declared != _UNKNOWN_SIZE and len(data) < declared:
        warnings.append(
            "{!r} declares {:.2f} s of audio but holds only {:.2f} s; the file was cut "
            "short and only the part that is there was analysed".format(
                name, declared / float(frame_bytes * rate), usable / float(frame_bytes * rate)
            )
        )
    raw = data[:usable]
    flat = _decode_float(raw, width) if is_float else decode_pcm(raw, width)
    return WavData(
        samples=flat.reshape(-1, channels),
        sample_rate=int(rate),
        bits=int(bits) if bits else width * 8,
        is_float=is_float,
        warnings=warnings,
    )


def write_wav(path: PathLike, samples: np.ndarray, sample_rate: int, bits: int = 16) -> None:
    """Write float samples in [-1, 1] to ``path`` as a WAV file.

    ``samples`` is 1-D (mono) or ``(n_samples, n_channels)``. ``bits`` is 8, 16,
    24 or 32 for integer PCM, or ``-32`` / ``-64`` for IEEE float. Values
    outside the representable range are clipped to the rails.
    """
    data = np.array(samples, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("samples must be 1-D or (n_samples, n_channels)")
    channels = data.shape[1]
    if bits in (-32, -64):
        width = abs(bits) // 8
        tag = WAVE_FORMAT_IEEE_FLOAT
        payload = data.astype("<f4" if width == 4 else "<f8").tobytes()
    elif bits in (8, 16, 24, 32):
        width = bits // 8
        tag = WAVE_FORMAT_PCM
        top = 2 ** (bits - 1)
        ints = np.clip(np.round(data * top), -top, top - 1).astype(np.int64)
        if bits == 8:
            payload = (ints + 128).astype(np.uint8).tobytes()
        elif bits == 16:
            payload = ints.astype("<i2").tobytes()
        elif bits == 32:
            payload = ints.astype("<i4").tobytes()
        else:
            wide = ints.astype("<i4").reshape(-1, 1).view(np.uint8).reshape(-1, 4)
            payload = np.ascontiguousarray(wide[:, :3]).tobytes()
    else:
        raise ValueError("bits must be 8, 16, 24, 32, -32 or -64, not {!r}".format(bits))
    block_align = width * channels
    fmt = struct.pack(
        "<HHIIHH", tag, channels, int(sample_rate), int(sample_rate) * block_align, block_align, width * 8
    )
    pad = b"\x00" if len(payload) % 2 else b""
    riff_size = 4 + (8 + len(fmt)) + (8 + len(payload) + len(pad))
    with open(os.fspath(path), "wb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE")
        handle.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        handle.write(b"data" + struct.pack("<I", len(payload)) + payload + pad)
