"""Turn whatever the caller hands us into mono float samples plus a sample rate.

Three input shapes are accepted everywhere in this package: a path to a ``.wav``
file, a numpy array, or a ``(samples, sample_rate)`` pair. The caller's array is
never written to - every path here ends in a fresh ``float64`` buffer that this
package owns.

The WAV reader is a small RIFF parser rather than :mod:`wave`, because the
stdlib module refuses IEEE-float and WAVE_FORMAT_EXTENSIBLE files, which is what
a lot of recorders and editors actually write.
"""

from __future__ import annotations

import logging
import os
import struct
import wave
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PCM = 0x0001
_IEEE_FLOAT = 0x0003
_EXTENSIBLE = 0xFFFE

MAX_CHANNELS = 32
"""Above this, a 2-D array's short axis is not plausibly a channel axis."""


class _Fmt:
    """The parsed contents of a WAV format chunk."""

    __slots__ = ("code", "channels", "sample_rate", "bits")

    def __init__(self, code: int, channels: int, sample_rate: int, bits: int) -> None:
        self.code = code
        self.channels = channels
        self.sample_rate = sample_rate
        self.bits = bits


def _parse_fmt(body: bytes, path: str) -> _Fmt:
    """Read a WAV format chunk body, resolving WAVE_FORMAT_EXTENSIBLE."""
    if len(body) < 16:
        raise ValueError(
            "{}: the WAV format chunk is {} bytes, too short to describe the "
            "audio".format(path, len(body))
        )
    code, channels, sample_rate, _byte_rate, _align, bits = struct.unpack_from(
        "<HHIIHH", body, 0
    )
    if code == _EXTENSIBLE:
        if len(body) < 40:
            raise ValueError(
                "{}: the WAV file says WAVE_FORMAT_EXTENSIBLE but the format "
                "chunk has no sub-format GUID".format(path)
            )
        code = struct.unpack_from("<H", body, 24)[0]
    return _Fmt(code, channels, sample_rate, bits)


def _decode_pcm(raw: bytes, fmt: _Fmt, path: str) -> np.ndarray:
    """Turn the data chunk bytes into float samples in roughly -1 to 1."""
    if fmt.code == _IEEE_FLOAT:
        if fmt.bits == 32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if fmt.bits == 64:
            return np.frombuffer(raw, dtype="<f8").astype(np.float64)
        raise ValueError(
            "{}: {}-bit float WAV is not something this reader knows; 32 and 64 "
            "bit are".format(path, fmt.bits)
        )
    if fmt.code != _PCM:
        raise ValueError(
            "{}: WAV format code {} is neither PCM (1) nor IEEE float (3); "
            "re-export the file as plain PCM WAV".format(path, fmt.code)
        )
    if fmt.bits == 8:
        # 8-bit WAV is unsigned with a 128 offset; every other width is signed.
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if fmt.bits == 16:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if fmt.bits == 24:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        value = (value ^ 0x800000) - 0x800000  # sign-extend 24 bits into int32
        return value.astype(np.float64) / 8388608.0
    if fmt.bits == 32:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(
        "{}: {}-bit PCM WAV is not something this reader knows; 8, 16, 24 and 32 "
        "bit are".format(path, fmt.bits)
    )


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a ``.wav`` file into ``(samples, sample_rate)``.

    Args:
        path: Path to a RIFF/WAVE file.

    Returns:
        A 2-D ``(n_samples, n_channels)`` float64 array and the sample rate.

    Raises:
        ValueError: The file is not a WAV file, or carries a codec this reader
            does not decode (anything but PCM and IEEE float).
    """
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(
                "{}: this is not a RIFF/WAVE file (it does not start with the "
                "RIFF...WAVE marker)".format(path)
            )
        fmt = None  # type: Optional[_Fmt]
        raw = b""
        while True:
            head = handle.read(8)
            if len(head) < 8:
                break
            chunk_id = head[0:4]
            size = struct.unpack("<I", head[4:8])[0]
            if chunk_id == b"fmt ":
                fmt = _parse_fmt(handle.read(size), path)
            elif chunk_id == b"data":
                # A streamed file can declare a bogus size; trust what is there.
                raw = handle.read(size)
            else:
                handle.seek(size, os.SEEK_CUR)
            if size % 2:  # RIFF chunks are padded to an even length
                handle.seek(1, os.SEEK_CUR)
    if fmt is None:
        raise ValueError("{}: the WAV file has no format chunk".format(path))
    if fmt.channels < 1:
        raise ValueError(
            "{}: the WAV file declares {} channels".format(path, fmt.channels)
        )
    if fmt.sample_rate <= 0:
        raise ValueError(
            "{}: the WAV file declares a sample rate of {}".format(
                path, fmt.sample_rate
            )
        )
    frame_bytes = fmt.channels * max(fmt.bits // 8, 1)
    usable = len(raw) - (len(raw) % frame_bytes)
    if usable != len(raw):
        logger.warning(
            "%s: the data chunk ends mid-frame; %d trailing bytes ignored",
            path,
            len(raw) - usable,
        )
    flat = _decode_pcm(raw[:usable], fmt, path)
    return flat.reshape(-1, fmt.channels), int(fmt.sample_rate)


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float samples to a 16-bit PCM ``.wav`` file.

    Values outside -1 to 1 are clipped rather than allowed to wrap round.

    Args:
        path: Destination path.
        samples: 1-D float array.
        sample_rate: Frames per second.
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def _scale_integers(array: np.ndarray) -> np.ndarray:
    """Map an integer array onto -1 to 1 using the full range of its dtype."""
    info = np.iinfo(array.dtype)
    if info.min == 0:  # unsigned: silence sits at the middle of the range
        middle = (float(info.max) + 1.0) / 2.0
        return (array.astype(np.float64) - middle) / middle
    return array.astype(np.float64) / float(-info.min)


def _to_mono(array: np.ndarray, notes: List[str], source: str) -> np.ndarray:
    """Collapse a multi-channel array to one channel by averaging."""
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(
            "audio has {} dimensions; a recording is 1-D (mono) or 2-D "
            "(samples x channels)".format(array.ndim)
        )
    rows, cols = array.shape
    if cols == 1:
        return array[:, 0]
    if rows == 1:
        return array[0, :]
    # Whichever axis is short is the channel axis: a recording has far more
    # samples than channels.
    if cols <= rows and cols <= MAX_CHANNELS:
        channels, mono = cols, array.mean(axis=1)
    elif rows <= MAX_CHANNELS:
        channels, mono = rows, array.mean(axis=0)
    else:
        raise ValueError(
            "audio is a {}x{} array and neither axis is short enough to be "
            "channels (at most {}); pass mono samples".format(rows, cols, MAX_CHANNELS)
        )
    notes.append(
        "{}: {} channels were mixed down to mono by averaging".format(source, channels)
    )
    return mono


def load_audio(
    audio: Any, sample_rate: Optional[int] = None
) -> Tuple[np.ndarray, int, str, List[str]]:
    """Normalise any accepted input into mono float64 samples.

    Args:
        audio: A path to a ``.wav`` file, a numpy array (1-D mono, or 2-D
            samples-by-channels), or a ``(samples, sample_rate)`` pair.
        sample_rate: Frames per second. Required when ``audio`` is a bare array,
            and must agree with the file or pair when given alongside one.

    Returns:
        ``(mono, sample_rate, source, notes)`` where ``mono`` is a freshly
        allocated, writable float64 array this package owns, ``source`` names the
        input for messages, and ``notes`` records anything decided for the caller.

    Raises:
        ValueError: The input shape, dtype or sample rate does not make sense.
        FileNotFoundError: A path was given and no file is there.
    """
    notes = []  # type: List[str]
    source = "<array>"

    if isinstance(audio, tuple):
        if len(audio) != 2:
            raise ValueError(
                "a tuple input must be (samples, sample_rate); got {} "
                "items".format(len(audio))
            )
        data, pair_rate = audio
        pair_rate = int(pair_rate)
        if sample_rate is not None and int(sample_rate) != pair_rate:
            raise ValueError(
                "sample_rate={} was passed but the (samples, sample_rate) pair "
                "says {}; drop one of them".format(int(sample_rate), pair_rate)
            )
        sample_rate = pair_rate
        audio = data

    if isinstance(audio, (str, bytes, os.PathLike)):
        path = os.fspath(audio)
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        if not os.path.exists(path):
            raise FileNotFoundError("no such audio file: {}".format(path))
        frames, file_rate = read_wav(path)
        if sample_rate is not None and int(sample_rate) != file_rate:
            raise ValueError(
                "sample_rate={} was passed but {} is recorded at {} Hz; drop "
                "sample_rate and the file's own rate is used".format(
                    int(sample_rate), path, file_rate
                )
            )
        sample_rate = file_rate
        source = os.path.basename(path) or path
        array = frames
    else:
        array = np.asarray(audio)

    if array.dtype == np.dtype(object) or array.dtype.kind in "USVM":
        raise ValueError(
            "audio has dtype {}; samples must be numeric".format(array.dtype)
        )
    if array.dtype.kind == "c":
        raise ValueError(
            "audio is complex; a recording is real-valued, so take the real part "
            "or the magnitude first"
        )

    if sample_rate is None:
        raise ValueError(
            "sample_rate is required when audio is an array; pass "
            "sample_rate=16000, or hand in (samples, 16000), or a .wav path"
        )
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive; got {}".format(sample_rate))

    # Scale to -1 .. 1 before mixing, so the integer range is read off the
    # caller's own dtype rather than off the float that averaging produces.
    if array.dtype.kind in "iu":
        scaled = _scale_integers(array)
    else:
        # copy=True is the whole point: the caller's array is never written to,
        # and nothing downstream inherits a read-only or shared buffer.
        scaled = np.array(array, dtype=np.float64, copy=True)
    mono = _to_mono(scaled, notes, source)
    mono = np.ascontiguousarray(np.array(mono, dtype=np.float64, copy=True))

    if mono.size and not np.all(np.isfinite(mono)):
        bad = int(np.count_nonzero(~np.isfinite(mono)))
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        notes.append(
            "{}: {} non-finite samples (NaN or inf) were treated as "
            "silence".format(source, bad)
        )
    return mono, sample_rate, source, notes
