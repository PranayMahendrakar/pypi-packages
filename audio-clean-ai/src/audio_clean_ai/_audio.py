"""Reading and writing audio, and turning any accepted input into a frame matrix.

Three input shapes are accepted everywhere in this package: a path to a ``.wav``
file, a numpy array, or a ``(samples, sample_rate)`` pair. Everything ends up as
a fresh ``(n_samples, n_channels)`` float64 matrix that this package owns, so the
caller's array is never written to, and a read-only array (pandas 3 hands those
out from ``.to_numpy()``) is as good as any other.

The layout of the caller's input is remembered, so the cleaned audio goes back
in exactly the same shape it came in: 1-D stays 1-D, samples-by-channels stays
samples-by-channels, channels-by-samples stays channels-by-samples.

The WAV reader is a small RIFF parser rather than :mod:`wave`, because the
stdlib module refuses IEEE-float and WAVE_FORMAT_EXTENSIBLE files, which is what
a lot of recorders and editors actually write. Writing uses :mod:`wave`.
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

MIN_SAMPLE_RATE = 2000
"""Below this there is no speech band left to protect, so the rate is refused."""

WRITE_BITS = (8, 16, 24, 32)
"""The PCM widths :func:`write_wav` can produce."""


class Loaded:
    """Audio normalised for processing, plus what is needed to hand it back.

    Attributes:
        frames: ``(n_samples, n_channels)`` float64, owned by this package.
        sample_rate: Frames per second.
        source: A short name for the input, used in messages.
        notes: Things decided on the caller's behalf while loading.
        channel_axis: ``None`` for 1-D input, ``1`` for samples-by-channels,
            ``0`` for channels-by-samples.
        positive_rail: The largest value the input format can hold. For 8-bit
            WAV this is 127/128, not 1: the format is asymmetric.
        negative_rail: The smallest value the input format can hold.
        bits: PCM bit depth when the input was a PCM WAV file, else ``None``.
    """

    __slots__ = (
        "frames",
        "sample_rate",
        "source",
        "notes",
        "channel_axis",
        "positive_rail",
        "negative_rail",
        "bits",
    )

    def __init__(
        self,
        frames: np.ndarray,
        sample_rate: int,
        source: str,
        notes: List[str],
        channel_axis: Optional[int],
        positive_rail: float,
        negative_rail: float,
        bits: Optional[int],
    ) -> None:
        self.frames = frames
        self.sample_rate = sample_rate
        self.source = source
        self.notes = notes
        self.channel_axis = channel_axis
        self.positive_rail = positive_rail
        self.negative_rail = negative_rail
        self.bits = bits

    def restore(self, frames: np.ndarray) -> np.ndarray:
        """Put a ``(n_samples, n_channels)`` matrix back into the caller's layout."""
        return to_layout(frames, self.channel_axis)


def to_layout(frames: np.ndarray, channel_axis: Optional[int]) -> np.ndarray:
    """Turn ``(n_samples, n_channels)`` into the layout ``channel_axis`` names."""
    if channel_axis is None:
        return np.ascontiguousarray(frames[:, 0])
    if channel_axis == 0:
        return np.ascontiguousarray(frames.T)
    return np.ascontiguousarray(frames)


def from_layout(audio: np.ndarray, channel_axis: Optional[int]) -> np.ndarray:
    """The inverse of :func:`to_layout`: back to ``(n_samples, n_channels)``."""
    if channel_axis is None or audio.ndim == 1:
        return audio.reshape(-1, 1)
    if channel_axis == 0:
        return audio.T
    return audio


# --------------------------------------------------------------------------- WAV


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


def _decode(raw: bytes, fmt: _Fmt, path: str) -> Tuple[np.ndarray, float]:
    """Turn data chunk bytes into floats, and say where the positive rail is."""
    if fmt.code == _IEEE_FLOAT:
        if fmt.bits == 32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float64), 1.0
        if fmt.bits == 64:
            return np.frombuffer(raw, dtype="<f8").astype(np.float64), 1.0
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
        # 8-bit WAV is unsigned with a 128 offset, so it reaches -1 but stops at
        # +127/128. Every other width is signed and has the same asymmetry,
        # just by a smaller margin.
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        return samples, 127.0 / 128.0
    if fmt.bits == 16:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        return samples, 32767.0 / 32768.0
    if fmt.bits == 24:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        value = (value ^ 0x800000) - 0x800000  # sign-extend 24 bits into int32
        return value.astype(np.float64) / 8388608.0, 8388607.0 / 8388608.0
    if fmt.bits == 32:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        return samples, 2147483647.0 / 2147483648.0
    raise ValueError(
        "{}: {}-bit PCM WAV is not something this reader knows; 8, 16, 24 and 32 "
        "bit are".format(path, fmt.bits)
    )


def read_wav(path: str) -> Tuple[np.ndarray, int, Optional[int], float]:
    """Read a ``.wav`` file.

    Args:
        path: Path to a RIFF/WAVE file.

    Returns:
        ``(frames, sample_rate, pcm_bits, positive_rail)``: a 2-D
        ``(n_samples, n_channels)`` float64 array, the sample rate, the PCM bit
        depth (``None`` for float files), and the largest value the format holds.

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
    flat, positive_rail = _decode(raw[:usable], fmt, path)
    bits = fmt.bits if fmt.code == _PCM else None
    return flat.reshape(-1, fmt.channels), int(fmt.sample_rate), bits, positive_rail


def write_wav(path: Any, frames: np.ndarray, sample_rate: int, bits: int = 16) -> str:
    """Write float samples to a PCM ``.wav`` file with the stdlib :mod:`wave` module.

    Samples are scaled by ``2**(bits-1) - 1`` on both sides, so +1.0 and -1.0
    both land inside the format and nothing ever wraps round or clips. (Scaling
    by ``2**(bits-1)`` would put +1.0 one step past the positive rail.)

    Args:
        path: Destination path.
        frames: 1-D mono samples or a 2-D ``(n_samples, n_channels)`` array.
        sample_rate: Frames per second.
        bits: 8, 16, 24 or 32.

    Returns:
        The path written, as a string.

    Raises:
        ValueError: ``bits`` is not a width the wave module can write, or the
            array is not 1-D or 2-D.
    """
    if bits not in WRITE_BITS:
        raise ValueError(
            "bits must be one of {}; got {}".format(", ".join(map(str, WRITE_BITS)), bits)
        )
    data = np.asarray(frames, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(
            "write_wav needs 1-D samples or a (samples, channels) array; got shape "
            "{}".format(data.shape)
        )
    clipped = np.clip(np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    scale = float(2 ** (bits - 1) - 1)
    ints = np.round(clipped * scale).astype(np.int64)
    if bits == 8:
        payload = (ints + 128).astype(np.uint8).tobytes()
    elif bits == 16:
        payload = ints.astype("<i2").tobytes()
    elif bits == 24:
        as32 = np.ascontiguousarray(ints.astype("<i4")).reshape(-1)
        payload = as32.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        payload = ints.astype("<i4").tobytes()
    target = os.fspath(path)
    if isinstance(target, bytes):
        target = target.decode("utf-8", "replace")
    with open(target, "wb") as handle:
        writer = wave.open(handle, "wb")
        try:
            writer.setnchannels(int(data.shape[1]))
            writer.setsampwidth(bits // 8)
            writer.setframerate(int(sample_rate))
            writer.writeframes(payload)
        finally:
            writer.close()
    return target


# ------------------------------------------------------------------------ arrays


def _integer_rails(dtype: np.dtype) -> Tuple[float, float, float]:
    """``(offset, scale, positive_rail)`` that map an integer dtype onto -1 to 1."""
    info = np.iinfo(dtype)
    if info.min == 0:  # unsigned: silence sits at the middle of the range
        middle = (float(info.max) + 1.0) / 2.0
        return middle, middle, (float(info.max) - middle) / middle
    scale = float(-info.min)
    return 0.0, scale, float(info.max) / scale


def _as_frames(array: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
    """View a 1-D or 2-D array as ``(n_samples, n_channels)`` and say how."""
    if array.ndim == 1:
        return array[:, None], None
    if array.ndim != 2:
        raise ValueError(
            "audio has {} dimensions; a recording is 1-D (mono) or 2-D "
            "(samples x channels, or channels x samples)".format(array.ndim)
        )
    rows, cols = array.shape
    # Whichever axis is short is the channel axis: a recording has far more
    # samples than channels.
    if 1 <= cols <= MAX_CHANNELS and cols <= rows:
        return array, 1
    if 1 <= rows <= MAX_CHANNELS:
        return array.T, 0
    if 1 <= cols <= MAX_CHANNELS:
        return array, 1
    raise ValueError(
        "audio is a {}x{} array and neither axis is a plausible channel count "
        "(1 to {}); pass mono samples or a (samples, channels) array".format(
            rows, cols, MAX_CHANNELS
        )
    )


def load_audio(audio: Any, sample_rate: Optional[int] = None) -> Loaded:
    """Normalise any accepted input into an owned ``(n_samples, n_channels)`` matrix.

    Args:
        audio: A path to a ``.wav`` file, a numpy array (1-D mono, or 2-D in
            either orientation), or a ``(samples, sample_rate)`` pair.
        sample_rate: Frames per second. Required when ``audio`` is a bare array,
            and must agree with the file or pair when given alongside one.

    Returns:
        A :class:`Loaded` holding a freshly allocated, writable float64 matrix.

    Raises:
        ValueError: The input shape, dtype or sample rate does not make sense.
        FileNotFoundError: A path was given and no file is there.
    """
    notes = []  # type: List[str]
    source = "<array>"
    bits = None  # type: Optional[int]
    positive_rail, negative_rail = 1.0, -1.0

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

    from_file = isinstance(audio, (str, bytes, os.PathLike))
    if from_file:
        path = os.fspath(audio)
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        if not os.path.exists(path):
            raise FileNotFoundError("no such audio file: {}".format(path))
        file_frames, file_rate, bits, positive_rail = read_wav(path)
        if sample_rate is not None and int(sample_rate) != file_rate:
            raise ValueError(
                "sample_rate={} was passed but {} is recorded at {} Hz; drop "
                "sample_rate and the file's own rate is used".format(
                    int(sample_rate), path, file_rate
                )
            )
        sample_rate = file_rate
        source = os.path.basename(path) or path
        array = file_frames
    else:
        array = np.asarray(audio)

    if array.dtype == np.dtype(object) or array.dtype.kind in "USVMmb":
        raise ValueError(
            "audio has dtype {}; samples must be numbers".format(array.dtype)
        )
    if array.dtype.kind == "c":
        raise ValueError(
            "audio is complex; a recording is real-valued, so take the real part "
            "first"
        )

    if sample_rate is None:
        raise ValueError(
            "sample_rate is required when audio is an array; pass "
            "sample_rate=16000, or hand in (samples, 16000), or a .wav path"
        )
    sample_rate = int(sample_rate)
    if sample_rate < MIN_SAMPLE_RATE:
        raise ValueError(
            "sample_rate={} Hz is too low for speech audio; this package works "
            "from {} Hz up (8 kHz to 48 kHz is the tested range)".format(
                sample_rate, MIN_SAMPLE_RATE
            )
        )
    if not 8000 <= sample_rate <= 48000:
        notes.append(
            "{} Hz is outside the tested 8 kHz to 48 kHz range; it is processed "
            "the same way".format(sample_rate)
        )

    if array.dtype.kind in "iu":
        offset, scale, positive_rail = _integer_rails(array.dtype)
        # astype always copies, so the caller's integers are never touched.
        scaled = (array.astype(np.float64) - offset) / scale
    else:
        # copy=True is the whole point: the caller's array is never written to,
        # and nothing downstream inherits a read-only or shared buffer.
        scaled = np.array(array, dtype=np.float64, copy=True)

    frames, channel_axis = _as_frames(scaled)
    if from_file and frames.shape[1] == 1:
        channel_axis = None  # a mono file comes back as plain 1-D samples
    frames = np.ascontiguousarray(np.array(frames, dtype=np.float64, copy=True))

    if frames.size and not np.all(np.isfinite(frames)):
        bad = int(np.count_nonzero(~np.isfinite(frames)))
        frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        notes.append(
            "{} non-finite samples (NaN or inf) were found in the input and replaced with silence".format(bad)
        )
    return Loaded(
        frames=frames,
        sample_rate=sample_rate,
        source=source,
        notes=notes,
        channel_axis=channel_axis,
        positive_rail=positive_rail,
        negative_rail=negative_rail,
        bits=bits,
    )


def rail_hits(
    frames: np.ndarray, positive_rail: float, negative_rail: float
) -> Tuple[int, int]:
    """Count samples sitting at or beyond each rail, measured separately.

    The two rails are counted apart because formats are asymmetric: 8-bit WAV
    reaches -1.0 but stops at +127/128, so a single ``abs(x) >= 1`` test would
    never see the positive side clip at all.
    """
    if frames.size == 0:
        return 0, 0
    tolerance = 1e-9
    positive = int(np.count_nonzero(frames >= positive_rail - tolerance))
    negative = int(np.count_nonzero(frames <= negative_rail + tolerance))
    return positive, negative
