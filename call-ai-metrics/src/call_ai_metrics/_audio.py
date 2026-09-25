"""Turn call audio into per-channel frame power.

Everything downstream of this module works on the loudness of short frames, never
on the waveform, so a long recording is read in blocks and never held in memory as
float64. Three input shapes arrive here: a path to a ``.wav`` file, a 2-D numpy
array (samples x channels, or channels x samples), and a list of 1-D channels.

The caller's arrays are only ever read. Blocks are copied into buffers this module
owns before anything is computed on them, which also means a read-only array (for
example one that pandas 3 handed out from ``.to_numpy()``) works unchanged.

The WAV reader is a small RIFF parser rather than :mod:`wave`, because the stdlib
module refuses IEEE-float and WAVE_FORMAT_EXTENSIBLE files, and because call
recorders very often write G.711 mu-law or A-law, which it does not decode either.
"""

from __future__ import annotations

import logging
import os
import struct
from typing import Any, Iterator, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

FRAME_S = 0.01
"""Hop between analysis frames, in seconds. Every boundary is accurate to this."""

MAX_CHANNELS = 16
"""More channels than this is not a call recording."""

SILENT_DB = -120.0
"""The level reported for a frame with no signal at all."""

_BLOCK_FRAMES = 4096  # analysis frames per block read from disk or an array

_PCM = 0x0001
_IEEE_FLOAT = 0x0003
_ALAW = 0x0006
_MULAW = 0x0007
_EXTENSIBLE = 0xFFFE


def _mulaw_table() -> np.ndarray:
    """ITU-T G.711 mu-law code -> linear value, scaled to -1 .. 1."""
    code = (~np.arange(256, dtype=np.int32)) & 0xFF
    magnitude = (((code & 0x0F) << 3) + 0x84) << ((code & 0x70) >> 4)
    linear = np.where(code & 0x80, 0x84 - magnitude, magnitude - 0x84)
    return linear.astype(np.float64) / 32768.0


def _alaw_table() -> np.ndarray:
    """ITU-T G.711 A-law code -> linear value, scaled to -1 .. 1."""
    code = np.arange(256, dtype=np.int32) ^ 0x55
    seg = (code & 0x70) >> 4
    base = (code & 0x0F) << 4
    magnitude = np.where(seg == 0, base + 8, (base + 0x108) << np.maximum(seg - 1, 0))
    linear = np.where(code & 0x80, magnitude, -magnitude)
    return linear.astype(np.float64) / 32768.0


_MULAW_TABLE = _mulaw_table()
_ALAW_TABLE = _alaw_table()


class WavFormat:
    """What the header of a WAV file says about its samples."""

    __slots__ = ("code", "channels", "sample_rate", "bits", "width", "data_offset", "data_size")

    def __init__(
        self,
        code: int,
        channels: int,
        sample_rate: int,
        bits: int,
        width: int,
        data_offset: int,
        data_size: int,
    ) -> None:
        self.code = code
        self.channels = channels
        self.sample_rate = sample_rate
        self.bits = bits
        self.width = width
        self.data_offset = data_offset
        self.data_size = data_size


def read_wav_format(path: str) -> WavFormat:
    """Parse the RIFF header of ``path`` without reading the audio.

    Raises:
        ValueError: The file is not a WAV file, or uses a codec this reader does
            not decode (anything but PCM, IEEE float, mu-law and A-law).
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(
                "{}: this is not a RIFF/WAVE file (it does not start with the "
                "RIFF...WAVE marker)".format(path)
            )
        fmt_body: Optional[bytes] = None
        data_offset = -1
        data_size = 0
        while True:
            head = handle.read(8)
            if len(head) < 8:
                break
            chunk_id = head[0:4]
            size = struct.unpack("<I", head[4:8])[0]
            if chunk_id == b"fmt ":
                fmt_body = handle.read(size)
            elif chunk_id == b"data":
                data_offset = handle.tell()
                # A streamed or truncated file can declare more than is there.
                data_size = max(0, min(size, file_size - data_offset))
                handle.seek(data_size, os.SEEK_CUR)
            else:
                handle.seek(size, os.SEEK_CUR)
            if size % 2:  # RIFF chunks are padded to an even length
                handle.seek(1, os.SEEK_CUR)
            if fmt_body is not None and data_offset >= 0:
                break

    if fmt_body is None or len(fmt_body) < 16:
        raise ValueError("{}: the WAV file has no usable format chunk".format(path))
    if data_offset < 0:
        raise ValueError("{}: the WAV file has no data chunk".format(path))
    code, channels, sample_rate, _byte_rate, align, bits = struct.unpack_from(
        "<HHIIHH", fmt_body, 0
    )
    if code == _EXTENSIBLE:
        if len(fmt_body) < 26:
            raise ValueError(
                "{}: the WAV file says WAVE_FORMAT_EXTENSIBLE but carries no "
                "sub-format".format(path)
            )
        code = struct.unpack_from("<H", fmt_body, 24)[0]
    if channels < 1:
        raise ValueError("{}: the WAV file declares {} channels".format(path, channels))
    if sample_rate <= 0:
        raise ValueError(
            "{}: the WAV file declares a sample rate of {}".format(path, sample_rate)
        )
    if align and align % channels == 0:
        width = align // channels
    else:
        width = max(bits // 8, 1)
    known = {
        _PCM: (1, 2, 3, 4),
        _IEEE_FLOAT: (4, 8),
        _ALAW: (1,),
        _MULAW: (1,),
    }
    if code not in known:
        raise ValueError(
            "{}: WAV format code {} is not PCM, IEEE float, mu-law or A-law; "
            "re-export the file as plain PCM WAV".format(path, code)
        )
    if width not in known[code]:
        raise ValueError(
            "{}: {}-byte samples are not something this reader decodes for WAV "
            "format code {}".format(path, width, code)
        )
    return WavFormat(code, channels, int(sample_rate), bits, width, data_offset, data_size)


def _decode(raw: bytes, fmt: WavFormat) -> np.ndarray:
    """Decode whole frames of WAV data into a (samples, channels) float64 array."""
    if fmt.code == _IEEE_FLOAT:
        dtype = "<f4" if fmt.width == 4 else "<f8"
        flat = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    elif fmt.code == _MULAW:
        flat = _MULAW_TABLE[np.frombuffer(raw, dtype=np.uint8)]
    elif fmt.code == _ALAW:
        flat = _ALAW_TABLE[np.frombuffer(raw, dtype=np.uint8)]
    elif fmt.width == 1:
        # 8-bit PCM is unsigned around 128, so the positive rail stops at 127/128.
        flat = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif fmt.width == 2:
        flat = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif fmt.width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        value = (value ^ 0x800000) - 0x800000  # sign-extend 24 bits
        flat = value.astype(np.float64) / 8388608.0
    else:
        flat = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    return flat.reshape(-1, fmt.channels)


def iter_wav_blocks(path: str, fmt: WavFormat, block_samples: int) -> Iterator[np.ndarray]:
    """Yield the audio of ``path`` as float64 ``(samples, channels)`` blocks."""
    frame_bytes = fmt.width * fmt.channels
    usable = fmt.data_size - fmt.data_size % frame_bytes
    if usable != fmt.data_size:
        logger.warning(
            "%s: the data chunk ends mid-frame; %d trailing bytes ignored",
            path,
            fmt.data_size - usable,
        )
    step = max(block_samples, 1) * frame_bytes
    with open(path, "rb") as handle:
        handle.seek(fmt.data_offset)
        remaining = usable
        while remaining > 0:
            raw = handle.read(min(step, remaining))
            if not raw:
                break
            raw = raw[: len(raw) - len(raw) % frame_bytes]
            remaining -= len(raw)
            if raw:
                yield _decode(raw, fmt)


class FrameMeter:
    """Accumulate per-frame signal power from blocks of samples.

    Power is the variance of each frame, so a DC offset (common on cheap
    recorders and on 8-bit files) does not read as sound.
    """

    def __init__(self, n_channels: int, hop: int) -> None:
        self.n_channels = n_channels
        self.hop = hop
        self.n_samples = 0
        self.non_finite = 0
        self._carry = np.zeros((0, n_channels), dtype=np.float64)
        self._chunks: List[np.ndarray] = []

    def feed(self, block: np.ndarray) -> None:
        """Add a ``(samples, channels)`` float64 block that this meter owns."""
        if block.size == 0:
            return
        bad = ~np.isfinite(block)
        if bad.any():
            self.non_finite += int(np.count_nonzero(bad))
            block = np.where(bad, 0.0, block)
        self.n_samples += block.shape[0]
        if self._carry.shape[0]:
            block = np.concatenate([self._carry, block], axis=0)
        n_full = block.shape[0] // self.hop
        if n_full:
            full = block[: n_full * self.hop].reshape(n_full, self.hop, self.n_channels)
            self._chunks.append(full.var(axis=1))
        self._carry = block[n_full * self.hop :].copy()

    def finish(self) -> np.ndarray:
        """Return frame power as a ``(channels, frames)`` float64 array."""
        if self._carry.shape[0]:
            self._chunks.append(self._carry.var(axis=0, keepdims=True))
            self._carry = np.zeros((0, self.n_channels), dtype=np.float64)
        if not self._chunks:
            return np.zeros((self.n_channels, 0), dtype=np.float64)
        return np.ascontiguousarray(np.concatenate(self._chunks, axis=0).T)


class CallAudio:
    """Frame power of every channel of a call, plus where it came from."""

    __slots__ = ("power", "sample_rate", "hop", "n_samples", "source", "kind", "notes")

    def __init__(
        self,
        power: np.ndarray,
        sample_rate: int,
        hop: int,
        n_samples: int,
        source: str,
        kind: str,
        notes: List[str],
    ) -> None:
        self.power = power
        self.sample_rate = sample_rate
        self.hop = hop
        self.n_samples = n_samples
        self.source = source
        self.kind = kind
        self.notes = notes

    @property
    def n_channels(self) -> int:
        """How many parties (channels) the recording carries."""
        return int(self.power.shape[0])

    @property
    def hop_s(self) -> float:
        """Seconds between frame starts."""
        return self.hop / float(self.sample_rate)

    @property
    def duration_s(self) -> float:
        """Length of the recording in seconds."""
        return self.n_samples / float(self.sample_rate)


def _hop_for(sample_rate: int) -> int:
    hop = int(round(sample_rate * FRAME_S))
    if hop < 4:
        raise ValueError(
            "sample_rate={} is too low for speech; a call recording is 8000 Hz or "
            "more".format(sample_rate)
        )
    return hop


def mono_hint(source: str) -> str:
    """The message for a recording that has only one channel."""
    return (
        "{} has a single channel, so both parties are mixed together and this "
        "package cannot tell them apart. Run a diarizer on it first "
        "(speaker-diarize-lite, pyannote or any other) and pass its segments as "
        "[(start_s, end_s, speaker), ...]".format(source)
    )


def load_wav(path: str, sample_rate: Optional[int]) -> CallAudio:
    """Read a multi-channel ``.wav`` call recording into frame power."""
    fmt = read_wav_format(path)
    source = os.path.basename(path) or path
    if sample_rate is not None and int(sample_rate) != fmt.sample_rate:
        raise ValueError(
            "sample_rate={} was passed but {} is recorded at {} Hz; drop "
            "sample_rate and the file's own rate is used".format(
                int(sample_rate), source, fmt.sample_rate
            )
        )
    if fmt.channels == 1:
        raise ValueError(mono_hint(source))
    if fmt.channels > MAX_CHANNELS:
        raise ValueError(
            "{} has {} channels; a call recording has one channel per party "
            "(at most {})".format(source, fmt.channels, MAX_CHANNELS)
        )
    hop = _hop_for(fmt.sample_rate)
    meter = FrameMeter(fmt.channels, hop)
    for block in iter_wav_blocks(path, fmt, hop * _BLOCK_FRAMES):
        meter.feed(block)
    notes: List[str] = []
    if meter.non_finite:
        notes.append(
            "{}: {} non-finite samples (NaN or inf) were treated as silence".format(
                source, meter.non_finite
            )
        )
    return CallAudio(
        meter.finish(), fmt.sample_rate, hop, meter.n_samples, source, "stereo-wav", notes
    )


def _numeric_1d(values: Any, index: int) -> np.ndarray:
    """View one channel as a 1-D numeric array without copying or writing it."""
    array = np.asarray(values)
    if array.dtype == np.dtype(object) or array.dtype.kind in "USVMm":
        raise ValueError(
            "channel {} has dtype {}; samples must be numeric".format(index + 1, array.dtype)
        )
    if array.dtype.kind == "c":
        raise ValueError("channel {} is complex; a recording is real-valued".format(index + 1))
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(
            "channel {} has shape {}; each channel must be 1-D samples".format(
                index + 1, array.shape
            )
        )
    return array


def _scaled_block(array: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Copy ``array[start:stop]`` into a fresh float64 buffer scaled to -1 .. 1."""
    part = array[start:stop]
    if part.dtype.kind == "b":
        return part.astype(np.float64)
    if part.dtype.kind in "iu":
        info = np.iinfo(part.dtype)
        if info.min == 0:  # unsigned: silence sits at the middle of the range
            middle = (float(info.max) + 1.0) / 2.0
            return (part.astype(np.float64) - middle) / middle
        return part.astype(np.float64) / float(-info.min)
    return np.array(part, dtype=np.float64, copy=True)


def split_channels(array: Any) -> List[np.ndarray]:
    """Split a 2-D sample array into its channels (the short axis is channels)."""
    data = np.asarray(array)
    if data.ndim == 1:
        raise ValueError(mono_hint("the array"))
    if data.ndim != 2:
        raise ValueError(
            "audio has {} dimensions; a call recording is 2-D (samples x "
            "channels)".format(data.ndim)
        )
    rows, cols = data.shape
    if rows == 0 or cols == 0:  # an empty recording: the non-empty axis is channels
        count = max(rows, cols)
        if count < 2 or count > MAX_CHANNELS:
            raise ValueError(
                "audio is an empty {}x{} array; an empty call is (0, 2)".format(rows, cols)
            )
        return [np.zeros(0, dtype=data.dtype) for _ in range(count)]
    if min(rows, cols) == 1:
        raise ValueError(mono_hint("the array"))
    if cols <= rows and cols <= MAX_CHANNELS:
        return [data[:, index] for index in range(cols)]
    if rows <= MAX_CHANNELS:
        return [data[index, :] for index in range(rows)]
    raise ValueError(
        "audio is a {}x{} array and neither axis is short enough to be channels "
        "(at most {})".format(rows, cols, MAX_CHANNELS)
    )


def load_channels(
    channels: Sequence[Any], sample_rate: Optional[int], source: str, kind: str
) -> CallAudio:
    """Measure a list of 1-D channels, one per party."""
    if sample_rate is None:
        raise ValueError(
            "sample_rate is required when the call is given as arrays; pass "
            "sample_rate=8000 (or whatever the recording uses)"
        )
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive; got {}".format(sample_rate))
    if len(channels) < 2:
        raise ValueError(mono_hint("the input"))
    if len(channels) > MAX_CHANNELS:
        raise ValueError(
            "{} channels were given; a call has one per party (at most {})".format(
                len(channels), MAX_CHANNELS
            )
        )
    hop = _hop_for(sample_rate)
    arrays = [_numeric_1d(values, index) for index, values in enumerate(channels)]
    notes: List[str] = []
    powers: List[np.ndarray] = []
    non_finite = 0
    step = hop * _BLOCK_FRAMES
    for array in arrays:
        meter = FrameMeter(1, hop)
        for start in range(0, array.shape[0], step):
            meter.feed(_scaled_block(array, start, start + step).reshape(-1, 1))
        non_finite += meter.non_finite
        powers.append(meter.finish()[0])
    lengths = [array.shape[0] for array in arrays]
    n_samples = max(lengths)
    if len(set(lengths)) > 1:
        notes.append(
            "the channels have different lengths ({} samples); the shorter ones "
            "were treated as silent after they end".format(
                ", ".join(str(length) for length in lengths)
            )
        )
    n_frames = max(power.shape[0] for power in powers)
    power = np.zeros((len(powers), n_frames), dtype=np.float64)
    for index, row in enumerate(powers):
        power[index, : row.shape[0]] = row
    if non_finite:
        notes.append(
            "{} non-finite samples (NaN or inf) were treated as silence".format(non_finite)
        )
    return CallAudio(power, sample_rate, hop, n_samples, source, kind, notes)


def power_to_db(power: np.ndarray) -> np.ndarray:
    """Frame power to dBFS, with silence pinned at :data:`SILENT_DB`."""
    return 10.0 * np.log10(np.maximum(power, 10.0 ** (SILENT_DB / 10.0)))


def smooth_rows(values: np.ndarray, frames: int) -> np.ndarray:
    """Centred moving average along the last axis of a 2-D array."""
    if frames <= 1 or values.shape[-1] == 0:
        return values.copy()
    kernel = np.ones(frames, dtype=np.float64) / frames
    before = (frames - 1) // 2
    after = frames - 1 - before
    # Direct convolution of non-negative power: a running cumsum would lose the
    # quiet frames to cancellation once the sum has passed a loud stretch.
    return np.stack(
        [
            np.convolve(np.pad(row, (before, after), mode="edge"), kernel, mode="valid")
            for row in values
        ]
    )
