"""Turn whatever the caller passed into one mono float64 signal and a sample rate.

Accepted: a path to a ``.wav`` file, a numpy array (or anything
``numpy.asarray`` understands) together with ``sample_rate``, or a
``(samples, sample_rate)`` tuple. The caller's array is never written to: a
private float64 copy is made first, which also keeps read-only arrays (pandas 3
``.to_numpy()`` under copy-on-write, memory-mapped files) working.
"""

from __future__ import annotations

import logging
import numbers
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from ._wav import rails, read_wav

logger = logging.getLogger(__name__)

MIN_SAMPLE_RATE = 2000
"""Below this there is no speech band left to describe."""

_MAX_CHANNELS = 16
_CLIP_WARN_FRACTION = 0.001
_CANCEL_RATIO = 0.25


@dataclass
class Loaded:
    """A mono signal ready for analysis, plus what was done to get it."""

    samples: np.ndarray
    sample_rate: int
    source: str
    file_id: str
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _as_rate(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("{} must be a number of samples per second, not {!r}".format(what, value))
    rate = float(value)
    if not np.isfinite(rate) or rate <= 0 or rate != int(rate):
        raise ValueError("{} must be a positive whole number of Hz, not {!r}".format(what, value))
    if rate < MIN_SAMPLE_RATE:
        raise ValueError(
            "{} is {} Hz; at least {} Hz is needed to describe a voice".format(what, int(rate), MIN_SAMPLE_RATE)
        )
    return int(rate)


def _file_id(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0] or "audio"
    return "_".join(stem.split()) or "audio"


def _to_float(array: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """A float64 copy scaled to [-1, 1], with the (negative, positive) rails of its encoding."""
    kind = array.dtype.kind
    if kind == "b":
        raise ValueError("audio samples must be numbers, not booleans")
    if kind == "u":
        bits = array.dtype.itemsize * 8
        half = float(2 ** (bits - 1))
        out = (np.array(array, dtype=np.float64, copy=True) - half) / half
        low, high = rails(bits, False)
        return out, low, high
    if kind == "i":
        bits = array.dtype.itemsize * 8
        out = np.array(array, dtype=np.float64, copy=True) / float(2 ** (bits - 1))
        low, high = rails(bits, False)
        return out, low, high
    if kind in "fc":
        if kind == "c":
            raise ValueError("audio samples must be real numbers, not complex")
        return np.array(array, dtype=np.float64, copy=True), -1.0, 1.0
    raise ValueError("audio samples must be numeric; got an array of dtype {}".format(array.dtype))


def _channels_last(data: np.ndarray, notes: List[str]) -> np.ndarray:
    """Return ``(n_samples, n_channels)`` from a 1-D or 2-D array of either orientation."""
    if data.ndim == 1:
        return data[:, None]
    if data.ndim != 2:
        raise ValueError(
            "audio must be 1-D (mono) or 2-D (samples x channels); got {} dimensions".format(data.ndim)
        )
    rows, cols = data.shape
    if cols <= _MAX_CHANNELS and rows >= cols:
        return data
    if rows <= _MAX_CHANNELS and cols > rows:
        notes.append("the array was shaped (channels, samples); it was read as {} channels".format(rows))
        return data.T
    raise ValueError(
        "cannot tell samples from channels in an array shaped {}; pass (n_samples, n_channels)".format(
            data.shape
        )
    )


def _mix_to_mono(data: np.ndarray, notes: List[str], warnings: List[str]) -> np.ndarray:
    channels = data.shape[1]
    if channels == 1:
        return data[:, 0]
    mono = data.mean(axis=1)
    notes.append("mixed {} channels to mono".format(channels))
    per_channel = np.sqrt(np.mean(data ** 2, axis=0)) if data.shape[0] else np.zeros(channels)
    loudest = int(np.argmax(per_channel)) if channels else 0
    mixed_rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    if per_channel.size and per_channel[loudest] > 0 and mixed_rms < _CANCEL_RATIO * float(per_channel.mean()):
        warnings.append(
            "the channels largely cancel when averaged (they look out of phase), so channel {} "
            "was analysed on its own instead of the mono mix".format(loudest)
        )
        return data[:, loudest].copy()
    return mono


def _clipping(data: np.ndarray, low: float, high: float, warnings: List[str]) -> None:
    """Warn when many samples sit on a rail. Each rail is measured on its own."""
    if data.size == 0:
        return
    tolerance = 1e-9
    at_low = int(np.count_nonzero(data <= low + tolerance))
    at_high = int(np.count_nonzero(data >= high - tolerance))
    total = data.size
    parts = []
    if at_low / total > _CLIP_WARN_FRACTION:
        parts.append("{:.2%} of samples on the negative rail".format(at_low / total))
    if at_high / total > _CLIP_WARN_FRACTION:
        parts.append("{:.2%} on the positive rail".format(at_high / total))
    if parts:
        warnings.append(
            "the recording is clipped ({}); distortion blurs the differences between "
            "voices".format(", ".join(parts))
        )


def load_audio(audio: Any, sample_rate: Optional[Any] = None) -> Loaded:
    """Normalise ``audio`` (path, array, or ``(samples, rate)``) to mono float64.

    Raises:
        ValueError: the sample rate is missing, contradictory or implausible, the
            array has an unusable shape or type, or it holds NaN or infinity.
        FileNotFoundError: a path was given that does not exist.
    """
    notes: List[str] = []
    warnings: List[str] = []
    if isinstance(audio, (str, os.PathLike)):
        name = os.fspath(audio)
        wav = read_wav(name)
        warnings.extend(wav.warnings)
        if sample_rate is not None and _as_rate(sample_rate, "sample_rate") != wav.sample_rate:
            raise ValueError(
                "{!r} is recorded at {} Hz but sample_rate={} was passed; this package does "
                "not resample, so leave sample_rate out for files".format(name, wav.sample_rate, sample_rate)
            )
        low, high = rails(wav.bits, wav.is_float)
        data = wav.samples
        rate = wav.sample_rate
        source = name
        file_id = _file_id(name)
    else:
        if (
            isinstance(audio, tuple)
            and len(audio) == 2
            and isinstance(audio[1], numbers.Real)
            and np.ndim(audio[0]) >= 1
        ):
            samples, tuple_rate = audio
            rate = _as_rate(tuple_rate, "the sample rate in (samples, sample_rate)")
            if sample_rate is not None and _as_rate(sample_rate, "sample_rate") != rate:
                raise ValueError(
                    "the (samples, sample_rate) tuple says {} Hz but sample_rate={} was also "
                    "passed; give it once".format(rate, sample_rate)
                )
        else:
            samples = audio
            if sample_rate is None:
                raise ValueError(
                    "sample_rate is required with an array: diarize(samples, sample_rate=16000), "
                    "or pass (samples, sample_rate)"
                )
            rate = _as_rate(sample_rate, "sample_rate")
        array = np.asarray(samples)
        data, low, high = _to_float(array)
        source = "array"
        file_id = "audio"

    data = _channels_last(data, notes)
    finite = np.isfinite(data)
    if not finite.all():
        bad = int(data.size - np.count_nonzero(finite))
        raise ValueError(
            "the audio holds {} non-finite sample(s) (NaN or infinity); clean it first".format(bad)
        )
    _clipping(data, low, high, warnings)
    mono = _mix_to_mono(data, notes, warnings)
    mono = np.ascontiguousarray(mono, dtype=np.float64)
    if mono.size:
        offset = float(np.mean(mono))
        if abs(offset) > 0.01:
            notes.append("removed a DC offset of {:.3f}".format(offset))
        mono = mono - offset
    for message in warnings:
        logger.debug("%s", message)
    return Loaded(samples=mono, sample_rate=rate, source=source, file_id=file_id, notes=notes, warnings=warnings)
