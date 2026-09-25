"""Getting samples in, from the three shapes a caller actually has them in.

A path to a ``.wav`` file, a numpy array of samples, or a ``(samples,
sample_rate)`` pair. WAV files are read with the standard library :mod:`wave`
module and converted here, 8, 16, 24 and 32 bit alike, so the package needs no
audio dependency at all.

Whatever comes in leaves as :class:`Audio`: one-dimensional float64 samples on a
-1.0 to 1.0 scale, a sample rate, and a list of notes recording every decision
taken on the caller's behalf. The caller's own array is never written to.
"""

from __future__ import annotations

import contextlib
import os
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Audio",
    "DEFAULT_SAMPLE_RATE",
    "WAV_SUFFIXES",
    "load_audio",
    "read_wav",
]

WAV_SUFFIXES = (".wav", ".wave")
"""File suffixes :func:`read_wav` will open."""

DEFAULT_SAMPLE_RATE = 16000
"""Rate assumed for bare sample arrays, always recorded as a note."""

SAMPLE_WIDTHS = {1: 8, 2: 16, 3: 24, 4: 32}
"""Bytes per sample mapped to the bit depth they mean."""

PLAUSIBLE_RATES = (4000, 192000)
"""Sample rates outside this range are accepted but noted as unusual."""

AudioInput = Union[str, "os.PathLike[str]", np.ndarray, Sequence[Any], Tuple[Any, Any]]


@dataclass
class Audio:
    """Mono float samples, ready to measure.

    Attributes:
        samples: one-dimensional float64 array, full scale at 1.0.
        sample_rate: samples per second.
        channels: how many channels the source had before the mixdown.
        source: a label for messages, usually a file name.
        notes: decisions taken while loading, in the order they were taken.
        assumed_sample_rate: True when no rate was given and the default was used.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int = 1
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    assumed_sample_rate: bool = False
    #: The largest positive value the source format can represent. Integer PCM is
    #: asymmetric: 8-bit stops at 127/128 on the positive side while the negative rail
    #: reaches -1.0 exactly. Clipping has to be measured against the rail the format
    #: actually has, or the positive half of a clipped 8-bit file is simply invisible.
    positive_rail: float = 1.0

    @property
    def n_samples(self) -> int:
        """How many samples the mono signal holds."""
        return int(self.samples.size)

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return float(self.samples.size) / float(self.sample_rate)

    @property
    def is_digital_silence(self) -> bool:
        """True when every sample is exactly zero."""
        return not bool(np.any(self.samples))

    def describe(self) -> str:
        """One line naming the recording, for a report headline."""
        layout = "mono"
        if self.channels > 1:
            layout = "{} channels mixed to mono".format(self.channels)
        parts = ["{:.2f} s".format(self.duration), "{} Hz".format(self.sample_rate), layout]
        if self.source:
            parts.append(str(self.source))
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe description of the recording as loaded."""
        return {
            "source": self.source,
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
            "samples": self.n_samples,
            "duration_s": round(self.duration, 4),
            "assumed_sample_rate": bool(self.assumed_sample_rate),
            "notes": list(self.notes),
        }


def _unsupported_path(path: str) -> ValueError:
    """The error for a path that is not a WAV file."""
    suffix = os.path.splitext(path)[1] or "no suffix"
    return ValueError(
        "speech-quality reads WAV files only (PCM 8, 16, 24 or 32 bit), and {} is {}. "
        "Convert it first, for example with ffmpeg -i input out.wav, or hand the samples "
        "in directly as a numpy array or a (samples, sample_rate) pair.".format(
            os.path.basename(path), suffix
        )
    )


def _decode_frames(raw: bytes, width: int) -> np.ndarray:
    """Turn raw WAV frame bytes into float64 samples on a -1.0 to 1.0 scale.

    Args:
        raw: the bytes :mod:`wave` handed over.
        width: bytes per sample, 1 to 4.

    Returns:
        A new float64 array; the input buffer is never written to.
    """
    if width == 1:
        # 8-bit WAV is unsigned, with 128 as the zero point.
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (values - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if width == 3:
        # wave hands 24-bit over as raw bytes: three little-endian bytes per sample.
        octets = np.frombuffer(raw, dtype=np.uint8)
        usable = (octets.size // 3) * 3
        triples = octets[:usable].reshape(-1, 3).astype(np.int32)
        packed = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        signed = np.where(packed >= (1 << 23), packed - (1 << 24), packed)
        return signed.astype(np.float64) / float(1 << 23)
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(
        "WAV sample width {} bytes is not supported; 8, 16, 24 and 32 bit PCM are".format(width)
    )


def _truncation_note(
    raw_bytes: int, width: int, channels: int, declared: int, rate: int
) -> List[str]:
    """A note when the header promises more audio than the file actually holds.

    A valid header over a short body is what a copy interrupted part way through
    leaves behind. :mod:`wave` hands over the bytes that are there without
    complaint, so five milliseconds of a four-second recording would otherwise
    be assessed as a perfectly good five-millisecond recording.

    Args:
        raw_bytes: how many bytes of frame data were actually read.
        width: bytes per sample.
        channels: channels declared in the header.
        declared: frame count declared in the header.
        rate: frames per second declared in the header.

    Returns:
        One note, or an empty list when the file holds what it promised.
    """
    frame_bytes = int(width) * max(1, int(channels))
    if frame_bytes <= 0 or rate <= 0 or declared <= 0:
        return []
    held = int(raw_bytes) // frame_bytes
    if held >= int(declared):
        return []
    return [
        "the file declares {:.3f} s but holds {:.3f} s; it looks truncated".format(
            float(declared) / float(rate), float(held) / float(rate)
        )
    ]


def read_wav(path: Union[str, "os.PathLike[str]"]) -> Audio:
    """Read a PCM WAV file with the standard library, no audio dependency.

    Args:
        path: a ``.wav`` file. Multi-channel files are mixed down to mono.

    Returns:
        The recording as :class:`Audio`. A file holding fewer frames than its
        header declares is read as far as it goes, with a note saying so.

    Raises:
        FileNotFoundError: no file at that path.
        ValueError: the path is not a WAV file, or the file is not readable PCM
            WAV (compressed, float and extensible formats included).
    """
    text = os.fspath(path)
    # The directory check comes first: a folder is not a file with a bad suffix,
    # and "." should not be told it has none.
    if os.path.isdir(text):
        raise ValueError("{} is a directory, not a WAV file".format(text))
    if os.path.splitext(text)[1].lower() not in WAV_SUFFIXES:
        raise _unsupported_path(text)
    if not os.path.exists(text):
        raise FileNotFoundError("no such file: {}".format(text))

    try:
        with contextlib.closing(wave.open(text, "rb")) as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            count = handle.getnframes()
            raw = handle.readframes(count)
    except wave.Error as exc:
        raise ValueError(
            "{} is not a readable PCM WAV file ({}). Only uncompressed PCM WAV is "
            "supported; re-encode it, for example with "
            "ffmpeg -i input -c:a pcm_s16le out.wav.".format(os.path.basename(text), exc)
        ) from None
    except EOFError:
        raise ValueError(
            "{} ends part way through its audio data and cannot be read".format(
                os.path.basename(text)
            )
        ) from None

    if width not in SAMPLE_WIDTHS:
        raise ValueError(
            "{} uses {} bytes per sample; 8, 16, 24 and 32 bit PCM WAV are supported".format(
                os.path.basename(text), width
            )
        )
    if channels < 1:
        raise ValueError("{} reports {} channels".format(os.path.basename(text), channels))
    if not raw:
        raise ValueError("{} holds no audio data".format(os.path.basename(text)))

    flat = _decode_frames(raw, width)
    notes = ["read as {}-bit PCM WAV".format(SAMPLE_WIDTHS[width])]
    notes.extend(_truncation_note(len(raw), width, channels, count, rate))
    if channels > 1:
        usable = (flat.size // channels) * channels
        if usable != flat.size:
            notes.append("the last frame was incomplete and was dropped")
        flat = flat[:usable].reshape(-1, channels).mean(axis=1)
        notes.append("{} channels were mixed down to mono by averaging them".format(channels))

    audio = _finish(
        np.array(flat, dtype=np.float64, copy=True),
        int(rate),
        channels=channels,
        source=os.path.basename(text),
        notes=notes,
    )
    bits = 8 * int(width)
    audio.positive_rail = float((1 << (bits - 1)) - 1) / float(1 << (bits - 1))
    return audio


def _as_pair(value: Any) -> Optional[Tuple[Any, Any]]:
    """Recognise a ``(samples, sample_rate)`` pair without eating a 2-sample signal."""
    if isinstance(value, tuple) and len(value) == 2:
        first, second = value
        looks_like_samples = isinstance(first, (np.ndarray, list, tuple))
        looks_like_rate = isinstance(second, (int, float, np.integer, np.floating))
        if looks_like_samples and looks_like_rate and not isinstance(second, bool):
            return first, second
    return None


def _to_mono(values: np.ndarray, notes: List[str]) -> Tuple[np.ndarray, int]:
    """Flatten a multi-channel array to mono, recording what was done."""
    if values.ndim == 1:
        return values, 1
    if values.ndim != 2:
        raise ValueError(
            "samples must be 1-D (mono) or 2-D (multi-channel), got {} dimensions".format(
                values.ndim
            )
        )
    rows, cols = values.shape
    if rows == 0 or cols == 0:
        raise ValueError("the recording has no samples")
    if cols <= rows:
        channels, mixed = cols, values.mean(axis=1)
    else:
        channels, mixed = rows, values.mean(axis=0)
    if channels == 1:
        return mixed, 1
    notes.append("{} channels were mixed down to mono by averaging them".format(channels))
    return mixed, int(channels)


def _scale_integers(values: np.ndarray, notes: List[str]) -> np.ndarray:
    """Scale an integer sample array to -1.0 to 1.0, noting the bit depth used."""
    kind = values.dtype.kind
    if kind not in "iu":
        return values.astype(np.float64, copy=True)
    info = np.iinfo(values.dtype)
    bits = values.dtype.itemsize * 8
    if kind == "u":
        middle = (float(info.max) + 1.0) / 2.0
        notes.append("samples were scaled from unsigned {}-bit to -1.0 to 1.0".format(bits))
        return (values.astype(np.float64) - middle) / middle
    notes.append("samples were scaled from signed {}-bit to -1.0 to 1.0".format(bits))
    return values.astype(np.float64) / (float(info.max) + 1.0)


def _finish(
    samples: np.ndarray,
    sample_rate: int,
    channels: int,
    source: Optional[str],
    notes: List[str],
    assumed_sample_rate: bool = False,
) -> Audio:
    """The last checks, shared by every input shape."""
    if samples.size == 0:
        raise ValueError("the recording has no samples")
    bad = ~np.isfinite(samples)
    if bool(bad.any()):
        count = int(bad.sum())
        samples = samples.copy()
        samples[bad] = 0.0
        notes.append(
            "{} sample(s) were not finite (NaN or infinity) and were read as "
            "silence".format(count)
        )
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        notes.append(
            "samples reach {:.3f}, past full scale; 1.0 is still treated as full scale "
            "and nothing was rescaled".format(peak)
        )
    low, high = PLAUSIBLE_RATES
    if not low <= sample_rate <= high:
        notes.append(
            "a sample rate of {} Hz is outside the usual {}-{} Hz range and is used as "
            "given".format(sample_rate, low, high)
        )
    return Audio(
        samples=samples,
        sample_rate=int(sample_rate),
        channels=int(channels),
        source=source,
        notes=notes,
        assumed_sample_rate=assumed_sample_rate,
    )


def load_audio(
    audio: AudioInput,
    sample_rate: Optional[float] = None,
    source: Optional[str] = None,
) -> Audio:
    """Turn any accepted input into mono float samples.

    Args:
        audio: a path to a ``.wav`` file, an array of samples, or a
            ``(samples, sample_rate)`` pair.
        sample_rate: samples per second, for the array form. A file carries its
            own rate, so one passed alongside a path is ignored with a note.
        source: a label for messages; defaults to the file name.

    Returns:
        The recording as :class:`Audio`, already mono and float64.

    Raises:
        FileNotFoundError: a path was given and nothing is there.
        ValueError: the path is not a WAV file, the samples are empty, or the
            sample rate is not a positive number.
        TypeError: the input is not one of the accepted shapes.
    """
    if isinstance(audio, Audio):
        return audio

    pair = _as_pair(audio)
    if pair is not None:
        samples, paired_rate = pair
        if sample_rate is not None and float(sample_rate) != float(paired_rate):
            raise ValueError(
                "two different sample rates were given: {} with the samples and {} as "
                "sample_rate".format(paired_rate, sample_rate)
            )
        return load_audio(samples, sample_rate=paired_rate, source=source)

    if isinstance(audio, (str, os.PathLike)):
        loaded = read_wav(audio)
        if sample_rate is not None and int(sample_rate) != loaded.sample_rate:
            loaded.notes.append(
                "the file says {} Hz, so the sample_rate={} passed with it was "
                "ignored".format(loaded.sample_rate, int(sample_rate))
            )
        if source is not None:
            loaded.source = source
        return loaded

    if isinstance(audio, (bytes, bytearray, memoryview)):
        raise TypeError(
            "raw bytes carry no sample format; hand in a .wav path, a numpy array of "
            "samples, or a (samples, sample_rate) pair"
        )

    try:
        values = np.asarray(audio)
    except Exception as exc:  # pragma: no cover - exotic objects
        raise TypeError(
            "audio must be a .wav path, a numpy array of samples, or a "
            "(samples, sample_rate) pair, got {}".format(type(audio).__name__)
        ) from exc
    if values.dtype.kind not in "iufb":
        raise TypeError(
            "audio samples must be numbers, got an array of {}; a .wav path, a numpy "
            "array or a (samples, sample_rate) pair are accepted".format(values.dtype)
        )
    if values.ndim == 0:
        raise ValueError("a single number is not a recording; pass an array of samples")

    notes: List[str] = []
    assumed = sample_rate is None
    if assumed:
        rate = DEFAULT_SAMPLE_RATE
        notes.append(
            "no sample rate was given, so {} Hz was assumed; pass sample_rate= for the "
            "frequency measures to mean anything".format(DEFAULT_SAMPLE_RATE)
        )
    else:
        rate_value = float(sample_rate)
        if not np.isfinite(rate_value) or rate_value <= 0:
            raise ValueError(
                "sample_rate must be a positive number of samples per second, got "
                "{!r}".format(sample_rate)
            )
        rate = int(round(rate_value))

    scaled = _scale_integers(values, notes)
    mono, channels = _to_mono(scaled, notes)
    return _finish(
        np.array(mono, dtype=np.float64, copy=True),
        rate,
        channels=channels,
        source=source,
        notes=notes,
        assumed_sample_rate=assumed,
    )
