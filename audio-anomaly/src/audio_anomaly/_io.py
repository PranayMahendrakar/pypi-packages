"""Turn whatever the caller passed into mono float64 samples plus the facts about them.

Three input shapes are accepted everywhere audio is taken - a ``.wav`` path, a
numpy array, or a ``(samples, sample_rate)`` pair - so ``detect``, ``Monitor``
and ``spectral_profile`` never disagree about what counts as audio.

Clipping is measured here, per channel and before the mix to mono, because a
clipped left channel averaged with a clean right one no longer touches any rail.
The rail itself comes from the format: 8-bit WAV tops out at +127/128, 16-bit at
+32767/32768, float at 1.0. A single fixed threshold such as 0.995 would miss
every clipped 8-bit file, since 127/128 = 0.9922 never reaches it.

The arrays handed back are always fresh ones. Nothing in this package writes
into an array the caller still holds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ._runs import mark_long_runs
from ._wav import read_wav

#: More channels than this and the array was almost certainly handed over
#: transposed; it is still mixed down, but the report says so.
MAX_SENSIBLE_CHANNELS = 16

#: A sample counts toward clipping only inside a run of at least this many
#: consecutive samples on the rail. One sample touching full scale is a peak;
#: a flat top is clipping.
MIN_CLIP_RUN = 3

#: Float audio has no integer rail; within this distance of 1.0 counts as on it.
#: 32767/32768 (16-bit audio scaled to float) is inside it.
FLOAT_RAIL_TOLERANCE = 1e-4

#: A float array peaking above this is not audio scaled to [-1, 1] (it is most
#: likely integer samples stored as floats), so clipping cannot be judged.
FLOAT_PEAK_LIMIT = 2.0

#: A saved spectral profile has exactly these columns.
PROFILE_COLUMNS = ("frequency_hz", "level_db", "spread_db")

#: The most bands a saved profile can have; anything longer is a waveform.
MAX_PROFILE_BANDS = 1024


@dataclass
class Rail:
    """Where full scale sits for one input, and how to describe it."""

    positive: float
    negative: float
    tolerance: float
    label: str

    def to_dict(self) -> dict:
        """JSON-safe form."""
        return {
            "positive": float(self.positive),
            "negative": float(self.negative),
            "label": self.label,
        }


@dataclass
class LoadedAudio:
    """One input, reduced to what the detector needs.

    Attributes:
        samples: fresh 1-D float64 mono samples.
        sample_rate: samples per second.
        channels: how many channels the input had before mixing.
        clipped: boolean per mono sample, True where any channel sat on its
            rail inside a flat run; None when clipping could not be judged.
        rail: the full-scale rail clipping was measured against.
        source: the file path, or None for an array.
        notes: plain-language notes about anything that was changed or assumed.
        warnings: notes about something the caller should fix.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int
    clipped: Optional[np.ndarray]
    rail: Rail
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.samples.size / float(self.sample_rate)


def _is_scalar_number(value: Any) -> bool:
    """True for a plain int/float-like scalar, excluding bools and arrays."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf"


def _looks_like_pair(audio: Any) -> bool:
    """True for a ``(samples, sample_rate)`` pair rather than a 2-sample clip."""
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        return False
    if _is_scalar_number(audio[0]):
        # Two bare numbers are a two-sample waveform, not a pair.
        return False
    return _is_scalar_number(audio[1])


def as_profile(obj: Any) -> Optional[np.ndarray]:
    """Return ``obj`` as a float64 copy if it is a saved spectral profile, else None.

    A profile is what :func:`audio_anomaly.spectral_profile` returns: a 2-D array
    with one row per band and the columns ``frequency_hz, level_db, spread_db``,
    frequencies positive and strictly rising. A waveform essentially never has
    that shape, and one short enough to (a few hundred samples) would be shorter
    than any useful analysis frame anyway.
    """
    if not isinstance(obj, np.ndarray) or obj.ndim != 2 or obj.shape[1] != 3:
        return None
    if not 1 <= obj.shape[0] <= MAX_PROFILE_BANDS or obj.dtype.kind not in "iuf":
        return None
    table = np.array(obj, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(table)):
        return None
    freqs = table[:, 0]
    if freqs[0] <= 0 or np.any(np.diff(freqs) <= 0) or np.any(table[:, 2] < 0):
        return None
    return table


def _check_rate(value: Any, label: str) -> int:
    """A positive integer sample rate, or a ValueError that says what was wrong."""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "%s has a sample rate of %r, which is not a number" % (label, value)
        ) from None
    if not np.isfinite(rate) or rate < 1:
        raise ValueError(
            "%s has a sample rate of %r Hz, which cannot be true; it must be a "
            "positive number of samples per second" % (label, value)
        )
    return int(round(rate))


def _integer_width(raw: np.ndarray, signed: bool) -> int:
    """The narrowest standard PCM width (16, 24 or 32 bits; 8 when unsigned) holding ``raw``."""
    if raw.size == 0:
        return 16
    if signed:
        peak = int(max(abs(int(raw.min())), abs(int(raw.max()))))
        for bits in (16, 24, 32):
            if peak <= 2 ** (bits - 1):
                return bits
    else:
        peak = int(raw.max())
        for bits in (8, 16, 24, 32):
            if peak <= 2 ** bits - 1:
                return bits
    return 64


def _to_float_samples(
    data: Any, label: str, notes: List[str]
) -> Tuple[np.ndarray, Optional[Rail]]:
    """A fresh float64 array from anything array-like, plus its rail if the dtype fixes one."""
    try:
        raw = np.asarray(data)
    except Exception as exc:  # numpy refuses very odd objects
        raise TypeError("%s could not be read as a numpy array: %s" % (label, exc)) from exc
    if raw.dtype.kind == "b":
        raise TypeError("%s holds booleans, not audio samples" % label)
    if raw.dtype.kind not in "iuf":
        raise TypeError(
            "%s holds %s values, not real numbers; pass a numeric array, a .wav "
            "path, or (samples, sample_rate)" % (label, raw.dtype)
        )
    # copy=True keeps the caller's array untouched, including the already-float64
    # case where asarray would otherwise hand back the original object.
    samples = np.array(raw, dtype=np.float64, copy=True)
    if raw.dtype.kind in "iu":
        signed = raw.dtype.kind == "i"
        bits = raw.dtype.itemsize * 8
        if bits == 64 or not isinstance(data, np.ndarray):
            # No audio format stores 64-bit integers, and a Python list of ints
            # says nothing about its width: use the narrowest one that fits.
            bits = _integer_width(raw, signed)
            notes.append(
                "%s was %s, so its bit depth was inferred from the values: they fit "
                "%d-bit %s PCM, and were scaled as that"
                % (
                    label,
                    "a list of integers" if not isinstance(data, np.ndarray) else "64-bit integers",
                    bits,
                    "signed" if signed else "unsigned",
                )
            )
        if signed:
            # Integer PCM: scale so the negative rail is exactly -1.0.
            full = float(2 ** (bits - 1))
            samples /= full
            return samples, Rail(
                (full - 1.0) / full,
                -1.0,
                0.5 / full,
                "%d-bit integer rail (+%d/%d, -1)" % (bits, int(full) - 1, int(full)),
            )
        half = float(2 ** (bits - 1))
        samples = (samples - half) / half
        return samples, Rail(
            (half - 1.0) / half,
            -1.0,
            0.5 / half,
            "%d-bit unsigned rail (+%d/%d, -1)" % (bits, int(half) - 1, int(half)),
        )
    return samples, None


def _orient(samples: np.ndarray, label: str, notes: List[str]) -> np.ndarray:
    """Return ``(n_samples, n_channels)``, turning a ``(channels, samples)`` array round."""
    if samples.ndim == 0:
        return samples.reshape(1, 1)
    if samples.ndim == 1:
        return samples.reshape(-1, 1)
    if samples.ndim != 2:
        raise ValueError(
            "%s has %d dimensions; audio is 1-D (mono) or 2-D (samples x "
            "channels)" % (label, samples.ndim)
        )
    rows, cols = samples.shape
    if rows == 0 or cols == 0:
        # No samples at all (a WAV file holding zero frames, say): empty mono.
        return samples.reshape(0, 1)
    if cols > rows:
        # Handed over as (channels, samples); the long axis is always time.
        notes.append(
            "%s was shaped (%d, %d); the longer axis was taken as time" % (label, rows, cols)
        )
        return samples.T
    return samples


def _clip_mask(channels: np.ndarray, rail: Rail) -> np.ndarray:
    """True for every sample where any channel sits on the rail inside a flat run."""
    hit = np.zeros(channels.shape[0], dtype=bool)
    top = rail.positive - rail.tolerance
    bottom = rail.negative + rail.tolerance
    for c in range(channels.shape[1]):
        column = channels[:, c]
        on_rail = (column >= top) | (column <= bottom)
        if on_rail.any():
            hit |= mark_long_runs(on_rail, MIN_CLIP_RUN)
    return hit


def load_audio(
    audio: Any,
    sample_rate: Optional[float] = None,
    label: str = "audio",
) -> LoadedAudio:
    """Read any accepted input into a :class:`LoadedAudio`.

    Args:
        audio: a ``.wav`` path, a numpy array of samples (``(n,)`` or
            ``(n, channels)``), or a ``(samples, sample_rate)`` pair.
        sample_rate: samples per second. Required for a bare array; an input
            that carries its own rate keeps it, and the disagreement is noted.
        label: how this input is named in messages, e.g. ``"reference"``.

    Raises:
        ValueError: the rate is missing or impossible, or the file is unreadable.
        TypeError: the object is not audio at all.
    """
    notes: List[str] = []
    warnings: List[str] = []
    given_rate = None if sample_rate is None else _check_rate(sample_rate, label)
    source: Optional[str] = None
    rail: Optional[Rail] = None

    if as_profile(audio) is not None:
        raise ValueError(
            "%s looks like a saved spectral profile (one row per band: frequency_hz, "
            "level_db, spread_db), not a recording; pass it as reference= instead"
            % label
        )

    if _looks_like_pair(audio):
        data = audio[0]
        rate = _check_rate(audio[1], label)
        if given_rate is not None and given_rate != rate:
            notes.append(
                "%s carried its own sample rate of %d Hz, which was used instead "
                "of the sample_rate=%d that was passed" % (label, rate, given_rate)
            )
        samples, rail = _to_float_samples(data, label, notes)
    elif isinstance(audio, (str, os.PathLike)):
        source = os.fspath(audio)
        wav = read_wav(source)
        samples, rate = wav.samples.copy(), wav.sample_rate
        notes.extend(wav.notes)
        warnings.extend(wav.warnings)
        if wav.is_float:
            rail = Rail(1.0, -1.0, FLOAT_RAIL_TOLERANCE, "%d-bit float full scale (+/-1.0)" % wav.bits)
        else:
            full = float(2 ** (wav.bits - 1))
            rail = Rail(
                (full - 1.0) / full,
                -1.0,
                0.5 / full,
                "%d-bit PCM rail (+%d/%d, -1)" % (wav.bits, int(full) - 1, int(full)),
            )
        if given_rate is not None and given_rate != rate:
            notes.append(
                "%s is a file recorded at %d Hz, which was used instead of the "
                "sample_rate=%d that was passed" % (label, rate, given_rate)
            )
    elif isinstance(audio, (np.ndarray, Sequence)) or hasattr(audio, "__array__"):
        if given_rate is None:
            raise ValueError(
                "%s is an array, so its sample rate is unknown: pass "
                "sample_rate=44100 (or whatever it was recorded at), or hand over "
                "(samples, sample_rate)" % label
            )
        samples, rail = _to_float_samples(audio, label, notes)
        rate = given_rate
    else:
        raise TypeError(
            "%s is a %s; pass a .wav path, a numpy array, or (samples, "
            "sample_rate)" % (label, type(audio).__name__)
        )

    channels = _orient(samples, label, notes)
    n_channels = int(channels.shape[1])

    bad = ~np.isfinite(channels)
    n_bad = int(bad.sum())
    if n_bad:
        # A NaN would poison every frame it touches and every statistic after
        # it, so it is read as silence and reported, never propagated.
        channels = np.where(bad, 0.0, channels)
        notes.append(
            "%s held %d sample%s that %s not a finite number (NaN or infinity); "
            "%s read as silence"
            % (label, n_bad, "" if n_bad == 1 else "s", "was" if n_bad == 1 else "were",
               "it was" if n_bad == 1 else "they were")
        )

    clipped: Optional[np.ndarray]
    if rail is None:
        # A float array: full scale is 1.0, provided it really is scaled that way.
        peak = float(np.max(np.abs(channels))) if channels.size else 0.0
        rail = Rail(1.0, -1.0, FLOAT_RAIL_TOLERANCE, "float full scale (+/-1.0)")
        if peak > FLOAT_PEAK_LIMIT:
            warnings.append(
                "%s peaks at %.4g, far beyond full scale (1.0); it looks like integer "
                "samples stored as floats. It was analysed as given, but clipping "
                "cannot be judged - divide by 32768 (16-bit) or pass the integer "
                "array itself" % (label, peak)
            )
            clipped = None
        else:
            clipped = _clip_mask(channels, rail)
    else:
        clipped = _clip_mask(channels, rail)

    if n_channels > 1:
        if n_channels > MAX_SENSIBLE_CHANNELS:
            notes.append(
                "%s had %d channels, which is more than a recording usually has; "
                "they were averaged to mono anyway" % (label, n_channels)
            )
        else:
            notes.append("%s was %d-channel and was mixed down to mono" % (label, n_channels))
        mono = channels.mean(axis=1)
    else:
        mono = channels[:, 0]
    mono = np.array(mono, dtype=np.float64, copy=True)

    return LoadedAudio(
        samples=mono,
        sample_rate=rate,
        channels=n_channels,
        clipped=clipped,
        rail=rail,
        source=source,
        notes=notes,
        warnings=warnings,
    )
