"""Turn whatever the caller passed into ``(mono float64 samples, sample_rate)``.

Three input shapes are accepted everywhere audio is taken - a ``.wav`` path, a
numpy array, or a ``(samples, sample_rate)`` pair - so ``detect``, ``Monitor``
and ``spectral_profile`` never disagree about what counts as audio.

The array handed back is always a fresh one. Nothing in this package writes
into an array the caller still holds.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ._wav import read_wav

#: More channels than this and the array was almost certainly handed over
#: transposed; it is still mixed down, but the report says so.
MAX_SENSIBLE_CHANNELS = 16


def _is_scalar_number(value: Any) -> bool:
    """True for a plain int/float-like scalar, excluding bools and arrays."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return isinstance(value, np.ndarray) and value.ndim == 0


def _looks_like_pair(audio: Any) -> bool:
    """True for a ``(samples, sample_rate)`` pair rather than a 2-sample clip."""
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        return False
    if _is_scalar_number(audio[0]):
        # Two bare numbers are a two-sample waveform, not a pair.
        return False
    return _is_scalar_number(audio[1])


def _to_float_samples(data: Any, label: str) -> np.ndarray:
    """A fresh float64 array from anything array-like, scaled to roughly [-1, 1]."""
    try:
        raw = np.asarray(data)
    except Exception as exc:  # pragma: no cover - numpy refuses very odd objects
        raise TypeError(
            "%s could not be read as a numpy array: %s" % (label, exc)
        ) from exc
    if raw.dtype.kind == "b":
        raise TypeError("%s holds booleans, not audio samples" % label)
    if raw.dtype == object or raw.dtype.kind in "USMmVc":
        raise TypeError(
            "%s holds %s values, not numbers; pass a numeric array, a .wav path, "
            "or (samples, sample_rate)" % (label, raw.dtype)
        )
    # copy=True is what keeps the caller's array untouched, including the
    # already-float64 case where asarray would otherwise hand back the original.
    samples = np.array(raw, dtype=np.float64, copy=True)
    if raw.dtype.kind == "i":
        # Integer PCM: scale by the format's full scale so +/-1.0 means the rail.
        samples /= float(abs(np.iinfo(raw.dtype).min))
    elif raw.dtype.kind == "u":
        half = (float(np.iinfo(raw.dtype).max) + 1.0) / 2.0
        samples = (samples - half) / half
    return samples


def _mix_to_mono(samples: np.ndarray, label: str, notes: List[str]) -> np.ndarray:
    """Average multi-channel audio down to one channel, recording that it happened."""
    if samples.ndim == 1:
        return np.ascontiguousarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(
            "%s has %d dimensions; audio is 1-D (mono) or 2-D (samples x "
            "channels)" % (label, samples.ndim)
        )
    rows, cols = samples.shape
    if cols <= rows:
        channels, mono = cols, samples.mean(axis=1)
    else:
        # Handed over as (channels, samples); the long axis is always time.
        channels, mono = rows, samples.mean(axis=0)
    if channels > 1:
        if channels > MAX_SENSIBLE_CHANNELS:
            notes.append(
                "%s had %d channels, which is more than a recording usually has; "
                "they were averaged to mono anyway" % (label, channels)
            )
        else:
            notes.append(
                "%s was %d-channel and was mixed down to mono" % (label, channels)
            )
    return np.ascontiguousarray(mono, dtype=np.float64)


def load_audio(
    audio: Any,
    sample_rate: Optional[int] = None,
    label: str = "audio",
) -> Tuple[np.ndarray, int, List[str]]:
    """Return ``(mono_samples, sample_rate, notes)`` for any accepted input.

    Args:
        audio: a ``.wav`` path, a numpy array of samples, or a
            ``(samples, sample_rate)`` pair.
        sample_rate: samples per second. Required for a bare array; an input
            that carries its own rate keeps it, and the disagreement is noted.
        label: how this input is named in messages, e.g. ``"reference"``.

    Returns:
        ``(samples, sample_rate, notes)``: a fresh 1-D float64 array, the rate
        in Hz, and plain-language notes about anything that was changed.

    Raises:
        ValueError: the rate is missing or impossible, or the file is unreadable.
        TypeError: the object is not audio at all.
    """
    notes: List[str] = []
    given_rate = None if sample_rate is None else int(sample_rate)

    if _looks_like_pair(audio):
        data, pair_rate = audio[0], int(audio[1])
        if given_rate is not None and given_rate != pair_rate:
            notes.append(
                "%s carried its own sample rate of %d Hz, which was used instead "
                "of the sample_rate=%d that was passed"
                % (label, pair_rate, given_rate)
            )
        samples = _to_float_samples(data, label)
        rate = pair_rate
    elif isinstance(audio, (str, os.PathLike)):
        path = os.fspath(audio)
        samples, rate = read_wav(path)
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
        samples = _to_float_samples(audio, label)
        rate = given_rate
    else:
        raise TypeError(
            "%s is a %s; pass a .wav path, a numpy array, or (samples, "
            "sample_rate)" % (label, type(audio).__name__)
        )

    if rate <= 0:
        raise ValueError(
            "%s has a sample rate of %d Hz, which cannot be true; it must be a "
            "positive number of samples per second" % (label, rate)
        )

    samples = _mix_to_mono(samples, label, notes)

    bad = ~np.isfinite(samples)
    n_bad = int(bad.sum())
    if n_bad:
        # A NaN would poison every frame it touches and every statistic after
        # it, so it is read as silence and reported, never propagated.
        samples[bad] = 0.0
        notes.append(
            "%s held %d sample%s that was not a finite number (NaN or infinity); "
            "they were read as silence"
            % (label, n_bad, "" if n_bad == 1 else "s")
        )
    return samples, rate, notes


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linearly resample ``samples`` from ``from_rate`` to ``to_rate``.

    Linear interpolation is enough here: the result is only ever used to build
    an average spectrum, where the modest high-frequency roll-off it introduces
    is far smaller than the departure the detector is looking for.
    """
    if from_rate == to_rate or samples.size == 0:
        return samples
    duration = samples.size / float(from_rate)
    n_out = int(round(duration * to_rate))
    if n_out < 1:
        return samples[:0]
    source_t = np.arange(samples.size, dtype=np.float64) / float(from_rate)
    target_t = np.arange(n_out, dtype=np.float64) / float(to_rate)
    return np.interp(target_t, source_t, samples)
