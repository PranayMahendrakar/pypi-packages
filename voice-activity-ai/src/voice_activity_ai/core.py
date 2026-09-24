"""The public entry points: :func:`detect`, :func:`speech_ratio`, :func:`trim_silence`.

Everything here is a thin, well-validated shell around three steps that live in
their own modules: load the audio (:mod:`voice_activity_ai._audio`), measure
every frame (:mod:`voice_activity_ai._features`), and decide
(:mod:`voice_activity_ai._detect`).
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional

import numpy as np

from ._audio import load_audio
from ._detect import (
    find_runs,
    majority,
    noise_profile,
    smooth,
    speech_score,
    threshold_for,
)
from ._features import frame_features
from ._result import Segment, VoiceActivity

logger = logging.getLogger(__name__)

MIN_FRAME_MS = 1.0
"""Shorter than this and a frame holds too few samples to measure anything."""

MAX_FRAME_MS = 1000.0
"""Longer than this and a frame is longer than most of the words in it."""


def _check_number(name: str, value: Any, low: float, high: float) -> float:
    """Return ``value`` as a float, or say clearly why it is not usable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "{} must be a number; got {!r}".format(name, value)
        ) from None
    if not math.isfinite(number):
        raise ValueError("{} must be a finite number; got {}".format(name, number))
    if not (low <= number <= high):
        raise ValueError(
            "{} must be between {:g} and {:g}; got {:g}".format(name, low, high, number)
        )
    return number


def detect(
    audio: Any,
    *,
    sample_rate: Optional[int] = None,
    frame_ms: float = 30.0,
    sensitivity: float = 0.5,
    min_speech_ms: float = 200.0,
    min_silence_ms: float = 300.0
) -> VoiceActivity:
    """Find the stretches of a recording where someone is actually speaking.

    Every frame is scored on how far it stands out from *this recording's own*
    noise floor in loudness, in spectral fine structure and in zero-crossing
    rate, so a quiet phone recording and a loud studio one are judged the same
    way and no absolute level is baked in. The raw verdict is then smoothed:
    short gaps are bridged before short runs are dropped, so one breath does not
    split a sentence and one door slam does not become a segment.

    Args:
        audio: A path to a ``.wav`` file, a numpy array of samples (1-D mono or
            2-D samples-by-channels), or a ``(samples, sample_rate)`` pair.
            Integer arrays are scaled by their dtype range. The caller's array is
            never written to.
        sample_rate: Frames per second. Required when ``audio`` is a bare array;
            for a file or a pair it must either be omitted or agree.
        frame_ms: Length of one analysis frame in milliseconds. Frames do not
            overlap, so this is also the resolution of every reported boundary.
        sensitivity: 0 to 1. Higher calls more of the recording speech. The
            default of 0.5 is tuned so ordinary room noise stays out.
        min_speech_ms: Speech runs shorter than this are dropped, so a single
            loud transient does not become a segment.
        min_silence_ms: Gaps *inside* speech shorter than this are bridged, so a
            breath or a stop consonant does not split one sentence into two.

    Returns:
        A :class:`~voice_activity_ai.VoiceActivity`. A recording shorter than one
        frame gives an empty result rather than an error, and a recording with
        nothing above the threshold gives no segments rather than one long one.

    Raises:
        ValueError: An argument is out of range, or the audio cannot be read as
            real-valued samples with a sample rate.
        FileNotFoundError: A path was given and no file is there.

    Example:
        >>> import numpy as np
        >>> from voice_activity_ai import detect
        >>> quiet = np.zeros(16000)
        >>> detect(quiet, sample_rate=16000).n_segments
        0
    """
    frame_ms = _check_number("frame_ms", frame_ms, MIN_FRAME_MS, MAX_FRAME_MS)
    sensitivity = _check_number("sensitivity", sensitivity, 0.0, 1.0)
    min_speech_ms = _check_number("min_speech_ms", min_speech_ms, 0.0, 1e6)
    min_silence_ms = _check_number("min_silence_ms", min_silence_ms, 0.0, 1e6)

    samples, rate, source, notes = load_audio(audio, sample_rate)
    notes = list(notes)

    frame_length = int(round(rate * frame_ms / 1000.0))
    if frame_length < 1:
        frame_length = 1
        notes.append(
            "frame_ms={:g} is under one sample at {} Hz, so frames are one "
            "sample long".format(frame_ms, rate)
        )
    actual_frame_ms = 1000.0 * frame_length / float(rate)
    duration_s = float(samples.size) / float(rate)
    threshold = threshold_for(sensitivity)

    frames, energy_db, zcr, flatness_db = frame_features(samples, frame_length)
    noise = noise_profile(energy_db, zcr, flatness_db)
    scores = speech_score(energy_db, zcr, flatness_db, noise)

    if frames.shape[0] == 0 and samples.size:
        notes.append(
            "the recording is {:.3f}s, shorter than one {:.0f} ms frame, so "
            "there was nothing to score".format(duration_s, actual_frame_ms)
        )

    # A minimum of zero means "no minimum", and one frame is the smallest run
    # that can exist, so it is the right floor for min_speech_frames.
    min_speech_frames = max(1, int(round(min_speech_ms / actual_frame_ms)))
    min_silence_frames = int(round(min_silence_ms / actual_frame_ms))
    raw_mask = scores >= threshold
    # Thin the flicker out before bridging, then bridge, then drop what is left
    # too short. Doing the density vote last would be too late: by then the
    # bridging has already turned scattered frames into a long segment.
    voted = majority(raw_mask, min_speech_frames)
    mask = smooth(voted, min_speech_frames, min_silence_frames)

    frame_times = np.arange(mask.size, dtype=np.float64) * frame_length / float(rate)
    segments = []  # type: List[Segment]
    for start, stop in find_runs(mask):
        start_s = start * frame_length / float(rate)
        end_s = min(stop * frame_length / float(rate), duration_s)
        segments.append(
            Segment(
                start_s=float(start_s),
                end_s=float(end_s),
                duration_s=float(max(end_s - start_s, 0.0)),
                confidence=float(np.clip(np.mean(scores[start:stop]), 0.0, 1.0)),
            )
        )

    logger.debug(
        "%s: %d frames, %d segments, noise floor %.1f dBFS, threshold %.2f",
        source,
        mask.size,
        len(segments),
        noise.energy_db,
        threshold,
    )
    return VoiceActivity(
        segments=segments,
        mask=mask,
        frame_times=frame_times,
        scores=scores,
        sample_rate=int(rate),
        frame_ms=float(actual_frame_ms),
        duration_s=float(duration_s),
        threshold=float(threshold),
        noise_floor_dbfs=float(noise.energy_db),
        source=source,
        notes=notes,
        samples=samples,
    )


def speech_ratio(audio: Any, **kwargs: Any) -> float:
    """Return the share of a recording that is speech, from 0 to 1.

    Args:
        audio: Anything :func:`detect` accepts.
        **kwargs: Passed straight through to :func:`detect`.

    Returns:
        ``0.0`` for silence, ``1.0`` for a recording that is speech throughout.

    Example:
        >>> import numpy as np
        >>> from voice_activity_ai import speech_ratio
        >>> speech_ratio(np.zeros(16000), sample_rate=16000)
        0.0
    """
    return detect(audio, **kwargs).speech_ratio


def trim_silence(audio: Any, **kwargs: Any) -> np.ndarray:
    """Return the recording with leading and trailing silence removed.

    Args:
        audio: Anything :func:`detect` accepts.
        **kwargs: Passed straight through to :func:`detect`.

    Returns:
        A fresh 1-D float64 array of mono samples. Empty when no speech was
        found. The caller's array is never modified.
    """
    return detect(audio, **kwargs).trim()
