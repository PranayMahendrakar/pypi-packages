"""Turn per-frame measurements into a speech / not-speech decision.

The decision has two halves, and both matter.

**Scoring.** Every frame gets one score in ``[0, 1]`` built from three parts:

* *loudness*, measured against the recording's own noise floor rather than
  against an absolute level, so a quiet phone recording and a loud studio one
  are judged on the same footing;
* *tonality*, from spectral flatness. Noise spreads its power evenly across the
  spectrum; speech concentrates it into harmonics and formants. This is the part
  that keeps a slammed door out and lets a whisper in, and it is the only part
  that is judged absolutely as well as relatively - because a recording that is
  *entirely* speech has no silence to compare against, and a recording that is
  entirely noise must not be rescued by comparing noise with itself;
* *busyness*, from the zero-crossing rate, scored as distance from the noise
  floor's own rate. Fricatives sit far above a low rumble and voiced speech far
  below a hiss, so the useful signal is the distance, not the direction.

The three are added, but not as equals: tonality is weighted so that loudness and
busyness **together** fall short of the default threshold. Structure is therefore
necessary and loudness alone is never sufficient, which is the whole reason a
door slam and an air conditioner stay out of the answer. See the weights below.

**Smoothing.** A raw per-frame verdict is jittery. Short gaps are bridged first,
so one breath does not split a sentence, and only then are short runs dropped,
so one loud frame is not a segment. That order is deliberate: doing it the other
way would delete the halves of a sentence before they had a chance to join up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

NOISE_QUANTILE = 0.20
"""Share of the quietest frames taken as the noise-floor sample."""

ENERGY_SCALE_DB = 8.0
"""Decibels above the noise floor that count as fully loud."""

FLATNESS_SCALE_DB = 5.0
"""Decibels of flatness difference that count as fully tonal."""

FLATNESS_GATE_DB = -5.0
"""Flatness above this is noise-like whatever the rest of the recording does."""

ZCR_SCALE = 0.08
"""Zero-crossing-rate distance from the noise floor that counts as fully busy."""

WEIGHT_TONALITY = 0.65
WEIGHT_ENERGY = 0.175
WEIGHT_ZCR = 0.175
"""Weights of the three parts of the score. They sum to 1.

Tonality carries most of the weight on purpose, and the split is the single most
important number in this package. ``WEIGHT_ENERGY + WEIGHT_ZCR`` is 0.35, which
is **below the default threshold of 0.40**: a frame that is loud and busy but
carries no spectral structure at all cannot reach the line no matter how loud it
is. That is what keeps a slammed door, a rumbling air conditioner and a passing
lorry out of the answer - all three are loud, and brown-ish noise has a wandering
level that puts stretches of it 12 dB or more over its own quiet fifth, so under
any weighting where loudness and busyness could add up on their own, a recording
of nothing but traffic comes back as somebody talking.

The guard is written into the constants rather than into a special case, so the
only way to lose it is to change these numbers. Above roughly ``sensitivity=0.6``
the threshold drops under 0.35 and the guard does relax - but by then the caller
has explicitly asked to be told about anything that stands out at all.
"""

THRESHOLD_QUIET = 0.65
"""Decision threshold at ``sensitivity=0`` - only obvious speech gets through."""

THRESHOLD_EAGER = 0.15
"""Decision threshold at ``sensitivity=1`` - almost anything gets through."""


@dataclass(frozen=True)
class NoiseProfile:
    """What the quietest frames of this particular recording look like."""

    energy_db: float
    zcr: float
    flatness_db: float
    n_frames: int


def threshold_for(sensitivity: float) -> float:
    """Map ``sensitivity`` in ``[0, 1]`` onto a decision threshold.

    Higher sensitivity means a lower threshold, so more of the recording is
    called speech. ``0.5`` sits in the middle at ``0.40``.
    """
    return THRESHOLD_QUIET - float(sensitivity) * (THRESHOLD_QUIET - THRESHOLD_EAGER)


def noise_profile(
    energy_db: np.ndarray, zcr: np.ndarray, flatness_db: np.ndarray
) -> NoiseProfile:
    """Describe the recording's noise floor from its quietest frames.

    The quietest fifth of the frames is taken as the noise sample and summarised
    with medians, which a handful of very loud or very quiet frames cannot drag
    around the way a mean can.

    Args:
        energy_db: Per-frame level in dBFS.
        zcr: Per-frame zero-crossing rate.
        flatness_db: Per-frame spectral flatness in dB.

    Returns:
        The profile. On an empty input it describes digital silence.
    """
    if energy_db.size == 0:
        return NoiseProfile(energy_db=-100.0, zcr=0.0, flatness_db=0.0, n_frames=0)
    cutoff = float(np.quantile(energy_db, NOISE_QUANTILE))
    index = np.nonzero(energy_db <= cutoff)[0]
    if index.size == 0:  # every frame above the quantile; take the single quietest
        index = np.array([int(np.argmin(energy_db))])
    return NoiseProfile(
        energy_db=float(np.median(energy_db[index])),
        zcr=float(np.median(zcr[index])),
        flatness_db=float(np.median(flatness_db[index])),
        n_frames=int(index.size),
    )


def speech_score(
    energy_db: np.ndarray,
    zcr: np.ndarray,
    flatness_db: np.ndarray,
    noise: NoiseProfile,
) -> np.ndarray:
    """Score every frame from 0 (indistinguishable from the noise floor) to 1.

    Args:
        energy_db: Per-frame level in dBFS.
        zcr: Per-frame zero-crossing rate.
        flatness_db: Per-frame spectral flatness in dB.
        noise: The recording's own noise floor, from :func:`noise_profile`.

    Returns:
        A 1-D float array of scores in ``[0, 1]``, one per frame.
    """
    if energy_db.size == 0:
        return np.zeros(0, dtype=np.float64)
    loud = np.clip((energy_db - noise.energy_db) / ENERGY_SCALE_DB, 0.0, 1.0)
    # Absolute: is this frame tonal in its own right? Relative: is it more tonal
    # than this recording's noise? Speech needs only one of the two to hold, so
    # wall-to-wall speech (no silence to compare with) and quiet speech over a
    # coloured noise floor are both caught.
    tonal_absolute = (FLATNESS_GATE_DB - flatness_db) / FLATNESS_SCALE_DB
    tonal_relative = (noise.flatness_db - flatness_db) / FLATNESS_SCALE_DB
    tonal = np.clip(np.maximum(tonal_absolute, tonal_relative), 0.0, 1.0)
    busy = np.clip(np.abs(zcr - noise.zcr) / ZCR_SCALE, 0.0, 1.0)
    return WEIGHT_ENERGY * loud + WEIGHT_TONALITY * tonal + WEIGHT_ZCR * busy


def find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """List the ``[start, stop)`` index pairs of every true run in ``mask``."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def majority(mask: np.ndarray, width: int) -> np.ndarray:
    """Keep a frame only if speech is what is mostly happening around it.

    This runs before any gap is bridged, and it is what stops the smoothing from
    backfiring. In a noisy recording the raw verdict flickers on for a frame
    here and a frame there; bridge every short gap between those and the flicker
    welds itself into one long stretch that then sails past the minimum length,
    which is how a detector ends up reporting a rumbling air conditioner as
    somebody talking. Real speech is not sparse like that - inside an utterance
    most frames are over the line, and the ones that are not sit among ones that
    are - so a majority vote over the shortest stretch the caller is willing to
    call speech separates the two before bridging can do any harm.

    The window is truncated rather than padded at the two ends of the recording,
    so a segment that starts in the first frame is judged on the frames that
    exist rather than against silence that does not.

    Args:
        mask: Raw per-frame verdict.
        width: Window in frames. ``1`` or less leaves ``mask`` alone, which is
            what a caller who asked for no minimum speech length wants.

    Returns:
        A new boolean array; ``mask`` is left alone.
    """
    out = np.asarray(mask, dtype=bool)
    if width <= 1 or out.size == 0:
        return np.array(out, dtype=bool, copy=True)
    half = width // 2
    running = np.concatenate(([0], np.cumsum(out.astype(np.int64))))
    index = np.arange(out.size)
    low = np.maximum(index - half, 0)
    high = np.minimum(index + half + 1, out.size)
    votes = running[high] - running[low]
    return (2 * votes) >= (high - low)


def smooth(
    mask: np.ndarray, min_speech_frames: int, min_silence_frames: int
) -> np.ndarray:
    """Bridge short gaps, then drop short runs.

    That order is deliberate. Doing it the other way would delete the halves of
    a sentence before they had a chance to join up.

    Args:
        mask: Per-frame verdict, already thinned by :func:`majority`.
        min_speech_frames: Runs shorter than this stop being speech.
        min_silence_frames: Interior gaps shorter than this are filled in.

    Returns:
        A new boolean array; ``mask`` is left alone.
    """
    out = np.array(mask, dtype=bool, copy=True)
    if out.size == 0:
        return out
    for start, stop in find_runs(~out):
        # Leading and trailing silence is real silence, not a gap in a sentence.
        if start == 0 or stop == out.size:
            continue
        if (stop - start) < min_silence_frames:
            out[start:stop] = True
    for start, stop in find_runs(out):
        if (stop - start) < min_speech_frames:
            out[start:stop] = False
    return out
