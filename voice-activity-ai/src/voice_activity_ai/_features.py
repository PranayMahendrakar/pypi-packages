"""Per-frame measurements: how loud, how busy, how noise-like.

Energy on its own is a bad voice detector. A slammed door is loud and is not
speech; a whisper is quiet and is. So every frame is measured three ways and the
decision in :mod:`voice_activity_ai._detect` uses all three:

* **short-time energy** - how loud the frame is, in dBFS.
* **zero-crossing rate** - how often the waveform changes sign, which separates
  hiss and fricatives from voiced sound and from low rumble.
* **spectral flatness** - the ratio of the geometric to the arithmetic mean of
  the power spectrum, in dB. A flat spectrum lands near 0 dB; one whose power is
  concentrated into peaks lands far below.

Flatness here is measured **after the frame's own broad spectral shape is
divided out**, and that detail is the difference between a detector that works
in a real room and one that does not. Measured raw, every coloured noise looks
tonal: a fan, a rumbling air conditioner and traffic through a window all put
most of their power at the bottom of the spectrum, so raw flatness calls them
"structured" and they read as wall-to-wall speech.

The shape that gets divided out is a quadratic fitted to the log spectrum
against **log frequency**. That choice is the whole trick. Noise that is
coloured like a power law - pink, brown, and everything real that sits between
them - is a straight line in log-log, so a straight line is exactly what comes
out, and the quadratic term absorbs the gentle bow that a room's rolloff adds on
top. What a smooth curve cannot follow is the comb of harmonics and the peaks
and valleys of formants, so those survive the division. After this step white,
pink, brown, high-passed hiss, low-passed rumble and a mains hum all land in a
narrow band between -2 and -5 dB, and only genuinely structured sound goes
further.

Every frame has its own mean removed before any of the three is measured. A DC
offset is not sound - no speaker moves for it and no microphone should have
produced it - but left in place it corrupts all three measurements at once: it
inflates the energy of a frame that is acoustically silent, it hides real sign
changes from the zero-crossing count, and, worst of all, windowing a constant
leaves a sharp peak at the bottom of the spectrum that reads as pure tone. A
recording with a stuck DC bias would otherwise be reported as speech from end to
end.

Frames are laid end to end with no overlap, so frame ``i`` covers samples
``[i * frame_length, (i + 1) * frame_length)`` and nothing is counted twice.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

EPS = 1e-20
"""Added inside every logarithm so digital silence gives a number, not ``-inf``."""

SILENT_DBFS = -100.0
"""Energy reported for a frame of exact zeros, and the clamp for anything below."""

ENVELOPE_DEGREE = 2
"""Degree of the log-frequency polynomial taken as the spectral envelope.

One term would be enough for a pure power law. The second absorbs a room's
gentle rolloff. A third starts to follow formants, which is exactly what must
be left behind, so two is the ceiling rather than a starting point.
"""

MIN_SPECTRUM_BINS = 4
"""Below this many usable bins a frame carries no spectrum worth measuring."""


def frame_signal(samples: np.ndarray, frame_length: int) -> np.ndarray:
    """Cut a 1-D signal into non-overlapping frames.

    Args:
        samples: 1-D float array.
        frame_length: Samples per frame; must be at least 1.

    Returns:
        A ``(n_frames, frame_length)`` array. A tail shorter than one frame is
        dropped, so a signal shorter than ``frame_length`` yields a
        ``(0, frame_length)`` array rather than an error.

    Raises:
        ValueError: ``frame_length`` is less than one sample.
    """
    if frame_length < 1:
        raise ValueError("frame_length must be at least 1 sample")
    n_frames = int(samples.size // frame_length)
    if n_frames == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    usable = n_frames * frame_length
    return np.ascontiguousarray(samples[:usable]).reshape(n_frames, frame_length)


def centre_frames(frames: np.ndarray) -> np.ndarray:
    """Remove each frame's own mean, so a DC offset cannot be mistaken for sound.

    Args:
        frames: A ``(n_frames, frame_length)`` array.

    Returns:
        A new array of the same shape; ``frames`` is left alone.
    """
    if frames.shape[0] == 0:
        return np.zeros(frames.shape, dtype=np.float64)
    return frames - frames.mean(axis=1, keepdims=True)


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Root-mean-square level of every frame, in dBFS (``0`` = full scale).

    Expects frames that have already been centred by :func:`centre_frames`, so
    the level reported is of the sound in the frame and not of any DC bias.
    """
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    power = np.mean(np.square(frames), axis=1)
    db = 10.0 * np.log10(power + EPS)
    return np.maximum(db, SILENT_DBFS)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Share of neighbouring sample pairs that change sign, in ``[0, 1]``.

    Expects frames that have already been centred by :func:`centre_frames`, so a
    DC offset or a slow drift cannot hide the crossings that are actually there.
    """
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)
    signs = np.signbit(frames)
    changes = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    return changes.astype(np.float64) / float(frames.shape[1] - 1)


def _log_frequency_basis(n_bins: int, degree: int) -> np.ndarray:
    """Return the design matrix for a polynomial in log frequency.

    The log-frequency axis is standardised to mean zero and unit spread before
    the powers are taken, which keeps the fit well conditioned whether there are
    four bins or four thousand.

    Args:
        n_bins: Number of spectrum bins, excluding DC.
        degree: Degree of the polynomial.

    Returns:
        An ``(n_bins, degree + 1)`` array.
    """
    axis = np.log(np.arange(1, n_bins + 1, dtype=np.float64))
    spread = float(axis.std())
    axis = (axis - axis.mean()) / (spread if spread > 0.0 else 1.0)
    return np.vander(axis, degree + 1)


def _remove_spectral_envelope(log_power: np.ndarray, degree: int) -> np.ndarray:
    """Subtract each frame's own smooth log-frequency trend from its spectrum.

    The trend is a least-squares polynomial fit, so it has no window to slide
    off the end of the band - which matters, because a moving average over a
    steeply tilted spectrum leaves a large false peak at each edge, and that
    artefact alone is enough to make brown noise look like continuous speech.

    Args:
        log_power: A ``(n_frames, n_bins)`` array of natural-log power.
        degree: Degree of the polynomial to remove.

    Returns:
        The residual, same shape, with mean close to zero along each row.
    """
    basis = _log_frequency_basis(log_power.shape[1], degree)
    # The pseudo-inverse is computed once for the whole recording and applied to
    # every frame at once: one deterministic decomposition, no per-frame solve.
    coefficients = log_power @ np.linalg.pinv(basis).T
    return log_power - coefficients @ basis.T


def spectral_flatness(frames: np.ndarray) -> np.ndarray:
    """Fine-structure spectral flatness of every frame, in dB.

    ``0`` means the spectrum has no fine structure left once its overall shape
    is accounted for, which is what noise of any colour looks like. Values well
    below zero mean the power is concentrated into peaks and valleys, which is
    what harmonics and formants look like.

    The frame is Hamming-windowed before the transform - its narrow main lobe
    keeps neighbouring harmonics apart, which a Hann window of this length
    blurs together - and the DC bin is dropped, because it carries offset rather
    than timbre.

    Args:
        frames: A ``(n_frames, frame_length)`` array.

    Returns:
        A 1-D array with one value per frame, at most ``0.0``.
    """
    n_frames, frame_length = frames.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    if frame_length < 4:
        # Two or three samples carry no usable spectrum; call them noise-like
        # rather than inventing structure out of a two-bin transform.
        return np.zeros(n_frames, dtype=np.float64)
    window = np.hamming(frame_length)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.square(np.abs(spectrum))[:, 1:] + EPS
    if power.shape[1] <= ENVELOPE_DEGREE + 1 or power.shape[1] < MIN_SPECTRUM_BINS:
        # With no more bins than the fit has terms there is no residual left to
        # measure, so the honest answer is "no structure found".
        return np.zeros(n_frames, dtype=np.float64)
    residual = _remove_spectral_envelope(np.log(power), ENVELOPE_DEGREE)
    whitened = np.exp(residual)
    log_mean = np.mean(np.log(whitened), axis=1)
    arithmetic = np.mean(whitened, axis=1)
    flatness_db = 10.0 * (log_mean / np.log(10.0)) - 10.0 * np.log10(arithmetic)
    # A frame of exact zeros is perfectly flat by this definition; keep that,
    # but never let rounding push the value above the 0 dB ceiling.
    return np.minimum(flatness_db, 0.0)


def frame_features(
    samples: np.ndarray, frame_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure every frame of ``samples`` three ways.

    Args:
        samples: 1-D mono float samples.
        frame_length: Samples per frame.

    Returns:
        ``(frames, energy_db, zcr, flatness_db)``; ``frames`` are the centred
        frames the measurements were taken from, and the last three are 1-D
        arrays with one entry per frame.
    """
    frames = centre_frames(frame_signal(samples, frame_length))
    return (
        frames,
        short_time_energy(frames),
        zero_crossing_rate(frames),
        spectral_flatness(frames),
    )
