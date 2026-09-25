"""The entry points: assess one recording, assess a pile of them, ask one number.

Everything the package does in anger goes through :func:`assess`. It loads the
audio, frames it once, takes the spectra once, runs the seven measures over that
shared work and assembles the report. :func:`assess_batch` does the same for a
list and keeps going when one file is unreadable. :func:`signal_to_noise` and
:func:`estimate_noise_floor` are there for callers who want one number rather
than a verdict.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ._frames import Frames, Spectra, frame_signal, take_spectra
from .audio import Audio, AudioInput, load_audio
from .measures import (
    measure_bandwidth,
    measure_clipping,
    measure_dynamics,
    measure_level,
    measure_noise,
    measure_silence,
    measure_speech,
    noise_stats,
)
from .report import (
    MIN_COVERAGE,
    AudioReport,
    BatchReport,
    Metric,
    build_report,
    measured_coverage,
)
from .thresholds import MEASURES, Thresholds, ThresholdsLike, resolve_thresholds

__all__ = [
    "assess",
    "assess_batch",
    "estimate_noise_floor",
    "signal_to_noise",
]

logger = logging.getLogger(__name__)

_MEASURE_ORDER = {name: index for index, name in enumerate(MEASURES)}


def _prepare(
    audio: AudioInput,
    sample_rate: Optional[float],
    thresholds: ThresholdsLike,
    source: Optional[str],
) -> Tuple[Audio, Thresholds, Frames]:
    """Load the audio, settle the thresholds and frame the signal once."""
    loaded = load_audio(audio, sample_rate=sample_rate, source=source)
    limits = resolve_thresholds(thresholds)
    frames = frame_signal(loaded.samples, loaded.sample_rate, limits)
    return loaded, limits, frames


def _collect_issues(metrics: Dict[str, Metric], digital_silence: bool) -> List[str]:
    """The problem sentences, worst first.

    Ordered by score, so the thing most wrong is read first, with the fixed
    measure order breaking ties. A measure that could not be taken is never an
    issue: it has nothing to report, which is not the same as failing.
    """
    failing = [metric for metric in metrics.values() if not metric.ok and metric.measured]
    failing.sort(key=lambda metric: (metric.score, _MEASURE_ORDER.get(metric.name, 99)))
    issues = [metric.message for metric in failing]

    coverage = measured_coverage(metrics)
    if coverage < MIN_COVERAGE:
        untaken = sorted(name for name, metric in metrics.items() if not metric.measured)
        issues.insert(
            0,
            "there is not enough recording here to judge: {} of the {} measures had "
            "nothing to work with ({}), so no verdict is offered".format(
                len(untaken), len(metrics), ", ".join(untaken)
            ),
        )
    if digital_silence:
        issues.insert(
            0,
            "this recording is digital silence: every sample is exactly zero, so there "
            "is nothing here to transcribe or publish",
        )
    return issues


def _run_measures(
    audio: Audio, frames: Frames, spectra: Spectra, limits: Thresholds
) -> Dict[str, Metric]:
    """Every measure, in the order a report reads them."""
    return {
        "level": measure_level(audio, limits),
        "clipping": measure_clipping(audio, limits),
        "noise": measure_noise(audio, frames, limits),
        "silence": measure_silence(audio, frames, limits),
        "speech": measure_speech(audio, frames, spectra, limits),
        "dynamics": measure_dynamics(audio, frames, limits),
        "bandwidth": measure_bandwidth(audio, frames, spectra, limits),
    }


def assess(
    audio: AudioInput,
    *,
    sample_rate: Optional[float] = None,
    thresholds: ThresholdsLike = None,
    source: Optional[str] = None,
) -> AudioReport:
    """Measure whether a voice recording is clean enough to transcribe or publish.

    Seven measures are taken - level, clipping, noise, silence, speech, dynamics
    and bandwidth - each reporting the raw number it found, a 0-100 score and a
    sentence saying what it means. The overall score is their weighted average.

    Args:
        audio: a path to a ``.wav`` file, a numpy array of samples, or a
            ``(samples, sample_rate)`` pair. Multi-channel input is mixed down
            to mono and the mixdown is recorded in ``report.notes``. The
            caller's array is never written to.
        sample_rate: samples per second, needed for the array form; a file
            carries its own. Without one, 16000 Hz is assumed and said so.
        thresholds: a :class:`~speech_quality.thresholds.Thresholds`, or a dict
            of individual overrides such as ``{"min_snr_db": 20}``.
        source: a label for the report; defaults to the file name.

    Returns:
        An :class:`~speech_quality.report.AudioReport`.

    Raises:
        FileNotFoundError: a path was given and nothing is there.
        ValueError: the path is not a WAV file, the file is not readable PCM
            WAV, the samples are empty, or a threshold override is unknown.
        TypeError: the input is not one of the accepted shapes.
    """
    loaded, limits, frames = _prepare(audio, sample_rate, thresholds, source)
    spectra = take_spectra(loaded.samples, frames, limits)
    metrics = _run_measures(loaded, frames, spectra, limits)
    issues = _collect_issues(metrics, loaded.is_digital_silence)
    logger.debug(
        "assessed %s: %d samples at %d Hz, %d frames, %d spectra",
        loaded.source or "recording",
        loaded.n_samples,
        loaded.sample_rate,
        frames.count,
        spectra.count,
    )
    return build_report(
        metrics=metrics,
        issues=issues,
        notes=loaded.notes,
        source=loaded.source,
        duration=loaded.duration,
        sample_rate=loaded.sample_rate,
        channels=loaded.channels,
        digital_silence=loaded.is_digital_silence,
        usable_score=limits.usable_score,
    )


def _label_for(item: Any, index: int) -> str:
    """A name to print for a batch item that may never have had one."""
    if isinstance(item, (str, os.PathLike)):
        return os.path.basename(os.fspath(item))
    return "item {}".format(index)


def _split_labelled(item: Any, index: int) -> Tuple[str, Any]:
    """Pull a ``(label, recording)`` pair apart, leaving anything else alone."""
    if isinstance(item, tuple) and len(item) == 2:
        first, second = item
        rate_like = isinstance(second, (int, float)) and not isinstance(second, bool)
        if isinstance(first, str) and not rate_like:
            return first, second
    return _label_for(item, index), item


def assess_batch(
    items: Iterable[Any],
    *,
    sample_rate: Optional[float] = None,
    thresholds: ThresholdsLike = None,
) -> BatchReport:
    """Assess every recording in ``items`` and keep going when one cannot be read.

    Args:
        items: recordings, each anything :func:`assess` accepts. A
            ``(label, recording)`` pair labels that recording explicitly, which
            is how arrays get names.
        sample_rate: applied to every item that needs one.
        thresholds: applied to every item.

    Returns:
        A :class:`~speech_quality.report.BatchReport`. Unreadable items land in
        ``.failures`` as ``(label, reason)`` rather than raising, so one bad
        file does not lose the other ninety-nine.

    Raises:
        TypeError: ``items`` is a single path rather than a list of them, which
            would otherwise assess it one character at a time.
    """
    if isinstance(items, (str, bytes, bytearray, os.PathLike)):
        raise TypeError(
            "assess_batch wants a list of recordings, not one path; call assess() for a "
            "single recording, or pass [path]"
        )

    results: List[AudioReport] = []
    failures: List[Tuple[str, str]] = []
    for index, item in enumerate(items):
        label, recording = _split_labelled(item, index)
        try:
            report = assess(
                recording,
                sample_rate=sample_rate,
                thresholds=thresholds,
                source=label,
            )
        except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
            logger.warning("could not assess %s: %s", label, exc)
            failures.append((label, str(exc)))
            continue
        results.append(report)
    return BatchReport(results=results, failures=failures)


def _floor_and_snr(
    audio: AudioInput,
    sample_rate: Optional[float],
    thresholds: ThresholdsLike,
) -> Tuple[float, float]:
    """The noise floor in dBFS and the signal-to-noise ratio in dB."""
    _loaded, limits, frames = _prepare(audio, sample_rate, thresholds, None)
    stats = noise_stats(frames, limits)
    return stats.floor_dbfs, stats.snr_db


def signal_to_noise(
    audio: AudioInput,
    *,
    sample_rate: Optional[float] = None,
    thresholds: ThresholdsLike = None,
) -> float:
    """How far the voice rises above the noise under it, in dB.

    The loudest decile of frames stands for the voice and the quietest for the
    floor, so no voice-activity decision is needed and none can go wrong.

    Args:
        audio: anything :func:`assess` accepts.
        sample_rate: samples per second, for the array form.
        thresholds: overrides, notably the two percentiles.

    Returns:
        The ratio in dB. ``0.0`` for a recording whose level never drops, which
        is what digital silence and an unbroken tone both look like.
    """
    return _floor_and_snr(audio, sample_rate, thresholds)[1]


def estimate_noise_floor(
    audio: AudioInput,
    *,
    sample_rate: Optional[float] = None,
    thresholds: ThresholdsLike = None,
) -> float:
    """The level of the quiet parts of a recording, in dBFS.

    Args:
        audio: anything :func:`assess` accepts.
        sample_rate: samples per second, for the array form.
        thresholds: overrides, notably ``noise_floor_percentile``.

    Returns:
        The floor in dBFS, never ``-inf``: digital silence reports
        :data:`~speech_quality.thresholds.DB_FLOOR` instead.
    """
    return _floor_and_snr(audio, sample_rate, thresholds)[0]
