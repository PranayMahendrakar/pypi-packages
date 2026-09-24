"""The object :func:`voice_activity_ai.detect` hands back.

A :class:`VoiceActivity` is meant to answer the obvious questions without any
further work: where is the speech, how much of the recording is speech, and give
me the audio with the dead air at the ends taken off. It also carries enough of
its own working - the per-frame verdict, the frame clock, the noise floor it
measured and the threshold it used - that a surprising answer can be explained
rather than just disbelieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One stretch of the recording that was judged to be speech."""

    start_s: float
    """Where the segment begins, in seconds from the start of the recording."""

    end_s: float
    """Where the segment ends, in seconds from the start of the recording."""

    duration_s: float
    """The length of the segment, in seconds."""

    confidence: float
    """Mean frame score over the segment, from 0 (barely) to 1 (unmistakably)."""

    def to_dict(self) -> Dict[str, float]:
        """Return the segment as a JSON-safe dict."""
        return {
            "start_s": round(float(self.start_s), 4),
            "end_s": round(float(self.end_s), 4),
            "duration_s": round(float(self.duration_s), 4),
            "confidence": round(float(self.confidence), 4),
        }


def _format_time(seconds: float) -> str:
    """Render seconds as m:ss.s so a long recording stays readable."""
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    return "{}:{:04.1f}".format(minutes, seconds - minutes * 60)


@dataclass(frozen=True)
class VoiceActivity:
    """Where speech is, and where it is not, in one recording."""

    segments: List[Segment]
    """The speech segments, in time order. Empty when nothing was speech."""

    mask: np.ndarray
    """Boolean array, one entry per analysed frame: True where speech was found."""

    frame_times: np.ndarray
    """Start time in seconds of each analysed frame, same length as ``mask``."""

    scores: np.ndarray
    """Raw frame score in ``[0, 1]`` before smoothing, same length as ``mask``."""

    sample_rate: int
    """Frames per second of the audio that was analysed."""

    frame_ms: float
    """Actual length of one frame in milliseconds, after rounding to samples."""

    duration_s: float
    """Length of the whole recording in seconds, tail included."""

    threshold: float
    """Score a frame had to beat, derived from ``sensitivity``."""

    noise_floor_dbfs: float
    """Level of the recording's own noise floor, in dBFS."""

    source: str = "<array>"
    """Where the audio came from, for messages: a file name or ``<array>``."""

    notes: List[str] = field(default_factory=list)
    """Anything decided on the caller's behalf, such as a mixdown to mono."""

    samples: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """The mono float samples that were analysed. Owned by this result."""

    @property
    def n_segments(self) -> int:
        """How many speech segments were found."""
        return len(self.segments)

    @property
    def total_speech_s(self) -> float:
        """Total speech time in seconds, across all segments."""
        return float(sum(segment.duration_s for segment in self.segments))

    @property
    def total_silence_s(self) -> float:
        """Everything that is not speech, in seconds. Never negative."""
        return max(float(self.duration_s) - self.total_speech_s, 0.0)

    @property
    def speech_ratio(self) -> float:
        """Share of the recording that is speech, from 0 to 1.

        A recording with no duration gives ``0.0`` rather than a division error.
        """
        if self.duration_s <= 0.0:
            return 0.0
        return min(self.total_speech_s / float(self.duration_s), 1.0)

    def trim(self) -> np.ndarray:
        """Return the samples with leading and trailing silence removed.

        Silence *between* segments is kept, because removing that would change
        how the recording sounds rather than just where it starts and stops.

        Returns:
            A fresh 1-D float64 array. Empty when no speech was found, which is
            the honest answer for a recording that is silence all the way
            through.
        """
        if not self.segments:
            return np.zeros(0, dtype=np.float64)
        start = int(round(self.segments[0].start_s * self.sample_rate))
        stop = int(round(self.segments[-1].end_s * self.sample_rate))
        start = max(start, 0)
        stop = min(max(stop, start), int(self.samples.size))
        return np.array(self.samples[start:stop], dtype=np.float64, copy=True)

    def summary(self) -> str:
        """Return a short human-readable report in plain ASCII."""
        lines = []
        if not self.mask.size:
            lines.append(
                "voice activity: nothing to analyse  [{}, {:.3f}s at {} Hz]".format(
                    self.source, self.duration_s, self.sample_rate
                )
            )
            lines.append(
                "  the recording is shorter than one {:.0f} ms frame, so no frame "
                "could be scored".format(self.frame_ms)
            )
        elif not self.segments:
            lines.append(
                "voice activity: no speech found  [{}, {:.3f}s at {} Hz]".format(
                    self.source, self.duration_s, self.sample_rate
                )
            )
            lines.append(
                "  {} frames scored, best {:.2f}, threshold {:.2f}; noise floor "
                "{:.1f} dBFS".format(
                    self.mask.size,
                    float(self.scores.max()),
                    self.threshold,
                    self.noise_floor_dbfs,
                )
            )
        else:
            lines.append(
                "voice activity: {:.1f}% speech in {} segment{}  "
                "[{}, {:.3f}s at {} Hz]".format(
                    100.0 * self.speech_ratio,
                    self.n_segments,
                    "" if self.n_segments == 1 else "s",
                    self.source,
                    self.duration_s,
                    self.sample_rate,
                )
            )
            lines.append(
                "  speech {:.2f}s, silence {:.2f}s; noise floor {:.1f} dBFS, "
                "threshold {:.2f}".format(
                    self.total_speech_s,
                    self.total_silence_s,
                    self.noise_floor_dbfs,
                    self.threshold,
                )
            )
            shown = self.segments[:10]
            for number, segment in enumerate(shown, start=1):
                lines.append(
                    "  {:>3}. {} - {}  ({:.2f}s, confidence {:.2f})".format(
                        number,
                        _format_time(segment.start_s),
                        _format_time(segment.end_s),
                        segment.duration_s,
                        segment.confidence,
                    )
                )
            if len(self.segments) > len(shown):
                lines.append(
                    "  ... and {} more segments".format(len(self.segments) - len(shown))
                )
        for note in self.notes:
            lines.append("  note: {}".format(note))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return the whole result as a JSON-safe dict.

        The per-frame arrays are left out: they are as long as the recording and
        belong in :attr:`mask` and :attr:`frame_times`, not in a report.
        """
        return {
            "source": self.source,
            "sample_rate": int(self.sample_rate),
            "duration_s": round(float(self.duration_s), 4),
            "frame_ms": round(float(self.frame_ms), 4),
            "n_frames": int(self.mask.size),
            "n_segments": int(self.n_segments),
            "speech_ratio": round(float(self.speech_ratio), 4),
            "total_speech_s": round(float(self.total_speech_s), 4),
            "total_silence_s": round(float(self.total_silence_s), 4),
            "threshold": round(float(self.threshold), 4),
            "noise_floor_dbfs": round(float(self.noise_floor_dbfs), 2),
            "segments": [segment.to_dict() for segment in self.segments],
            "notes": list(self.notes),
        }
