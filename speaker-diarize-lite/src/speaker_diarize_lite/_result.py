"""The result objects: :class:`Segment` and :class:`Diarization`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_SUMMARY_SEGMENTS = 25


@dataclass(frozen=True)
class Segment:
    """One stretch of one person talking.

    Attributes:
        start_s: start time in seconds from the beginning of the recording.
        end_s: end time in seconds.
        speaker: label such as ``"SPEAKER_00"``; speakers are numbered in the
            order they first talk.
        confidence: 0 to 1, how much closer this stretch sits to its own
            speaker than to the next most similar one. 1.0 when only one
            speaker was found. It ranks segments by how clear-cut they are; it
            is not a calibrated probability.
    """

    start_s: float
    end_s: float
    speaker: str
    confidence: float

    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return self.end_s - self.start_s

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the segment."""
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "speaker": self.speaker,
            "confidence": round(self.confidence, 3),
        }


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 3)


@dataclass
class Diarization:
    """Who spoke when, and how that was worked out.

    Attributes:
        segments: non-overlapping :class:`Segment` objects sorted by start time.
        estimated_speakers: True when the number of speakers was estimated from
            the recording, False when it was given as ``num_speakers``.
        duration_s: length of the recording in seconds.
        speech_s: seconds the energy detector judged to be someone talking.
        sample_rate: sample rate of the analysed signal.
        method: ``"spectral"`` (hand-built log mel-band features) or
            ``"embed"`` (the caller's embedding function).
        requested_speakers: ``num_speakers`` as passed, or None.
        evidence_speakers: how many speakers the clustering itself supports,
            whatever was requested.
        separation: smallest separation between any two reported speakers, in
            pooled spreads; for one speaker, the best any two-way split
            reached. None when there was nothing to compare, or when the voices
            had no spread at all to measure against.
        separation_threshold: separation two voices need to count as distinct.
        windows: number of analysis windows clustered.
        file_id: recording name used in RTTM output.
        warnings: problems that make the result less trustworthy.
        notes: what was done along the way, for the record.
        settings: the parameters used.
    """

    segments: List[Segment]
    estimated_speakers: bool
    duration_s: float
    speech_s: float
    sample_rate: int
    method: str
    requested_speakers: Optional[int] = None
    evidence_speakers: int = 0
    separation: Optional[float] = None
    separation_threshold: float = 0.0
    windows: int = 0
    file_id: str = "audio"
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def speakers(self) -> List[str]:
        """Speaker labels in order of first appearance."""
        seen: List[str] = []
        for segment in self.segments:
            if segment.speaker not in seen:
                seen.append(segment.speaker)
        return seen

    @property
    def num_speakers(self) -> int:
        """How many distinct speakers the segments hold."""
        return len(self.speakers)

    @property
    def speaking_time(self) -> Dict[str, float]:
        """Seconds of talking per speaker, in order of first appearance."""
        totals: Dict[str, float] = {name: 0.0 for name in self.speakers}
        for segment in self.segments:
            totals[segment.speaker] += segment.duration_s
        return {name: round(value, 3) for name, value in totals.items()}

    def speaker_at(self, time_s: float) -> Optional[str]:
        """The speaker talking at ``time_s``, or None if nobody is."""
        for segment in self.segments:
            if segment.start_s <= time_s < segment.end_s:
                return segment.speaker
        return None

    def to_rttm(self, file_id: Optional[str] = None) -> str:
        """The segments in RTTM, the plain-text format diarization tools exchange.

        One ``SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>``
        line per segment. ``file_id`` defaults to the recording's file name
        without extension (``"audio"`` for arrays); whitespace in it is
        replaced by underscores because RTTM fields are space separated.
        """
        name = file_id if file_id is not None else self.file_id
        name = "_".join(str(name).split()) or "audio"
        lines = [
            "SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>".format(
                name, segment.start_s, segment.duration_s, segment.speaker
            )
            for segment in self.segments
        ]
        return "".join(line + "\n" for line in lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of everything in the result."""
        return {
            "file_id": self.file_id,
            "duration_s": round(self.duration_s, 3),
            "speech_s": round(self.speech_s, 3),
            "sample_rate": int(self.sample_rate),
            "method": self.method,
            "num_speakers": self.num_speakers,
            "estimated_speakers": bool(self.estimated_speakers),
            "requested_speakers": self.requested_speakers,
            "evidence_speakers": int(self.evidence_speakers),
            "separation": _finite_or_none(self.separation),
            "separation_threshold": round(float(self.separation_threshold), 3),
            "windows": int(self.windows),
            "speakers": self.speakers,
            "speaking_time": self.speaking_time,
            "segments": [segment.to_dict() for segment in self.segments],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "settings": dict(self.settings),
        }

    def _count_line(self) -> str:
        count = self.num_speakers
        if count == 0:
            return "no speakers found"
        noun = "speaker" if count == 1 else "speakers"
        how = "estimated from the recording" if self.estimated_speakers else "as requested"
        return "{} {} ({})".format(count, noun, how)

    def _separation_line(self) -> Optional[str]:
        if self.windows < 2:
            return None
        bar = "distinct at {:.1f} or more".format(self.separation_threshold)
        if self.separation is None or not math.isfinite(self.separation):
            if self.num_speakers >= 2:
                return "separation: complete (no spread within each voice to measure against)"
            return None
        if self.num_speakers >= 2:
            return "separation: closest pair of voices {:.1f} spreads apart ({})".format(self.separation, bar)
        return "separation: best split into two voices {:.1f} spreads apart ({})".format(self.separation, bar)

    def summary(self) -> str:
        """A short human-readable report in plain ASCII punctuation."""
        method = (
            "hand-built spectral features (log mel-band energies and deltas)"
            if self.method == "spectral"
            else "the supplied embedding function"
        )
        lines = [
            "speaker-diarize-lite: {} in {:.1f} s of audio ({:.1f} s of speech)".format(
                self._count_line(), self.duration_s, self.speech_s
            ),
            "recording: {}".format(self.file_id),
            "method: {}, {} windows".format(method, self.windows),
        ]
        separation = self._separation_line()
        if separation:
            lines.append(separation)
        if self.requested_speakers is not None and self.evidence_speakers != self.requested_speakers:
            lines.append("the recording itself supports {} speaker(s)".format(self.evidence_speakers))
        if self.segments:
            total = sum(self.speaking_time.values()) or 1.0
            lines.append("speaking time:")
            for name, seconds in self.speaking_time.items():
                count = sum(1 for s in self.segments if s.speaker == name)
                lines.append(
                    "  {}  {:7.1f} s  {:5.1f}%  {} segment{}".format(
                        name, seconds, 100.0 * seconds / total, count, "" if count == 1 else "s"
                    )
                )
            lines.append("segments:")
            for segment in self.segments[:_SUMMARY_SEGMENTS]:
                lines.append(
                    "  {:8.2f} - {:8.2f}  {}  confidence {:.2f}".format(
                        segment.start_s, segment.end_s, segment.speaker, segment.confidence
                    )
                )
            if len(self.segments) > _SUMMARY_SEGMENTS:
                lines.append("  ... and {} more (see to_dict() or to_rttm())".format(
                    len(self.segments) - _SUMMARY_SEGMENTS))
        else:
            lines.append("no speech segments found")
        if self.warnings:
            lines.append("warnings:")
            lines.extend("  - {}".format(message) for message in self.warnings)
        if self.notes:
            lines.append("notes:")
            lines.extend("  - {}".format(message) for message in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            "Diarization(num_speakers={}, estimated_speakers={}, segments={}, "
            "speech_s={:.1f}, duration_s={:.1f}, method={!r})".format(
                self.num_speakers, self.estimated_speakers, len(self.segments),
                self.speech_s, self.duration_s, self.method,
            )
        )
