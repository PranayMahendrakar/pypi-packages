"""The result objects: one :class:`Event` per anomaly, one :class:`AudioAnomalyReport` per run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

#: Every kind of anomaly the detector reports, in the order summaries list them.
KINDS = ("burst", "tonal", "level_rise", "level_drop", "dropout", "clipping")

#: One line per kind, used by :meth:`AudioAnomalyReport.summary` and the README.
KIND_MEANINGS = {
    "burst": "a short broadband sound: an impact, knock, click or rattle",
    "tonal": "narrowband energy that was not there before: a whine, hum or squeal",
    "level_rise": "the sound got louder across much of the spectrum and stayed louder",
    "level_drop": "the overall level fell well below normal",
    "dropout": "the signal cut out to (near) silence in the middle of the sound",
    "clipping": "samples flattened against the full-scale rail",
}

#: Summaries list at most this many events; the rest stay in ``.anomalies``.
MAX_LISTED = 15


@dataclass(frozen=True)
class Event:
    """One anomaly.

    Attributes:
        start_s: when it starts, in seconds from the start of the recording.
        end_s: when it ends, in seconds.
        kind: one of ``burst``, ``tonal``, ``level_rise``, ``level_drop``,
            ``dropout``, ``clipping``.
        score: how far from normal it went, in the same units as ``sensitivity``
            (robust standard deviations, and at least 2 dB per unit).
        message: one plain-English sentence saying what happened.
    """

    start_s: float
    end_s: float
    kind: str
    score: float
    message: str

    def __post_init__(self) -> None:
        # Plain Python types, whatever numpy scalars were passed in.
        object.__setattr__(self, "start_s", float(self.start_s))
        object.__setattr__(self, "end_s", float(self.end_s))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "message", str(self.message))

    @property
    def duration_s(self) -> float:
        """How long the event lasts, in seconds."""
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        """JSON-safe form."""
        return {
            "start_s": round(float(self.start_s), 4),
            "end_s": round(float(self.end_s), 4),
            "duration_s": round(float(self.duration_s), 4),
            "kind": self.kind,
            "score": round(float(self.score), 3),
            "message": self.message,
        }

    @property
    def heading(self) -> str:
        """The kind, time span and score on one short line."""
        return "%s at %.3f-%.3f s, score %.1f" % (self.kind, self.start_s, self.end_s, self.score)

    def __str__(self) -> str:
        return "%s: %s" % (self.heading, self.message)


@dataclass
class AudioAnomalyReport:
    """Everything one run of the detector found, and how it got there.

    Attributes:
        anomalies: the events, in time order.
        scores: one score per analysis frame; a frame at or above
            ``sensitivity`` is anomalous. Empty when the recording is shorter
            than one frame.
        frame_times: the centre of each frame, in seconds, aligned with ``scores``.
        anomaly_ratio: share of the recording's duration covered by anomalies,
            0.0 to 1.0.
        sample_rate: samples per second of the analysed audio.
        duration_s: length of the analysed audio, in seconds.
        frame_ms: analysis frame length, in milliseconds.
        sensitivity: the threshold the scores were judged against.
        mode: ``"self"`` when the recording was judged against its own typical
            sound, ``"reference"`` when against a known-good recording or profile.
        compared_against: what "normal" meant for this run, in words.
        source: the file that was analysed, or None for an array.
        notes: what was assumed, converted or skipped along the way.
        warnings: things worth fixing before trusting the result.
    """

    anomalies: List[Event]
    scores: np.ndarray
    frame_times: np.ndarray
    anomaly_ratio: float
    sample_rate: int
    duration_s: float
    frame_ms: float
    sensitivity: float
    mode: str
    compared_against: str
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        """How many analysis frames were scored."""
        return int(self.scores.size)

    @property
    def max_score(self) -> float:
        """The highest frame score, 0.0 when nothing was scored."""
        return float(np.max(self.scores)) if self.scores.size else 0.0

    @property
    def counts(self) -> Dict[str, int]:
        """How many events of each kind, in :data:`KINDS` order, only kinds that occurred."""
        tally: Dict[str, int] = {}
        for e in self.anomalies:
            tally[e.kind] = tally.get(e.kind, 0) + 1
        order = {kind: i for i, kind in enumerate(KINDS)}
        return dict(sorted(tally.items(), key=lambda item: order.get(item[0], len(KINDS))))

    @property
    def has_anomalies(self) -> bool:
        """True when at least one anomaly was found."""
        return bool(self.anomalies)

    def loudest(self, n: int = 5) -> List[Event]:
        """The ``n`` anomalies with the highest scores, strongest first."""
        n = max(0, int(n))
        ranked = sorted(self.anomalies, key=lambda e: (-e.score, e.start_s, e.kind))
        return ranked[:n]

    def of_kind(self, kind: str) -> List[Event]:
        """The anomalies of one kind, in time order."""
        return [e for e in self.anomalies if e.kind == kind]

    def _where(self) -> str:
        name = self.source if self.source else "the audio"
        return "%s (%.2f s at %d Hz)" % (name, self.duration_s, self.sample_rate)

    def summary(self) -> str:
        """A short human-readable report, plain ASCII punctuation throughout."""
        lines: List[str] = []
        n = len(self.anomalies)
        if n:
            lines.append(
                "audio-anomaly: %d anomal%s in %s"
                % (n, "y" if n == 1 else "ies", self._where())
            )
        else:
            lines.append("audio-anomaly: no anomalies in %s" % self._where())
        lines.append("Normal means: %s" % self.compared_against)

        if self.n_frames == 0:
            lines.append(
                "Nothing was analysed: the audio is shorter than one %g ms frame."
                % self.frame_ms
            )
        elif n == 0:
            lines.append(
                "Nothing departed from normal: the highest frame score was %.1f "
                "against a threshold of %.1f." % (self.max_score, self.sensitivity)
            )
        else:
            parts = ["%d %s" % (count, kind) for kind, count in self.counts.items()]
            lines.append(
                "Flagged %.1f%% of the recording (%s); highest frame score %.1f, "
                "threshold %.1f." % (
                    100.0 * self.anomaly_ratio,
                    ", ".join(parts),
                    self.max_score,
                    self.sensitivity,
                )
            )
            lines.append("")
            if n > MAX_LISTED:
                shown = sorted(self.loudest(MAX_LISTED), key=lambda e: (e.start_s, e.kind))
                lines.append(
                    "The %d strongest of %d anomalies, in time order:" % (MAX_LISTED, n)
                )
            else:
                shown = list(self.anomalies)
                lines.append("Anomalies, in time order:")
            for i, event in enumerate(shown, 1):
                lines.append("%3d. %s" % (i, event.heading))
                lines.append("     %s" % event.message)
            if n > MAX_LISTED:
                lines.append("     ... %d more in report.anomalies" % (n - MAX_LISTED))
            lines.append("")
            lines.append("What the kinds mean:")
            for kind in self.counts:
                if kind in KIND_MEANINGS:
                    lines.append("  %-10s  %s" % (kind, KIND_MEANINGS[kind]))

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend("  - %s" % w for w in self.warnings)
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend("  - %s" % note for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Everything in the report as JSON-safe Python types."""
        return {
            "source": self.source,
            "mode": self.mode,
            "compared_against": self.compared_against,
            "sample_rate": int(self.sample_rate),
            "duration_s": round(float(self.duration_s), 4),
            "frame_ms": float(self.frame_ms),
            "sensitivity": float(self.sensitivity),
            "n_frames": self.n_frames,
            "anomaly_ratio": round(float(self.anomaly_ratio), 6),
            "max_score": round(self.max_score, 3),
            "counts": dict(self.counts),
            "anomalies": [e.to_dict() for e in self.anomalies],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "frame_times": [round(float(t), 4) for t in self.frame_times],
            "scores": [round(float(s), 3) for s in self.scores],
        }

    def __repr__(self) -> str:
        return "AudioAnomalyReport(%d anomalies, %.2f s, mode=%r, anomaly_ratio=%.3f)" % (
            len(self.anomalies),
            self.duration_s,
            self.mode,
            self.anomaly_ratio,
        )
