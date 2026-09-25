"""What an assessment hands back: one measure, one recording, one batch.

A number on its own is not an answer. Every measure carries the raw value it
found, a 0-100 score, whether that passes, and a sentence saying what it means,
so a report can be read straight out rather than looked up in documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ._scoring import clamp, round_or_none
from .thresholds import DECISIVE_MEASURES, MEASURE_WEIGHTS, MEASURES, grade_for

__all__ = [
    "Metric",
    "AudioReport",
    "BatchReport",
    "blocking_measures",
    "measured_coverage",
    "overall_score",
]

MIN_COVERAGE = 0.5
"""Share of the measure weight that must have been taken for a verdict to mean anything."""


@dataclass
class Metric:
    """One measurement, scored and explained.

    Attributes:
        value: the headline raw number, in the unit named by ``unit``. ``None``
            when the recording gave nothing to measure.
        score: 0-100, higher is better.
        ok: True when this measure is not a problem, False when it is, and
            ``None`` when the measure could not be taken at all - which is not
            a pass and must not read as one.
        message: one sentence a human can act on.
        name: which measure this is.
        unit: the unit ``value`` is in, such as ``"dBFS"`` or ``"share"``.
        measured: False when the recording could not support the measurement,
            in which case ``score`` is a neutral placeholder rather than a
            judgement.
        details: every other number this measure produced.
    """

    value: Optional[float]
    score: float
    ok: Optional[bool]
    message: str
    name: str = ""
    unit: str = ""
    measured: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = clamp(self.score)
        self.ok = None if self.ok is None else bool(self.ok)
        self.measured = bool(self.measured)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping of this measure."""
        return {
            "name": self.name,
            "value": round_or_none(self.value),
            "unit": self.unit,
            "score": round(float(self.score), 1),
            "ok": None if self.ok is None else bool(self.ok),
            "measured": bool(self.measured),
            "message": self.message,
            "details": dict(self.details),
        }

    def line(self) -> str:
        """One plain-text line for a summary."""
        if self.value is None:
            shown = "n/a"
        elif self.unit == "share":
            shown = "{:.1%}".format(float(self.value))
        elif abs(float(self.value)) >= 1000:
            shown = "{:.0f} {}".format(float(self.value), self.unit).strip()
        else:
            shown = "{:.2f} {}".format(float(self.value), self.unit).strip()
        if self.ok is None:
            # Not a pass and not a failure: nothing was measured, and a reader
            # scanning this column must not come away thinking it passed.
            mark = "--  "
        else:
            mark = "ok  " if self.ok else "FAIL"
        return "  {} {:<10} {:>5.1f}  {:<14} {}".format(
            mark, self.name, self.score, shown, self.message
        )


def measured_coverage(metrics: Dict[str, Metric]) -> float:
    """How much of the measure weight was actually taken, 0.0 to 1.0.

    A recording too short, too quiet or too strange for most of the measures
    comes back with neutral placeholder scores, which average out to a
    respectable-looking number that means nothing. Coverage is what lets a
    report refuse to call such a recording usable.

    Args:
        metrics: the measures, keyed by name.

    Returns:
        The measured share of the total weight; ``0.0`` for an empty mapping.
    """
    measured = 0.0
    total = 0.0
    for name, metric in metrics.items():
        weight = float(MEASURE_WEIGHTS.get(name, 1.0))
        total += weight
        if metric.measured:
            measured += weight
    if total <= 0.0:
        return 0.0
    return measured / total


def blocking_measures(metrics: Dict[str, Metric]) -> List[str]:
    """The decisive measures that failed, in reading order.

    A weighted average lets six good measures carry one catastrophic one: with
    the shipped weights a single measure scoring zero still leaves the overall
    score in the eighties, so a 7% clipped take, a telephone-band file and a
    recording that is not speech at all all came back "usable". The measures
    named in :data:`~speech_quality.thresholds.DECISIVE_MEASURES` describe
    faults nothing later puts right, so one of them failing settles the verdict
    on its own.

    Args:
        metrics: the measures, keyed by name.

    Returns:
        The names that failed, in the order the report reads them. A measure
        that could not be taken never blocks: it has nothing to report, which
        is not the same as failing.
    """
    return [
        name
        for name in DECISIVE_MEASURES
        if name in metrics and metrics[name].measured and not metrics[name].ok
    ]


def _join(names: Sequence[str]) -> str:
    '''"a", "a and b", or "a, b and c".'''
    items = [str(name) for name in names]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "{} and {}".format(", ".join(items[:-1]), items[-1])


def overall_score(metrics: Dict[str, Metric]) -> float:
    """Weighted average of the measure scores, 0-100.

    Args:
        metrics: the measures, keyed by name.

    Returns:
        The overall score. An empty mapping scores 0.
    """
    total = 0.0
    weight_sum = 0.0
    for name, metric in metrics.items():
        weight = float(MEASURE_WEIGHTS.get(name, 1.0))
        total += weight * float(metric.score)
        weight_sum += weight
    if weight_sum <= 0.0:
        return 0.0
    return clamp(total / weight_sum)


@dataclass
class AudioReport:
    """What one recording measured, scored and explained.

    Attributes:
        score: 0-100 overall, the weighted average of the measures.
        grade: "A" (best) through "F".
        usable: True when the recording is good enough to transcribe or
            publish: the overall score clears ``usable_score``, the recording
            is not digital silence, most of the measures could be taken, and
            none of the decisive measures failed.
        metrics: every measure, keyed by name.
        blocking: the decisive measures that failed, which is why ``usable``
            can be False on a recording whose overall score looks respectable.
        issues: what is wrong, worst first, one sentence each.
        notes: decisions taken while loading, such as a stereo mixdown.
        source: a label for the recording, usually the file name.
        duration: length in seconds.
        sample_rate: samples per second.
        channels: channels in the source, before any mixdown.
        digital_silence: True when every sample was exactly zero.
        coverage: share of the measure weight that could actually be taken.
            Below :data:`MIN_COVERAGE` the recording is never called usable,
            because most of the score would be placeholder.
    """

    score: float
    grade: str
    usable: bool
    metrics: Dict[str, Metric]
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    blocking: List[str] = field(default_factory=list)
    source: Optional[str] = None
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    digital_silence: bool = False
    coverage: float = 1.0

    @property
    def label(self) -> str:
        """The name to print for this recording."""
        return self.source or "recording"

    def metric(self, name: str) -> Metric:
        """One measure by name.

        Raises:
            KeyError: there is no such measure.
        """
        if name not in self.metrics:
            raise KeyError(
                "no measure named {!r}; this report has {}".format(
                    name, ", ".join(sorted(self.metrics))
                )
            )
        return self.metrics[name]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping of the whole report."""
        return {
            "source": self.source,
            "score": round(float(self.score), 1),
            "grade": self.grade,
            "usable": bool(self.usable),
            "blocking": list(self.blocking),
            "digital_silence": bool(self.digital_silence),
            "coverage": round(float(self.coverage), 4),
            "duration_s": round(float(self.duration), 4),
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
            "metrics": {name: self.metrics[name].to_dict() for name in self.metrics},
            "issues": list(self.issues),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """The whole report as plain text, ASCII punctuation only."""
        verdict = "usable" if self.usable else "NOT usable"
        head = "{}: {} (score {:.1f} / 100, grade {})".format(
            self.label, verdict, self.score, self.grade
        )
        shape = "  {:.2f} s, {} Hz, {}".format(
            self.duration,
            self.sample_rate,
            "mono" if self.channels <= 1 else "{} channels".format(self.channels),
        )
        lines = [head, shape]
        if self.blocking:
            lines.append(
                "  not usable: {} failed, and no later pass puts that right".format(
                    _join(self.blocking)
                )
            )
        if self.coverage < 1.0:
            lines.append(
                "  {:.0%} of the measures could be taken; the rest had nothing to work "
                "with".format(self.coverage)
            )
        lines.append("measures:")
        for name in MEASURES:
            if name in self.metrics:
                lines.append(self.metrics[name].line())
        for name in sorted(set(self.metrics) - set(MEASURES)):
            lines.append(self.metrics[name].line())
        if self.issues:
            lines.append("issues ({}):".format(len(self.issues)))
            lines.extend("  - {}".format(text) for text in self.issues)
        else:
            lines.append("issues: none")
        if self.notes:
            lines.append("notes:")
            lines.extend("  - {}".format(text) for text in self.notes)
        return "\n".join(lines)


@dataclass
class BatchReport:
    """What a set of recordings measured, and which of them need attention.

    Attributes:
        results: one report per recording that could be read, in input order.
        failures: ``(label, reason)`` for every item that could not be read.
    """

    results: List[AudioReport] = field(default_factory=list)
    failures: List[Any] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    @property
    def usable(self) -> List[AudioReport]:
        """The recordings good enough to use."""
        return [report for report in self.results if report.usable]

    @property
    def unusable(self) -> List[AudioReport]:
        """The recordings that are not good enough to use."""
        return [report for report in self.results if not report.usable]

    @property
    def mean_score(self) -> float:
        """Average overall score, 0.0 when nothing could be read."""
        if not self.results:
            return 0.0
        return float(sum(report.score for report in self.results) / len(self.results))

    def worst(self, n: int = 5) -> List[AudioReport]:
        """The ``n`` lowest-scoring recordings, worst first.

        Args:
            n: how many to return. Values below 1 return an empty list.

        Returns:
            Reports sorted by score, ties broken by label so the order is
            stable across runs.
        """
        if n < 1:
            return []
        ordered = sorted(self.results, key=lambda report: (report.score, report.label))
        return ordered[: int(n)]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping of the batch."""
        return {
            "count": len(self.results),
            "usable": len(self.usable),
            "unusable": len(self.unusable),
            "mean_score": round(self.mean_score, 1),
            "results": [report.to_dict() for report in self.results],
            "failures": [
                {"source": label, "error": reason} for label, reason in self.failures
            ],
        }

    def summary(self, worst: int = 5) -> str:
        """The batch as plain text: the headline, then the worst offenders.

        Args:
            worst: how many low scorers to list.
        """
        if not self.results and not self.failures:
            return "no recordings were assessed"
        lines = [
            "{} recording(s): {} usable, {} not, mean score {:.1f} / 100".format(
                len(self.results), len(self.usable), len(self.unusable), self.mean_score
            )
        ]
        listed = self.worst(worst)
        if listed:
            lines.append("worst {}:".format(len(listed)))
            for report in listed:
                first = report.issues[0] if report.issues else "nothing wrong"
                lines.append(
                    "  {:>5.1f} {}  {}  {}".format(
                        report.score, report.grade, report.label, first
                    )
                )
        if self.failures:
            lines.append("could not be read ({}):".format(len(self.failures)))
            for label, reason in self.failures:
                lines.append("  {}: {}".format(label, reason))
        return "\n".join(lines)


def build_report(
    metrics: Dict[str, Metric],
    issues: Sequence[str],
    notes: Iterable[str],
    source: Optional[str],
    duration: float,
    sample_rate: int,
    channels: int,
    digital_silence: bool,
    usable_score: float,
) -> AudioReport:
    """Assemble an :class:`AudioReport` from finished measures.

    Args:
        metrics: the measures, keyed by name.
        issues: problem sentences, already ordered worst first.
        notes: decisions taken while loading.
        source: label for the recording.
        duration: length in seconds.
        sample_rate: samples per second.
        channels: channels before any mixdown.
        digital_silence: True when the recording was all zeros.
        usable_score: the score at or above which a recording is usable.

    Returns:
        The finished report. Digital silence scores 0 whatever the individual
        measures said, a recording most of the measures could not touch is
        never called usable, and neither is one where a decisive measure
        failed, however well the other six average out.
    """
    coverage = measured_coverage(metrics)
    # Digital silence has no quality to score: saying "48 out of 100" about a file
    # of zeros would be arithmetic pretending to be a judgement.
    score = 0.0 if digital_silence else overall_score(metrics)
    blocking = blocking_measures(metrics)
    usable = bool(
        score >= float(usable_score)
        and not digital_silence
        and coverage >= MIN_COVERAGE
        and not blocking
    )
    return AudioReport(
        score=score,
        grade=grade_for(score),
        usable=usable,
        metrics=metrics,
        issues=list(issues),
        notes=list(notes),
        blocking=blocking,
        source=source,
        duration=float(duration),
        sample_rate=int(sample_rate),
        channels=int(channels),
        digital_silence=bool(digital_silence),
        coverage=coverage,
    )
