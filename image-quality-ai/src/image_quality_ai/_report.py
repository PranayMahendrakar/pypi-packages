"""The result objects: :class:`Metric`, :class:`QualityReport`, :class:`BatchReport`.

The report is meant to explain itself. Every measure carries the raw number it
computed, the 0-100 score that number became, and a sentence naming the boundary
it was judged against, so nobody has to open the source to find out why a photo
got the verdict it got.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

from ._thresholds import DEFAULT_THRESHOLDS, Thresholds

#: Overall score at or above which each letter grade starts.
GRADE_CUTOFFS = ((90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"), (0.0, "F"))

#: How many images a batch summary lists under "Worst first:" when nobody says
#: otherwise. The CLI's ``--worst N`` says otherwise.
DEFAULT_WORST = 5

_WRAP_WIDTH = 94
_LABEL_WIDTH = 11


def grade_for(score: float) -> str:
    """Letter grade for a 0 to 100 score."""
    for cutoff, letter in GRADE_CUTOFFS:
        if score >= cutoff:
            return letter
    return "F"


def _bar(value: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(100.0, value)) / 100.0 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass
class Metric:
    """One measure: what was computed, what it scored, and why.

    Attributes:
        value: the headline raw number, in the units named in :attr:`message`.
        score: that number placed on a 0 to 100 scale. 50 always means the
            measure landed exactly on its borderline threshold.
        ok: whether the score cleared ``thresholds.measure_ok_score``.
        message: one sentence naming the raw numbers and the boundary used.
        details: every other number this measure produced.
        applies: ``False`` when the measure could not be judged at all, for
            instance framing on a frame with no detail in it. Measures that do
            not apply are left out of the overall score instead of dragging it
            down, and their :attr:`score` of 50 is a placeholder, not a verdict.
        name: the measure's name.
    """

    value: float
    score: float
    ok: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    applies: bool = True
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the metric."""
        return {
            "name": self.name,
            "value": float(self.value),
            "score": round(float(self.score), 1),
            "ok": bool(self.ok),
            "applies": bool(self.applies),
            "message": self.message,
            "details": {
                key: (float(item) if isinstance(item, float) else item)
                for key, item in self.details.items()
            },
        }

    def __str__(self) -> str:                      # pragma: no cover - cosmetic
        return "{0}: {1:.1f}/100 - {2}".format(self.name or "metric", self.score, self.message)


@dataclass
class QualityReport:
    """What :func:`image_quality_ai.assess` gives back.

    Attributes:
        score: overall quality, 0 to 100, a weighted average of the measures
            that applied.
        grade: ``"A"`` to ``"F"`` for :attr:`score`.
        usable: whether the photo is good enough to pass on to a model.
        metrics: measure name to :class:`Metric`.
        issues: what is wrong, in plain language, worst first.
        image: how the file was read - size, channels, bit depth, EXIF.
        thresholds: the boundaries this photo was judged against.
        failed_gates: the measures a photo cannot fail and still be usable,
            usually ``["sharpness"]`` or ``["exposure"]``. When this is not
            empty, :attr:`score` was scaled down to keep it below the usable
            mark, so the number, the grade and :attr:`usable` always agree.
    """

    score: float
    grade: str
    usable: bool
    metrics: Dict[str, Metric] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    image: Dict[str, Any] = field(default_factory=dict)
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    failed_gates: List[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        """The file path, or a label for an in-memory image."""
        return str(self.image.get("source", "<image>"))

    def explain(self, name: str) -> str:
        """The sentence behind one measure, e.g. ``report.explain("noise")``."""
        try:
            return self.metrics[name].message
        except KeyError:
            raise KeyError(
                "no measure named {0!r}; try one of: {1}".format(
                    name, ", ".join(sorted(self.metrics))
                )
            ) from None

    def scores(self) -> Dict[str, float]:
        """Just the 0-100 score of each measure, for a quick comparison."""
        return {name: round(float(metric.score), 1) for name, metric in self.metrics.items()}

    def summary(self) -> str:
        """The human-readable report, plain ASCII so any console can print it."""
        verdict = "usable" if self.usable else "not usable"
        head = "Image quality: {0:.1f} / 100 (grade {1}) - {2}".format(
            self.score, self.grade, verdict
        )
        shape = "{0}x{1}".format(self.image.get("width", 0), self.image.get("height", 0))
        channels = {1: "greyscale", 3: "colour", 4: "colour with alpha"}.get(
            int(self.image.get("channels", 3)), "colour"
        )
        depth = "{0}-bit".format(self.image.get("bit_depth", 8))
        detail = "  {0}: {1}, {2}, {3}".format(self.source, shape, channels, depth)
        if self.image.get("orientation_applied"):
            detail += ", EXIF rotation applied"
        lines = [head, detail]
        if self.failed_gates:
            if len(self.failed_gates) > 1:
                gate = (
                    "{0} are pass/fail gates: a frame that fails either carries no usable "
                    "detail, so the score above is held below {1:g} whatever the other "
                    "measures say."
                )
            else:
                gate = (
                    "{0} is a pass/fail gate: a frame that fails it carries no usable "
                    "detail, so the score above is held below {1:g} whatever the other "
                    "measures say."
                )
            lines.extend(
                _wrapped(
                    "  ",
                    gate.format(
                        " and ".join(self.failed_gates), self.thresholds.usable_score
                    ),
                )
            )
        lines.append("")

        for name, metric in self.metrics.items():
            if metric.applies:
                left = "  {0}{1:5.1f} {2}  ".format(
                    name.ljust(_LABEL_WIDTH), metric.score, _bar(metric.score)
                )
            else:
                left = "  {0}{1}  ".format(name.ljust(_LABEL_WIDTH), "   -  [not judged]")
            lines.extend(_wrapped(left, metric.message))

        for note in self.image.get("notes", []):
            lines.append("  note: {0}".format(note))

        if self.issues:
            lines.append("")
            lines.append("  What is wrong, worst first:")
            for index, issue in enumerate(self.issues, start=1):
                lines.extend(_wrapped("    {0}. ".format(index), issue))
        else:
            lines.append("")
            lines.append("  No problems found.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole report."""
        return {
            "source": self.source,
            "score": round(float(self.score), 1),
            "grade": self.grade,
            "usable": bool(self.usable),
            "image": dict(self.image),
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "issues": list(self.issues),
            "failed_gates": list(self.failed_gates),
            "thresholds_changed": self.changed_thresholds(),
        }

    def to_json(self, **kwargs: Any) -> str:
        """:meth:`to_dict` as a JSON string, safe for any alphabet."""
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    def changed_thresholds(self) -> Dict[str, float]:
        """Only the thresholds that differ from the defaults, so JSON stays small."""
        return {
            name: float(getattr(self.thresholds, name))
            for name in Thresholds.field_names()
            if getattr(self.thresholds, name) != getattr(DEFAULT_THRESHOLDS, name)
        }

    def __str__(self) -> str:                      # pragma: no cover - cosmetic
        return self.summary()


def _wrapped(prefix: str, text: str) -> List[str]:
    """``prefix`` then ``text``, wrapped and hanging-indented under the prefix.

    The prefix counts towards the line width, so nothing overflows a terminal
    just because the label in front of it was long.
    """
    indent = " " * len(prefix)
    width = max(24, _WRAP_WIDTH - len(prefix))
    wrapped = textwrap.wrap(text, width=width) or [""]
    return [prefix + wrapped[0]] + [indent + line for line in wrapped[1:]]


@dataclass
class BatchReport:
    """What :func:`image_quality_ai.assess_batch` gives back.

    Attributes:
        results: one :class:`QualityReport` per image that could be read, in the
            order the images were given.
        failures: ``{"source": ..., "error": ...}`` for each image that could
            not be read. A batch does not stop for one corrupt file the way a
            single :func:`~image_quality_ai.assess` call does.
    """

    results: List[QualityReport] = field(default_factory=list)
    failures: List[Dict[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[QualityReport]:
        return iter(self.results)

    def __getitem__(self, index: int) -> QualityReport:
        return self.results[index]

    @property
    def usable(self) -> List[QualityReport]:
        """The reports that passed."""
        return [report for report in self.results if report.usable]

    @property
    def rejected(self) -> List[QualityReport]:
        """The reports that did not pass."""
        return [report for report in self.results if not report.usable]

    def worst(self, n: int = DEFAULT_WORST) -> List[QualityReport]:
        """The ``n`` lowest-scoring reports, worst first.

        Ties are broken by source name so the order is the same on every run.
        """
        if n < 0:
            raise ValueError("n must be zero or more, got {0}".format(n))
        ordered = sorted(self.results, key=lambda report: (report.score, report.source))
        return ordered[:n]

    def mean_score(self) -> float:
        """Average overall score, or 0.0 for an empty batch."""
        if not self.results:
            return 0.0
        return sum(report.score for report in self.results) / len(self.results)

    def grade_counts(self) -> Dict[str, int]:
        """How many images got each letter grade."""
        counts = {letter: 0 for _, letter in GRADE_CUTOFFS}
        for report in self.results:
            counts[report.grade] = counts.get(report.grade, 0) + 1
        return counts

    def to_frame(self) -> Any:
        """One row per image as a pandas DataFrame.

        pandas is not a dependency of this package. Install it, or use
        :meth:`to_dict`, which needs nothing extra.

        Raises:
            ImportError: if pandas is not installed.
        """
        try:
            import pandas
        except ImportError:                        # pragma: no cover - depends on env
            raise ImportError(
                "to_frame() needs pandas, which image-quality-ai does not install. "
                "Run `pip install pandas`, or use to_dict() instead."
            ) from None
        return pandas.DataFrame(self.rows())

    def rows(self) -> List[Dict[str, Any]]:
        """One flat dict per image: the table behind :meth:`to_frame`."""
        rows: List[Dict[str, Any]] = []
        for report in self.results:
            row: Dict[str, Any] = {
                "source": report.source,
                "score": round(float(report.score), 1),
                "grade": report.grade,
                "usable": bool(report.usable),
                "width": report.image.get("width"),
                "height": report.image.get("height"),
            }
            for name, metric in report.metrics.items():
                row[name] = round(float(metric.score), 1)
                row["{0}_value".format(name)] = float(metric.value)
            row["issues"] = len(report.issues)
            row["top_issue"] = report.issues[0] if report.issues else ""
            rows.append(row)
        return rows

    def summary(self) -> str:
        """The human-readable batch report, plain ASCII."""
        return self._summary_text(DEFAULT_WORST)

    def _summary_text(self, worst: int) -> str:
        """:meth:`summary` with the length of the worst-first list chosen.

        The CLI's ``--worst N`` asks for N and :meth:`summary` asks for the
        default five, and either way the report carries one worst-first list -
        not a five-long one followed by a longer one repeating it. ``worst=0``
        leaves the list out.
        """
        total = len(self.results)
        if not total and not self.failures:
            return "No images were assessed."
        usable = len(self.usable)
        head = "{0} image(s) assessed: {1} usable, {2} not usable".format(
            total, usable, total - usable
        )
        if self.failures:
            head += ", {0} could not be read".format(len(self.failures))
        lines = [head]
        if total:
            grades = self.grade_counts()
            spread = " ".join(
                "{0}:{1}".format(letter, grades.get(letter, 0)) for _, letter in GRADE_CUTOFFS
            )
            lines.append(
                "  average score {0:.1f}, grades {1}".format(self.mean_score(), spread)
            )
            listed = self.worst(max(0, int(worst)))
            if listed:
                lines.append("")
                lines.append("  Worst first:")
                for report in listed:
                    issue = report.issues[0] if report.issues else "no problems found"
                    lines.extend(
                        _wrapped(
                            "    {0:5.1f} {1}  ".format(report.score, report.grade),
                            "{0}: {1}".format(report.source, issue),
                        )
                    )
        if self.failures:
            lines.append("")
            lines.append("  Could not be read:")
            for failure in self.failures:
                lines.extend(_wrapped("    ", failure.get("error", "unknown error")))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole batch."""
        return {
            "count": len(self.results),
            "usable": len(self.usable),
            "not_usable": len(self.results) - len(self.usable),
            "mean_score": round(self.mean_score(), 1),
            "grades": self.grade_counts(),
            "results": [report.to_dict() for report in self.results],
            "failures": [dict(failure) for failure in self.failures],
        }

    def to_json(self, **kwargs: Any) -> str:
        """:meth:`to_dict` as a JSON string, safe for any alphabet."""
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    def __str__(self) -> str:                      # pragma: no cover - cosmetic
        return self.summary()
