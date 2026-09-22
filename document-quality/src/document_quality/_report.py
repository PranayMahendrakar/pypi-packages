"""The objects handed back, and the text they turn themselves into.

Nothing here measures anything. These are the shapes a caller holds: one
:class:`Measure` per thing that was looked at, one :class:`Issue` per thing
worth doing something about, a :class:`PageReport` for one page and a
:class:`BatchReport` for a pile of them.

Two rules run through all of it. Every verdict carries the number it was made
from and the boundary it was compared against, so nothing has to be taken on
trust. And every problem carries a ``fix``: not "contrast is low" but "rescan
with the lid closed", not "the page is crooked" but "deskew by 2.30 degrees".
A report that only names problems leaves the work undone.

All text is plain ASCII, and JSON is written with ``ensure_ascii=False``, so a
page called ``facture_reçu.png`` survives a pipe on any console.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Severities, worst first. ``info`` is a statement of fact, not a complaint.
SEVERITIES = ("failure", "warning", "info")
_SEVERITY_RANK = {name: index for index, name in enumerate(SEVERITIES)}

#: What each page kind means, for anyone printing the word on its own.
PAGE_KINDS = {
    "document": "a page of text, which is what this package is for",
    "blank": "a sheet with nothing on it",
    "photograph": "a photograph or picture rather than a document page",
}


def _round(value: Optional[float], places: int = 4) -> Optional[float]:
    """Round for JSON, leaving ``None`` alone."""
    return None if value is None else round(float(value), places)


@dataclass(frozen=True)
class Issue:
    """One thing wrong with a page, and what to do about it.

    Attributes:
        kind: stable machine name, e.g. ``"skew"`` or ``"show_through"``.
            Safe to switch on; the message wording is not.
        severity: ``"failure"`` (OCR will suffer), ``"warning"`` (worth
            knowing) or ``"info"`` (a fact, not a fault).
        message: what was measured, in one sentence, naming the number and the
            boundary it missed.
        fix: the concrete remedy, e.g. ``"deskew by 2.30 degrees"`` or
            ``"rescan at 300 dpi"``. Never empty.
    """

    kind: str
    severity: str
    message: str
    fix: str

    @property
    def rank(self) -> int:
        """Sort key: 0 for a failure, 1 for a warning, 2 for information."""
        return _SEVERITY_RANK.get(self.severity, len(SEVERITIES))

    def __str__(self) -> str:
        return "[{0}] {1} {2} Fix: {3}".format(
            self.severity, self.kind + ":", self.message, self.fix
        )

    def to_dict(self) -> Dict[str, str]:
        """JSON-safe dict of this issue."""
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
        }


@dataclass(frozen=True)
class Measure:
    """One thing that was looked at, with the number behind it.

    Attributes:
        name: stable machine name, e.g. ``"sharpness"``.
        value: the raw quantity measured, in the unit named by ``unit``, or
            ``None`` when the measure did not apply.
        unit: what ``value`` is counted in, e.g. ``"degrees"`` or ``"dpi"``.
        score: 0-100, or ``None`` when the measure is descriptive rather than
            a verdict, or did not apply. A measure scores 60 or more exactly
            when it is inside its limit.
        ok: whether this measure passed. ``True`` for a measure that did not
            apply: something unmeasured is not something wrong.
        applies: ``False`` when there was nothing to measure, e.g. resolution
            on a file that carries no dpi.
        message: one sentence naming the number and the boundary used.
        details: any supporting numbers worth keeping.
    """

    name: str
    value: Optional[float]
    unit: str
    score: Optional[float]
    ok: bool
    applies: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of this measure."""
        return {
            "name": self.name,
            "value": _round(self.value),
            "unit": self.unit,
            "score": _round(self.score, 1),
            "ok": bool(self.ok),
            "applies": bool(self.applies),
            "message": self.message,
            "details": {
                key: (_round(item) if isinstance(item, float) else item)
                for key, item in self.details.items()
            },
        }


@dataclass
class PageReport:
    """What one page is worth OCRing, and what to do if it is not.

    The three numbers most callers want are :attr:`ocr_ready`, :attr:`score`
    and :attr:`issues`. Everything else is there so a disputed verdict can be
    traced to the quantity behind it.
    """

    #: Where the page came from: a path, or a description of an in-memory image.
    source: str
    #: Page width in pixels, after any EXIF rotation was applied.
    width: int
    #: Page height in pixels, after any EXIF rotation was applied.
    height: int
    #: ``"document"``, ``"blank"`` or ``"photograph"``. See :data:`PAGE_KINDS`.
    kind: str
    #: Whether this page is worth sending to an OCR engine.
    ocr_ready: bool
    #: Overall quality for OCR, 0 to 100.
    score: float
    #: How far the text is turned counter-clockwise from horizontal.
    #: ``image.rotate(-report.skew_degrees)`` puts the page straight.
    skew_degrees: float
    #: Height of an inked text line in pixels, or ``None`` if no lines were found.
    estimated_text_height_px: Optional[float]
    #: Problems worth acting on, worst first. Each carries its own fix.
    issues: List[Issue] = field(default_factory=list)
    #: Every measure by name, whether it passed or not.
    measures: Dict[str, Measure] = field(default_factory=dict)
    #: Dots per inch used, or ``None`` when the file claimed none and none was
    #: given. ``None`` means resolution advice was left out, not guessed.
    dpi: Optional[float] = None
    #: ``"argument"``, ``"image metadata"`` or ``None``.
    dpi_source: Optional[str] = None
    #: The numbers behind the :attr:`kind` decision.
    evidence: Dict[str, float] = field(default_factory=dict)
    #: Anything the caller should know that is not a fault, e.g. that EXIF
    #: orientation was applied before measuring.
    notes: List[str] = field(default_factory=list)

    # -- the quick questions ----------------------------------------------

    @property
    def is_blank(self) -> bool:
        """True when the sheet carries nothing. Not a fault, just empty."""
        return self.kind == "blank"

    @property
    def is_document(self) -> bool:
        """True when this looks like a page of text at all."""
        return self.kind == "document"

    @property
    def failures(self) -> List[Issue]:
        """Issues that will cost you OCR accuracy."""
        return [item for item in self.issues if item.severity == "failure"]

    @property
    def warnings(self) -> List[Issue]:
        """Issues worth knowing about that are not disqualifying."""
        return [item for item in self.issues if item.severity == "warning"]

    @property
    def worst(self) -> Optional[Issue]:
        """The single issue to deal with first, or ``None`` if there are none."""
        return self.issues[0] if self.issues else None

    @property
    def fixes(self) -> List[str]:
        """Just the remedies, worst first, with duplicates dropped."""
        seen: List[str] = []
        for item in self.issues:
            if item.fix not in seen:
                seen.append(item.fix)
        return seen

    @property
    def megapixels(self) -> float:
        """Page area in megapixels."""
        return (self.width * self.height) / 1e6

    @property
    def page_inches(self) -> Optional[Sequence[float]]:
        """Page size in inches, or ``None`` when the dpi is unknown."""
        if not self.dpi:
            return None
        return (self.width / self.dpi, self.height / self.dpi)

    def scores(self) -> Dict[str, float]:
        """Every measure that produced a score, by name."""
        return {
            name: measure.score
            for name, measure in self.measures.items()
            if measure.score is not None
        }

    def explain(self, name: str) -> str:
        """The message behind one measure.

        Raises:
            KeyError: if ``name`` is not a measure on this report.
        """
        if name not in self.measures:
            raise KeyError(
                "no measure named {0!r}; this page has {1}".format(
                    name, ", ".join(sorted(self.measures))
                )
            )
        return self.measures[name].message

    # -- the human answer --------------------------------------------------

    def headline(self) -> str:
        """The one-line verdict."""
        if self.kind == "blank":
            verdict = "blank, nothing to OCR"
        elif self.kind == "photograph":
            verdict = "not a document page"
        elif self.ocr_ready:
            verdict = "ready to OCR"
        else:
            verdict = "not ready to OCR"
        return "{0}: {1} - score {2:.0f} of 100".format(
            self.source, verdict, self.score
        )

    def _description(self) -> str:
        """The second line: what this page physically is."""
        parts = ["{0} x {1} px".format(self.width, self.height)]
        if self.dpi:
            parts.append("{0:g} dpi (from {1})".format(self.dpi, self.dpi_source))
            inches = self.page_inches
            if inches:
                parts.append("{0:.1f} x {1:.1f} in".format(inches[0], inches[1]))
        else:
            parts.append("dpi unknown, so resolution advice is left out")
        return "{0} page, {1}.".format(self.kind.capitalize(), ", ".join(parts))

    def _geometry_line(self) -> str:
        """The third line: what the text on it looks like."""
        if self.estimated_text_height_px is None:
            return "No rows of text were found, so text size is not reported."
        return "Text lines about {0:.0f} px tall, skew {1:+.2f} degrees.".format(
            self.estimated_text_height_px, self.skew_degrees
        )

    def summary(self) -> str:
        """The whole report as plain text, verdict first."""
        lines = [self.headline(), self._description(), self._geometry_line()]
        for note in self.notes:
            lines.append("Note: {0}".format(note))
        if self.issues:
            lines.append("")
            lines.append("What to do, worst first:")
            for item in self.issues:
                lines.append("  [{0}] {1}".format(item.severity, item.message))
                lines.append("      fix: {0}".format(item.fix))
        else:
            lines.append("")
            lines.append("Nothing to fix.")
        scored = [item for item in self.measures.values() if item.applies]
        if scored:
            lines.append("")
            lines.append("Measures:")
            width = max(len(item.name) for item in scored)
            for item in scored:
                value = "  n/a" if item.value is None else "{0:7.3f}".format(item.value)
                score = "   -" if item.score is None else "{0:4.0f}".format(item.score)
                lines.append(
                    "  {0:<{1}} {2} {3}  {4}".format(
                        item.name, width, value, score, item.message
                    )
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """The whole report as a JSON-safe dict."""
        inches = self.page_inches
        return {
            "source": self.source,
            "kind": self.kind,
            "ocr_ready": bool(self.ocr_ready),
            "score": _round(self.score, 1),
            "width": int(self.width),
            "height": int(self.height),
            "megapixels": _round(self.megapixels, 3),
            "dpi": _round(self.dpi, 2),
            "dpi_source": self.dpi_source,
            "page_inches": None if inches is None else [
                _round(inches[0], 2), _round(inches[1], 2)
            ],
            "skew_degrees": _round(self.skew_degrees, 2),
            "estimated_text_height_px": _round(self.estimated_text_height_px, 1),
            "issues": [item.to_dict() for item in self.issues],
            "fixes": self.fixes,
            "measures": {
                name: measure.to_dict() for name, measure in self.measures.items()
            },
            "evidence": {key: _round(value) for key, value in self.evidence.items()},
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        """:meth:`to_dict` as JSON, UTF-8 safe."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        return self.headline()


@dataclass
class BatchReport:
    """Many pages at once, with the ones needing attention pulled to the front."""

    #: One report per page that could be read, in the order given.
    reports: List[PageReport] = field(default_factory=list)
    #: One entry per page that could not be read: ``{"source", "error"}``.
    failures: List[Dict[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.reports)

    def __iter__(self) -> Iterator[PageReport]:
        return iter(self.reports)

    def __getitem__(self, index: int) -> PageReport:
        return self.reports[index]

    @property
    def not_ready(self) -> List[PageReport]:
        """Pages that should not be sent to an OCR engine, worst first.

        Blank pages are in here - there is nothing on them to read - but they
        are not failures, and :attr:`blank` separates them out.
        """
        return sorted(
            [item for item in self.reports if not item.ocr_ready],
            key=lambda item: item.score,
        )

    @property
    def ready(self) -> List[PageReport]:
        """Pages worth OCRing, best first."""
        return sorted(
            [item for item in self.reports if item.ocr_ready],
            key=lambda item: -item.score,
        )

    @property
    def blank(self) -> List[PageReport]:
        """Pages with nothing on them."""
        return [item for item in self.reports if item.kind == "blank"]

    @property
    def photographs(self) -> List[PageReport]:
        """Pages that are not document pages at all."""
        return [item for item in self.reports if item.kind == "photograph"]

    def mean_score(self) -> Optional[float]:
        """Average score across the pages that were read, or ``None`` if none were."""
        if not self.reports:
            return None
        return sum(item.score for item in self.reports) / len(self.reports)

    def issue_counts(self) -> Dict[str, int]:
        """How many pages carry each kind of issue, most common first."""
        counts: Dict[str, int] = {}
        for report in self.reports:
            for kind in {item.kind for item in report.issues}:
                counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))

    def kind_counts(self) -> Dict[str, int]:
        """How many pages of each kind, in the order of :data:`PAGE_KINDS`."""
        counts = {name: 0 for name in PAGE_KINDS}
        for report in self.reports:
            counts[report.kind] = counts.get(report.kind, 0) + 1
        return counts

    def rows(self) -> List[Dict[str, Any]]:
        """One flat dict per page: the shape a spreadsheet or DataFrame wants."""
        return [
            {
                "source": item.source,
                "kind": item.kind,
                "ocr_ready": item.ocr_ready,
                "score": _round(item.score, 1),
                "skew_degrees": _round(item.skew_degrees, 2),
                "text_height_px": _round(item.estimated_text_height_px, 1),
                "dpi": _round(item.dpi, 2),
                "worst_issue": item.worst.kind if item.worst else "",
                "fix": item.worst.fix if item.worst else "",
            }
            for item in self.reports
        ]

    def summary(self, worst: int = 5) -> str:
        """The batch as plain text: the count, then the pages needing work."""
        if not self.reports and not self.failures:
            return "No pages were assessed."
        counts = self.kind_counts()
        ready = len(self.ready)
        lines = [
            "{0} page(s) assessed: {1} ready to OCR, {2} not.".format(
                len(self.reports), ready, len(self.reports) - ready
            )
        ]
        mean = self.mean_score()
        detail = "Mean score {0:.0f} of 100.".format(mean) if mean is not None else ""
        lines.append(
            "{0} document, {1} blank, {2} not a document page. {3}".format(
                counts.get("document", 0), counts.get("blank", 0),
                counts.get("photograph", 0), detail,
            ).strip()
        )
        if self.failures:
            lines.append("{0} file(s) could not be read.".format(len(self.failures)))

        problems = self.not_ready
        if problems and worst > 0:
            lines.append("")
            lines.append("Needing attention, worst first:")
            for item in problems[:worst]:
                fix = item.worst.fix if item.worst else "nothing to fix"
                lines.append(
                    "  {0:5.0f}  {1:<11} {2}".format(item.score, item.kind, item.source)
                )
                lines.append("         fix: {0}".format(fix))
            if len(problems) > worst:
                lines.append("  ... and {0} more.".format(len(problems) - worst))
        counts_by_issue = self.issue_counts()
        if counts_by_issue:
            lines.append("")
            lines.append("Issues by how many pages have them:")
            for kind, count in counts_by_issue.items():
                lines.append("  {0:<16} {1}".format(kind, count))
        for failure in self.failures:
            lines.append("  unreadable: {0}: {1}".format(
                failure.get("source", "?"), failure.get("error", "")))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """The whole batch as a JSON-safe dict."""
        return {
            "pages": len(self.reports),
            "ready": len(self.ready),
            "not_ready": len(self.not_ready),
            "mean_score": _round(self.mean_score(), 1),
            "kind_counts": self.kind_counts(),
            "issue_counts": self.issue_counts(),
            "reports": [item.to_dict() for item in self.reports],
            "failures": list(self.failures),
        }

    def to_json(self, indent: int = 2) -> str:
        """:meth:`to_dict` as JSON, UTF-8 safe."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        return self.summary()
