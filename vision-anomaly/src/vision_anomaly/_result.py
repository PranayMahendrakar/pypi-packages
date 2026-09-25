"""The objects a caller actually reads: :class:`Box`, :class:`AnomalyResult`,
:class:`BatchReport`.

A number on its own - "15.3" - tells nobody anything. Every result here carries
the threshold it was judged against, what the fitted set itself scores, which
feature families moved, and which grid cells moved most, so the answer to "why?"
never requires opening the source.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ._features import GROUP_DESCRIPTIONS, GROUP_NAMES, FeatureSpace
from ._profile import Deviation

#: How sharply :attr:`AnomalyResult.confidence` climbs away from the threshold.
#: At the threshold itself confidence is 0.5 - the verdict is a coin toss, and
#: saying so is the point. One threshold's worth either side takes it to about
#: 0.95.
CONFIDENCE_SHARPNESS = 1.8

#: How many grid cells :attr:`AnomalyResult.regions` lists by default.
DEFAULT_REGIONS = 3

#: How many feature families :attr:`AnomalyResult.reasons` names by default.
DEFAULT_REASONS = 3


@dataclass(frozen=True)
class Box:
    """One grid cell that departed from normal, and by how much.

    The coordinates are pixels on the square analysis grid every image is
    resampled onto, not on the caller's original. :meth:`scaled` converts.

    Attributes:
        row: the cell's row in the coarse grid, counting from the top.
        col: the cell's column, counting from the left.
        x: left edge, in analysis-grid pixels.
        y: top edge, in analysis-grid pixels.
        width: cell width, in analysis-grid pixels.
        height: cell height, in analysis-grid pixels.
        score: how far this cell's features lie outside normal, in robust sigmas.
        grid: cells on a side of the grid this box came from, so :meth:`scaled`
            can map it onto an image of any size.
    """

    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    score: float
    grid: int = 1

    def scaled(self, width: int, height: int) -> "Box":
        """The same cell as pixels on an image of ``width`` by ``height``.

        Boxes come back on the square analysis grid, which is not the shape of
        the picture the caller has in front of them. This maps a box onto that
        picture so it can be drawn on it.

        Args:
            width: pixel width of the image to map onto.
            height: pixel height of the image to map onto.

        Returns:
            A new box; this one is unchanged.
        """
        cells = max(1, int(self.grid))
        step_x = float(width) / float(cells)
        step_y = float(height) / float(cells)
        left = int(round(self.col * step_x))
        top = int(round(self.row * step_y))
        return Box(
            row=self.row,
            col=self.col,
            x=left,
            y=top,
            width=max(1, int(round((self.col + 1) * step_x)) - left),
            height=max(1, int(round((self.row + 1) * step_y)) - top),
            score=self.score,
            grid=cells,
        )

    def label(self) -> str:
        """``row 1, column 2`` - how a person refers to the cell."""
        return "row {0}, column {1}".format(self.row, self.col)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the box."""
        return {
            "row": int(self.row),
            "col": int(self.col),
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "score": round(float(self.score), 3),
            "grid": int(self.grid),
        }


@dataclass
class AnomalyResult:
    """The verdict on one image, with everything needed to argue with it.

    Attributes:
        source: what was scored - a path, or a label for an array.
        anomalous: whether :attr:`score` reached :attr:`threshold`.
        score: how far the image sits from normal, in robust sigmas. Zero means
            every feature landed inside the band the fitted set defines.
        threshold: the sensitivity the verdict used.
        confidence: 0.0 to 1.0, how decisively the score sits on its side of the
            threshold. 0.5 means it landed on the line.
        regions: the grid cells that departed most, worst first, so a person can
            see where to look.
        reasons: plain sentences naming the feature families that moved.
        group_scores: every family's score, including the quiet ones.
        fit_typical: what a typical fitted image scores against the profile.
        fit_worst: the highest score any fitted image gets - the honest floor
            below which nothing can be called unusual.
        n_fitted: how many images the profile was fitted on.
        notes: how this image was read, including any resizing.
        warnings: anything that makes this verdict less trustworthy, carried
            down from the profile.
        width: source pixel width.
        height: source pixel height.
        channels: 1 for greyscale, 3 for colour, 4 when alpha was present.
    """

    source: str
    anomalous: bool
    score: float
    threshold: float
    confidence: float
    regions: List[Box] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    group_scores: Dict[str, float] = field(default_factory=dict)
    fit_typical: float = 0.0
    fit_worst: float = 0.0
    n_fitted: int = 0
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    channels: int = 3

    @property
    def verdict(self) -> str:
        """``"anomalous"`` or ``"normal"``."""
        return "anomalous" if self.anomalous else "normal"

    def summary(self) -> str:
        """Human-readable text, plain ASCII, safe on any console."""
        lines = [
            "{0}: {1} (score {2:.2f}, threshold {3:.2f}, confidence {4:.2f})".format(
                self.source,
                "ANOMALOUS" if self.anomalous else "normal",
                self.score,
                self.threshold,
                self.confidence,
            )
        ]
        if self.n_fitted:
            lines.append(
                "compared against {0} known-good image{1}, which themselves score "
                "{2:.2f} typical and {3:.2f} at worst".format(
                    self.n_fitted,
                    "" if self.n_fitted == 1 else "s",
                    self.fit_typical,
                    self.fit_worst,
                )
            )
        if self.reasons:
            lines.append("what moved:")
            for reason in self.reasons:
                lines.append("  " + reason)
        elif not self.anomalous:
            lines.append(
                "nothing moved: every feature stayed inside the band the "
                "known-good set defines"
            )
        if self.regions:
            lines.append("where it moved most:")
            for box in self.regions:
                lines.append(
                    "  {0:<20} {1:>4} sigmas  (analysis-grid box x={2} y={3} "
                    "{4}x{5})".format(
                        box.label(),
                        "{0:.1f}".format(box.score),
                        box.x,
                        box.y,
                        box.width,
                        box.height,
                    )
                )
        for note in self.notes:
            lines.append("note: " + note)
        for warning in self.warnings:
            lines.append("warning: " + warning)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the whole result."""
        return {
            "source": self.source,
            "anomalous": bool(self.anomalous),
            "verdict": self.verdict,
            "score": round(float(self.score), 3),
            "threshold": round(float(self.threshold), 3),
            "confidence": round(float(self.confidence), 3),
            "regions": [box.to_dict() for box in self.regions],
            "reasons": list(self.reasons),
            "group_scores": {
                name: round(float(value), 3)
                for name, value in self.group_scores.items()
            },
            "fit_typical": round(float(self.fit_typical), 3),
            "fit_worst": round(float(self.fit_worst), 3),
            "n_fitted": int(self.n_fitted),
            "width": int(self.width),
            "height": int(self.height),
            "channels": int(self.channels),
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


def _shared_notes(results: Sequence["AnomalyResult"]) -> List[str]:
    """Notes every result in a run carries, in the order the first one has them.

    A note on one image describes that image. A note on all of them describes
    the run, and is worth saying once at the top rather than never.
    """
    if not results:
        return []
    common = set(results[0].notes)
    for result in results[1:]:
        common &= set(result.notes)
        if not common:
            return []
    return [note for note in results[0].notes if note in common]


@dataclass
class BatchReport:
    """Every result from one run, plus the totals a person reads first.

    Iterating a report yields its :class:`AnomalyResult` objects in the order
    they were given, and indexing works, so ``report[0]`` and ``for result in
    report`` both do the obvious thing.

    Attributes:
        results: one entry per image scored, in the order they came in.
        threshold: the sensitivity every verdict used.
        n_fitted: how many images the profile was fitted on.
        fit_typical: what a typical fitted image scores.
        fit_worst: the highest score any fitted image gets.
        warnings: anything that makes the whole run less trustworthy.
    """

    results: List[AnomalyResult] = field(default_factory=list)
    threshold: float = 0.0
    n_fitted: int = 0
    fit_typical: float = 0.0
    fit_worst: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[AnomalyResult]:
        return iter(self.results)

    def __getitem__(self, index: int) -> AnomalyResult:
        return self.results[index]

    @property
    def anomalous(self) -> List[AnomalyResult]:
        """Only the results that were flagged, worst first."""
        return sorted(
            (result for result in self.results if result.anomalous),
            key=lambda result: result.score,
            reverse=True,
        )

    @property
    def normal(self) -> List[AnomalyResult]:
        """Only the results that were not flagged, in the order given."""
        return [result for result in self.results if not result.anomalous]

    @property
    def n_anomalous(self) -> int:
        """How many images were flagged."""
        return sum(1 for result in self.results if result.anomalous)

    @property
    def scores(self) -> List[float]:
        """Every score, in the order the images were given."""
        return [float(result.score) for result in self.results]

    @property
    def worst(self) -> Optional[AnomalyResult]:
        """The highest-scoring result, or ``None`` if nothing was scored."""
        if not self.results:
            return None
        return max(self.results, key=lambda result: result.score)

    def summary(self, limit: int = 5) -> str:
        """Human-readable text, plain ASCII, safe on any console.

        Args:
            limit: how many flagged images to describe in full.
        """
        total = len(self.results)
        lines = [
            "{0} image{1} checked against {2} known-good, threshold {3:.2f}: "
            "{4} anomalous".format(
                total,
                "" if total == 1 else "s",
                self.n_fitted,
                self.threshold,
                self.n_anomalous,
            )
        ]
        if self.n_fitted:
            lines.append(
                "the known-good images themselves score {0:.2f} typical, {1:.2f} "
                "at worst".format(self.fit_typical, self.fit_worst)
            )
        for warning in self.warnings:
            lines.append("warning: " + warning)
        # A caveat that applies to every image in the run belongs at the top,
        # where it is read even when nothing was flagged and no per-image block
        # is printed at all. "everything was resampled onto one square grid" is
        # exactly that kind of caveat, and it used to reach nobody.
        for note in _shared_notes(self.results):
            lines.append("note: " + note)
        flagged = self.anomalous
        if not flagged:
            if total:
                lines.append(
                    "nothing unusual; the highest score was {0:.2f}".format(
                        max(self.scores)
                    )
                )
            return "\n".join(lines)
        lines.append("")
        for result in flagged[: max(0, int(limit))]:
            lines.append(result.summary())
            lines.append("")
        if len(flagged) > max(0, int(limit)):
            lines.append(
                "... and {0} more anomalous image{1}".format(
                    len(flagged) - limit, "" if len(flagged) - limit == 1 else "s"
                )
            )
        return "\n".join(lines).rstrip()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the whole report."""
        return {
            "n_images": len(self.results),
            "n_anomalous": self.n_anomalous,
            "threshold": round(float(self.threshold), 3),
            "n_fitted": int(self.n_fitted),
            "fit_typical": round(float(self.fit_typical), 3),
            "fit_worst": round(float(self.fit_worst), 3),
            "warnings": list(self.warnings),
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """The report as a JSON string, non-ASCII characters left intact."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def confidence_for(score: float, threshold: float) -> float:
    """How decisively a score sits on its side of the threshold, 0.0 to 1.0.

    Confidence is in the verdict, not in the anomaly: a score of zero against a
    threshold of three is a confident "normal". Landing exactly on the threshold
    gives 0.5, which is the honest answer for a borderline image.
    """
    if threshold <= 0.0:
        return 1.0
    margin = abs(float(score) - float(threshold)) / float(threshold)
    return round(0.5 + 0.5 * math.tanh(CONFIDENCE_SHARPNESS * margin), 4)


def boxes_from(
    deviation: Deviation, space: FeatureSpace, limit: int = DEFAULT_REGIONS
) -> List[Box]:
    """The grid cells that departed most, worst first.

    Cells that did not depart at all are left out rather than padded in, so an
    empty list means "nowhere in particular", which is the truth for an image
    that belongs.
    """
    scores = np.asarray(deviation.cell_scores, dtype=np.float64)
    if scores.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    grid = int(space.config.grid)
    boxes: List[Box] = []
    for cell in order[: max(0, int(limit))]:
        value = float(scores[cell])
        if value <= 0.0:
            break
        left, top, width, height = space.cell_box(int(cell))
        row, col = divmod(int(cell), grid)
        boxes.append(
            Box(
                row=row,
                col=col,
                x=left,
                y=top,
                width=width,
                height=height,
                score=value,
                grid=grid,
            )
        )
    return boxes


def reasons_from(
    deviation: Deviation, space: FeatureSpace, limit: int = DEFAULT_REASONS
) -> List[str]:
    """One sentence per feature family that moved, worst family first.

    Each sentence names the family, says in words what that family measures, and
    points at the single feature inside it that moved furthest, with the
    direction and the grid cell. That is enough for a person to go and look at
    the right part of the picture.
    """
    excess = np.asarray(deviation.excess, dtype=np.float64)
    z = np.asarray(deviation.z, dtype=np.float64)
    ranked = sorted(
        deviation.group_scores.items(), key=lambda item: item[1], reverse=True
    )
    grid = int(space.config.grid)
    out: List[str] = []
    for group, value in ranked[: max(0, int(limit))]:
        if value <= 0.0:
            break
        index = space.group_index(group)
        if index.size == 0:                 # pragma: no cover - every family has features
            continue
        worst = int(index[int(np.argmax(excess[index]))])
        cell = space.cell_of[worst]
        if cell < 0:
            where = "over the whole frame"
        else:
            row, col = divmod(int(cell), grid)
            where = "at row {0}, column {1}".format(row, col)
        out.append(
            "{0} ({1:.1f} sigmas): {2}; furthest is {3}, {4:.1f} sigmas {5} normal "
            "{6}".format(
                group,
                value,
                GROUP_DESCRIPTIONS.get(group, group),
                space.names[worst],
                abs(z[worst]),
                "above" if z[worst] >= 0 else "below",
                where,
            )
        )
    return out


def build_result(
    *,
    source: str,
    deviation: Deviation,
    space: FeatureSpace,
    threshold: float,
    fit_typical: float,
    fit_worst: float,
    n_fitted: int,
    notes: Sequence[str] = (),
    warnings: Sequence[str] = (),
    size: Tuple[int, int] = (0, 0),
    channels: int = 3,
    max_regions: int = DEFAULT_REGIONS,
    max_reasons: int = DEFAULT_REASONS,
) -> AnomalyResult:
    """Assemble the public result object from a raw deviation."""
    score = float(deviation.score)
    anomalous = score >= float(threshold)
    return AnomalyResult(
        source=source,
        anomalous=anomalous,
        score=score,
        threshold=float(threshold),
        confidence=confidence_for(score, threshold),
        regions=boxes_from(deviation, space, max_regions),
        reasons=reasons_from(deviation, space, max_reasons),
        group_scores={
            name: float(deviation.group_scores.get(name, 0.0)) for name in GROUP_NAMES
        },
        fit_typical=float(fit_typical),
        fit_worst=float(fit_worst),
        n_fitted=int(n_fitted),
        notes=[str(note) for note in notes],
        warnings=[str(note) for note in warnings],
        width=int(size[0]),
        height=int(size[1]),
        channels=int(channels),
    )
