"""The result object: :class:`CropResult`.

A crop is a decision, and a decision nobody can check is not worth much. Every
result carries the box it chose, the strategy that chose it, the raw window
scores behind that choice, and a confidence number that says plainly how much
better the chosen window is than simply cutting out the middle.
"""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

#: Confidence at or above which each label starts.
CONFIDENCE_LABELS = (
    (0.35, "strong"),
    (0.15, "moderate"),
    (0.02, "weak"),
    (0.0, "none"),
)

_WRAP_WIDTH = 92
_INDENT = " " * 14


def confidence_label(confidence: float) -> str:
    """Plain word for a 0 to 1 confidence: strong, moderate, weak or none."""
    for cutoff, label in CONFIDENCE_LABELS:
        if confidence >= cutoff:
            return label
    return "none"          # pragma: no cover - the table ends at 0.0


@dataclass
class CropResult:
    """What :func:`smart_crop_ai.crop` gives back.

    Attributes:
        image: the cropped ``PIL.Image.Image``. A new image every time; the one
            you passed in is never touched.
        box: ``(left, top, right, bottom)`` in source pixels, after EXIF
            orientation was applied. Always inside the source.
        strategy_used: the strategy that actually picked the box. Usually the
            one you asked for, but ``"center"`` when the energy map turned out
            to be flat, and ``"whole_image"`` when the target was larger than
            the source.
        confidence: 0 to 1. The share of the chosen window's detail that a
            centre crop of the same size would have missed. 0 means the crop is
            no better than cutting out the middle, and the notes say why.
        strategy_requested: what the caller asked for, including ``"auto"``.
        source_size: ``(width, height)`` of the source, after EXIF orientation.
        target_size: ``(width, height)`` that was asked for, in source pixels.
        padding: the padding fraction used.
        scores: the raw numbers behind the decision - ``best_window`` and
            ``center_window`` (mean energy per pixel in each, 0 to 1),
            ``energy_peak``, ``energy_spread`` and ``window_spread`` (how far
            the best window rose above the worst one). Empty for a crop that
            needed no search, such as ``strategy="center"``.
        notes: anything the caller should know, in plain language.
        source: the file path, or ``"<image>"`` for an in-memory image.
        destination: where the crop was written, for
            :func:`smart_crop_ai.crop_to_file`.
    """

    image: Image.Image
    box: Tuple[int, int, int, int]
    strategy_used: str
    confidence: float
    strategy_requested: str = "auto"
    source_size: Tuple[int, int] = (0, 0)
    target_size: Tuple[int, int] = (0, 0)
    padding: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    source: str = "<image>"
    destination: Optional[str] = None

    @property
    def size(self) -> Tuple[int, int]:
        """The ``(width, height)`` of the crop that was taken."""
        left, top, right, bottom = self.box
        return (right - left, bottom - top)

    @property
    def offset(self) -> Tuple[int, int]:
        """The ``(left, top)`` of the crop inside the source."""
        return (self.box[0], self.box[1])

    @property
    def covers(self) -> float:
        """Share of the source area the crop keeps, 0 to 1."""
        source_area = float(self.source_size[0] * self.source_size[1])
        if source_area <= 0.0:                 # pragma: no cover - guarded upstream
            return 0.0
        width, height = self.size
        return float(width * height) / source_area

    @property
    def moved(self) -> bool:
        """True when the box is not the plain centre crop of the same size."""
        width, height = self.size
        centre = (
            (self.source_size[0] - width) // 2,
            (self.source_size[1] - height) // 2,
        )
        return self.offset != centre

    @property
    def confidence_label(self) -> str:
        """One of ``"strong"``, ``"moderate"``, ``"weak"``, ``"none"``."""
        return confidence_label(self.confidence)

    def confidence_sentence(self) -> str:
        """One sentence saying what :attr:`confidence` actually measured."""
        if self.confidence <= 0.0:
            return "none - this crop is no better than a centre crop of the same size"
        return (
            "{0} - a centre crop of the same size would have missed "
            "{1:.0f}% of the detail this window keeps".format(
                self.confidence_label, self.confidence * 100.0
            )
        )

    def summary(self) -> str:
        """Human readable report of the crop, in plain ASCII."""
        width, height = self.size
        lines = [
            "smart-crop-ai: {0}".format(self.source),
            "  source      {0} x {1} pixels".format(self.source_size[0], self.source_size[1]),
            "  asked for   {0} x {1} pixels".format(self.target_size[0], self.target_size[1]),
            "  crop        {0} x {1} at ({2}, {3}), box {4}".format(
                width, height, self.offset[0], self.offset[1], tuple(self.box)
            ),
            "  strategy    {0}{1}".format(
                self.strategy_used,
                ""
                if self.strategy_used == self.strategy_requested
                else " (asked for: {0})".format(self.strategy_requested),
            ),
            "  confidence  {0:.2f}  {1}".format(
                self.confidence, self.confidence_sentence()
            ),
            "  keeps       {0:.0f}% of the source area, {1}".format(
                self.covers * 100.0,
                "off centre" if self.moved else "centred",
            ),
        ]
        if self.destination:
            lines.append("  written to  {0}".format(self.destination))
        for index, note in enumerate(self.notes):
            label = "  notes       " if index == 0 else _INDENT
            lines.append(
                textwrap.fill(
                    note,
                    width=_WRAP_WIDTH,
                    initial_indent=label,
                    subsequent_indent=_INDENT,
                )
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of everything except the pixels."""
        return {
            "source": self.source,
            "destination": self.destination,
            "source_size": list(self.source_size),
            "target_size": list(self.target_size),
            "box": list(self.box),
            "size": list(self.size),
            "offset": list(self.offset),
            "strategy_used": self.strategy_used,
            "strategy_requested": self.strategy_requested,
            "confidence": round(float(self.confidence), 4),
            "confidence_label": self.confidence_label,
            "padding": float(self.padding),
            "covers": round(self.covers, 4),
            "moved": bool(self.moved),
            "scores": {
                key: round(float(value), 6) for key, value in self.scores.items()
            },
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        """:meth:`to_dict` as JSON text, non-ASCII kept as-is."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str, **kwargs: Any) -> str:
        """Write the cropped image to ``path``. Returns the path written.

        Alpha is composited onto white when the destination format cannot store
        it (JPEG, BMP, PPM), and a note says so. Missing parent directories are
        created.
        """
        from ._images import flatten_for_format   # local import, avoids a cycle

        destination = str(path)
        image, flattened = flatten_for_format(self.image, destination)
        directory = os.path.dirname(destination)
        if directory:
            os.makedirs(directory, exist_ok=True)
        image.save(destination, **kwargs)
        if flattened:
            note = (
                "Transparency was composited onto white because "
                "{0} files cannot store an alpha channel.".format(
                    os.path.splitext(destination)[1].lstrip(".").upper() or "these"
                )
            )
            if note not in self.notes:
                self.notes.append(note)
        self.destination = destination
        return destination

    def __repr__(self) -> str:                      # pragma: no cover - cosmetic
        width, height = self.size
        return (
            "CropResult(box={0}, size={1}x{2}, strategy_used={3!r}, "
            "confidence={4:.2f})".format(
                tuple(self.box), width, height, self.strategy_used, self.confidence
            )
        )
