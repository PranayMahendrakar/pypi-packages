"""The result objects: :class:`Step` and :class:`CleanResult`.

A cleaning pipeline that quietly does five things to a page is impossible to
debug when the OCR comes back wrong, because there is no way to tell which of
the five ruined it. So every step reports itself, including the steps that
decided to do nothing, and each carries the reason in a sentence a person can
read. ``result.steps`` is the whole record, in the order it happened.
"""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

#: The pipeline's steps, in the order they run.
STEP_NAMES = ("grayscale", "deskew", "border", "denoise", "threshold", "upscale")

_WRAP_WIDTH = 92
_INDENT = " " * 16


@dataclass
class Step:
    """One stage of the pipeline, and what it did or did not do.

    Attributes:
        name: one of :data:`STEP_NAMES`.
        applied: whether this step changed the page.
        detail: a sentence saying what it did, or why it did nothing. Never
            empty - "skipped" on its own is the thing this class exists to
            prevent.
    """

    name: str
    applied: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the step."""
        return {"name": self.name, "applied": bool(self.applied), "detail": self.detail}

    def __str__(self) -> str:
        return "{0:<10} {1:<8} {2}".format(
            self.name, "applied" if self.applied else "skipped", self.detail
        )


@dataclass
class CleanResult:
    """What :func:`ocr_cleaner.clean` gives back.

    Attributes:
        image: the cleaned page, as a ``PIL.Image.Image`` in mode ``L``. A new
            image every time; the one you passed in is never touched. After a
            threshold it holds only 0 and 255.
        steps: every stage of the pipeline in order, applied or not, each with
            its reason. See :class:`Step`.
        skew_corrected_degrees: how far the page was actually turned, positive
            counter-clockwise. ``0.0`` when deskewing was off, skipped or
            unnecessary; :attr:`estimated_skew_degrees` holds what was measured
            either way.
        estimated_text_height_px: height of a line of text in source pixels,
            ascender top to descender foot, or ``None`` when no lines were
            found. This is the number the median window, the adaptive threshold
            window and the upscale decision are all sized from.
        page_kind: ``"document"``, ``"blank"`` or ``"photograph"``.
        page_kind_detail: one sentence on how that was decided.
        estimated_skew_degrees: the measured skew, whether or not it was
            corrected.
        source_size: ``(width, height)`` of the input, after EXIF orientation.
        output_size: ``(width, height)`` of :attr:`image`.
        dpi: the input resolution, if the caller said what it was.
        output_dpi: the resolution of :attr:`image`, if that is known.
        threshold_mode: the mode asked for: ``"adaptive"``, ``"otsu"`` or
            ``"none"``.
        threshold_level: the grey level an ``"otsu"`` cut used, else ``None``.
        notes: anything the caller should know, in plain language.
        source: the file path, or ``"<image>"`` for an in-memory image.
        destination: where the page was written, for
            :func:`ocr_cleaner.clean_file`.
    """

    image: Image.Image
    steps: List[Step] = field(default_factory=list)
    skew_corrected_degrees: float = 0.0
    estimated_text_height_px: Optional[float] = None
    page_kind: str = "document"
    page_kind_detail: str = ""
    estimated_skew_degrees: float = 0.0
    source_size: Tuple[int, int] = (0, 0)
    output_size: Tuple[int, int] = (0, 0)
    dpi: Optional[float] = None
    output_dpi: Optional[float] = None
    threshold_mode: str = "adaptive"
    threshold_level: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    source: str = "<image>"
    destination: Optional[str] = None

    # -- looking things up ------------------------------------------------- #

    def step(self, name: str) -> Optional[Step]:
        """The :class:`Step` called ``name``, or ``None`` if it never ran."""
        for entry in self.steps:
            if entry.name == name:
                return entry
        return None

    @property
    def applied(self) -> List[str]:
        """Names of the steps that changed the page, in order."""
        return [entry.name for entry in self.steps if entry.applied]

    @property
    def skipped(self) -> List[str]:
        """Names of the steps that decided to do nothing, in order."""
        return [entry.name for entry in self.steps if not entry.applied]

    @property
    def is_blank(self) -> bool:
        """True when the page was found to be blank and left alone."""
        return self.page_kind == "blank"

    @property
    def is_photograph(self) -> bool:
        """True when the page did not look like a document and was left alone."""
        return self.page_kind == "photograph"

    @property
    def binary(self) -> bool:
        """True when :attr:`image` came out black and white."""
        entry = self.step("threshold")
        return bool(entry is not None and entry.applied)

    @property
    def changed(self) -> bool:
        """True when any step past ``grayscale`` changed the page."""
        return any(name != "grayscale" for name in self.applied)

    # -- saying what happened ---------------------------------------------- #

    def headline(self) -> str:
        """The one-line answer: what the page was, and what was done to it."""
        did = [name for name in self.applied if name != "grayscale"]
        if not did:
            return "{0}: {1} page, left as it came".format(self.source, self.page_kind)
        return "{0}: {1} page, {2}".format(self.source, self.page_kind, ", ".join(did))

    def summary(self) -> str:
        """Human readable report of the whole run, in plain ASCII."""
        height = self.estimated_text_height_px
        lines = [
            "ocr-cleaner: {0}".format(self.source),
            "  page        {0}, {1}".format(self.page_kind, self.page_kind_detail),
            "  size        {0} x {1} in, {2} x {3} out{4}".format(
                self.source_size[0], self.source_size[1],
                self.output_size[0], self.output_size[1],
                "" if self.dpi is None else ", {0:g} dpi in{1}".format(
                    self.dpi,
                    ""
                    if self.output_dpi is None or self.output_dpi == self.dpi
                    else ", {0:g} dpi out".format(self.output_dpi),
                ),
            ),
            "  text        {0}".format(
                "no lines of text found"
                if height is None
                else "lines about {0:.0f} px tall".format(height)
            ),
            "  skew        {0:+.2f} degrees measured, {1}".format(
                self.estimated_skew_degrees,
                "corrected by {0:+.2f}".format(-self.skew_corrected_degrees)
                if self.skew_corrected_degrees
                else "not corrected",
            ),
            "  result      {0}".format(
                "black and white" if self.binary else "greyscale"
            ),
        ]
        if self.destination:
            lines.append("  written to  {0}".format(self.destination))
        lines.append("  steps")
        for entry in self.steps:
            lines.append(
                textwrap.fill(
                    entry.detail,
                    width=_WRAP_WIDTH,
                    initial_indent="    {0:<10}{1:<7} ".format(
                        entry.name, "applied" if entry.applied else "skipped"
                    ),
                    subsequent_indent=_INDENT + "     ",
                )
            )
        for index, note in enumerate(self.notes):
            label = "  notes       " if index == 0 else " " * 14
            lines.append(
                textwrap.fill(
                    note, width=_WRAP_WIDTH, initial_indent=label, subsequent_indent=" " * 14
                )
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of everything except the pixels."""
        height = self.estimated_text_height_px
        return {
            "source": self.source,
            "destination": self.destination,
            "page_kind": self.page_kind,
            "page_kind_detail": self.page_kind_detail,
            "source_size": list(self.source_size),
            "output_size": list(self.output_size),
            "dpi": None if self.dpi is None else float(self.dpi),
            "output_dpi": None if self.output_dpi is None else float(self.output_dpi),
            "estimated_skew_degrees": round(float(self.estimated_skew_degrees), 3),
            "skew_corrected_degrees": round(float(self.skew_corrected_degrees), 3),
            "estimated_text_height_px": None if height is None else round(float(height), 2),
            "threshold_mode": self.threshold_mode,
            "threshold_level": self.threshold_level,
            "binary": self.binary,
            "applied": self.applied,
            "skipped": self.skipped,
            "steps": [entry.to_dict() for entry in self.steps],
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        """:meth:`to_dict` as JSON text, non-ASCII kept as it is."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Any, **kwargs: Any) -> str:
        """Write :attr:`image` to ``path``. Returns the path written.

        Missing parent directories are created. When the page carries a known
        output resolution and the format can store one, it is written into the
        file so the next tool in the chain does not have to be told again.
        """
        destination = os.fspath(path)
        directory = os.path.dirname(destination)
        if directory:
            os.makedirs(directory, exist_ok=True)
        options = dict(kwargs)
        if self.output_dpi and "dpi" not in options:
            suffix = os.path.splitext(destination)[1].lower()
            if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                options["dpi"] = (float(self.output_dpi), float(self.output_dpi))
        try:
            self.image.save(destination, **options)
        except (OSError, ValueError):
            options.pop("dpi", None)     # some plugins refuse a dpi they cannot store
            self.image.save(destination, **options)
        self.destination = destination
        return destination

    def __repr__(self) -> str:           # pragma: no cover - cosmetic
        return (
            "CleanResult(page_kind={0!r}, applied={1}, skew_corrected_degrees="
            "{2:+.2f}, size={3}x{4})".format(
                self.page_kind, self.applied, self.skew_corrected_degrees,
                self.output_size[0], self.output_size[1],
            )
        )
