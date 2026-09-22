"""Region maths, the four redaction methods, and the result object.

Everything public in the package funnels through :class:`Redactor`; the module
level :func:`redact` and :func:`redact_file` are the one-line front doors.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._detect import BUILTIN_DETECTORS, HEURISTIC_CAVEAT, detect_faces, detect_plates
from ._image import Loaded, load_image, save_image

LOGGER = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

#: What each ``method`` does, in one line. Used by the CLI help and the README.
METHOD_HELP = {
    "blur": "smooth the region until detail is gone (reversible in principle - see below)",
    "pixelate": "replace each block of pixels with that block's average colour",
    "fill": "paint the region a single flat colour (mid grey by default)",
    "blackout": "paint the region solid black",
}

#: The four accepted values of ``method``.
METHODS = tuple(METHOD_HELP)

#: ``fill`` and ``blackout`` throw every pixel away; ``pixelate`` keeps only a
#: block average. ``blur`` is a linear filter and is NOT in this set.
IRREVERSIBLE_METHODS = ("pixelate", "fill", "blackout")

_DEFAULT_FILL = (128, 128, 128)


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #

def _looks_like_single_box(regions: Any) -> bool:
    """True for ``(10, 10, 40, 40)`` and False for ``[(10, 10, 40, 40)]``."""
    if isinstance(regions, (str, bytes)) or not isinstance(regions, (list, tuple, np.ndarray)):
        return False
    if len(regions) != 4:
        return False
    return all(isinstance(value, (int, float, np.integer, np.floating)) for value in regions)


def parse_box(raw: Any, label: str) -> Tuple[float, float, float, float]:
    """Validate one ``(left, top, right, bottom)``; raise a clear ``ValueError``."""
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            f"{label} is a string; a box is 4 numbers (left, top, right, bottom), "
            f"for example (10, 20, 90, 100). Got {raw!r}"
        )
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} is not a box: expected 4 numbers (left, top, right, bottom), "
            f"got {raw!r}"
        ) from None
    if len(values) != 4:
        raise ValueError(
            f"{label} has {len(values)} value(s); a box is exactly 4 "
            f"(left, top, right, bottom). Got {raw!r}"
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite number: {raw!r}")
    left, top, right, bottom = values
    if right < left:
        raise ValueError(
            f"{label} is inverted: right ({right:g}) is less than left ({left:g}). "
            "A box is (left, top, right, bottom) with left <= right."
        )
    if bottom < top:
        raise ValueError(
            f"{label} is inverted: bottom ({bottom:g}) is less than top ({top:g}). "
            "A box is (left, top, right, bottom) with top <= bottom."
        )
    return left, top, right, bottom


def parse_regions(regions: Any, label: str = "region") -> List[Tuple[float, float, float, float]]:
    """Validate a list of boxes. A single flat 4-number box is accepted too."""
    if regions is None:
        return []
    if _looks_like_single_box(regions):
        return [parse_box(regions, f"{label} 0")]
    try:
        items = list(regions)
    except TypeError:
        raise ValueError(
            f"{label}s must be a list of boxes, for example [(10, 20, 90, 100)]; "
            f"got {type(regions).__name__}"
        ) from None
    return [parse_box(item, f"{label} {index}") for index, item in enumerate(items)]


def expand_box(
    box: Tuple[float, float, float, float], expand: float
) -> Tuple[float, float, float, float]:
    """Push each edge outward by ``expand`` times the box's width or height."""
    left, top, right, bottom = box
    grow_x = (right - left) * expand
    grow_y = (bottom - top) * expand
    return (left - grow_x, top - grow_y, right + grow_x, bottom + grow_y)


def clip_box(box: Tuple[float, float, float, float], width: int, height: int) -> Box:
    """Round outward, then clamp to the image. May come back with zero area."""
    left, top, right, bottom = box
    return (
        int(min(max(math.floor(left), 0), width)),
        int(min(max(math.floor(top), 0), height)),
        int(min(max(math.ceil(right), 0), width)),
        int(min(max(math.ceil(bottom), 0), height)),
    )


def _contains(outer: Box, inner: Box) -> bool:
    return (
        outer[0] <= inner[0] and outer[1] <= inner[1]
        and outer[2] >= inner[2] and outer[3] >= inner[3]
    )


def dedupe_boxes(boxes: Sequence[Box]) -> List[Box]:
    """Drop exact duplicates and any box fully inside another, biggest first."""
    ordered = sorted(
        dict.fromkeys(tuple(box) for box in boxes),
        key=lambda box: -((box[2] - box[0]) * (box[3] - box[1])),
    )
    kept: List[Box] = []
    for box in ordered:
        if not any(_contains(other, box) for other in kept):
            kept.append(box)
    return kept


# --------------------------------------------------------------------------- #
# pixel operations
# --------------------------------------------------------------------------- #

def _box_blur_axis(data: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """One box-blur pass along ``axis``, in O(1) per pixel via a running sum."""
    if radius < 1:
        return data
    length = data.shape[axis]
    pad = [(0, 0)] * data.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(data, pad, mode="edge")
    sums = np.cumsum(padded, axis=axis, dtype=np.float64)
    head_shape = list(sums.shape)
    head_shape[axis] = 1
    sums = np.concatenate([np.zeros(head_shape, dtype=sums.dtype), sums], axis=axis)
    window = 2 * radius + 1
    high = np.take(sums, np.arange(window, window + length), axis=axis)
    low = np.take(sums, np.arange(0, length), axis=axis)
    return ((high - low) / window).astype(np.float32)


def gaussian_blur(patch: np.ndarray, radius: float) -> np.ndarray:
    """Blur ``patch`` (float32, ``(h, w, c)``) by roughly ``radius`` pixels.

    Three box-blur passes, the standard gaussian approximation used by SVG
    filters. It is fast whatever the radius, and visually indistinguishable from
    a true gaussian at these strengths.
    """
    sigma = max(float(radius), 0.0) / 2.0
    if sigma <= 0:
        return patch
    box_radius = max(1, int(round((math.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))
    out = patch.astype(np.float32)
    for _ in range(3):
        out = _box_blur_axis(out, box_radius, 0)
        out = _box_blur_axis(out, box_radius, 1)
    return out


def pixelate(patch: np.ndarray, block: int) -> np.ndarray:
    """Replace every ``block`` x ``block`` tile of ``patch`` with its mean colour."""
    height, width = patch.shape[:2]
    block = max(1, min(int(block), max(height, width)))
    out = np.empty_like(patch, dtype=np.float32)
    source = patch.astype(np.float32)
    for top in range(0, height, block):
        bottom = min(top + block, height)
        for left in range(0, width, block):
            right = min(left + block, width)
            out[top:bottom, left:right] = source[top:bottom, left:right].mean(axis=(0, 1))
    return out


def blur_radius_for(width: int, height: int, strength: float) -> float:
    """How far to blur a ``width`` x ``height`` region at ``strength``."""
    shortest = float(min(width, height))
    return float(min(max(1.0, strength * shortest / 4.0), max(1.0, shortest)))


def block_size_for(width: int, height: int, strength: float) -> int:
    """How big a pixelation block to use for a ``width`` x ``height`` region."""
    shortest = min(width, height)
    return int(max(2, min(round(strength * shortest / 3.0), max(1, shortest))))


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DetectorReport:
    """What one detector did: how many boxes it returned, or how it failed."""

    name: str
    boxes: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "boxes": self.boxes, "error": self.error}


@dataclass
class RedactResult:
    """Everything ``redact`` did, and the redacted image.

    ``image`` is the same kind of object you passed in: a numpy array in gives a
    numpy array out, a Pillow image gives a Pillow image, and a path gives a
    Pillow image. The object you passed in is never modified.
    """

    image: Any
    boxes: List[Box] = field(default_factory=list)
    detector_used: Optional[str] = None
    method: str = "blur"
    strength: float = 0.9
    expand: float = 0.08
    width: int = 0
    height: int = 0
    mode: str = "RGB"
    changed: bool = False
    detections: List[DetectorReport] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source: Optional[str] = None
    saved_to: Optional[str] = None

    @property
    def count(self) -> int:
        """How many regions were actually redacted."""
        return len(self.boxes)

    @property
    def irreversible(self) -> bool:
        """True when the method throws the original pixels away for good."""
        return self.method in IRREVERSIBLE_METHODS

    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)``, the order Pillow uses."""
        return (self.width, self.height)

    def save(self, dst: Any) -> Path:
        """Write the redacted image to ``dst``; returns the path written."""
        pil = self.image
        if not hasattr(pil, "save"):
            from PIL import Image

            array = np.asarray(self.image)
            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            if array.dtype != np.uint8:
                array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255)
                array = array.astype(np.uint8)
            pil = Image.fromarray(array, mode="L" if array.ndim == 2 else self.mode)
        path, warning = save_image(pil, dst)
        if warning and warning not in self.warnings:
            self.warnings.append(warning)
        self.saved_to = str(path)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of everything except the pixels."""
        return {
            "count": self.count,
            "boxes": [list(box) for box in self.boxes],
            "method": self.method,
            "strength": self.strength,
            "expand": self.expand,
            "irreversible": self.irreversible,
            "changed": self.changed,
            "detector_used": self.detector_used,
            "detections": [report.to_dict() for report in self.detections],
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "source": self.source,
            "saved_to": self.saved_to,
        }

    def to_json(self, indent: int = 2) -> str:
        """``to_dict()`` as JSON text, non-ASCII kept as-is."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """A short human-readable report. Plain ASCII punctuation throughout."""
        lines: List[str] = []
        if self.count:
            lines.append(
                f"image-redactor: {self.count} region(s) redacted with {self.method} "
                f"(strength {self.strength:g}, expand {self.expand:g})"
            )
        else:
            lines.append(
                "image-redactor: nothing to redact - 0 regions found, "
                "the image is returned unchanged"
            )
        lines.append(f"  image: {self.width}x{self.height} {self.mode}")
        if self.source:
            lines.append(f"  source: {self.source}")
        if self.detector_used:
            lines.append(f"  detector: {self.detector_used}")
            for report in self.detections:
                if report.error:
                    lines.append(f"    {report.name}: FAILED - {report.error}")
                else:
                    lines.append(f"    {report.name}: {report.boxes} box(es)")
        else:
            lines.append("  detector: none (the regions you passed were used as given)")
        if self.count:
            shown = ", ".join(str(tuple(box)) for box in self.boxes[:6])
            if self.count > 6:
                shown += f", ... {self.count - 6} more"
            lines.append(f"  boxes: {shown}")
            lines.append(
                "  pixels changed: "
                + ("yes" if self.changed else "no (the region was already flat)")
            )
            lines.append(
                "  reversible: "
                + (
                    "no, the original pixels are gone"
                    if self.irreversible
                    else "blur is a linear filter, so treat it as obscured, not destroyed"
                )
            )
        if self.saved_to:
            lines.append(f"  saved to: {self.saved_to}")
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        for error in self.errors:
            lines.append(f"  error: {error}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RedactResult(count={self.count}, method={self.method!r}, "
            f"size={self.width}x{self.height}, changed={self.changed})"
        )


# --------------------------------------------------------------------------- #
# the redactor
# --------------------------------------------------------------------------- #

def _detector_name(detector: Callable[[Any], Any]) -> str:
    """A readable name for a detector, marking the built-ins as built-in."""
    for builtin in BUILTIN_DETECTORS:
        if detector is builtin:
            return f"built-in {builtin.__name__}"
    name = getattr(detector, "__name__", None)
    if not name:
        name = getattr(type(detector), "__name__", "detector")
    return str(name)


class Redactor:
    """The redactor behind :func:`redact`, for when you want to keep settings.

    ``redactor = Redactor(method="pixelate", strength=1.0)`` then
    ``redactor.redact(image)`` for each image. ``fill_color`` only affects
    ``method="fill"``.
    """

    def __init__(
        self,
        *,
        method: str = "blur",
        strength: float = 0.9,
        expand: float = 0.08,
        detector: Any = None,
        fill_color: Sequence[int] = _DEFAULT_FILL,
    ) -> None:
        self.method = self._check_method(method)
        self.strength = self._check_unit(strength, "strength")
        self.expand = self._check_expand(expand)
        self.detector = detector
        self.fill_color = tuple(int(value) for value in fill_color)
        if len(self.fill_color) != 3:
            raise ValueError(
                f"fill_color must be 3 numbers (red, green, blue); got {tuple(fill_color)!r}"
            )
        if any(not 0 <= value <= 255 for value in self.fill_color):
            raise ValueError(
                f"fill_color values must be between 0 and 255; got {self.fill_color!r}"
            )

    # -- validation -------------------------------------------------------- #

    @staticmethod
    def _check_method(method: Any) -> str:
        if not isinstance(method, str) or method.lower() not in METHODS:
            raise ValueError(
                f"method must be one of {', '.join(METHODS)}; got {method!r}"
            )
        return method.lower()

    @staticmethod
    def _check_unit(value: Any, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number between 0 and 1; got {value!r}") from None
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1; got {value!r}")
        return number

    @staticmethod
    def _check_expand(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"expand must be a number, for example 0.08 for 8 percent; got {value!r}"
            ) from None
        if not math.isfinite(number) or number < 0:
            raise ValueError(
                f"expand must be 0 or more (it grows each box); got {value!r}"
            )
        return number

    # -- detection --------------------------------------------------------- #

    @staticmethod
    def _as_detector_list(detector: Any) -> List[Callable[[Any], Any]]:
        if detector is None:
            return []
        if callable(detector):
            return [detector]
        if isinstance(detector, (list, tuple)):
            for item in detector:
                if not callable(item):
                    raise ValueError(
                        "detector must be a callable, or a list of callables, each "
                        f"taking the image and returning boxes; got {item!r} in the list"
                    )
            return list(detector)
        raise ValueError(
            "detector must be a callable taking the image and returning "
            f"[(left, top, right, bottom), ...], or a list of such callables; "
            f"got {type(detector).__name__}"
        )

    def _run_detectors(
        self, detectors: Sequence[Callable[[Any], Any]], loaded: Loaded
    ) -> Tuple[List[Tuple[float, float, float, float]], List[DetectorReport], List[str]]:
        """Run every detector, keeping going when one of them blows up."""
        found: List[Tuple[float, float, float, float]] = []
        reports: List[DetectorReport] = []
        errors: List[str] = []
        handed = loaded.to_pil()
        for detector in detectors:
            name = _detector_name(detector)
            try:
                raw = detector(handed)
                boxes = parse_regions(raw, label=f"box from {name}")
            except Exception as exc:  # noqa: BLE001 - one bad detector must not sink the rest
                message = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("detector %s failed: %s", name, message)
                reports.append(DetectorReport(name=name, boxes=0, error=message))
                errors.append(f"detector {name} failed and was skipped ({message})")
                continue
            reports.append(DetectorReport(name=name, boxes=len(boxes)))
            found.extend(boxes)
        return found, reports, errors

    # -- pixels ------------------------------------------------------------ #

    def _apply(self, array: np.ndarray, box: Box) -> None:
        """Redact one already-clipped box, in place, on our own copy."""
        left, top, right, bottom = box
        colour_channels = 3 if array.shape[2] == 4 else array.shape[2]
        patch = array[top:bottom, left:right, :colour_channels]
        width, height = right - left, bottom - top

        if self.method == "blackout":
            patch[:] = 0
            return
        if self.method == "fill":
            if colour_channels == 1:
                grey = int(round(sum(self.fill_color) / 3.0))
                patch[:] = grey
            else:
                patch[:] = np.array(self.fill_color, dtype=np.uint8)
            return
        if self.method == "pixelate":
            block = block_size_for(width, height, self.strength)
            patch[:] = np.clip(np.rint(pixelate(patch, block)), 0, 255).astype(np.uint8)
            return
        radius = blur_radius_for(width, height, self.strength)
        patch[:] = np.clip(np.rint(gaussian_blur(patch.astype(np.float32), radius)), 0, 255).astype(
            np.uint8
        )

    # -- the work ---------------------------------------------------------- #

    def redact(self, image: Any, *, regions: Any = None, detector: Any = None) -> RedactResult:
        """Redact ``image``; see :func:`redact` for the full description."""
        loaded = load_image(image)
        original = loaded.array
        working = original.copy()

        warnings: List[str] = []
        errors: List[str] = []
        detections: List[DetectorReport] = []

        given = parse_regions(regions)
        chosen = detector if detector is not None else self.detector
        detectors = self._as_detector_list(chosen)

        asked_for_none = chosen is not None and not detectors
        used_builtin = False
        if not given and not detectors and not asked_for_none:
            detectors = list(BUILTIN_DETECTORS)
            used_builtin = True

        detected: List[Tuple[float, float, float, float]] = []
        detector_used: Optional[str] = None
        if detectors:
            detected, detections, errors = self._run_detectors(detectors, loaded)
            detector_used = ", ".join(report.name for report in detections)
            if used_builtin:
                warnings.append(HEURISTIC_CAVEAT)
                if loaded.mode == "L":
                    warnings.append(
                        "this image is greyscale, so the built-in face heuristic "
                        "(which works on skin tone) found nothing by definition"
                    )

        raw_boxes = given + detected
        boxes: List[Box] = []
        dropped_outside = 0
        dropped_empty = 0
        clipped = 0
        for box in raw_boxes:
            if box[2] <= box[0] or box[3] <= box[1]:
                dropped_empty += 1
                continue
            grown = expand_box(box, self.expand)
            cut = clip_box(grown, loaded.width, loaded.height)
            if cut[2] <= cut[0] or cut[3] <= cut[1]:
                dropped_outside += 1
                continue
            if (
                box[0] < 0 or box[1] < 0
                or box[2] > loaded.width or box[3] > loaded.height
            ):
                clipped += 1
            boxes.append(cut)

        boxes = dedupe_boxes(boxes)
        for box in boxes:
            self._apply(working, box)

        if dropped_outside:
            warnings.append(
                f"{dropped_outside} region(s) fell entirely outside the "
                f"{loaded.width}x{loaded.height} image and were dropped"
            )
        if dropped_empty:
            warnings.append(
                f"{dropped_empty} region(s) had zero width or height and were dropped"
            )
        if clipped:
            warnings.append(
                f"{clipped} region(s) reached past the image edge and were clipped to it"
            )
        if not boxes:
            warnings.append(
                "no regions were redacted; pass regions=[(left, top, right, bottom), ...] "
                "or detector=your_model to hide something"
            )

        return RedactResult(
            image=loaded.to_output(working),
            boxes=boxes,
            detector_used=detector_used,
            method=self.method,
            strength=self.strength,
            expand=self.expand,
            width=loaded.width,
            height=loaded.height,
            mode=loaded.mode,
            changed=not np.array_equal(original, working),
            detections=detections,
            warnings=warnings,
            errors=errors,
            source=loaded.source,
        )


# --------------------------------------------------------------------------- #
# front doors
# --------------------------------------------------------------------------- #

def redact(
    image: Any,
    *,
    regions: Any = None,
    detector: Any = None,
    method: str = "blur",
    strength: float = 0.9,
    expand: float = 0.08,
) -> RedactResult:
    """Hide parts of an image and report exactly what was hidden.

    ``image`` is a path, a ``PIL.Image.Image`` or a numpy array (greyscale, RGB
    or RGBA). It is never modified; the redacted copy comes back as
    ``result.image``, the same kind of object you passed in.

    ``regions`` is ``[(left, top, right, bottom), ...]`` for the parts you
    already know about. A single box may be passed unwrapped. Boxes reaching
    past the edge are clipped; a box with ``right < left`` (or ``bottom < top``)
    raises ``ValueError``.

    ``detector`` is any ``callable(image) -> [(left, top, right, bottom), ...]``,
    or a list of them, and is how you plug in a real face or plate model without
    this package depending on one. Detectors are handed a ``PIL.Image.Image``;
    ``numpy.asarray(img)`` is one line away. A detector that raises is caught,
    recorded in ``result.detections`` and ``result.errors``, and the others still
    run. When you pass neither ``regions`` nor ``detector``, the weak built-in
    heuristics run as a fallback and say so in ``result.warnings``.

    ``method`` is ``blur``, ``pixelate``, ``fill`` or ``blackout``. ``strength``
    (0 to 1) sets how hard ``blur`` and ``pixelate`` hit; ``fill`` and
    ``blackout`` ignore it and always replace every pixel. ``expand`` grows each
    box by that fraction of its own width and height first, because a tight box
    leaves identifying edges behind.

    With nothing to redact you get the image back unchanged, ``count == 0`` and a
    ``summary()`` that says so; it is not an error.
    """
    return Redactor(method=method, strength=strength, expand=expand).redact(
        image, regions=regions, detector=detector
    )


def redact_file(src: Any, dst: Any, **kwargs: Any) -> RedactResult:
    """Redact the image at ``src`` and write the result to ``dst``.

    Takes the same keyword arguments as :func:`redact`. Missing folders in
    ``dst`` are created. The format follows the ``dst`` suffix; writing an
    image with transparency to ``.jpg`` drops the alpha channel and records a
    warning rather than failing.
    """
    result = redact(src, **kwargs)
    result.save(dst)
    return result


__all__ = [
    "BUILTIN_DETECTORS",
    "DetectorReport",
    "HEURISTIC_CAVEAT",
    "IRREVERSIBLE_METHODS",
    "METHODS",
    "METHOD_HELP",
    "RedactResult",
    "Redactor",
    "detect_faces",
    "detect_plates",
    "redact",
    "redact_file",
]
