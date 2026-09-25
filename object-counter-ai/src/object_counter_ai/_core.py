"""``count()``: one image in, one explained number out."""
from __future__ import annotations

import logging
import math
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ._classical import MIN_CONTRAST, ClassicalOutcome, count_blobs
from ._detector import clean, describe_error, parse_output
from ._images import Pixels, copy_for_detector, open_source, to_pixels
from ._region import Region, parse_region
from ._result import METHOD_CLASSICAL, METHOD_DETECTOR, CountResult

logger = logging.getLogger(__name__)

Detector = Callable[[Any], Any]
CLASSICAL_LABEL = "blob"


def _check_area(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a non-negative number of pixels or None, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number of pixels, got {value!r}")
    return float(value)


def check_options(detector: Any, min_area: Any, max_area: Any
                  ) -> Tuple[Optional[Detector], Optional[float], Optional[float]]:
    """Validate the options shared by :func:`count` and :class:`Counter`."""
    if detector is not None and not callable(detector):
        raise TypeError(
            "detector must be a callable detector(image) -> list of boxes, "
            f"not {type(detector).__name__}"
        )
    lo = _check_area(min_area, "min_area")
    hi = _check_area(max_area, "max_area")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"min_area ({lo:g}) is larger than max_area ({hi:g})")
    return detector, lo, hi


class _RegionCache:
    """Rasterises a region once per image size."""

    def __init__(self, region: Optional[Region]):
        self.region = region
        self._masks: Dict[Tuple[int, int], np.ndarray] = {}

    def mask(self, height: int, width: int) -> Optional[np.ndarray]:
        if self.region is None:
            return None
        key = (height, width)
        if key not in self._masks:
            self._masks[key] = self.region.pixel_mask(height, width)
        return self._masks[key]


def _image_shape(image: Any) -> Tuple[int, int, str]:
    """``(width, height, kind)`` without converting the whole image."""
    if isinstance(image, Image.Image):
        w, h = image.size
        kind = {"L": "grey", "1": "grey", "LA": "grey+alpha", "RGBA": "rgba", "RGB": "rgb",
                "P": "palette", "PA": "palette+alpha", "I;16": "grey 16-bit", "I": "grey 32-bit",
                "F": "grey float", "CMYK": "cmyk"}.get(image.mode, image.mode.lower())
        return int(w), int(h), kind
    arr = image if isinstance(image, np.ndarray) else np.asarray(image)
    if arr.size == 0:
        return 0, 0, "grey"
    if arr.ndim == 2:
        return int(arr.shape[1]), int(arr.shape[0]), "grey"
    if arr.ndim == 3:
        if arr.shape[2] in (1, 2, 3, 4):
            kind = {1: "grey", 2: "grey+alpha", 3: "rgb", 4: "rgba"}[arr.shape[2]]
            return int(arr.shape[1]), int(arr.shape[0]), kind
        if arr.shape[0] in (1, 2, 3, 4):
            kind = {1: "grey", 2: "grey+alpha", 3: "rgb", 4: "rgba"}[arr.shape[0]]
            return int(arr.shape[2]), int(arr.shape[1]), kind
    raise ValueError(f"cannot read an image size from an array of shape {arr.shape}")


def _empty_result(method: str, notes: List[str], region: Optional[Region]) -> CountResult:
    return CountResult(
        count=0, boxes=[], labels=[], scores=[], areas=[], centroids=[],
        confidence=1.0, method=method, image_size=(0, 0), image_kind="empty",
        notes=notes + ["nothing to count, so the count is 0"],
        region=region.to_json() if region is not None else None,
        details={"region_text": region.describe()} if region is not None else {},
    )


# --------------------------------------------------------------------------- detector path

def _run_detector(source: Any, detector: Detector, min_area: Optional[float],
                  max_area: Optional[float], regions: _RegionCache) -> CountResult:
    region = regions.region
    opened, loaded = open_source(source)
    width, height, kind = _image_shape(opened)
    region_json = region.to_json() if region is not None else None
    details: Dict[str, Any] = {"region_text": region.describe()} if region is not None else {}
    if width == 0 or height == 0:
        return _empty_result(METHOD_DETECTOR, ["the image has no pixels; the detector was not called"],
                             region)
    notes: List[str] = []
    try:
        raw = detector(copy_for_detector(opened))
        parsed = parse_output(raw)
    except Exception as exc:  # noqa: BLE001 - a user detector may raise anything
        message = describe_error(exc)
        logger.warning("object-counter-ai: %s", message)
        return CountResult(
            count=0, boxes=[], labels=[], scores=[], areas=[], centroids=[],
            confidence=0.0, method=METHOD_DETECTOR, image_size=(width, height),
            image_kind=kind, notes=notes, error=message, region=region_json,
            details=details,
        )
    detections, stats = clean(parsed, width, height)
    details.update(received=stats.received, clipped=stats.clipped, swapped=stats.swapped,
                   outside=stats.outside, non_finite=stats.non_finite)
    if stats.swapped:
        notes.append(f"{stats.swapped} box(es) had right < left or bottom < top; corners were swapped")
    if stats.non_finite:
        notes.append(f"{stats.non_finite} box(es) held NaN or infinity and were dropped")

    kept = []
    by_area = by_region = 0
    for det in detections:
        left, top, right, bottom = det.box
        area = (right - left) * (bottom - top)
        if (min_area is not None and area < min_area) or (max_area is not None and area > max_area):
            by_area += 1
            continue
        kept.append((det, area, ((left + right) / 2.0, (top + bottom) / 2.0)))
    if region is not None and region.kind == "mask":
        regions.mask(height, width)          # raises if the mask is not the image's size
    if region is not None and kept:
        xs = np.array([c[0] for _, _, c in kept])
        ys = np.array([c[1] for _, _, c in kept])
        inside = region.contains(xs, ys)
        by_region = int((~inside).sum())
        kept = [k for k, keep in zip(kept, inside) if keep]
    details.update(dropped_by_area=by_area, dropped_by_region=by_region,
                   min_area=min_area, max_area=max_area)
    if region is not None:
        notes.append("a detection is inside the region when its box centre is")

    scores = [det.score for det, _, _ in kept]
    known = [min(max(s, 0.0), 1.0) for s in scores if s is not None]
    if not kept:
        confidence: Optional[float] = None
        if stats.received == 0:
            notes.append("the detector found nothing")
    elif len(known) == len(kept):
        confidence = float(np.mean(known))
    elif known:
        confidence = float(np.mean(known))
        notes.append(f"{len(kept) - len(known)} of {len(kept)} detections had no score")
    else:
        confidence = None
    return CountResult(
        count=len(kept),
        boxes=[det.box for det, _, _ in kept],
        labels=[det.label for det, _, _ in kept],
        scores=scores,
        areas=[area for _, area, _ in kept],
        centroids=[centre for _, _, centre in kept],
        confidence=None if confidence is None else round(confidence, 4),
        method=METHOD_DETECTOR, image_size=(width, height), image_kind=kind,
        notes=notes, region=region_json, details=details,
    )


# --------------------------------------------------------------------------- classical path

def _classical_confidence(out: ClassicalOutcome, notes: List[str]) -> Tuple[float, Dict[str, float]]:
    factors: Dict[str, float] = {}
    blobs = out.blobs
    n = len(blobs)
    threshold = max(out.threshold, 1e-9)
    if out.floor > MIN_CONTRAST and out.floor >= out.otsu:
        factors["noise"] = 0.8
        notes.append(
            f"the image is noisy: the threshold ({out.threshold:.0f}) was set by the noise, "
            "so faint things may be missed"
        )
    if n == 0:
        if out.specks:
            factors["specks"] = 0.8
            notes.append(f"{out.specks} speck(s) under min_area were ignored; nothing larger was found")
        if out.too_large:
            factors["too_large"] = 0.6
            notes.append(f"{out.too_large} blob(s) over max_area were ignored")
        if out.foreground_fraction == 0 and out.counted_pixels:
            notes.append("no pixel stands out from the background")
    else:
        contrast = float(np.median([b.contrast for b in blobs]))
        sep = min(1.0, max(0.0, (contrast - threshold) / threshold / 0.8))
        factors["separation"] = round(sep, 3)
        if sep < 0.6:
            notes.append(
                f"the blobs are only just above the threshold (median contrast {contrast:.0f} "
                f"vs threshold {out.threshold:.0f}); edges are uncertain"
            )
        if n >= 3:
            areas = np.array([b.area for b in blobs], dtype=np.float64)
            cv = float(areas.std() / areas.mean())
            if cv > 0.35:
                factors["sizes"] = round(max(0.4, 1.0 - (cv - 0.35)), 3)
                notes.append(
                    f"blob sizes vary a lot (coefficient of variation {cv:.2f}); merged or broken "
                    "blobs look like this, so do genuinely mixed sizes"
                )
        pieces = sum(1 for b in blobs if b.split)
        if pieces:
            factors["splits"] = round(1.0 - 0.25 * pieces / n, 3)
            notes.append(f"{pieces} blob(s) came from cutting touching shapes at a neck; check those")
        edge = sum(1 for b in blobs if b.touches_edge)
        if edge:
            factors["image_edge"] = round(1.0 - 0.3 * edge / n, 3)
            notes.append(f"{edge} blob(s) touch the image edge and may be cut off")
        redge = sum(1 for b in blobs if b.touches_region_edge)
        if redge:
            factors["region_edge"] = round(1.0 - 0.3 * redge / n, 3)
            notes.append(f"{redge} blob(s) touch the region boundary and may be cut off")
        if out.foreground_fraction > 0.4:
            factors["coverage"] = 0.5
            notes.append(
                f"blobs cover {out.foreground_fraction:.0%} of the counted area; the counter "
                "assumes the background is most of the picture"
            )
    confidence = 1.0
    for value in factors.values():
        confidence *= value
    return round(confidence, 2), factors


def _run_classical(source: Any, min_area: Optional[float], max_area: Optional[float],
                   regions: _RegionCache) -> CountResult:
    region = regions.region
    pixels: Pixels = to_pixels(source)
    if pixels.empty:
        return _empty_result(METHOD_CLASSICAL, list(pixels.notes), region)
    h, w = pixels.height, pixels.width
    region_mask = regions.mask(h, w)
    usable = pixels.valid if region_mask is None else (pixels.valid & region_mask)
    out = count_blobs(pixels.data, usable, region_mask, min_area, max_area)
    notes = list(pixels.notes) + list(out.notes)
    confidence, factors = _classical_confidence(out, notes)
    blobs = out.blobs
    details: Dict[str, Any] = {
        "threshold": round(out.threshold, 2),
        "otsu": round(out.otsu, 2),
        "floor": round(out.floor, 2),
        "noise": round(out.noise, 3),
        "background": {
            "polarity": out.background_polarity,
            "model": out.background_model,
            "level": round(out.background_level, 2),
            "variation": round(out.background_variation, 2),
        },
        "min_area": out.min_area,
        "max_area": out.max_area,
        "min_area_was_default": min_area is None,
        "specks": out.specks,
        "too_large": out.too_large,
        "split_blobs": out.split_blobs,
        "split_into": out.split_into,
        "foreground_fraction": round(out.foreground_fraction, 4),
        "counted_pixels": out.counted_pixels,
        "confidence_factors": factors,
        "contrast": [round(b.contrast, 2) for b in blobs],
        "split": [b.split for b in blobs],
    }
    if region is not None:
        details["region_text"] = region.describe()
    return CountResult(
        count=len(blobs),
        boxes=[(b.left, b.top, b.right, b.bottom) for b in blobs],
        labels=[CLASSICAL_LABEL] * len(blobs),
        scores=[None] * len(blobs),
        areas=[float(b.area) for b in blobs],
        centroids=[(round(b.cx, 3), round(b.cy, 3)) for b in blobs],
        confidence=confidence,
        method=METHOD_CLASSICAL,
        image_size=(w, h),
        image_kind=pixels.kind,
        notes=notes,
        region=region.to_json() if region is not None else None,
        details=details,
    )


def run_count(image: Any, detector: Optional[Detector], min_area: Optional[float],
              max_area: Optional[float], regions: _RegionCache) -> CountResult:
    """The engine behind :func:`count` and :meth:`Counter.update` (options already checked)."""
    if detector is not None:
        return _run_detector(image, detector, min_area, max_area, regions)
    return _run_classical(image, min_area, max_area, regions)


def count(image: Any, *, detector: Optional[Detector] = None, min_area: Optional[float] = None,
          max_area: Optional[float] = None, region: Any = None) -> CountResult:
    """Count things in one image.

    ``image`` is a numpy array (grey, RGB or RGBA), a ``PIL.Image`` or a file path.
    With ``detector`` (any callable ``detector(image) -> boxes``), its boxes are
    counted, clipped to the image, and filtered by ``min_area``/``max_area`` (box
    area) and ``region`` (box centre). Without one, the built-in classical counter
    counts high-contrast blobs against the background; ``min_area``/``max_area``
    are then blob areas in pixels. ``region`` is a box ``(left, top, right,
    bottom)``, a polygon ``[(x, y), ...]`` or a boolean mask.
    """
    detector, lo, hi = check_options(detector, min_area, max_area)
    return run_count(image, detector, lo, hi, _RegionCache(parse_region(region)))
