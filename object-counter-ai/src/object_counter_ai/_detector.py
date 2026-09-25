"""Reading whatever a user's detector returns, and cleaning it up.

A detector is any callable ``detector(image)``. It may return:

* a list of boxes ``(left, top, right, bottom)``;
* a list of ``(box, label, score)``, ``(box, label)`` or ``(box, score)``;
* a list of dicts with ``box`` (or ``bbox``) and optional ``label`` and ``score``;
* a dict of parallel lists ``{"boxes": ..., "labels": ..., "scores": ...}``,
  the shape torchvision detection models produce;
* a numpy array shaped ``(n, 4)``, ``(n, 5)`` (plus score) or ``(n, 6)``
  (plus score and class id), the shape many YOLO exports produce.

Boxes are in pixels. Anything outside the image is clipped to it; boxes that
end up with no area are dropped, and the result says how many.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LABEL = "object"

Box = Tuple[float, float, float, float]


class DetectorOutputError(ValueError):
    """The detector returned something that is not a list of boxes."""


@dataclass
class Detection:
    """One cleaned-up detection."""

    box: Box
    label: str
    score: Optional[float]


@dataclass
class CleanStats:
    """What cleaning changed."""

    received: int = 0
    clipped: int = 0
    swapped: int = 0
    outside: int = 0
    non_finite: int = 0


def _is_number(value: Any) -> bool:
    return isinstance(value, (Real, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _as_box(value: Any, where: str) -> Box:
    try:
        seq = list(value)
    except TypeError:
        raise DetectorOutputError(f"{where}: a box must be 4 numbers, got {type(value).__name__}") from None
    if len(seq) != 4 or not all(_is_number(v) for v in seq):
        raise DetectorOutputError(
            f"{where}: a box must be 4 numbers (left, top, right, bottom), got {value!r}"
        )
    left, top, right, bottom = (float(v) for v in seq)
    return left, top, right, bottom


def _as_label(value: Any) -> str:
    if value is None:
        return DEFAULT_LABEL
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text or DEFAULT_LABEL


def _as_score(value: Any, where: str) -> Optional[float]:
    if value is None:
        return None
    if not _is_number(value):
        raise DetectorOutputError(f"{where}: a score must be a number, got {value!r}")
    score = float(value)
    return score if math.isfinite(score) else None


def _from_rows(rows: np.ndarray) -> List[Tuple[Box, str, Optional[float]]]:
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] not in (4, 5, 6):
        raise DetectorOutputError(
            f"an array of detections must be shaped (n, 4), (n, 5) or (n, 6), got {rows.shape}"
        )
    out = []
    for row in rows.astype(np.float64):
        box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        score = float(row[4]) if rows.shape[1] >= 5 and math.isfinite(row[4]) else None
        label = _as_label(row[5]) if rows.shape[1] == 6 else DEFAULT_LABEL
        out.append((box, label, score))
    return out


def _item(item: Any, index: int) -> Tuple[Box, str, Optional[float]]:
    where = f"detection {index}"
    if isinstance(item, dict):
        box = item.get("box", item.get("bbox"))
        if box is None:
            raise DetectorOutputError(f"{where}: a dict detection needs a 'box' (or 'bbox') key")
        return (_as_box(box, where), _as_label(item.get("label")),
                _as_score(item.get("score"), where))
    if isinstance(item, np.ndarray) and item.ndim == 1 and item.size in (4, 5, 6):
        return _from_rows(item)[0]
    try:
        parts = list(item)
    except TypeError:
        raise DetectorOutputError(
            f"{where}: expected a box or (box, label, score), got {type(item).__name__}"
        ) from None
    if len(parts) == 4 and all(_is_number(v) for v in parts):
        return _as_box(parts, where), DEFAULT_LABEL, None
    if len(parts) in (5, 6) and all(_is_number(v) for v in parts):
        return _from_rows(np.array(parts, dtype=np.float64))[0]
    if len(parts) == 3:
        return _as_box(parts[0], where), _as_label(parts[1]), _as_score(parts[2], where)
    if len(parts) == 2:
        box = _as_box(parts[0], where)
        extra = parts[1]
        if _is_number(extra):
            return box, DEFAULT_LABEL, _as_score(extra, where)
        return box, _as_label(extra), None
    raise DetectorOutputError(
        f"{where}: expected a box (left, top, right, bottom) or (box, label, score), got {item!r}"
    )


def parse_output(raw: Any) -> List[Tuple[Box, str, Optional[float]]]:
    """Turn a detector's return value into ``[(box, label, score), ...]``."""
    if raw is None:
        raise DetectorOutputError(
            "the detector returned None; return a list of boxes, or an empty list when "
            "nothing is found"
        )
    if isinstance(raw, dict):
        if "boxes" not in raw:
            raise DetectorOutputError(
                "a dict from the detector needs a 'boxes' key (and optional 'labels', 'scores')"
            )
        boxes = _to_list(raw["boxes"])
        labels = _to_list(raw.get("labels")) if raw.get("labels") is not None else None
        scores = _to_list(raw.get("scores")) if raw.get("scores") is not None else None
        for name, values in (("labels", labels), ("scores", scores)):
            if values is not None and len(values) != len(boxes):
                raise DetectorOutputError(
                    f"the detector returned {len(boxes)} boxes but {len(values)} {name}"
                )
        return [
            (_as_box(box, f"detection {i}"),
             _as_label(labels[i]) if labels is not None else DEFAULT_LABEL,
             _as_score(scores[i], f"detection {i}") if scores is not None else None)
            for i, box in enumerate(boxes)
        ]
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return []
        return _from_rows(raw)
    if isinstance(raw, (str, bytes)):
        raise DetectorOutputError("the detector returned text; return a list of boxes")
    try:
        items = list(raw)
    except TypeError:
        raise DetectorOutputError(
            f"the detector returned {type(raw).__name__}; return a list of boxes"
        ) from None
    return [_item(item, i) for i, item in enumerate(items)]


def _to_list(values: Any) -> List[Any]:
    if hasattr(values, "detach"):          # torch tensors, without importing torch
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy") and not isinstance(values, np.ndarray):
        values = values.numpy()
    if isinstance(values, np.ndarray):
        return list(values)
    return list(values)


def clean(parsed: Sequence[Tuple[Box, str, Optional[float]]], width: int, height: int
          ) -> Tuple[List[Detection], CleanStats]:
    """Order corners, clip to the image and drop boxes with no area left."""
    stats = CleanStats(received=len(parsed))
    out: List[Detection] = []
    for (left, top, right, bottom), lab, score in parsed:
        if not all(math.isfinite(v) for v in (left, top, right, bottom)):
            stats.non_finite += 1
            continue
        if left > right or top > bottom:
            stats.swapped += 1
            left, right = min(left, right), max(left, right)
            top, bottom = min(top, bottom), max(top, bottom)
        cl = min(max(left, 0.0), float(width))
        cr = min(max(right, 0.0), float(width))
        ct = min(max(top, 0.0), float(height))
        cb = min(max(bottom, 0.0), float(height))
        if cr - cl <= 0 or cb - ct <= 0:
            stats.outside += 1
            continue
        if (cl, ct, cr, cb) != (left, top, right, bottom):
            stats.clipped += 1
        out.append(Detection((cl, ct, cr, cb), lab, score))
    return out, stats


def describe_error(exc: BaseException) -> str:
    """A one-line description of a detector failure."""
    if isinstance(exc, DetectorOutputError):
        return f"detector output could not be read: {exc}"
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"detector raised {type(exc).__name__}" + (f": {text}" if text else "")
