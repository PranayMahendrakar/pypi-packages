"""The built-in detectors. Read the warning below before relying on either one.

These are deliberately simple pixel heuristics, not models:

* :func:`detect_faces` looks for connected blobs of skin-like colour whose shape
  is roughly face-shaped.
* :func:`detect_plates` looks for bright, wide, solidly filled rectangles.

They will miss dark skin under poor light, faces in profile, faces behind
glasses or masks, plates that are dirty, angled or not bright, and anything
small. They will also fire on hands, wooden furniture, sand, skin-toned walls
and white signage. They exist so that a redaction pipeline can be wired up and
tested end to end, and so that obvious cases can be hidden. **They are not a
privacy control and must not be used for compliance.** Pass a real model through
``detector=`` for that.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ._image import load_image

LOGGER = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

#: Detection runs on a downscaled copy; this caps its longest side in pixels.
WORK_SIZE = 256

#: The honest one-line caveat, reused by the result object and the CLI.
HEURISTIC_CAVEAT = (
    "the built-in detectors are weak pixel heuristics, not models: "
    "they miss faces and plates and they fire on things that are neither. "
    "Pass detector=your_model for anything that matters."
)


def _downscale(array: np.ndarray) -> Tuple[np.ndarray, int]:
    """Nearest-neighbour downscale by an integer step. Returns ``(small, step)``."""
    height, width = array.shape[:2]
    longest = max(height, width)
    step = max(1, int(np.ceil(longest / float(WORK_SIZE))))
    if step == 1:
        return array, 1
    return array[::step, ::step], step


def _row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open ``[start, end)`` spans of True in a 1-D boolean row."""
    if not row.any():
        return []
    width = row.shape[0]
    edges = np.diff(row.astype(np.int8))
    starts = (np.flatnonzero(edges == 1) + 1).tolist()
    ends = (np.flatnonzero(edges == -1) + 1).tolist()
    if row[0]:
        starts.insert(0, 0)
    if row[-1]:
        ends.append(width)
    return list(zip(starts, ends))


def connected_components(mask: np.ndarray) -> List[Dict[str, Any]]:
    """Label 8-connected True blobs in ``mask``.

    Returns one dict per blob with ``box`` (left, top, right, bottom, half-open),
    ``area`` in pixels and ``fill`` (area divided by the box area). Implemented by
    unioning row runs so it stays fast without scipy.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:  # pragma: no cover - internal guard
        raise ValueError("connected_components needs a 2-dimensional mask")
    height = mask.shape[0]

    parent: List[int] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    runs: List[Tuple[int, int, int]] = []          # (row, start, end)
    previous: List[int] = []                       # run ids on the row above
    for y in range(height):
        current: List[int] = []
        for start, end in _row_runs(mask[y]):
            run_id = len(runs)
            runs.append((y, start, end))
            parent.append(run_id)
            for other in previous:
                _, o_start, o_end = runs[other]
                if o_start - 1 < end and start - 1 < o_end:   # 8-connected
                    union(run_id, other)
            current.append(run_id)
        previous = current

    blobs: Dict[int, Dict[str, Any]] = {}
    for run_id, (y, start, end) in enumerate(runs):
        root = find(run_id)
        blob = blobs.get(root)
        if blob is None:
            blobs[root] = {
                "left": start,
                "top": y,
                "right": end,
                "bottom": y + 1,
                "area": end - start,
            }
            continue
        blob["left"] = min(blob["left"], start)
        blob["right"] = max(blob["right"], end)
        blob["bottom"] = max(blob["bottom"], y + 1)
        blob["area"] += end - start

    out: List[Dict[str, Any]] = []
    for blob in blobs.values():
        width = blob["right"] - blob["left"]
        tall = blob["bottom"] - blob["top"]
        box_area = max(1, width * tall)
        out.append({
            "box": (blob["left"], blob["top"], blob["right"], blob["bottom"]),
            "area": int(blob["area"]),
            "fill": blob["area"] / float(box_area),
        })
    out.sort(key=lambda item: -item["area"])
    return out


def _to_rgb(array: np.ndarray, mode: str) -> np.ndarray:
    """Drop alpha and fan grey out to three channels, as float32."""
    data = array.astype(np.float32)
    if mode == "L":
        return np.repeat(data, 3, axis=2)
    return data[:, :, :3]


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """A boolean skin-tone mask for a float32 ``(h, w, 3)`` RGB array.

    Two rules ORed together: the classic Kovac RGB thresholds, which work on
    lighter skin under daylight, and a normalised-rgb ratio rule, which picks up
    some of the darker and warmer-lit skin the first rule drops. Both are crude.
    """
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    peak = np.maximum(np.maximum(red, green), blue)
    floor = np.minimum(np.minimum(red, green), blue)

    kovac = (
        (red > 95) & (green > 40) & (blue > 20)
        & ((peak - floor) > 15)
        & (np.abs(red - green) > 15)
        & (red > green) & (red > blue)
    )

    total = red + green + blue + 1e-6
    r_norm, g_norm = red / total, green / total
    ratio = (
        (r_norm > 0.33) & (r_norm < 0.52)
        & (g_norm > 0.26) & (g_norm < 0.37)
        & (red > green) & (green >= blue)
        & (peak > 40)
    )
    return kovac | ratio


def _scale_box(box: Box, step: int, width: int, height: int) -> Box:
    """Map a box from the downscaled working image back to full resolution."""
    left, top, right, bottom = box
    return (
        int(max(0, left * step)),
        int(max(0, top * step)),
        int(min(width, right * step)),
        int(min(height, bottom * step)),
    )


def detect_faces(image: Any) -> List[Box]:
    """Find face-shaped blobs of skin-like colour. A weak heuristic, not a model.

    Returns ``[(left, top, right, bottom), ...]`` in the coordinates of the image
    you passed in, largest first. Accepts a path, a Pillow image or a numpy array.

    Greyscale images always return ``[]``: there is no colour to judge skin tone
    by. See the module docstring for everything else this gets wrong; it is for
    hiding obvious cases and for exercising a pipeline, never for compliance.
    """
    loaded = load_image(image)
    if loaded.mode == "L":
        LOGGER.debug("detect_faces: greyscale image has no colour to judge skin tone by")
        return []
    small, step = _downscale(loaded.array)
    rgb = _to_rgb(small, loaded.mode)
    mask = skin_mask(rgb)

    pixels = float(mask.shape[0] * mask.shape[1])
    if not mask.any() or mask.sum() > 0.6 * pixels:
        # Nothing, or almost everything: either way the heuristic has nothing to say.
        return []

    min_area = max(9.0, 0.0015 * pixels)
    boxes: List[Box] = []
    for blob in connected_components(mask):
        if blob["area"] < min_area:
            continue
        left, top, right, bottom = blob["box"]
        width, height = right - left, bottom - top
        if width < 3 or height < 3:
            continue
        aspect = width / float(height)
        if not 0.45 <= aspect <= 1.6:            # faces are roughly as tall as wide
            continue
        if blob["fill"] < 0.45:                  # a face blob is solid, not a spider
            continue
        boxes.append(_scale_box(blob["box"], step, loaded.width, loaded.height))
    return boxes


def detect_plates(image: Any) -> List[Box]:
    """Find bright, wide, solidly filled rectangles. A weak heuristic, not a model.

    Returns ``[(left, top, right, bottom), ...]`` in the coordinates of the image
    you passed in, largest first. This finds a clean, well-lit, roughly front-on
    plate, and it equally finds a white sign, a window or a sheet of paper. It
    reads no characters and confirms nothing. Use a real plate model via
    ``detector=`` when it matters.
    """
    loaded = load_image(image)
    small, step = _downscale(loaded.array)
    rgb = _to_rgb(small, loaded.mode)
    luma = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

    cutoff = max(150.0, min(float(luma.mean() + 1.5 * luma.std()), 245.0))
    mask = luma >= cutoff

    pixels = float(mask.shape[0] * mask.shape[1])
    if not mask.any() or mask.sum() > 0.5 * pixels:
        return []

    min_area = max(12.0, 0.0008 * pixels)
    boxes: List[Box] = []
    for blob in connected_components(mask):
        if blob["area"] < min_area:
            continue
        left, top, right, bottom = blob["box"]
        width, height = right - left, bottom - top
        if width < 6 or height < 2:
            continue
        aspect = width / float(height)
        if not 1.8 <= aspect <= 7.0:             # plates are wide
            continue
        if blob["fill"] < 0.6:                   # and rectangular
            continue
        boxes.append(_scale_box(blob["box"], step, loaded.width, loaded.height))
    return boxes


#: The detectors ``redact`` falls back to when you give it no regions and no detector.
BUILTIN_DETECTORS = (detect_faces, detect_plates)
