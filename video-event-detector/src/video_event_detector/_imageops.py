"""Small numpy image operations: smoothing, mask cleaning, labelling, alignment.

These replace the handful of OpenCV / scipy calls a motion pipeline usually
leans on. They work on the small working image, so plain numpy is fast enough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _sum3(values: np.ndarray, edge: bool) -> np.ndarray:
    """Sum over each 3x3 neighbourhood, replicating the edge or treating it as zero."""
    rows, cols = values.shape
    if rows < 2 or cols < 2:
        padded = np.pad(values, 1, mode="edge" if edge else "constant")
        across = padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]
        return across[:-2] + across[1:-1] + across[2:]
    across = np.empty_like(values)
    np.add(values[:, :-2], values[:, 1:-1], out=across[:, 1:-1])
    across[:, 1:-1] += values[:, 2:]
    across[:, 0] = values[:, 0] + values[:, 1]
    across[:, -1] = values[:, -2] + values[:, -1]
    if edge:
        across[:, 0] += values[:, 0]
        across[:, -1] += values[:, -1]
    total = np.empty_like(values)
    np.add(across[:-2], across[1:-1], out=total[1:-1])
    total[1:-1] += across[2:]
    total[0] = across[0] + across[1]
    total[-1] = across[-2] + across[-1]
    if edge:
        total[0] += across[0]
        total[-1] += across[-1]
    return total


def box3(image: np.ndarray) -> np.ndarray:
    """3x3 mean with edge replication; returns a new float32 array."""
    total = _sum3(image.astype(np.float32, copy=False), edge=True)
    total *= np.float32(1.0 / 9.0)
    return total


def count3(mask: np.ndarray) -> np.ndarray:
    """Number of set pixels in each 3x3 neighbourhood (zero outside the image)."""
    return _sum3(mask.astype(np.int16), edge=False)


def clean(mask: np.ndarray) -> np.ndarray:
    """Drop isolated speckle and fill pinholes: a 3x3 majority-style filter."""
    count = count3(mask)
    return (mask & (count >= 4)) | (count >= 6)


def dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a mask by one pixel in every direction."""
    return count3(mask) > 0


def erode(mask: np.ndarray) -> np.ndarray:
    """Shrink a mask by one pixel in every direction."""
    return count3(mask) == 9


def gradient(image: np.ndarray) -> np.ndarray:
    """Absolute central-difference gradient, x plus y."""
    out = np.zeros(image.shape, dtype=np.float32)
    if image.shape[1] > 2:
        out[:, 1:-1] += np.abs(image[:, 2:] - image[:, :-2]) * 0.5
    if image.shape[0] > 2:
        out[1:-1, :] += np.abs(image[2:, :] - image[:-2, :]) * 0.5
    return out


@dataclass(frozen=True)
class Blob:
    """One 8-connected region of a mask, in working-image pixels."""

    label: int
    area: int
    top: int
    left: int
    bottom: int
    right: int
    cy: float
    cx: float

    @property
    def height(self) -> int:
        """Rows spanned, inclusive."""
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        """Columns spanned, inclusive."""
        return self.right - self.left + 1

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """(top, left, bottom, right), inclusive."""
        return (self.top, self.left, self.bottom, self.right)


def label(mask: np.ndarray) -> Tuple[np.ndarray, List[Blob]]:
    """8-connected component labelling by run-length union-find.

    Returns:
        A label image (int32, -1 where the mask is off) and one :class:`Blob`
        per component, ordered by label.
    """
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    run_rows, run_starts = np.nonzero(step == 1)
    _, run_ends = np.nonzero(step == -1)  # exclusive end column
    runs = int(run_rows.size)
    if runs == 0:
        return labels, []

    parent = list(range(runs))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    first = np.searchsorted(run_rows, np.arange(height + 1)).tolist()
    starts = run_starts.tolist()
    ends = run_ends.tolist()
    for row in range(1, height):
        a, a_end = first[row - 1], first[row]
        b, b_end = first[row], first[row + 1]
        if a == a_end or b == b_end:
            continue
        i, j = a, b
        while i < a_end and j < b_end:
            if starts[i] <= ends[j] and starts[j] <= ends[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if ends[i] < ends[j]:
                i += 1
            else:
                j += 1

    roots = np.array([find(i) for i in range(runs)], dtype=np.int64)
    _, run_label = np.unique(roots, return_inverse=True)
    run_label = run_label.astype(np.int64).ravel()
    count = int(run_label.max()) + 1
    lengths = (run_ends - run_starts).astype(np.int64)

    area = np.bincount(run_label, weights=lengths, minlength=count)
    top = np.full(count, height, dtype=np.int64)
    bottom = np.full(count, -1, dtype=np.int64)
    left = np.full(count, width, dtype=np.int64)
    right = np.full(count, -1, dtype=np.int64)
    np.minimum.at(top, run_label, run_rows)
    np.maximum.at(bottom, run_label, run_rows)
    np.minimum.at(left, run_label, run_starts)
    np.maximum.at(right, run_label, run_ends - 1)
    sum_y = np.bincount(run_label, weights=run_rows * lengths, minlength=count)
    sum_x = np.bincount(
        run_label, weights=(run_starts + run_ends - 1) * 0.5 * lengths, minlength=count
    )

    total = int(lengths.sum())
    flat_start = run_rows.astype(np.int64) * width + run_starts
    offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    flat = np.repeat(flat_start, lengths) + offsets
    labels.ravel()[flat] = np.repeat(run_label, lengths).astype(np.int32)

    blobs = [
        Blob(
            label=k,
            area=int(area[k]),
            top=int(top[k]),
            left=int(left[k]),
            bottom=int(bottom[k]),
            right=int(right[k]),
            cy=float(sum_y[k] / area[k]),
            cx=float(sum_x[k] / area[k]),
        )
        for k in range(count)
    ]
    return labels, blobs


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection over union of two inclusive (top, left, bottom, right) boxes."""
    inter = box_intersection(a, b)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(area_a + area_b - inter)


def box_intersection(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    """Pixels shared by two inclusive (top, left, bottom, right) boxes."""
    rows = min(a[2], b[2]) - max(a[0], b[0]) + 1
    cols = min(a[3], b[3]) - max(a[1], b[1]) + 1
    if rows <= 0 or cols <= 0:
        return 0
    return rows * cols


_WINDOWS: Dict[Tuple[int, int], np.ndarray] = {}


def _window(shape: Tuple[int, int]) -> np.ndarray:
    window = _WINDOWS.get(shape)
    if window is None:
        window = np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)
        if len(_WINDOWS) > 16:
            _WINDOWS.clear()
        _WINDOWS[shape] = window
    return window


def phase_correlate(reference: np.ndarray, image: np.ndarray) -> Tuple[float, float, float]:
    """Global translation of ``image`` relative to ``reference``.

    Returns:
        ``(dy, dx, peak)``: ``image`` looks like ``reference`` moved down by
        ``dy`` and right by ``dx`` pixels (sub-pixel), and the height of the
        correlation peak (near 1 for a clean shift, near 0 for no match).
    """
    window = _window(reference.shape)
    a = (reference - float(reference.mean())) * window
    b = (image - float(image.mean())) * window
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fb * np.conj(fa)
    magnitude = np.abs(cross)
    scale = float(magnitude.max()) if magnitude.size else 0.0
    if scale <= 1e-9:
        return 0.0, 0.0, 0.0
    cross /= magnitude + scale * 1e-6
    surface = np.fft.irfft2(cross, s=reference.shape)
    flat = int(np.argmax(surface))
    rows, cols = surface.shape
    py, px = divmod(flat, cols)
    peak = float(surface[py, px])

    def refine(minus: float, centre: float, plus: float) -> float:
        denom = minus - 2.0 * centre + plus
        if denom >= -1e-12:
            return 0.0
        return max(-0.5, min(0.5, 0.5 * (minus - plus) / denom))

    dy = py + refine(surface[(py - 1) % rows, px], peak, surface[(py + 1) % rows, px])
    dx = px + refine(surface[py, (px - 1) % cols], peak, surface[py, (px + 1) % cols])
    if dy > rows / 2.0:
        dy -= rows
    if dx > cols / 2.0:
        dx -= cols
    return float(dy), float(dx), peak


def shift_image(image: np.ndarray, dy: float, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """Move ``image`` down by ``dy`` and right by ``dx`` pixels (bilinear).

    Returns:
        The moved image and a mask of the pixels that had real image content
        to move in (the uncovered border is False).
    """
    rows, cols = image.shape
    ys = np.arange(rows, dtype=np.float64) - dy
    xs = np.arange(cols, dtype=np.float64) - dx
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    y0c = np.clip(y0, 0, rows - 1)
    y1c = np.clip(y0 + 1, 0, rows - 1)
    x0c = np.clip(x0, 0, cols - 1)
    x1c = np.clip(x0 + 1, 0, cols - 1)
    top = image[np.ix_(y0c, x0c)] * (1.0 - wx) + image[np.ix_(y0c, x1c)] * wx
    low = image[np.ix_(y1c, x0c)] * (1.0 - wx) + image[np.ix_(y1c, x1c)] * wx
    out = (top * (1.0 - wy) + low * wy).astype(np.float32)
    margin_y = int(math.ceil(abs(dy))) + 1
    margin_x = int(math.ceil(abs(dx))) + 1
    valid = np.zeros((rows, cols), dtype=bool)
    if rows > 2 * margin_y and cols > 2 * margin_x:
        valid[margin_y: rows - margin_y, margin_x: cols - margin_x] = True
    return out, valid
