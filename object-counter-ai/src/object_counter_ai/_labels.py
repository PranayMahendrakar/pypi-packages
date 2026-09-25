"""Connected components in plain numpy, no scipy.

The mask is cut into horizontal runs of foreground pixels, runs in neighbouring
rows that touch are joined, and the joins are resolved with vectorised
min-label propagation. Work is proportional to the number of runs rather than
the number of pixels, so a 2000 x 2000 frame with a few hundred parts labels in
a few tens of milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Components:
    """Every connected component of a boolean mask.

    Arrays are indexed by component number minus one; components are numbered
    in reading order of their first pixel (top row first, then left to right),
    so the numbering is deterministic.
    """

    shape: Tuple[int, int]
    count: int
    run_rows: np.ndarray      # row of each run
    run_starts: np.ndarray    # first column of each run
    run_ends: np.ndarray      # one past the last column of each run
    run_labels: np.ndarray    # component number (1-based) of each run
    areas: np.ndarray         # pixels per component
    tops: np.ndarray
    lefts: np.ndarray
    bottoms: np.ndarray       # exclusive
    rights: np.ndarray        # exclusive
    row_centres: np.ndarray   # centroid row (pixel centres at +0.5)
    col_centres: np.ndarray   # centroid column (pixel centres at +0.5)
    first_rows: np.ndarray    # a pixel that belongs to each component
    first_cols: np.ndarray

    def label_image(self) -> np.ndarray:
        """An int32 image, 0 for background and 1..count for the components."""
        height, width = self.shape
        out = np.zeros(height * width, dtype=np.int32)
        if self.run_rows.size:
            lengths = self.run_ends - self.run_starts
            base = np.repeat(self.run_rows * width + self.run_starts, lengths)
            offsets = np.arange(base.size) - np.repeat(np.cumsum(lengths) - lengths, lengths)
            out[base + offsets] = np.repeat(self.run_labels, lengths)
        return out.reshape(height, width)


def _find_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    rows, starts = np.nonzero(step == 1)
    _, ends = np.nonzero(step == -1)
    return rows.astype(np.int64), starts.astype(np.int64), ends.astype(np.int64)


def _resolve(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Union the pairs (a[i], b[i]) over nodes 0..n-1 and return a root per node."""
    parent = np.arange(n, dtype=np.int64)
    if a.size == 0:
        return parent
    while True:
        pa = parent[a]
        pb = parent[b]
        low = np.minimum(pa, pb)
        new = parent.copy()
        np.minimum.at(new, pa, low)
        np.minimum.at(new, pb, low)
        while True:
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, parent):
            return parent
        parent = new


def label(mask: np.ndarray, connectivity: int = 8) -> Components:
    """Label the connected components of a 2-D boolean ``mask``.

    ``connectivity`` is 8 (diagonal neighbours touch) or 4.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = mask.shape
    rows, starts, ends = _find_runs(np.asarray(mask, dtype=bool))
    n = rows.size
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    if n == 0:
        return Components((height, width), 0, empty_i, empty_i, empty_i, empty_i,
                          empty_i, empty_i, empty_i, empty_i, empty_i, empty_f, empty_f,
                          empty_i, empty_i)

    stride = width + 2
    key_starts = rows * stride + starts
    key_ends = rows * stride + ends
    previous = (rows - 1) * stride
    if connectivity == 8:
        lo = np.searchsorted(key_ends, previous + starts, side="left")
        hi = np.searchsorted(key_starts, previous + ends, side="right")
    else:
        lo = np.searchsorted(key_ends, previous + starts, side="right")
        hi = np.searchsorted(key_starts, previous + ends, side="left")
    counts = np.where(rows > 0, np.maximum(hi - lo, 0), 0)
    total = int(counts.sum())
    if total:
        b_idx = np.repeat(np.arange(n, dtype=np.int64), counts)
        a_idx = np.repeat(lo, counts) + (
            np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        )
    else:
        a_idx = b_idx = empty_i
    roots = _resolve(n, a_idx, b_idx)

    # Number components 1..k in reading order of their first run.
    unique_roots, first_index, inverse = np.unique(roots, return_index=True, return_inverse=True)
    order = np.argsort(first_index, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    run_labels = rank[inverse.reshape(-1)] + 1
    k = int(unique_roots.size)

    lengths = (ends - starts).astype(np.float64)
    idx = run_labels - 1
    areas = np.bincount(idx, weights=lengths, minlength=k)
    tops = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    bottoms = np.zeros(k, dtype=np.int64)
    lefts = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    rights = np.zeros(k, dtype=np.int64)
    np.minimum.at(tops, idx, rows)
    np.maximum.at(bottoms, idx, rows + 1)
    np.minimum.at(lefts, idx, starts)
    np.maximum.at(rights, idx, ends)
    row_sum = np.bincount(idx, weights=(rows + 0.5) * lengths, minlength=k)
    col_sum = np.bincount(idx, weights=(starts + ends) / 2.0 * lengths, minlength=k)
    first_runs = first_index[order]
    return Components(
        shape=(height, width),
        count=k,
        run_rows=rows,
        run_starts=starts,
        run_ends=ends,
        run_labels=run_labels.astype(np.int64),
        areas=areas.astype(np.int64),
        tops=tops,
        lefts=lefts,
        bottoms=bottoms,
        rights=rights,
        row_centres=row_sum / areas,
        col_centres=col_sum / areas,
        first_rows=rows[first_runs],
        first_cols=starts[first_runs],
    )
