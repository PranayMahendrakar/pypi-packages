"""Runs of True in a boolean array: finding them, keeping long ones, closing small gaps."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return ``[(start, stop), ...]`` for every run of True, ``stop`` exclusive."""
    flags = np.asarray(mask, dtype=bool).ravel()
    if flags.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def mark_long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """A copy of ``mask`` that keeps only the runs of at least ``min_len`` Trues."""
    flags = np.asarray(mask, dtype=bool).ravel()
    if flags.size == 0:
        return flags.copy()
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    keep = (stops - starts) >= int(min_len)
    delta = np.zeros(flags.size + 1, dtype=np.int64)
    np.add.at(delta, starts[keep], 1)
    np.add.at(delta, stops[keep], -1)
    return np.cumsum(delta[:-1]) > 0


def merge_runs(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    """Join runs separated by at most ``max_gap`` elements."""
    merged: List[Tuple[int, int]] = []
    for start, stop in sorted(runs):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged
