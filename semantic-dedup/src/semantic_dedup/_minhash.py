"""MinHash + LSH banding: cheap candidate pairs for large inputs.

Nothing here decides anything on its own. It proposes pairs that are *likely*
similar; the caller then scores every candidate with the exact cosine, so the
similarities reported to the user are never estimates.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ._vectors import SparseMatrix

__all__ = ["minhash_signatures", "candidate_pairs", "band_shape", "jaccard"]

_PRIME = 2147483647  # 2**31 - 1, Mersenne prime
_SENTINEL = np.uint64(2**64 - 1)
_CHUNK_CELLS = 4_000_000


def minhash_signatures(
    matrix: SparseMatrix, num_perm: int = 128, random_state: int = 0
) -> np.ndarray:
    """A ``(n_rows, num_perm)`` MinHash signature matrix over the feature sets.

    Rows with no features keep the sentinel value, so they never collide with a
    real document by accident.
    """
    rng = np.random.default_rng(random_state)
    # Multiply-shift hashing in wrapping uint64: no modulo in the inner loop.
    a = (rng.integers(0, 2**62, size=num_perm, dtype=np.int64).astype(np.uint64)
         * np.uint64(2)) + np.uint64(1)
    b = rng.integers(0, 2**62, size=num_perm, dtype=np.int64).astype(np.uint64)
    signatures = np.full((matrix.n_rows, num_perm), _SENTINEL, dtype=np.uint64)
    if matrix.n_rows == 0 or matrix.indices.size == 0:
        return signatures

    lengths = np.diff(matrix.indptr)
    rows_per_chunk = max(1, _CHUNK_CELLS // max(int(lengths.max()) * num_perm, 1))
    for first in range(0, matrix.n_rows, rows_per_chunk):
        last = min(matrix.n_rows, first + rows_per_chunk)
        start, stop = int(matrix.indptr[first]), int(matrix.indptr[last])
        if start == stop:
            continue
        features = matrix.indices[start:stop].astype(np.uint64)
        hashed = (features[:, None] * a[None, :] + b[None, :]) >> np.uint64(24)
        keep = lengths[first:last] > 0
        starts = (matrix.indptr[first:last] - start)[keep]
        block = np.minimum.reduceat(hashed, starts, axis=0)
        chunk = signatures[first:last]
        chunk[keep] = block
        signatures[first:last] = chunk
    return signatures


def band_shape(num_perm: int, target: float) -> Tuple[int, int]:
    """Pick ``(bands, rows_per_band)`` whose LSH curve turns on near ``target``."""
    best = (num_perm, 1)
    best_gap = float("inf")
    for rows in range(1, num_perm + 1):
        if num_perm % rows:
            continue
        bands = num_perm // rows
        turn_on = (1.0 / bands) ** (1.0 / rows)
        gap = abs(turn_on - target)
        if gap < best_gap:
            best_gap = gap
            best = (bands, rows)
    return best


def candidate_pairs(
    signatures: np.ndarray,
    threshold: float,
    max_pairs: int = 2_000_000,
    max_bucket: int = 256,
) -> Tuple[List[Tuple[int, int]], bool]:
    """Candidate ``(i, j)`` pairs from LSH banding, and whether a cap was hit."""
    n, num_perm = signatures.shape
    if n < 2:
        return [], False
    # Jaccard is a harsher scale than cosine, so aim low and keep recall high.
    target = max(0.2, min(0.95, threshold / (2.0 - threshold) - 0.1))
    bands, rows = band_shape(num_perm, target)

    encoded: List[np.ndarray] = []
    found = 0
    truncated = False
    for band in range(bands):
        block = signatures[:, band * rows : (band + 1) * rows]
        # Fold the band into one key, staying well inside int64 at every step.
        keys = np.zeros(n, dtype=np.int64)
        for column in range(rows):
            cell = (block[:, column] % np.uint64(_PRIME)).astype(np.int64)
            keys = (keys * 31 + cell) % _PRIME
        order = np.argsort(keys, kind="stable")
        boundaries = np.flatnonzero(np.diff(keys[order])) + 1
        for group in np.split(order, boundaries):
            size = group.size
            if size < 2:
                continue
            members = np.sort(group)
            if size <= max_bucket:
                left, right = np.triu_indices(size, k=1)
                encoded.append(members[left].astype(np.int64) * n + members[right])
                found += left.size
            else:
                # A bucket of thousands of look-alikes would cost m*(m-1)/2
                # candidates to say one thing. Score every pair inside a bounded
                # head, then link the tail to the smallest member so the
                # connected component still forms at linear cost.
                head = members[:max_bucket]
                left, right = np.triu_indices(max_bucket, k=1)
                encoded.append(head[left].astype(np.int64) * n + head[right])
                tail = members[max_bucket:].astype(np.int64)
                encoded.append(int(members[0]) * n + tail)
                found += int(left.size) + int(tail.size)
            if found >= max_pairs:
                truncated = True
                break
        if found >= max_pairs:
            break

    if not encoded:
        return [], truncated
    flat = np.unique(np.concatenate(encoded))
    if flat.size > max_pairs:
        flat = flat[:max_pairs]
        truncated = True
    return [(int(value // n), int(value % n)) for value in flat], truncated


def jaccard(matrix: SparseMatrix, i: int, j: int) -> float:
    """Exact Jaccard similarity of two rows' feature sets.

    Two rows that have no features at all (punctuation- or emoji-only text)
    share no evidence of anything, so they score 0.0 rather than the vacuous
    1.0 of "two empty sets are equal". Callers that want identical text to
    score 1.0 compare the normalized strings before getting here.
    """
    left, _ = matrix.row(i)
    right, _ = matrix.row(j)
    union = np.union1d(left, right).size
    if union == 0:
        return 0.0
    return float(np.intersect1d(left, right, assume_unique=True).size / union)
