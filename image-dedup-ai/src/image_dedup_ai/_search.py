"""Finding every pair of hashes within a hamming radius, without comparing all pairs.

Small sets (and loose thresholds, where no shortcut is exact) are compared
exhaustively in memory-bounded blocks. Large sets use multi-index hashing: split
the bits into ``max_dist + 1`` disjoint chunks; by the pigeonhole principle two
hashes within ``max_dist`` bits of each other agree exactly on at least one
chunk, so only hashes that share a chunk value are ever compared. Every
candidate is then verified with its exact hamming distance, so the result is
identical to the exhaustive search, just faster.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")

#: Below this many distinct hashes the exhaustive search is already fast.
BRUTE_FORCE_MAX = 4096
#: Upper bound on elements in one temporary comparison block (memory ~ 8 bytes each).
BLOCK_ELEMENTS = 2_000_000

Pairs = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _empty() -> Pairs:
    e = np.zeros(0, dtype=np.int64)
    return e, e.copy(), e.copy()


def popcount(x: np.ndarray, *, native: Optional[bool] = None) -> np.ndarray:
    """Set bits in each element of a uint64 array (same shape, small unsigned ints)."""
    use_native = _HAS_BITWISE_COUNT if native is None else (native and _HAS_BITWISE_COUNT)
    if use_native:
        return np.bitwise_count(x)
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP8[x.view(np.uint8)].reshape(x.shape + (8,)).sum(axis=-1, dtype=np.uint16)


def to_words(packed: np.ndarray) -> np.ndarray:
    """``(n, nbytes)`` uint8 hash rows -> ``(n, words)`` uint64, zero padded."""
    packed = np.asarray(packed, dtype=np.uint8)
    n, nbytes = packed.shape
    words = max(1, -(-nbytes // 8))
    buf = np.zeros((n, words * 8), dtype=np.uint8)
    buf[:, :nbytes] = packed
    return buf.view("<u8").reshape(n, words)


def distances_to(words: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Hamming distance from one ``(words,)`` hash to every row of ``words``."""
    if words.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return popcount(words ^ query[None, :]).sum(axis=-1, dtype=np.int64)


def pairs_within(words: np.ndarray, idx: np.ndarray, max_dist: int) -> Pairs:
    """All pairs ``(i, j, d)`` among rows ``idx`` (ascending) with hamming ``d <= max_dist``, ``i < j``."""
    g = int(idx.size)
    if g < 2:
        return _empty()
    sub = words[idx]
    width = sub.shape[1]
    rows = max(1, BLOCK_ELEMENTS // max(1, g * width))
    out_i: List[np.ndarray] = []
    out_j: List[np.ndarray] = []
    out_d: List[np.ndarray] = []
    for start in range(0, g - 1, rows):
        stop = min(g - 1, start + rows)
        a = sub[start:stop]
        b = sub[start + 1:]
        dist = popcount(a[:, None, :] ^ b[None, :, :]).sum(axis=-1, dtype=np.int64)
        r, c = np.nonzero(dist <= max_dist)
        upper = c >= r  # column c is row start+1+c, which must come after row start+r
        r, c = r[upper], c[upper]
        if r.size:
            out_i.append(idx[start + r])
            out_j.append(idx[start + 1 + c])
            out_d.append(dist[r, c])
    if not out_i:
        return _empty()
    return (
        np.concatenate(out_i).astype(np.int64),
        np.concatenate(out_j).astype(np.int64),
        np.concatenate(out_d).astype(np.int64),
    )


def _mih_worthwhile(n: int, bits: int, max_dist: int) -> bool:
    chunks = max_dist + 1
    if chunks > bits:
        return False
    chunk_bits = bits // chunks
    # Expected speed-up over the exhaustive search on uniform hashes: 2**b / chunks.
    return n > BRUTE_FORCE_MAX and (2.0 ** min(chunk_bits, 60)) / chunks >= 4.0


def similar_pairs(packed: np.ndarray, bits: int, max_dist: int, *, strategy: str = "auto") -> Pairs:
    """Every pair of rows of ``packed`` (``(n, nbytes)`` uint8) within ``max_dist`` bits.

    ``strategy`` is ``"auto"``, ``"brute"`` (exhaustive) or ``"mih"`` (multi-index
    hashing); all three return exactly the same pairs, as ``(i, j, d)`` with ``i < j``.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    n = packed.shape[0]
    if n < 2 or max_dist < 0:
        return _empty()
    words = to_words(packed)
    if strategy not in ("auto", "brute", "mih"):
        raise ValueError(f"strategy must be 'auto', 'brute' or 'mih'; got {strategy!r}")
    use_mih = strategy == "mih" or (strategy == "auto" and _mih_worthwhile(n, bits, max_dist))
    chunks = max_dist + 1
    if not use_mih or chunks > bits:
        return pairs_within(words, np.arange(n, dtype=np.int64), max_dist)

    unpacked = np.unpackbits(packed, axis=1)[:, :bits]
    found_i: List[np.ndarray] = []
    found_j: List[np.ndarray] = []
    found_d: List[np.ndarray] = []
    for chunk in range(chunks):
        cols = np.arange(chunk, bits, chunks)  # interleaved bits: less correlated than runs
        key_bytes = np.ascontiguousarray(np.packbits(unpacked[:, cols], axis=1))
        keys = key_bytes.view(np.dtype((np.void, key_bytes.shape[1]))).ravel()
        _, inverse = np.unique(keys, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        order = np.argsort(inverse, kind="stable")
        sorted_keys = inverse[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts.tolist(), ends.tolist()):
            if e - s < 2:
                continue
            i, j, d = pairs_within(words, order[s:e], max_dist)  # stable sort keeps rows ascending
            if i.size:
                found_i.append(i)
                found_j.append(j)
                found_d.append(d)
    if not found_i:
        return _empty()
    i = np.concatenate(found_i)
    j = np.concatenate(found_j)
    d = np.concatenate(found_d)
    code = i * np.int64(n) + j
    _, first = np.unique(code, return_index=True)
    return i[first], j[first], d[first]


def cosine_pairs(vectors: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs of unit-normalised rows with cosine similarity ``>= threshold`` (exhaustive)."""
    n = vectors.shape[0]
    if n < 2:
        e = np.zeros(0, dtype=np.int64)
        return e, e.copy(), np.zeros(0, dtype=np.float64)
    rows = max(1, BLOCK_ELEMENTS // n)
    out_i: List[np.ndarray] = []
    out_j: List[np.ndarray] = []
    out_s: List[np.ndarray] = []
    for start in range(0, n - 1, rows):
        stop = min(n - 1, start + rows)
        sims = vectors[start:stop] @ vectors[start + 1:].T
        r, c = np.nonzero(sims >= threshold - 1e-6)
        upper = c >= r
        r, c = r[upper], c[upper]
        if r.size:
            out_i.append(start + r)
            out_j.append(start + 1 + c)
            out_s.append(sims[r, c].astype(np.float64))
    if not out_i:
        e = np.zeros(0, dtype=np.int64)
        return e, e.copy(), np.zeros(0, dtype=np.float64)
    return (
        np.concatenate(out_i).astype(np.int64),
        np.concatenate(out_j).astype(np.int64),
        np.clip(np.concatenate(out_s), -1.0, 1.0),
    )


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected-component label (the smallest member index) for each of ``n`` nodes."""
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(i.tolist(), j.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    return np.array([find(x) for x in range(n)], dtype=np.int64)
