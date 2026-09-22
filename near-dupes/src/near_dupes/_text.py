"""Character-shingle similarity for text: normalization, MinHash, LSH, exact Jaccard.

Everything here works on plain Python strings and numpy arrays; nothing is
random unless it is seeded through ``random_state``.
"""
from __future__ import annotations

import functools
import math
import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

import numpy as np

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]|_")

_U64 = np.uint64
_MERSENNE_PRIME = _U64((1 << 61) - 1)
_MAX_HASH = _U64((1 << 32) - 1)
_MIX_C1 = _U64(0xBF58476D1CE4E5B9)
_MIX_C2 = _U64(0x94D049BB133111EB)
_HASH_SEED = _U64(0x9E3779B97F4A7C15)
_SIG_ROW_CHUNK = 512  # rows of (shingle x permutation) work per numpy batch; small keeps the temporaries in cache
_EXACT_WORK_BUDGET = 150_000_000  # shingle lookups the exact all-pairs pass may spend

PairList = List[Tuple[int, int, float]]


# --------------------------------------------------------------------------- #
# Normalization and shingling
# --------------------------------------------------------------------------- #
def normalize_text(value: object, *, lowercase: bool = True, strip_punctuation: bool = False) -> str:
    """Return the canonical comparison form of ``value``.

    ``None``/NaN become ``""``; other non-strings go through ``str()``.
    Applies NFKC, case folding (``lowercase``), punctuation removal
    (``strip_punctuation``) and whitespace collapsing.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        if isinstance(value, float) and math.isnan(value):
            return ""
        value = str(value)
    value = unicodedata.normalize("NFKC", value)
    if lowercase:
        value = value.casefold()
    if strip_punctuation:
        value = _PUNCT_RE.sub(" ", value)
    return _WS_RE.sub(" ", value).strip()


def _mix64(z: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on a uint64 array (bijective, wraps mod 2**64)."""
    z = (z ^ (z >> _U64(30))) * _MIX_C1
    z = (z ^ (z >> _U64(27))) * _MIX_C2
    return z ^ (z >> _U64(31))


def _hash_columns(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Order-sensitive 64-bit hash of the rows formed by aligned uint64 columns."""
    h = np.full(columns[0].shape[0], _HASH_SEED, dtype=np.uint64)
    for col in columns:
        h = _mix64(h ^ col)
    return h


def shingle_hashes(text: str, n: int) -> np.ndarray:
    """Sorted, unique 64-bit ids of the character ``n``-gram shingles of ``text``.

    A text shorter than ``n`` characters yields a single shingle (the whole
    text); an empty text yields no shingles.
    """
    if not text:
        return np.empty(0, dtype=np.uint64)
    cps = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32).astype(np.uint64)
    width = min(n, cps.size)
    count = cps.size - width + 1
    return np.unique(_hash_columns([cps[k : k + count] for k in range(width)]))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard similarity of two sorted unique id arrays."""
    if a.size == 0 and b.size == 0:
        return 1.0
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size > b.size:
        a, b = b, a
    pos = np.searchsorted(b, a)
    pos[pos == b.size] = 0
    inter = int(np.count_nonzero(b[pos] == a))
    return inter / (a.size + b.size - inter)


def text_similarity(a: object, b: object, *, n_gram: int = 3, normalize: bool = True) -> float:
    """Jaccard similarity of the character n-gram shingles of two texts (0..1)."""
    if normalize:
        a, b = normalize_text(a), normalize_text(b)
    else:
        a, b = ("" if a is None else str(a)), ("" if b is None else str(b))
    return jaccard(shingle_hashes(a, n_gram), shingle_hashes(b, n_gram))


# --------------------------------------------------------------------------- #
# MinHash
# --------------------------------------------------------------------------- #
class MinHasher:
    """Fixed family of ``num_perm`` universal-hash permutations seeded by ``random_state``."""

    def __init__(self, num_perm: int = 128, random_state: int = 0) -> None:
        if num_perm < 2:
            raise ValueError("num_perm must be at least 2")
        self.num_perm = int(num_perm)
        self.random_state = int(random_state)
        rng = np.random.default_rng(self.random_state)
        # a*hv + b never overflows uint64: a < 2**31, hv < 2**32, b < 2**31
        self.a = rng.integers(1, 1 << 31, size=self.num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=self.num_perm, dtype=np.uint64)

    def _rows(self, hv: np.ndarray) -> np.ndarray:
        m = hv[:, None] * self.a[None, :]
        m += self.b[None, :]
        m %= _MERSENNE_PRIME
        m &= _MAX_HASH
        return m

    def signatures(self, sets: Sequence[np.ndarray]) -> np.ndarray:
        """MinHash signatures, shape ``(len(sets), num_perm)`` uint32.

        ``sets`` are arrays of non-negative integer ids below 2**32 (any
        integer dtype). Empty sets get an all-max signature.
        """
        n = len(sets)
        out = np.full((n, self.num_perm), (1 << 32) - 1, dtype=np.uint32)
        start = 0
        while start < n:
            end, total = start, 0
            while end < n and (total == 0 or total + sets[end].size <= _SIG_ROW_CHUNK):
                total += sets[end].size
                end += 1
            idx = [i for i in range(start, end) if sets[i].size]
            if idx:
                if len(idx) == 1 and sets[idx[0]].size > _SIG_ROW_CHUNK:
                    hv = np.asarray(sets[idx[0]], dtype=np.uint64)
                    sig = np.full(self.num_perm, (1 << 32) - 1, dtype=np.uint64)
                    for s in range(0, hv.size, _SIG_ROW_CHUNK):
                        np.minimum(sig, self._rows(hv[s : s + _SIG_ROW_CHUNK]).min(axis=0), out=sig)
                    out[idx[0]] = sig.astype(np.uint32)
                else:
                    sizes = np.array([sets[i].size for i in idx], dtype=np.intp)
                    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
                    hv = np.concatenate([np.asarray(sets[i], dtype=np.uint64) for i in idx])
                    reduced = np.minimum.reduceat(self._rows(hv), offsets, axis=0)
                    out[idx] = reduced.astype(np.uint32)
            start = end
        return out


def candidate_probability(similarity: float, bands: int, rows: int) -> float:
    """Probability that a pair with the given Jaccard similarity collides in some LSH band."""
    return 1.0 - (1.0 - similarity**rows) ** bands


@functools.lru_cache(maxsize=64)
def optimal_bands(num_perm: int, threshold: float, min_recall: float = 0.98) -> Tuple[int, int]:
    """Choose LSH ``(bands, rows)`` with ``bands * rows <= num_perm`` for ``threshold``.

    Every candidate pair is verified with exact Jaccard afterwards, so false
    positives only cost time while false negatives are lost for good. The
    choice therefore minimises the false-positive area under the S-curve
    ``1 - (1 - s**rows)**bands`` subject to a pair sitting exactly at the
    threshold being found with probability at least ``min_recall`` (pairs
    above the threshold are found with higher probability still).
    """
    xs = np.linspace(0.0, 1.0, 1001)
    dx = xs[1] - xs[0]
    below = xs < threshold
    best: Optional[Tuple[float, int, int]] = None
    fallback: Optional[Tuple[float, int, int]] = None
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            recall = candidate_probability(threshold, b, r)
            if fallback is None or recall > fallback[0]:
                fallback = (recall, b, r)
            if recall < min_recall:
                continue
            fp = float((1.0 - (1.0 - xs[below] ** r) ** b).sum() * dx)
            if best is None or fp < best[0]:
                best = (fp, b, r)
    chosen = best if best is not None else fallback
    assert chosen is not None
    return chosen[1], chosen[2]


# --------------------------------------------------------------------------- #
# Candidate pairs
# --------------------------------------------------------------------------- #
def pairs_sharing_key(keys: np.ndarray) -> np.ndarray:
    """All index pairs ``(i, j)``, ``i < j``, whose ``keys`` are equal. Shape ``(m, 2)``."""
    n = keys.size
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    bounds = np.flatnonzero(sk[1:] != sk[:-1]) + 1
    ends = np.concatenate((bounds, [n]))
    starts = np.concatenate(([0], bounds))
    run_end = np.repeat(ends, ends - starts)
    pos = np.arange(n)
    counts = run_end - pos - 1
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 2), dtype=np.int64)
    first = np.repeat(pos, counts)
    offsets = np.cumsum(counts) - counts
    second = first + 1 + (np.arange(total) - np.repeat(offsets, counts))
    i = order[first].astype(np.int64)
    j = order[second].astype(np.int64)
    return np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1)


def unique_pairs(chunks: Sequence[np.ndarray], n: int) -> np.ndarray:
    """Union of ``(m, 2)`` pair arrays, sorted lexicographically, duplicates removed."""
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    cat = np.concatenate(chunks)
    codes = np.unique(cat[:, 0] * np.int64(n) + cat[:, 1])
    return np.stack([codes // n, codes % n], axis=1)


def lsh_candidates(sigs: np.ndarray, bands: int, rows: int) -> np.ndarray:
    """Pairs that collide in at least one LSH band. Shape ``(m, 2)``, ``i < j``."""
    n = sigs.shape[0]
    found = []
    for b in range(bands):
        block = sigs[:, b * rows : (b + 1) * rows]
        keys = _hash_columns([block[:, c].astype(np.uint64) for c in range(block.shape[1])])
        found.append(pairs_sharing_key(keys))
    return unique_pairs(found, n)


def size_filter(cand: np.ndarray, lens: np.ndarray, threshold: float) -> np.ndarray:
    """Drop pairs whose set sizes alone rule out ``jaccard >= threshold``."""
    if cand.shape[0] == 0:
        return cand
    li, lj = lens[cand[:, 0]], lens[cand[:, 1]]
    ok = np.minimum(li, lj) >= threshold * np.maximum(li, lj) - 1e-9
    return cand[ok]


def all_pairs_candidates(sigs: np.ndarray, lens: np.ndarray, threshold: float, margin: float = 0.25) -> np.ndarray:
    """Every pair whose MinHash estimate is at least ``threshold - margin``.

    The screen for small inputs whose texts are too long for the plain exact
    pass: each pair is looked at, and the survivors are verified with exact
    Jaccard by the caller. With 128 permutations and the default margin a
    true near-duplicate at the threshold is dropped with probability far
    below 1e-8.
    """
    n, k = sigs.shape
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    min_eq = max(0, int(math.ceil((threshold - margin) * k - 1e-9)))
    rows_per_block = max(1, (8 << 20) // max(1, n * k))
    idx = np.arange(n)
    found = []
    for s in range(0, n, rows_per_block):
        blk = sigs[s : s + rows_per_block]
        b = blk.shape[0]
        eq = (blk[:, None, :] == sigs[None, :, :]).sum(axis=2)
        ok = eq >= min_eq
        ok &= idx[None, :] > np.arange(s, s + b)[:, None]
        li = lens[s : s + b][:, None]
        lj = lens[None, :]
        ok &= np.minimum(li, lj) >= threshold * np.maximum(li, lj) - 1e-9
        ii, jj = np.nonzero(ok)
        if ii.size:
            found.append(np.stack([ii.astype(np.int64) + s, jj.astype(np.int64)], axis=1))
    return unique_pairs(found, n)


def verify_pairs(sets: Sequence[np.ndarray], n_unique: int, cand: np.ndarray, lens: np.ndarray, threshold: float) -> PairList:
    """Exact Jaccard for candidate pairs; keeps those scoring at least ``threshold``.

    ``sets`` hold sorted unique ids in ``range(n_unique)``; ``cand`` is sorted by
    first index. Returns ``(i, j, score)`` tuples ordered by ``(i, j)``.
    """
    out: PairList = []
    if cand.shape[0] == 0:
        return out
    flag = np.zeros(n_unique, dtype=bool)
    first = cand[:, 0]
    bounds = np.flatnonzero(first[1:] != first[:-1]) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [cand.shape[0]]))
    for s, e in zip(starts, ends):
        i = int(first[s])
        js = cand[s:e, 1]
        parts = [sets[j] for j in js]
        sizes = np.array([p.size for p in parts], dtype=np.intp)
        offsets = np.cumsum(sizes) - sizes
        si = sets[i]
        flag[si] = True
        hits = flag[np.concatenate(parts)].astype(np.int32)
        flag[si] = False
        inter = np.add.reduceat(hits, offsets)
        union = int(lens[i]) + lens[js] - inter
        scores = inter / union
        for j, sc in zip(js.tolist(), scores.tolist()):
            if sc + 1e-9 >= threshold:
                out.append((i, j, float(sc)))
    return out


def all_pairs(lens: np.ndarray, threshold: float) -> np.ndarray:
    """Every pair ``(i, j)``, ``i < j``, that the size filter cannot rule out.

    Shape ``(m, 2)``, sorted lexicographically. Built in row blocks so the
    working memory stays small however many items there are.
    """
    n = lens.size
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    idx = np.arange(n)
    block = max(1, (4 << 20) // n)
    found = []
    for s in range(0, n, block):
        li = lens[s : s + block][:, None]
        ok = np.minimum(li, lens[None, :]) >= threshold * np.maximum(li, lens[None, :]) - 1e-9
        ok &= idx[None, :] > np.arange(s, s + li.shape[0])[:, None]
        ii, jj = np.nonzero(ok)  # row-major, so already ordered by (i, j)
        if ii.size:
            found.append(np.stack([ii.astype(np.int64) + s, jj.astype(np.int64)], axis=1))
    return np.concatenate(found) if found else np.empty((0, 2), dtype=np.int64)


def find_text_pairs(
    texts: Sequence[str],
    *,
    threshold: float,
    n_gram: int,
    num_perm: int = 128,
    random_state: int = 0,
    all_pairs_max: int = 2000,
    exact_work_budget: int = _EXACT_WORK_BUDGET,
) -> PairList:
    """Near-duplicate pairs among distinct, non-empty, already-normalized ``texts``.

    With ``len(texts) <= all_pairs_max`` every pair is scored with the exact
    Jaccard similarity of its character n-gram shingle sets (only a
    set-size bound prunes pairs, and that bound is exact too). Should the
    texts be so long that this exceeds ``exact_work_budget`` shingle lookups,
    every pair is first screened by its MinHash estimate with a wide safety
    margin and the survivors are scored exactly. Larger inputs go through
    MinHash + LSH banding to propose candidates, again scored exactly.

    Returns ``(i, j, score)`` with ``i < j`` referring to positions in ``texts``.
    """
    m = len(texts)
    if m < 2:
        return []
    raw = [shingle_hashes(t, n_gram) for t in texts]
    lens = np.array([r.size for r in raw], dtype=np.int64)
    uniq, inv = np.unique(np.concatenate(raw), return_inverse=True)
    inv = np.asarray(inv, dtype=np.int32).ravel()
    sets = np.split(inv, np.cumsum(lens)[:-1])
    if m <= all_pairs_max:
        cand = all_pairs(lens, threshold)
        work = int(lens[cand[:, 1]].sum()) if cand.shape[0] else 0
        if work > exact_work_budget:
            sigs = MinHasher(num_perm, random_state).signatures(sets)
            cand = all_pairs_candidates(sigs, lens, threshold)
    else:
        sigs = MinHasher(num_perm, random_state).signatures(sets)
        bands, rows = optimal_bands(num_perm, threshold)
        cand = size_filter(lsh_candidates(sigs, bands, rows), lens, threshold)
    return verify_pairs(sets, int(uniq.size), cand, lens, threshold)
