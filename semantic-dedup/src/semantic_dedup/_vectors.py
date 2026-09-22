"""A minimal sparse TF-IDF matrix and the pair search that runs on top of it.

Only numpy is used, so this file carries its own tiny CSR/CSC representation
plus an inverted-index dot product. Rows are L2-normalized, which makes the
cosine similarity of two rows a plain dot product.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._text import feature_counts

__all__ = [
    "SparseMatrix",
    "build_tfidf",
    "pairs_above",
    "PREFIX_FILTER_MIN_ROWS",
    "candidate_similarities",
    "dense_pairs_above",
    "normalize_rows",
]

_EPS = 1e-9
#: A partial score below this counts as "shares nothing indexed".
_PARTIAL_EPS = 1e-12
#: Below this many rows the exhaustive scan is already cheap, so skip the prune.
PREFIX_FILTER_MIN_ROWS = 400
#: Give up on the prune (and scan exhaustively) past this many candidate pairs.
_MAX_CANDIDATES = 5_000_000


class SparseMatrix:
    """Compressed sparse rows: ``indptr``, ``indices`` and ``data``."""

    __slots__ = ("indptr", "indices", "data", "n_rows", "n_features", "_csc")

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        n_features: int,
    ) -> None:
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.n_rows = int(indptr.shape[0] - 1)
        self.n_features = int(n_features)
        self._csc: Tuple[np.ndarray, np.ndarray, np.ndarray] = ()  # type: ignore[assignment]

    def row(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(indices, data)`` of row ``i``."""
        start, stop = int(self.indptr[i]), int(self.indptr[i + 1])
        return self.indices[start:stop], self.data[start:stop]

    def csc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Column view: ``(col_indptr, row_indices, values)``, built once."""
        if isinstance(self._csc, tuple) and len(self._csc) == 3:
            return self._csc
        order = np.argsort(self.indices, kind="stable")
        col_rows = np.repeat(
            np.arange(self.n_rows, dtype=np.int64), np.diff(self.indptr)
        )[order]
        col_vals = self.data[order]
        counts = np.bincount(self.indices, minlength=self.n_features)
        col_indptr = np.zeros(self.n_features + 1, dtype=np.int64)
        np.cumsum(counts, out=col_indptr[1:])
        self._csc = (col_indptr, col_rows.astype(np.int64), col_vals)
        return self._csc


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize the rows of a dense array; all-zero rows stay all-zero."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"embed must return a 2-D array, got shape {arr.shape}")
    norms = np.sqrt((arr * arr).sum(axis=1))
    norms[norms < _EPS] = 1.0
    return arr / norms[:, None]


def build_tfidf(
    texts: Sequence[str],
    word_ngram: Tuple[int, int] = (1, 2),
    char_ngram: Tuple[int, int] = (3, 4),
) -> SparseMatrix:
    """Build the L2-normalized, sublinear TF-IDF matrix for ``texts``."""
    n = len(texts)
    vocabulary: Dict[str, int] = {}
    rows: List[Tuple[np.ndarray, np.ndarray]] = []
    for text in texts:
        counts = feature_counts(text, word_ngram=word_ngram, char_ngram=char_ngram)
        if not counts:
            rows.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
            continue
        idx = np.empty(len(counts), dtype=np.int64)
        val = np.empty(len(counts), dtype=np.float64)
        for position, (feature, count) in enumerate(counts.items()):
            fid = vocabulary.get(feature)
            if fid is None:
                fid = len(vocabulary)
                vocabulary[feature] = fid
            idx[position] = fid
            val[position] = 1.0 + np.log(count)
        rows.append((idx, val))

    n_features = max(len(vocabulary), 1)
    lengths = np.array([len(idx) for idx, _ in rows], dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    if rows:
        indices = np.concatenate([idx for idx, _ in rows]) if indptr[-1] else np.empty(0, dtype=np.int64)
        data = np.concatenate([val for _, val in rows]) if indptr[-1] else np.empty(0, dtype=np.float64)
    else:
        indices = np.empty(0, dtype=np.int64)
        data = np.empty(0, dtype=np.float64)

    document_frequency = np.bincount(indices, minlength=n_features).astype(np.float64)
    idf = np.log((1.0 + n) / (1.0 + document_frequency)) + 1.0
    data = data * idf[indices]

    # L2-normalize every row; rows with no features keep a zero vector.
    squared = np.zeros(n, dtype=np.float64)
    if data.size:
        row_of = np.repeat(np.arange(n, dtype=np.int64), lengths)
        np.add.at(squared, row_of, data * data)
        norms = np.sqrt(squared)
        norms[norms < _EPS] = 1.0
        data = data / norms[row_of]
    return SparseMatrix(indptr, indices.astype(np.int64), data, n_features)


def _ragged_offsets(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Flat positions of ``lengths[k]`` consecutive items beginning at ``starts[k]``."""
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(lengths)
    base = np.repeat(starts - (ends - lengths), lengths)
    return base + np.arange(total, dtype=np.int64)


def _score_block(
    matrix: SparseMatrix,
    first: int,
    last: int,
    index: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Dense ``(last - first, n_rows)`` similarity block for rows ``[first, last)``.

    ``index`` defaults to the matrix's own column view; pass a pruned one to get
    partial scores over just the entries that view still holds.
    """
    col_indptr, col_rows, col_vals = matrix.csc() if index is None else index
    n = matrix.n_rows
    height = last - first
    start, stop = int(matrix.indptr[first]), int(matrix.indptr[last])
    if start == stop:
        return np.zeros((height, n), dtype=np.float64)
    terms = matrix.indices[start:stop]
    weights = matrix.data[start:stop]
    local_row = np.repeat(
        np.arange(height, dtype=np.int64), np.diff(matrix.indptr[first : last + 1])
    )
    posting_len = col_indptr[terms + 1] - col_indptr[terms]
    take = _ragged_offsets(col_indptr[terms], posting_len)
    if take.size == 0:
        return np.zeros((height, n), dtype=np.float64)
    docs = col_rows[take]
    values = col_vals[take] * np.repeat(weights, posting_len)
    flat = np.repeat(local_row, posting_len) * n + docs
    scores = np.bincount(flat, weights=values, minlength=height * n)
    return scores.reshape(height, n)


def _block_bounds(
    matrix: SparseMatrix,
    budget: int = 4_000_000,
    index: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> List[Tuple[int, int]]:
    """Split rows into blocks whose inverted-index work stays under ``budget``."""
    col_indptr, _, _ = matrix.csc() if index is None else index
    posting_len = (col_indptr[matrix.indices + 1] - col_indptr[matrix.indices]).astype(np.int64)
    per_row_cost = np.zeros(matrix.n_rows, dtype=np.int64)
    if posting_len.size:
        row_of = np.repeat(np.arange(matrix.n_rows, dtype=np.int64), np.diff(matrix.indptr))
        np.add.at(per_row_cost, row_of, posting_len)
    bounds: List[Tuple[int, int]] = []
    start = 0
    cost = 0
    row_budget = max(1, budget // max(matrix.n_rows, 1))
    for i in range(matrix.n_rows):
        cost += int(per_row_cost[i])
        if (i + 1 - start) >= row_budget or cost >= budget:
            bounds.append((start, i + 1))
            start = i + 1
            cost = 0
    if start < matrix.n_rows:
        bounds.append((start, matrix.n_rows))
    return bounds


def _prefix_index(
    matrix: SparseMatrix, threshold: float
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """An inverted index over only the entries a match could actually need.

    Every row's features are visited most-common-first, accumulating
    ``value * max_value_of_that_feature`` - an upper bound on what that stretch
    of the row can contribute to any cosine. While the bound stays under
    ``threshold`` the entries are left out of the index: two rows whose only
    shared features sit inside both of their prefixes cannot reach the
    threshold, so the pair can be skipped without changing any answer. The
    features left out are the very common ones with the longest posting lists,
    which is where an exhaustive scan spends nearly all of its time.

    Returns ``None`` when nothing could be pruned, meaning the caller should
    just scan exhaustively.
    """
    n_features = matrix.n_features
    col_indptr, _, col_vals = matrix.csc()
    lengths = np.diff(col_indptr)
    maxweight = np.zeros(n_features, dtype=np.float64)
    nonempty = np.flatnonzero(lengths > 0)
    if nonempty.size == 0:
        return None
    maxweight[nonempty] = np.maximum.reduceat(col_vals, col_indptr[nonempty])

    # Most common feature first; feature id breaks ties so the order is fixed.
    order = np.lexsort((np.arange(n_features, dtype=np.int64), -lengths))
    rank = np.empty(n_features, dtype=np.int64)
    rank[order] = np.arange(n_features, dtype=np.int64)

    row_of = np.repeat(np.arange(matrix.n_rows, dtype=np.int64), np.diff(matrix.indptr))
    perm = np.lexsort((rank[matrix.indices], row_of))
    features = matrix.indices[perm]
    values = matrix.data[perm]
    running = np.empty(matrix.indices.size + 1, dtype=np.float64)
    running[0] = 0.0
    np.cumsum(values * maxweight[features], out=running[1:])
    bound = running[1:] - running[matrix.indptr[row_of]]

    keep = bound >= (threshold - _EPS)
    if keep.all():
        return None
    kept = np.flatnonzero(keep)
    kept_features = features[kept]
    kept_rows = row_of[kept]
    kept_values = values[kept]
    by_feature = np.lexsort((kept_rows, kept_features))
    index_ptr = np.zeros(n_features + 1, dtype=np.int64)
    np.cumsum(np.bincount(kept_features, minlength=n_features), out=index_ptr[1:])
    return index_ptr, kept_rows[by_feature], kept_values[by_feature]


def _prefix_candidates(
    matrix: SparseMatrix,
    index: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Optional[np.ndarray]:
    """``(k, 2)`` array of ``i < j`` pairs that share at least one indexed entry.

    ``None`` means there were too many to be worth it; scan exhaustively instead.
    """
    n = matrix.n_rows
    found: List[np.ndarray] = []
    total = 0
    for first, last in _block_bounds(matrix, index=index):
        block = _score_block(matrix, first, last, index)
        for local in range(last - first):
            i = first + local
            hits = np.flatnonzero(block[local] > _PARTIAL_EPS)
            if hits.size:
                hits = hits[hits != i]
            if hits.size == 0:
                continue
            low = np.minimum(hits, i).astype(np.int64)
            high = np.maximum(hits, i).astype(np.int64)
            found.append(low * n + high)
            total += int(hits.size)
        if total > _MAX_CANDIDATES:
            return None
    if not found:
        return np.empty((0, 2), dtype=np.int64)
    flat = np.unique(np.concatenate(found))
    return np.stack((flat // n, flat % n), axis=1)


def pairs_above(matrix: SparseMatrix, threshold: float) -> List[Tuple[int, int, float]]:
    """Every ``(i, j, cosine)`` with ``i < j`` and cosine at or above ``threshold``.

    Exhaustive either way: on a large matrix the pairs that provably cannot
    reach ``threshold`` are skipped first (see :func:`_prefix_index`), and every
    pair that survives is then scored with the same exact cosine.
    """
    if matrix.n_rows < 2:
        return []
    if matrix.n_rows >= PREFIX_FILTER_MIN_ROWS and threshold > 0.0:
        index = _prefix_index(matrix, threshold)
        if index is not None:
            candidates = _prefix_candidates(matrix, index)
            if candidates is not None:
                return candidate_similarities(matrix, candidates, threshold)
    return _pairs_above_exhaustive(matrix, threshold)


def _pairs_above_exhaustive(
    matrix: SparseMatrix, threshold: float
) -> List[Tuple[int, int, float]]:
    """Score every pair, with no pruning. The reference the prune must match."""
    out: List[Tuple[int, int, float]] = []
    if matrix.n_rows < 2:
        return out
    cutoff = threshold - _EPS
    for first, last in _block_bounds(matrix):
        block = _score_block(matrix, first, last)
        for local in range(last - first):
            i = first + local
            row = block[local]
            hits = np.nonzero(row[i + 1 :] >= cutoff)[0]
            for offset in hits:
                j = i + 1 + int(offset)
                out.append((i, j, float(min(1.0, row[j]))))
    return out


def candidate_similarities(
    matrix: SparseMatrix,
    candidates: Any,
    threshold: float,
    budget: int = 4_000_000,
) -> List[Tuple[int, int, float]]:
    """Exact cosine for the given candidate pairs, keeping those above ``threshold``.

    One row at a time is scattered into a dense buffer and every partner of that
    row is scored in a single vectorized pass, so the cost is proportional to the
    stored values touched rather than to the number of Python-level pairs.
    """
    out: List[Tuple[int, int, float]] = []
    if candidates is None or len(candidates) == 0:
        return out
    cutoff = threshold - _EPS
    pairs = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return out
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    lefts = pairs[order, 0]
    rights = pairs[order, 1]
    run_starts = np.flatnonzero(np.concatenate(([True], lefts[1:] != lefts[:-1])))
    run_ends = np.concatenate((run_starts[1:], [lefts.size]))

    indptr = matrix.indptr
    buffer = np.zeros(matrix.n_features, dtype=np.float64)
    for start, stop in zip(run_starts, run_ends):
        i = int(lefts[start])
        idx_i, val_i = matrix.row(i)
        if idx_i.size == 0:
            continue
        partners = rights[start:stop]
        lengths_all = (indptr[partners + 1] - indptr[partners]).astype(np.int64)
        buffer[idx_i] = val_i
        # Walk the partners in chunks so the gathered work stays bounded.
        cursor = 0
        total = partners.size
        while cursor < total:
            end = cursor
            taken = 0
            while end < total and (taken == 0 or taken + int(lengths_all[end]) <= budget):
                taken += int(lengths_all[end])
                end += 1
            js = partners[cursor:end]
            lengths = lengths_all[cursor:end]
            cursor = end
            if taken == 0:
                continue
            take = _ragged_offsets(indptr[js], lengths)
            contrib = buffer[matrix.indices[take]] * matrix.data[take]
            segment = np.repeat(np.arange(js.size, dtype=np.int64), lengths)
            scores = np.bincount(segment, weights=contrib, minlength=js.size)
            for hit in np.flatnonzero(scores >= cutoff):
                out.append((i, int(js[hit]), float(min(1.0, scores[hit]))))
        buffer[idx_i] = 0.0
    out.sort()
    return out


def dense_pairs_above(vectors: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    """Every ``(i, j, cosine)`` above ``threshold`` for L2-normalized dense rows."""
    out: List[Tuple[int, int, float]] = []
    n = vectors.shape[0]
    if n < 2:
        return out
    cutoff = threshold - _EPS
    step = max(1, int(2_000_000 // max(n, 1)))
    for first in range(0, n, step):
        last = min(n, first + step)
        block = vectors[first:last] @ vectors.T
        for local in range(last - first):
            i = first + local
            row = block[local]
            hits = np.nonzero(row[i + 1 :] >= cutoff)[0]
            for offset in hits:
                j = i + 1 + int(offset)
                out.append((i, j, float(min(1.0, row[j]))))
    return out
