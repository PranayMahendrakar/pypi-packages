"""Clustering in numpy: projection, k-means, agglomerative linkage, and a speaker count.

How many speakers is decided from the clustering itself. Windows are grouped
bottom-up (average linkage). For every cut of the tree into ``k`` groups, the
groups big enough to count as a voice are compared pairwise: project both onto
the line between their centres and measure how far apart the centres are in
units of their pooled spread along that line. A recording has as many
speakers as the largest cut whose voices are all at least ``threshold`` apart.

Splitting one voice in two always manufactures some separation - a median
split of a single bell curve already lands about 2.7 spreads apart, and on
synthetic single-voice recordings the best split lands between 2.5 and 5 - so
the default threshold of 6 sits above that. Clearly different voices land well
beyond it; similar voices land below it and are reported as one, which is the
honest failure mode. Three guards keep the count from inventing voices:

* a voice heard for less than ``RELIABLE_S`` seconds has its separation
  discounted, because a second or two of overlapping windows sits tight;
* a group that only ever appears as a short bridge between two other voices
  is the windows straddling a change of speaker, not a third voice;
* the voices must account for ``MIN_COVERAGE`` of all windows, so a deep cut
  that isolates a few tight cores of one voice is not taken for several.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

SEPARATION_THRESHOLD = 6.0
MAX_DIRECT = 400
"""More windows than this are first summarised by k-means micro-clusters."""
RELIABLE_S = 3.0
"""Seconds of a voice after which its spread is trusted without a discount."""
MIN_COVERAGE = 0.8
"""Share of windows the accepted voices must account for between them."""
MAX_COMPONENTS = 12
VARIANCE_KEPT = 0.9


def standardise(x: np.ndarray) -> np.ndarray:
    """Z-score each column; constant columns become zero."""
    if x.shape[0] == 0:
        return x.copy()
    centred = x - x.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    out = centred / scale
    out[:, (x.std(axis=0) < 1e-9)] = 0.0
    return out


def unit_rows(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length (zero rows stay zero), for cosine geometry."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms


def project(x: np.ndarray, max_components: int = MAX_COMPONENTS,
            variance_kept: float = VARIANCE_KEPT) -> np.ndarray:
    """Principal-component scores keeping ``variance_kept`` of the variance (not whitened)."""
    n = x.shape[0]
    if n < 2:
        return np.zeros((n, 1))
    centred = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    energy = s * s
    total = float(energy.sum())
    if total <= 1e-18:
        return np.zeros((n, 1))
    cumulative = np.cumsum(energy) / total
    keep = int(np.searchsorted(cumulative, variance_kept) + 1)
    keep = max(1, min(keep, max_components, n - 1, s.size))
    return u[:, :keep] * s[:keep]


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def kmeans(x: np.ndarray, k: int, *, init: Optional[np.ndarray] = None, random_state: int = 0,
           n_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ seeding (or the given centres). Returns (labels, centres)."""
    n = x.shape[0]
    k = max(1, min(int(k), n))
    if init is not None:
        centres = np.array(init, dtype=np.float64, copy=True)
    else:
        rng = np.random.default_rng(random_state)
        centres = np.empty((k, x.shape[1]))
        centres[0] = x[int(rng.integers(n))]
        closest = _sq_dists(x, centres[:1])[:, 0]
        for c in range(1, k):
            total = float(closest.sum())
            if total <= 0.0:
                centres[c] = x[int(rng.integers(n))]
            else:
                centres[c] = x[int(rng.choice(n, p=closest / total))]
            closest = np.minimum(closest, _sq_dists(x, centres[c:c + 1])[:, 0])
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(n_iter):
        dist = _sq_dists(x, centres)
        new = np.argmin(dist, axis=1)
        for c in range(centres.shape[0]):
            members = new == c
            if members.any():
                centres[c] = x[members].mean(axis=0)
            else:
                far = int(np.argmax(dist[np.arange(n), new]))
                centres[c] = x[far]
                new[far] = c
        if np.array_equal(new, labels):
            break
        labels = new
    return labels, centres


def average_linkage(x: np.ndarray, weights: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Bottom-up merges by average (UPGMA) linkage on Euclidean distance.

    Returns the ``n - 1`` merges as ``(kept, absorbed)`` index pairs, in order.
    Ties are broken by lowest index, so the result is deterministic.
    """
    n = x.shape[0]
    if n < 2:
        return []
    size = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).copy()
    dist = np.sqrt(_sq_dists(x, x))
    np.fill_diagonal(dist, np.inf)
    merges: List[Tuple[int, int]] = []
    alive = np.ones(n, dtype=bool)
    for _ in range(n - 1):
        flat = int(np.argmin(dist))
        i, j = divmod(flat, n)
        if i > j:
            i, j = j, i
        merges.append((i, j))
        merged = (size[i] * dist[i] + size[j] * dist[j]) / (size[i] + size[j])
        size[i] += size[j]
        alive[j] = False
        merged[~alive] = np.inf
        merged[i] = np.inf
        dist[i, :] = merged
        dist[:, i] = merged
        dist[j, :] = np.inf
        dist[:, j] = np.inf
    return merges


def cut(merges: Sequence[Tuple[int, int]], n: int, k: int) -> np.ndarray:
    """Labels ``0 .. k-1`` for ``n`` items after replaying the first ``n - k`` merges."""
    parent = np.arange(n)

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for i, j in list(merges)[: max(0, n - k)]:
        parent[root(j)] = root(i)
    roots = np.array([root(a) for a in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart two groups are along the line between their centres, in pooled spreads."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    direction = b.mean(axis=0) - a.mean(axis=0)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return 0.0
    direction /= length
    pa = a @ direction
    pb = b @ direction
    # The textbook pooled spread (each group weighted by its degrees of
    # freedom): a group of two or three windows can sit unusually tight by
    # chance and must not shrink the yardstick.
    dof = pa.size + pb.size - 2
    if dof < 1:
        return 0.0
    pooled = np.sqrt((pa.size * float(pa.var()) + pb.size * float(pb.var())) / float(dof))
    gap = abs(float(pb.mean() - pa.mean()))
    if pooled <= 1e-12 * max(gap, 1.0):
        return float("inf")
    return gap / pooled


@dataclass
class Clustering:
    """The outcome of clustering windows.

    Attributes:
        labels: speaker index per window.
        k: number of speakers.
        evidence: how many speakers the clustering itself supports.
        separation: smallest pairwise separation among the speakers when
            ``k >= 2``; for one speaker, the best separation any split achieved.
        points: the projected window coordinates the decision was made in.
    """

    labels: np.ndarray
    k: int
    evidence: int
    separation: float
    points: np.ndarray


def _tree(points: np.ndarray, random_state: int) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    """Linkage over the windows, or over k-means micro-clusters of them when there are many.

    Returns (item index of each window, merges, number of items).
    """
    n = points.shape[0]
    if n <= MAX_DIRECT:
        return np.arange(n), average_linkage(points), n
    micro, centres = kmeans(points, MAX_DIRECT, random_state=random_state, n_iter=30)
    used, item = np.unique(micro, return_inverse=True)
    centres = centres[used]
    sizes = np.bincount(item, minlength=used.size).astype(np.float64)
    return item, average_linkage(centres, sizes), used.size


def is_blend(labels: np.ndarray, region_ids: Optional[np.ndarray], group: int, max_run: int) -> bool:
    """True when every run of ``group`` is a short bridge between two other groups.

    Windows that straddle a change of speaker with no pause mix both voices,
    so they land between the two in feature space and can look like a third
    voice. Such a group only ever appears as a short run of windows squeezed
    between one speaker and another inside the same stretch of speech; a real
    voice has at least one run that is longer, starts or ends a stretch of
    speech, or returns to the same neighbour.
    """
    if region_ids is None:
        return False
    n = labels.size
    index = 0
    found = False
    while index < n:
        if labels[index] != group:
            index += 1
            continue
        end = index
        while end + 1 < n and labels[end + 1] == group and region_ids[end + 1] == region_ids[index]:
            end += 1
        found = True
        left = labels[index - 1] if index > 0 and region_ids[index - 1] == region_ids[index] else None
        right = labels[end + 1] if end + 1 < n and region_ids[end + 1] == region_ids[index] else None
        length = end - index + 1
        if left is None or right is None or left == right or length > max_run:
            return False
        index = end + 1
    return found


def heard_seconds(spans: np.ndarray, members: np.ndarray) -> float:
    """Seconds of audio covered by the windows in ``members`` (overlaps counted once)."""
    chosen = spans[members]
    if chosen.shape[0] == 0:
        return 0.0
    chosen = chosen[np.argsort(chosen[:, 0], kind="stable")]
    total = 0.0
    start, end = float(chosen[0, 0]), float(chosen[0, 1])
    for first, last in chosen[1:]:
        if first > end:
            total += end - start
            start, end = float(first), float(last)
        else:
            end = max(end, float(last))
    return total + (end - start)


def _pairwise_min(points: np.ndarray, labels: np.ndarray, groups: Sequence[int],
                  spans: Optional[np.ndarray] = None, reliable_s: float = RELIABLE_S) -> float:
    """Smallest separation between any two of ``groups``.

    With ``spans`` (each window's start and end in seconds), a pair where either
    voice is heard for less than ``reliable_s`` has its separation discounted by
    ``sqrt(heard / reliable_s)``: overlapping windows from a second or two of
    one voice are near copies of each other, sit unusually tight, and would
    otherwise make any split of them look convincing.
    """
    best = float("inf")
    heard = {}
    if spans is not None:
        heard = {g: heard_seconds(spans, labels == g) for g in groups}
    for index, a in enumerate(groups):
        for b in groups[index + 1:]:
            value = separation(points[labels == a], points[labels == b])
            if spans is not None and reliable_s > 0:
                shortest = min(heard[a], heard[b])
                if shortest < reliable_s:
                    value *= float(np.sqrt(max(shortest, 1e-9) / reliable_s))
            best = min(best, value)
    return best


def _refine(points: np.ndarray, labels: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    """Reassign every window to the nearest of the chosen groups' centres, then iterate."""
    centres = np.vstack([points[labels == g].mean(axis=0) for g in groups])
    refined, _ = kmeans(points, len(groups), init=centres, n_iter=50)
    counts = np.bincount(refined, minlength=len(groups))
    if (counts == 0).any():
        remap = {g: i for i, g in enumerate(groups)}
        nearest = np.argmin(_sq_dists(points, centres), axis=1)
        return np.array([remap.get(int(l), int(nearest[w])) for w, l in enumerate(labels)], dtype=np.int64)
    return refined


def cluster_windows(points: np.ndarray, *, num_speakers: Optional[int], max_speakers: int,
                    min_windows: int, threshold: float = SEPARATION_THRESHOLD,
                    random_state: int = 0, region_ids: Optional[np.ndarray] = None,
                    max_blend_run: int = 3, spans: Optional[np.ndarray] = None) -> Clustering:
    """Group windows into speakers, estimating how many when ``num_speakers`` is None.

    ``points`` must be in time order; ``region_ids`` gives the stretch of speech
    each window belongs to, which lets groups made only of windows straddling
    a change of speaker be recognised (see :func:`is_blend`). ``spans`` gives
    each window's start and end in seconds, so voices heard only briefly are
    held to a stricter standard (see :func:`_pairwise_min`).
    """
    n = points.shape[0]
    if n == 0:
        return Clustering(np.zeros(0, dtype=np.int64), 0, 0, 0.0, points)
    if n == 1:
        return Clustering(np.zeros(1, dtype=np.int64), 1, 1, 0.0, points)

    item, merges, n_items = _tree(points, random_state)
    evidence = 1
    evidence_labels = np.zeros(n, dtype=np.int64)
    evidence_groups: List[int] = [0]
    best_split = 0.0
    candidates: List[Tuple[np.ndarray, List[int]]] = []
    top = min(max(max_speakers, num_speakers or 1) + 4, n_items)
    for k in range(2, top + 1):
        labels = cut(merges, n_items, k)[item]
        counts = np.bincount(labels, minlength=k)
        blend = [g for g in range(k) if is_blend(labels, region_ids, g, max_blend_run)]
        big = [g for g in range(k) if counts[g] >= min_windows and g not in blend]
        if len(big) < 2:
            continue
        # A deep cut can leave a few tight cores of one voice standing apart while
        # most windows sit in crumbs; voices must account for nearly every window.
        if int(counts[big].sum()) < MIN_COVERAGE * n:
            continue
        score = _pairwise_min(points, labels, big, spans)
        if len(big) == 2:
            best_split = max(best_split, score)
        if score >= threshold and len(big) <= max_speakers:
            candidates.append((labels, big))
            if len(big) > evidence:
                evidence = len(big)
                evidence_labels = labels
                evidence_groups = big

    if num_speakers is None:
        # Refining reassigns every window (crumbs and blends too) to the chosen
        # voices, which can pull them together. Accept a count only if the voices
        # are still ``threshold`` apart afterwards; otherwise try fewer.
        candidates.sort(key=lambda c: -len(c[1]))
        for cand_labels, cand_groups in candidates:
            labels = _refine(points, cand_labels, cand_groups)
            k_found = int(np.unique(labels).size)
            if k_found < 2:
                continue
            score = _pairwise_min(points, labels, list(range(k_found)), spans)
            if score >= threshold:
                return Clustering(labels, k_found, k_found, score, points)
        return Clustering(np.zeros(n, dtype=np.int64), 1, 1, best_split, points)

    k = min(int(num_speakers), n)
    if k == 1:
        return Clustering(np.zeros(n, dtype=np.int64), 1, evidence, best_split, points)
    labels = cut(merges, n_items, min(k, n_items))[item]
    groups = sorted(range(int(labels.max()) + 1), key=lambda g: -int(np.count_nonzero(labels == g)))[:k]
    if len(groups) < k:
        # Fewer tree items than requested speakers: fall back to k-means on the windows.
        labels, _ = kmeans(points, k, random_state=random_state)
        groups = list(range(k))
    labels = _refine(points, labels, groups)
    k_found = int(np.unique(labels).size)
    return Clustering(labels, k_found, evidence, _pairwise_min(points, labels, list(range(k_found)), spans),
                      points)
