"""Find topic shifts from sentence-similarity valleys.

No model, no download: each sentence becomes a hashed bag-of-words vector, a
similarity is taken across every sentence boundary using a small window on each
side, and the boundaries that sit in a *valley* (a local minimum below a low
percentile of the whole document) are proposed as topic shifts.

The word hash is ``zlib.crc32``, not the built-in ``hash``, so results are
identical across processes and runs.
"""
from __future__ import annotations

import string
import zlib
from typing import Any, Callable, Optional, List, Optional, Sequence, Set, Tuple

import numpy as np

from ._segment import WORD_RE

DIMENSIONS = 256
_STRIP = string.punctuation + "‘’“”–—…、。！，？"

# Two similarities closer together than this count as the same number: a series
# that never varies has no valleys, so it has no topic shift.
FLAT_TOLERANCE = 1e-9


def _tokens(text: str) -> List[str]:
    words = []
    for raw in WORD_RE.findall(text.lower()):
        token = raw.strip(_STRIP)
        if token:
            words.append(token)
    return words


def sentence_vectors(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised hashed bag-of-words vectors, one row per text."""
    matrix = np.zeros((len(texts), DIMENSIONS), dtype=np.float64)
    for row, text in enumerate(texts):
        for word in _tokens(text):
            matrix[row, zlib.crc32(word.encode("utf-8")) % DIMENSIONS] += 1.0
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return matrix / norms[:, None]


def _vectors(texts: Sequence[str], embed: Optional[Callable[[Sequence[str]], Any]] = None) -> np.ndarray:
    """Sentence vectors from ``embed`` when given, else the hashed bag of words."""
    if embed is None:
        return sentence_vectors(texts)
    matrix = np.asarray(embed(list(texts)), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError(
            "embed must return one vector per sentence: expected {0} rows, got shape {1}".format(
                len(texts), getattr(matrix, "shape", type(matrix).__name__)
            )
        )
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return matrix / norms[:, None]


def boundary_similarities(
    texts: Sequence[str], window: int = 2, embed: Optional[Callable[[Sequence[str]], Any]] = None
) -> List[float]:
    """Cosine similarity across each boundary, comparing ``window`` units per side.

    ``embed`` replaces the built-in hashed bag-of-words with a real embedding model:
    a callable taking the list of sentences and returning one vector per sentence.
    Word overlap is a weak stand-in for meaning, so this is how the method earns its
    name; without it, sentences that say the same thing in different words look
    unrelated. See :func:`analyse_shifts` for what that costs in practice.
    """
    count = len(texts)
    if count < 2:
        return []
    vectors = _vectors(texts, embed)
    sims: List[float] = []
    for boundary in range(1, count):
        left = vectors[max(0, boundary - window):boundary].mean(axis=0)
        right = vectors[boundary:min(count, boundary + window)].mean(axis=0)
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(left, right) / (left_norm * right_norm)))
    return sims


def analyse_shifts(
    texts: Sequence[str],
    sensitivity: float = 30.0,
    window: int = 2,
    embed: Optional[Callable[[Sequence[str]], Any]] = None,
) -> Tuple[Set[int], List[float], Optional[str]]:
    """``(shifts, similarities, degenerate_reason)``.

    ``sensitivity`` is a percentile: only boundaries whose similarity falls in
    the lowest ``sensitivity`` percent of the document are considered, and of
    those only the local minima are kept.

    A percentile threshold is meaningless on a similarity series that never
    varies -- every boundary then sits both at the threshold and at a local
    minimum, so a naive reading finds a topic shift at *every* boundary, which
    is the same as finding none. Those runs are detected here, reported through
    the third return value and turned into "no shifts" so the caller falls back
    to packing whole sentences.
    """
    sims = boundary_similarities(texts, window=window, embed=embed)
    if len(sims) < 2:
        return set(), sims, None
    array = np.asarray(sims, dtype=float)
    if float(array.max() - array.min()) <= FLAT_TOLERANCE:
        return set(), sims, (
            "every boundary is equally similar (similarity {0:.2f} throughout), "
            "so the document has no similarity valley to cut at".format(float(array[0]))
        )
    threshold = float(np.percentile(array, sensitivity))
    shifts: Set[int] = set()
    last = len(sims) - 1
    for i, value in enumerate(sims):
        if value > threshold:
            continue
        before = sims[i - 1] if i > 0 else float("inf")
        after = sims[i + 1] if i < last else float("inf")
        if value <= before and value <= after:
            shifts.add(i + 1)
    if len(shifts) == len(sims):
        return set(), sims, (
            "a topic shift was found at all {0} boundaries, which is no signal "
            "at all".format(len(sims))
        )
    return shifts, sims, None


def topic_shifts(
    texts: Sequence[str],
    sensitivity: float = 30.0,
    window: int = 2,
    embed: Optional[Callable[[Sequence[str]], Any]] = None,
) -> Tuple[Set[int], List[float]]:
    """Indices where a new topic appears to start, plus the similarity series.

    Thin wrapper over :func:`analyse_shifts` that drops the degeneracy reason.
    """
    shifts, sims, _reason = analyse_shifts(
        texts, sensitivity=sensitivity, window=window, embed=embed
    )
    return shifts, sims
