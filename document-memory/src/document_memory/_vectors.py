"""The optional vector half of the ranking.

Vectors are only touched when ``Memory(embed=...)`` is given an embedding
callable.  They are stored as little-endian float32 blobs, L2-normalised on the
way in so that cosine similarity is a plain dot product on the way out.

numpy is used here and nowhere else in the package.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

DTYPE = np.dtype("<f4")


def as_matrix(vectors: Sequence, *, expected_dim: Optional[int] = None) -> np.ndarray:
    """Turn whatever ``embed`` returned into a 2-D, L2-normalised float32 array."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            "embed must return one vector per text (a 2-D array or a list of "
            f"equal-length sequences), got an array with {array.ndim} dimensions"
        )
    if array.shape[1] == 0:
        raise ValueError("embed returned zero-length vectors")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(
            f"embed returned {array.shape[1]}-dimensional vectors but this store "
            f"already holds {expected_dim}-dimensional ones; use a new store or a "
            "new namespace for a different model"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("embed returned a vector containing NaN or infinity")
    return normalize(array)


def normalize(array: np.ndarray) -> np.ndarray:
    """L2-normalise each row; an all-zero row is left as zeros (similarity 0)."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (array / norms).astype(np.float32, copy=False)


def embed_texts(
    embed: Callable[[List[str]], Sequence],
    texts: Sequence[str],
    *,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    """Call ``embed`` on a list of texts and validate the shape of the answer."""
    batch = list(texts)
    if not batch:
        return np.zeros((0, expected_dim or 1), dtype=np.float32)
    try:
        raw = embed(batch)
    except Exception as exc:  # pragma: no cover - depends on the user's model
        raise ValueError(f"embed raised {type(exc).__name__}: {exc}") from exc
    matrix = as_matrix(raw, expected_dim=expected_dim)
    if matrix.shape[0] != len(batch):
        raise ValueError(
            f"embed was given {len(batch)} text(s) and returned {matrix.shape[0]} "
            "vector(s); it must return exactly one vector per text"
        )
    return matrix


def to_blob(vector: np.ndarray) -> bytes:
    """One row as the bytes stored in the ``vector`` column."""
    return np.asarray(vector, dtype=DTYPE).tobytes()


def from_blobs(blobs: Sequence[bytes], dim: int) -> np.ndarray:
    """Stack stored blobs back into a matrix, zero-filling any short row."""
    out = np.zeros((len(blobs), dim), dtype=np.float32)
    for index, blob in enumerate(blobs):
        if not blob:
            continue
        row = np.frombuffer(blob, dtype=DTYPE)
        if row.size == dim:
            out[index] = row
    return out


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one normalised query against normalised rows."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ np.asarray(query, dtype=np.float32).reshape(-1)


def blend(lexical: float, similarity: float, weight: float) -> Tuple[float, float, float]:
    """Combine a normalised BM25 score with a cosine similarity.

    ``lexical`` is BM25 divided by the best BM25 among the candidates for this
    query, so it lands in ``[0, 1]`` with the strongest lexical match at 1.0.
    ``similarity`` is cosine in ``[-1, 1]``, clamped to ``[0, 1]``: anything
    pointing away from the query is simply "not similar", and a memory stored
    before the store had an embedding function (no vector, so cosine 0)
    contributes nothing rather than half a point.  The blend is::

        score = (1 - weight) * lexical + weight * vector

    with ``weight`` defaulting to 0.5, an even split between the two halves.
    ``weight=0`` is pure BM25, ``weight=1`` is pure cosine.  Returns
    ``(score, lexical, vector)``.
    """
    vector = min(1.0, max(0.0, float(similarity)))
    lexical = min(1.0, max(0.0, float(lexical)))
    weight = min(1.0, max(0.0, float(weight)))
    return ((1.0 - weight) * lexical + weight * vector, lexical, vector)
