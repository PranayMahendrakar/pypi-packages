"""Perceptual (difference) hashing for images. Needs the ``images`` extra (Pillow)."""
from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple

import numpy as np

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"})
HASH_BITS = 64
_HASH_SIZE = 8
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

PairList = List[Tuple[int, int, float]]


def _require_pil():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow
        raise ImportError(
            "Image support needs Pillow. Install it with: pip install \"near-dupes[images]\""
        ) from exc
    return Image, ImageOps


def is_pil_image(value: object) -> bool:
    """True for a ``PIL.Image.Image`` without importing Pillow."""
    return type(value).__module__.startswith("PIL.") and hasattr(value, "convert") and hasattr(value, "resize")


def looks_like_image_path(value: object) -> bool:
    """True for a str/PathLike whose suffix is a common raster image extension."""
    if isinstance(value, (str, os.PathLike)):
        return os.path.splitext(os.fspath(value))[1].lower() in IMAGE_SUFFIXES
    return False


def _to_gray_9x8(image, Image, ImageOps) -> np.ndarray:
    resample = getattr(Image, "Resampling", Image).LANCZOS
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # some formats carry unreadable EXIF; hashing still works
        pass
    gray = image.convert("L").resize((_HASH_SIZE + 1, _HASH_SIZE), resample)
    return np.asarray(gray, dtype=np.int16)


def dhash(image: object) -> int:
    """64-bit difference hash of an image path, ``PIL.Image`` or HxW[xC] numpy array.

    The image is converted to grayscale, resized to 9x8, and each bit records
    whether a pixel is brighter than its left neighbour. Similar images give
    hashes with a small Hamming distance.
    """
    Image, ImageOps = _require_pil()
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as im:
            arr = _to_gray_9x8(im, Image, ImageOps)
    elif isinstance(image, np.ndarray):
        arr = _to_gray_9x8(Image.fromarray(image), Image, ImageOps)
    elif is_pil_image(image):
        arr = _to_gray_9x8(image, Image, ImageOps)
    else:
        raise TypeError(f"Expected an image path, PIL.Image or numpy array, got {type(image).__name__}")
    bits = (arr[:, 1:] > arr[:, :-1]).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def dhash_many(items: Sequence[object]) -> np.ndarray:
    """dHash of every item as a uint64 array."""
    return np.array([dhash(x) for x in items], dtype=np.uint64)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits in each element of a uint64 array."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POPCOUNT[x.view(np.uint8)].reshape(x.shape + (8,)).sum(axis=-1, dtype=np.int64)


def hamming_similarity(h1: int, h2: int) -> float:
    """``1 - hamming(h1, h2) / 64`` for two 64-bit hashes."""
    return 1.0 - bin(int(h1) ^ int(h2)).count("1") / HASH_BITS


def hamming_pairs(hashes: np.ndarray, max_dist: int, *, brute_force_max: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """All pairs ``(i, j)``, ``i < j``, with Hamming distance ``<= max_dist``.

    Small inputs are compared exhaustively in numpy blocks. Larger inputs use
    pigeonhole banding: with ``max_dist + 1`` bit bands, two hashes within
    ``max_dist`` bits must agree on at least one band, so the search stays exact.
    Returns ``(pairs, distances)``.
    """
    from ._text import pairs_sharing_key, unique_pairs

    h = np.ascontiguousarray(hashes, dtype=np.uint64)
    n = h.size
    if n < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    max_dist = int(max_dist)
    n_bands = min(max_dist + 1, HASH_BITS)
    if n <= brute_force_max or n_bands > 16:
        idx = np.arange(n)
        block = max(1, (2 << 20) // n)
        found = []
        for s in range(0, n, block):
            blk = h[s : s + block]
            d = popcount64(blk[:, None] ^ h[None, :])
            ok = (d <= max_dist) & (idx[None, :] > np.arange(s, s + blk.shape[0])[:, None])
            ii, jj = np.nonzero(ok)
            if ii.size:
                found.append(np.stack([ii.astype(np.int64) + s, jj.astype(np.int64)], axis=1))
        pairs = unique_pairs(found, n)
    else:
        base, extra = divmod(HASH_BITS, n_bands)
        found = []
        shift = 0
        for band in range(n_bands):
            width = base + (1 if band < extra else 0)
            keys = (h >> np.uint64(shift)) & np.uint64((1 << width) - 1)
            found.append(pairs_sharing_key(keys))
            shift += width
        cand = unique_pairs(found, n)
        if cand.shape[0]:
            d = popcount64(h[cand[:, 0]] ^ h[cand[:, 1]])
            pairs = cand[d <= max_dist]
        else:
            pairs = cand
    dists = popcount64(h[pairs[:, 0]] ^ h[pairs[:, 1]]) if pairs.shape[0] else np.empty(0, dtype=np.int64)
    return pairs, dists


def find_image_pairs(hashes: np.ndarray, *, threshold: float) -> PairList:
    """Near-duplicate pairs among distinct hashes; score is ``1 - hamming / 64``."""
    max_dist = int(math.floor((1.0 - threshold) * HASH_BITS + 1e-9))
    pairs, dists = hamming_pairs(hashes, max_dist)
    return [(int(i), int(j), 1.0 - int(d) / HASH_BITS) for (i, j), d in zip(pairs.tolist(), dists.tolist())]
