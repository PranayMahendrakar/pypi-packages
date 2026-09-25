"""Perceptual hashes (pHash, dHash, aHash) computed with numpy and Pillow only.

Every image goes through one decode and one greyscale reduction to a square of
``8 * hash_size`` pixels; all three hashes are derived from that square, so
storing all three costs the same as storing one.
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

METHODS: Tuple[str, ...] = ("phash", "dhash", "ahash")

#: File extensions a directory scan treats as images. Other files in a scanned
#: directory are ignored (and counted); a file passed explicitly is always tried.
IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".apng", ".gif", ".bmp", ".dib",
        ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm", ".pnm", ".ico", ".tga",
        ".heic", ".heif", ".avif", ".jp2", ".j2k", ".pcx", ".sgi", ".icns",
    }
)

#: Standard deviation (in 0-255 grey levels) of the reduced image below which it
#: counts as "flat": blank, near-uniform, or with detail too small to survive the
#: reduction. The hash of a flat image is noise, so flat images only ever match
#: images with identical content.
FLAT_STD = 2.0

_RESAMPLING: Any = getattr(Image, "Resampling", Image)
_TRANSPOSE: Any = getattr(Image, "Transpose", Image)
_LANCZOS = _RESAMPLING.LANCZOS
_BOX = _RESAMPLING.BOX

# EXIF orientation tag value -> the transpose that shows the image upright.
_ORIENTATION_OPS = {
    2: _TRANSPOSE.FLIP_LEFT_RIGHT,
    3: _TRANSPOSE.ROTATE_180,
    4: _TRANSPOSE.FLIP_TOP_BOTTOM,
    5: _TRANSPOSE.TRANSPOSE,
    6: _TRANSPOSE.ROTATE_270,
    7: _TRANSPOSE.TRANSVERSE,
    8: _TRANSPOSE.ROTATE_90,
}

_DECODE_ERRORS = (OSError, SyntaxError, EOFError, ValueError, TypeError, IndexError, KeyError, struct.error)


@dataclass
class Fingerprint:
    """Everything the index stores about one image."""

    width: int
    height: int
    format: str
    digest: str
    flat: bool
    tone: float
    hashes: Dict[str, bytes]


@lru_cache(maxsize=16)
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix ``C`` of size ``n``, so that ``C @ x`` is the DCT of ``x``."""
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    c = np.cos(np.pi * (2.0 * i + 1.0) * k / (2.0 * n)) * np.sqrt(2.0 / n)
    c[0, :] = np.sqrt(1.0 / n)
    c.setflags(write=False)
    return c


def dct2(x: np.ndarray) -> np.ndarray:
    """Two-dimensional orthonormal DCT-II of a 2-D array."""
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    return dct_matrix(rows) @ x @ dct_matrix(cols).T


def _orientation(im: Image.Image) -> int:
    try:
        value = int(im.getexif().get(0x0112, 1))
    except Exception:  # corrupt EXIF is not a reason to skip the image
        return 1
    return value if value in _ORIENTATION_OPS else 1


def _to_grey(im: Image.Image) -> Image.Image:
    """A new 8-bit greyscale copy of ``im``; transparency is flattened onto white."""
    mode = im.mode
    if mode in ("I", "F") or mode.startswith("I;16"):
        arr = np.asarray(im, dtype=np.float64)
        lo, hi = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 0.0)
        if lo >= 0.0 and hi <= 255.0:
            pass
        elif lo >= 0.0 and hi <= 65535.0 and mode != "F":
            arr = arr / 257.0  # 16-bit samples
        else:
            arr = (arr - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(arr)
        return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "L")
    if mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info:
        try:
            rgba = im.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            return background.convert("L")
        except (ValueError, OSError):
            pass
    if mode in ("L", "RGB", "1", "P"):
        return im.convert("L")
    return im.convert("RGB").convert("L")


def _from_grey(grey: Image.Image, orientation: int, hash_size: int) -> Tuple[bool, float, Dict[str, bytes]]:
    side = 8 * hash_size
    base = grey.resize((side, side), _LANCZOS, reducing_gap=2.0)
    op = _ORIENTATION_OPS.get(orientation)
    if op is not None:
        base = base.transpose(op)
    a = np.asarray(base, dtype=np.float64)
    std = float(a.std())
    tone = float(a.mean())

    # pHash: DCT of a (4 * hash_size) square, low-frequency block against its median.
    p = a.reshape(4 * hash_size, 2, 4 * hash_size, 2).mean(axis=(1, 3))
    low = dct2(p)[:hash_size, :hash_size]
    phash = low > np.median(low)

    # dHash: is each pixel brighter than its left neighbour?
    d = np.asarray(base.resize((hash_size + 1, hash_size), _BOX), dtype=np.int16)
    dhash = d[:, 1:] > d[:, :-1]

    # aHash: is each cell brighter than the mean?
    m = a.reshape(hash_size, 8, hash_size, 8).mean(axis=(1, 3))
    ahash = m > m.mean()

    hashes = {
        "phash": np.packbits(phash.ravel()).tobytes(),
        "dhash": np.packbits(dhash.ravel()).tobytes(),
        "ahash": np.packbits(ahash.ravel()).tobytes(),
    }
    return std < FLAT_STD, tone, hashes


def _displayed_size(size: Tuple[int, int], orientation: int) -> Tuple[int, int]:
    w, h = size
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def fingerprint_bytes(data: bytes, hash_size: int) -> Fingerprint:
    """Fingerprint the bytes of an encoded image file. Raises ``ValueError`` if unreadable."""
    if not data:
        raise ValueError("empty file")
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format or ""
            orientation = _orientation(im)
            width, height = _displayed_size(im.size, orientation)
            if width == 0 or height == 0:
                raise ValueError("image has zero width or height")
            want = 16 * hash_size
            try:
                im.draft("L", (want, want))  # JPEG: decode straight to a small greyscale image
            except Exception:
                pass
            grey = _to_grey(im)
    except UnidentifiedImageError:
        raise ValueError("not an image file Pillow can read") from None
    except Image.DecompressionBombError as exc:
        raise ValueError(f"too large to decode safely ({exc})") from None
    except _DECODE_ERRORS as exc:
        raise ValueError(f"unreadable image ({type(exc).__name__}: {exc})") from None
    flat, tone, hashes = _from_grey(grey, orientation, hash_size)
    return Fingerprint(width, height, fmt, digest, flat, tone, hashes)


def fingerprint_file(path: str, hash_size: int) -> Fingerprint:
    """Fingerprint an image file on disk (opened read-only). Raises ``ValueError``/``OSError``."""
    with open(path, "rb") as fh:
        data = fh.read()
    return fingerprint_bytes(data, hash_size)


def pixel_digest(img: Image.Image) -> str:
    """Content digest of an in-memory image: mode, size, palette and pixels."""
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{img.mode}|{img.size[0]}x{img.size[1]}|".encode("ascii"))
    if img.mode in ("P", "PA"):
        h.update(bytes(img.getpalette() or []))
    h.update(img.tobytes())
    return h.hexdigest()


def fingerprint_image(img: Image.Image, hash_size: int) -> Fingerprint:
    """Fingerprint an in-memory PIL image. The image itself is never modified."""
    if not isinstance(img, Image.Image):
        raise TypeError(f"expected a PIL.Image.Image, got {type(img).__name__}")
    orientation = _orientation(img)
    width, height = _displayed_size(img.size, orientation)
    if width == 0 or height == 0:
        raise ValueError("image has zero width or height")
    digest = pixel_digest(img)
    flat, tone, hashes = _from_grey(_to_grey(img), orientation, hash_size)
    return Fingerprint(width, height, getattr(img, "format", None) or "", digest, flat, tone, hashes)


def upright_rgb(img: Image.Image) -> Image.Image:
    """A new RGB copy of ``img``, turned the way its EXIF orientation says it is displayed."""
    rgb = img.convert("RGB")
    op = _ORIENTATION_OPS.get(_orientation(img))
    return rgb.transpose(op) if op is not None else rgb


def open_upright_rgb(path: str) -> Image.Image:
    """Decode an image file fully and return an upright RGB copy (what ``embed`` receives)."""
    with open(path, "rb") as fh:
        data = fh.read()
    with Image.open(io.BytesIO(data)) as im:
        return upright_rgb(im)


def check_hash_size(hash_size: Any) -> int:
    """Validate ``hash_size`` (an int from 4 to 32) and return it."""
    if isinstance(hash_size, bool) or not isinstance(hash_size, (int, np.integer)):
        raise TypeError(f"hash_size must be an int, got {type(hash_size).__name__}")
    if not 4 <= int(hash_size) <= 32:
        raise ValueError(f"hash_size must be between 4 and 32, got {hash_size}")
    return int(hash_size)


def check_method(method: Any, allowed: Tuple[str, ...] = METHODS) -> str:
    """Validate a method name and return it."""
    if method not in allowed:
        raise ValueError(f"method must be one of {', '.join(repr(m) for m in allowed)}; got {method!r}")
    return str(method)


def hash_image(image: Any, method: str = "phash", *, hash_size: int = 8) -> str:
    """Hex string of one perceptual hash of an image path or ``PIL.Image``.

    Two hashes of the same ``method`` and ``hash_size`` compare by hamming distance;
    similarity is ``1 - hamming / hash_size**2``.
    """
    check_method(method)
    hash_size = check_hash_size(hash_size)
    if isinstance(image, Image.Image):
        fp = fingerprint_image(image, hash_size)
    else:
        fp = fingerprint_file(str(image), hash_size)
    return fp.hashes[method].hex()
