"""Getting any image into one predictable shape: RGB floats plus an alpha mask.

Nothing here decides anything about cropping. It opens files, applies EXIF
orientation before any measurement is taken, and flattens the zoo of Pillow
modes (L, LA, P, RGBA, CMYK, I, F, 1) into arrays the energy maps can read.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".jp2", ".png", ".ppm", ".tif", ".tiff", ".webp",
}

_EXIF_ORIENTATION_TAG = 0x0112
_WIDE_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")

# Pillow moved the resampling filters in 9.1; both spellings are supported.
_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def looks_like_image_path(path: str) -> bool:
    """True when ``path`` has a suffix Pillow normally reads."""
    return os.path.splitext(str(path))[1].lower() in IMAGE_SUFFIXES


def open_image(source: Any) -> Image.Image:
    """Open ``source`` as a PIL image.

    Accepts a ``PIL.Image.Image``, a path (``str`` or ``os.PathLike``) or a numpy
    array shaped ``(h, w)``, ``(h, w, 3)`` or ``(h, w, 4)``.
    """
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        return _image_from_array(source)
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path!r} does not exist")
        image = Image.open(path)
        image.load()
        return image
    raise TypeError(
        "image must be a PIL.Image, a file path or a numpy array, "
        f"not {type(source).__name__}"
    )


def _image_from_array(array: np.ndarray) -> Image.Image:
    if array.ndim not in (2, 3):
        raise ValueError(f"array images must be 2-D or 3-D, got shape {array.shape}")
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError(f"array images need 1, 3 or 4 channels, got {array.shape[2]}")
    if array.size == 0:
        raise ValueError("array image is empty")
    data = array
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    if data.dtype != np.uint8:
        floats = data.astype(np.float64)
        if floats.min() >= 0.0 and floats.max() <= 1.0:
            floats = floats * 255.0
        data = np.clip(floats, 0.0, 255.0).astype(np.uint8)
    mode = {2: "L", 3: "RGB", 4: "RGBA"}[data.ndim if data.ndim == 2 else data.shape[2]]
    return Image.fromarray(data, mode=mode)


def apply_exif_orientation(image: Image.Image) -> Tuple[Image.Image, bool]:
    """Rotate ``image`` the way its EXIF tag says, before anything else looks at it.

    Returns ``(image, was_rotated)``.
    """
    orientation = 1
    try:
        exif = image.getexif()
    except Exception:  # pragma: no cover - broken EXIF blocks in the wild
        exif = None
    if exif:
        try:
            orientation = int(exif.get(_EXIF_ORIENTATION_TAG, 1) or 1)
        except (TypeError, ValueError):  # pragma: no cover - nonsense tag value
            orientation = 1
    if orientation in (0, 1):
        return image, False
    rotated = ImageOps.exif_transpose(image)
    if rotated is None:  # pragma: no cover - only on very old Pillow
        return image, False
    return rotated, True


def normalize_mode(image: Image.Image) -> Image.Image:
    """Return ``image`` as RGB, or RGBA when it genuinely carries transparency."""
    mode = image.mode
    if mode in ("RGB", "RGBA"):
        return image
    if mode in _WIDE_MODES:
        return _stretch_wide_mode(image)
    wants_alpha = mode in ("LA", "PA") or (mode == "P" and "transparency" in image.info)
    return image.convert("RGBA" if wants_alpha else "RGB")


def _stretch_wide_mode(image: Image.Image) -> Image.Image:
    """16-bit and float images: stretch the used range into 8 bits, no clipping."""
    data = np.asarray(image).astype(np.float64)
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        data = np.zeros_like(data)
    else:
        data = (data - low) * (255.0 / (high - low))
    grey = Image.fromarray(np.clip(data, 0.0, 255.0).astype(np.uint8), mode="L")
    return grey.convert("RGB")


def working_arrays(
    image: Image.Image, max_side: int
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Downscale for analysis and split into ``(rgb, alpha, scale)``.

    ``rgb`` is ``(h, w, 3)`` float in ``[0, 1]``; ``alpha`` is ``(h, w)`` float in
    ``[0, 1]`` or ``None``; ``scale`` maps source pixels to working pixels.
    """
    prepared = normalize_mode(image)
    width, height = prepared.size
    if width < 1 or height < 1:
        raise ValueError("image has no pixels")
    scale = 1.0
    if max_side > 0 and max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        prepared = prepared.resize(target, _BOX)
        scale = prepared.size[0] / float(width)
    data = np.asarray(prepared, dtype=np.float64) / 255.0
    if data.ndim == 2:  # pragma: no cover - normalize_mode never returns 2-D
        data = np.repeat(data[:, :, None], 3, axis=2)
    alpha = None
    if data.shape[2] == 4:
        alpha = data[:, :, 3]
        data = data[:, :, :3]
    return np.ascontiguousarray(data), alpha, scale


def resize_exact(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """High quality resize to exactly ``size``."""
    return image.resize((int(size[0]), int(size[1])), _LANCZOS)


def flatten_for_format(image: Image.Image, path: str) -> Tuple[Image.Image, bool]:
    """Drop alpha when the destination format cannot store it (JPEG, BMP).

    Returns ``(image, was_flattened)``; transparency is composited onto white.
    """
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix not in (".jpg", ".jpeg", ".bmp", ".ppm") or image.mode not in ("RGBA", "LA", "PA", "P"):
        return image, False
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB"), True
