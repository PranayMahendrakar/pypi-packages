"""Getting any image in, as one predictable thing: an 8-bit luminance plane.

Nothing here decides anything about cleaning. It opens files, honours the EXIF
orientation tag before a single measurement is taken, and flattens the zoo of
Pillow modes (1, L, LA, P, RGB, RGBA, CMYK, I, F) down to mode ``L``.

Two rules hold everywhere in this package and start here:

* **The caller's image is never touched.** Every array this module hands out is
  a writable copy. ``np.asarray`` on a Pillow image can return a read-only view
  sharing the image's buffer, so writing into it either raises or silently
  edits the caller's picture. :func:`to_array` always copies.
* **Paper is measured, not assumed.** A scan of newsprint is not 255 white, and
  filling a rotation with white would print a bright frame around a grey page.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

#: Suffixes the CLI picks up when it is pointed at a directory.
IMAGE_SUFFIXES = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".jp2", ".png", ".pgm", ".ppm",
    ".tif", ".tiff", ".webp",
}

#: Percentile of the luminance plane taken as the paper level. Not the maximum:
#: one blown highlight or a torn white corner must not set the scale for a page.
PAPER_PERCENTILE = 85.0

#: Pillow modes carrying more than 8 bits per sample, which need stretching.
_WIDE_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")

# Pillow moved the resampling filters into an enum in 9.1; both spellings work.
_RESAMPLING = getattr(Image, "Resampling", Image)
BOX = getattr(_RESAMPLING, "BOX")
BICUBIC = getattr(_RESAMPLING, "BICUBIC")
LANCZOS = getattr(_RESAMPLING, "LANCZOS")


def looks_like_image_path(path: Any) -> bool:
    """True when ``path`` has a suffix Pillow normally reads."""
    return os.path.splitext(str(path))[1].lower() in IMAGE_SUFFIXES


def open_image(source: Any) -> Image.Image:
    """Open ``source`` as a PIL image, without modifying it.

    Args:
        source: a ``PIL.Image.Image``, a path (``str`` or ``os.PathLike``), or a
            numpy array shaped ``(h, w)``, ``(h, w, 1)``, ``(h, w, 3)`` or
            ``(h, w, 4)``.

    Raises:
        FileNotFoundError: if a path does not exist.
        TypeError: if ``source`` is none of the accepted kinds.
        ValueError: if an array has the wrong shape or no pixels.
    """
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        return image_from_array(source)
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise FileNotFoundError("{0!r} does not exist".format(path))
        image = Image.open(path)
        image.load()
        return image
    raise TypeError(
        "image must be a PIL.Image, a file path or a numpy array, not {0}".format(
            type(source).__name__
        )
    )


def image_from_array(array: np.ndarray) -> Image.Image:
    """Wrap a numpy array as a PIL image, copying rather than sharing memory."""
    if array.ndim not in (2, 3):
        raise ValueError(
            "array images must be 2-D or 3-D, got shape {0}".format(array.shape)
        )
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError(
            "array images need 1, 3 or 4 channels, got {0}".format(array.shape[2])
        )
    if array.size == 0:
        raise ValueError("array image is empty")
    data = np.array(array, copy=True)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    if data.dtype != np.uint8:
        floats = data.astype(np.float64)
        finite = floats[np.isfinite(floats)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            floats = floats * 255.0
        data = np.clip(np.nan_to_num(floats), 0.0, 255.0).astype(np.uint8)
    channels = 1 if data.ndim == 2 else data.shape[2]
    return Image.fromarray(data, mode={1: "L", 3: "RGB", 4: "RGBA"}[channels])


def apply_exif_orientation(image: Image.Image) -> Tuple[Image.Image, bool]:
    """Turn ``image`` the way its EXIF tag says. Returns ``(image, was_turned)``.

    A phone photograph of a page carries its orientation in EXIF rather than in
    the pixels, and measuring skew on the unturned pixels would report ninety
    degrees of it.
    """
    try:
        exif = image.getexif()
        orientation = int(exif.get(0x0112, 1) or 1) if exif else 1
    except Exception:            # pragma: no cover - broken EXIF blocks happen
        orientation = 1
    if orientation in (0, 1):
        return image, False
    turned = ImageOps.exif_transpose(image)
    if turned is None:           # pragma: no cover - only on very old Pillow
        return image, False
    return turned, True


def to_grayscale(image: Image.Image) -> Tuple[Image.Image, str]:
    """Return ``(luminance image in mode L, what had to be done)``.

    The description is plain English for the ``grayscale`` step, and reads
    ``"already 8-bit greyscale"`` when nothing had to change.
    """
    mode = image.mode
    if mode == "L":
        return image.copy(), "already 8-bit greyscale"
    if mode in _WIDE_MODES:
        return _stretch_wide_mode(image), "stretched {0} samples into 8 bits".format(mode)
    if mode == "1":
        return image.convert("L"), "expanded 1-bit input to 8-bit greyscale"
    if mode in ("LA", "PA", "RGBA") or (mode == "P" and "transparency" in image.info):
        flat = _flatten_alpha(image)
        return flat.convert("L"), "composited transparency onto white, then luminance"
    return image.convert("L"), "luminance from {0}".format(mode)


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite transparency onto white, the colour a scanner would have seen."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _stretch_wide_mode(image: Image.Image) -> Image.Image:
    """16-bit and float images: stretch the used range into 8 bits, no clipping."""
    data = np.array(image, dtype=np.float64, copy=True)
    finite = data[np.isfinite(data)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    if high - low < 1e-12:
        data = np.zeros_like(data)
    else:
        data = (np.nan_to_num(data, nan=low) - low) * (255.0 / (high - low))
    flat = np.clip(data, 0.0, 255.0).astype(np.uint8)
    if flat.ndim == 3:           # pragma: no cover - multi-band F images are rare
        flat = flat[:, :, 0]
    return Image.fromarray(flat, mode="L")


def to_array(image: Image.Image) -> np.ndarray:
    """A writable ``(h, w)`` uint8 copy of a mode ``L`` image.

    Always a copy. ``np.asarray`` of a Pillow image can hand back a read-only
    view onto the image's own buffer, and code that writes into it either
    raises "assignment destination is read-only" or quietly edits the caller's
    picture. Neither is acceptable, so the copy is not optional.
    """
    array = np.array(image, dtype=np.uint8, copy=True)
    if array.ndim != 2:          # pragma: no cover - callers pass mode L
        raise ValueError("expected a greyscale image, got shape {0}".format(array.shape))
    return array


def from_array(plane: np.ndarray) -> Image.Image:
    """A mode ``L`` image holding a copy of ``plane``."""
    return Image.fromarray(np.ascontiguousarray(plane, dtype=np.uint8), mode="L")


def local_mean(plane: np.ndarray, radius: int) -> np.ndarray:
    """Mean of ``plane`` over a box of ``radius`` around every pixel, as float64.

    Pillow's box blur runs a sliding sum in C, so the cost does not grow with
    the radius, and it repeats the edge pixels rather than shrinking the window
    at the margins. On a 12 megapixel page it beat an exact int64 integral image
    by eight times while differing from it by less than one grey level.
    """
    blurred = from_array(plane).filter(ImageFilter.BoxBlur(max(1, int(radius))))
    return to_array(blurred).astype(np.float64)


def paper_level(plane: np.ndarray) -> float:
    """The luminance of clean paper on this page, 0 to 255.

    A high percentile rather than the maximum, so one blown highlight cannot
    decide what "paper" means for a whole sheet.
    """
    if plane.size == 0:          # pragma: no cover - guarded by the entry points
        return 255.0
    return float(np.percentile(plane, PAPER_PERCENTILE))


def analysis_plane(plane: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Downscale ``plane`` for measurement. Returns ``(plane, scale)``.

    ``scale`` maps source pixels to analysis pixels, so a length measured on the
    returned plane is divided by it to get back to source pixels. A 4000 px scan
    is measured at 1600 px, which keeps the skew search and the line profile in
    milliseconds while still leaving far more rows per line of text than either
    needs.
    """
    height, width = plane.shape
    if max_side <= 0 or max(height, width) <= max_side:
        return plane, 1.0
    scale = max_side / float(max(height, width))
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    small = from_array(plane).resize(size, BOX)
    return to_array(small), small.size[0] / float(width)
