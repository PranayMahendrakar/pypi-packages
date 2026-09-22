"""Getting any scan into one predictable shape before anything is measured.

Nothing here judges a page. It opens files, applies EXIF orientation so every
later measurement refers to the upright image, flattens the zoo of Pillow modes
(L, LA, P, RGBA, CMYK, I, F, 1) down to 8-bit luminance, and builds the two
downscaled working planes the measurements share.

Three scales are kept on purpose:

``lum``
    Native resolution, 8-bit. Sharpness and clipping have to be read here -
    resizing an image changes exactly the thing they measure.
``work``
    Long edge capped at :data:`WORK_LONG_EDGE`. Lighting, show-through and the
    text mask live here; they care about page-sized structure, not pixels.
``fine``
    Long edge capped at :data:`SKEW_LONG_EDGE`. Skew and line pitch live here,
    small enough that a few hundred candidate angles cost milliseconds.

The caller's image is never modified. Every array handed out is freshly
allocated, and nothing in this package writes through a view of it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

#: File suffixes the CLI picks up when pointed at a directory.
IMAGE_SUFFIXES = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".jp2", ".png", ".pnm", ".ppm", ".pgm",
    ".tif", ".tiff", ".webp",
}

#: Long edge of the plane used for lighting, show-through and the text mask.
WORK_LONG_EDGE = 1024
#: Long edge of the plane used for skew, line pitch and text height.
SKEW_LONG_EDGE = 512
#: Long edge of the thumbnail used to ask "is this page in colour at all".
COLOUR_LONG_EDGE = 192
#: Anything claiming fewer dots per inch than this is metadata noise, not a dpi.
MIN_CREDIBLE_DPI = 24.0
#: Anything claiming more than this is metadata noise too.
MAX_CREDIBLE_DPI = 4800.0

_EXIF_ORIENTATION_TAG = 0x0112
_WIDE_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")

# Pillow moved the resampling filters in 9.1; both spellings are supported.
_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")


def looks_like_image_path(path: Any) -> bool:
    """True when ``path`` has a suffix Pillow normally reads."""
    return os.path.splitext(str(path))[1].lower() in IMAGE_SUFFIXES


def open_image(source: Any) -> Image.Image:
    """Open ``source`` as a PIL image without altering it.

    Accepts a ``PIL.Image.Image`` (returned as-is), a path (``str`` or
    ``os.PathLike``), or a numpy array shaped ``(h, w)``, ``(h, w, 1)``,
    ``(h, w, 3)`` or ``(h, w, 4)``.

    Raises:
        FileNotFoundError: if a path does not exist.
        TypeError: if ``source`` is none of the accepted kinds.
        ValueError: if an array has a shape no image could have.
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
    """Wrap a numpy array as a PIL image, copying so the caller's array is safe.

    Raises:
        ValueError: if the array is empty or has a shape no image could have.
    """
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
    data = array
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    if data.dtype != np.uint8:
        floats = np.asarray(data, dtype=np.float64)
        finite = floats[np.isfinite(floats)]
        low = float(finite.min()) if finite.size else 0.0
        high = float(finite.max()) if finite.size else 0.0
        if low >= 0.0 and high <= 1.0:
            floats = floats * 255.0
        data = np.clip(np.nan_to_num(floats), 0.0, 255.0).astype(np.uint8)
    else:
        data = data.copy()
    channels = 1 if data.ndim == 2 else data.shape[2]
    return Image.fromarray(
        np.ascontiguousarray(data), mode={1: "L", 3: "RGB", 4: "RGBA"}[channels]
    )


def apply_exif_orientation(image: Image.Image) -> Tuple[Image.Image, bool]:
    """Turn ``image`` the way its EXIF tag says, before anything else looks at it.

    Returns ``(image, was_turned)``. The original is never modified.
    """
    orientation = 1
    try:
        exif = image.getexif()
    except Exception:  # pragma: no cover - broken EXIF blocks exist in the wild
        exif = None
    if exif:
        try:
            orientation = int(exif.get(_EXIF_ORIENTATION_TAG, 1) or 1)
        except (TypeError, ValueError):  # pragma: no cover - nonsense tag value
            orientation = 1
    if orientation in (0, 1):
        return image, False
    turned = ImageOps.exif_transpose(image)
    if turned is None:  # pragma: no cover - only on very old Pillow
        return image, False
    return turned, True


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite transparency onto white: unscanned corners then read as paper."""
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba)


def _stretch_wide_mode(image: Image.Image) -> Image.Image:
    """16-bit and float scans: stretch the used range into 8 bits, no clipping."""
    data = np.asarray(image).astype(np.float64)
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        data = np.zeros_like(data)
    else:
        data = (data - low) * (255.0 / (high - low))
    return Image.fromarray(np.clip(data, 0.0, 255.0).astype(np.uint8), mode="L")


def _has_alpha(image: Image.Image) -> bool:
    """True when this mode actually carries transparency."""
    return image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )


def to_display_rgb(image: Image.Image) -> Image.Image:
    """``image`` as plain RGB, with transparency flattened onto white."""
    if image.mode in _WIDE_MODES:
        return _stretch_wide_mode(image).convert("RGB")
    if _has_alpha(image):
        return _flatten_alpha(image).convert("RGB")
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def to_luminance(image: Image.Image) -> np.ndarray:
    """Native-resolution 8-bit luminance as a freshly allocated ``(h, w)`` array."""
    if image.mode in _WIDE_MODES:
        grey = _stretch_wide_mode(image)
    elif _has_alpha(image):
        grey = _flatten_alpha(image).convert("L")
    elif image.mode == "L":
        grey = image
    else:
        grey = image.convert("L")
    return np.array(grey, dtype=np.uint8, copy=True)


def downscale_plane(lum: np.ndarray, long_edge: int) -> Tuple[np.ndarray, float]:
    """Box-filter ``lum`` down to ``long_edge``; return ``(float32 in 0-1, scale)``.

    ``scale`` maps native pixels to plane pixels, so a length measured on the
    plane becomes native pixels when divided by it. A page already smaller than
    ``long_edge`` comes back at its own size with ``scale`` 1.0.
    """
    height, width = lum.shape
    longest = max(height, width)
    if long_edge <= 0 or longest <= long_edge:
        return lum.astype(np.float32) / np.float32(255.0), 1.0
    ratio = long_edge / float(longest)
    target = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    small = Image.fromarray(lum, mode="L").resize(target, _BOX)
    plane = np.asarray(small, dtype=np.float32) / np.float32(255.0)
    return plane, small.size[0] / float(width)


def colour_spread(image: Image.Image) -> float:
    """How far from neutral grey this page is: 0 a grey scan, 1 fully saturated.

    Measured on a small thumbnail as the mean of ``max(r,g,b) - min(r,g,b)``,
    which paper and ink keep near zero even when the scanner was set to colour,
    and a photograph does not.
    """
    rgb = to_display_rgb(image)
    width, height = rgb.size
    longest = max(width, height)
    if longest > COLOUR_LONG_EDGE:
        ratio = COLOUR_LONG_EDGE / float(longest)
        rgb = rgb.resize(
            (max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))),
            _BOX,
        )
    data = np.asarray(rgb, dtype=np.int16)
    if data.ndim != 3:  # pragma: no cover - to_display_rgb always returns RGB
        return 0.0
    spread = data.max(axis=2) - data.min(axis=2)
    return float(spread.mean()) / 255.0


def dpi_from_image(image: Image.Image) -> Optional[float]:
    """The dots per inch the file claims, or ``None`` when it claims nothing.

    This is read, never guessed: a file with no resolution tag comes back
    ``None`` so the report can leave resolution advice out rather than invent it.
    """
    info = getattr(image, "info", None) or {}
    candidates = []
    raw = info.get("dpi")
    if raw is not None:
        candidates.extend(raw if isinstance(raw, (tuple, list)) else [raw])
    if not candidates and info.get("jfif_unit") == 1:
        candidates.extend([info.get("jfif_density"), info.get("jfif_x_density")])
    values = []
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if MIN_CREDIBLE_DPI <= value <= MAX_CREDIBLE_DPI:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def coerce_dpi(dpi: Any) -> Optional[float]:
    """Validate a caller-supplied dpi, returning it as a float or ``None``.

    Raises:
        ValueError: if ``dpi`` is not a positive, finite number.
    """
    if dpi is None:
        return None
    try:
        value = float(dpi)
    except (TypeError, ValueError):
        raise ValueError(
            "dpi must be a number or None, got {0!r}".format(dpi)
        ) from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError("dpi must be a positive number, got {0!r}".format(dpi))
    return value


@dataclass
class PagePlanes:
    """One page, prepared once and shared by every measurement."""

    width: int
    height: int
    lum: np.ndarray
    work: np.ndarray
    work_scale: float
    fine: np.ndarray
    fine_scale: float
    colour: float
    dpi: Optional[float]
    dpi_source: Optional[str]
    exif_applied: bool

    @property
    def megapixels(self) -> float:
        """Page area in megapixels."""
        return (self.width * self.height) / 1e6


def prepare(source: Any, dpi: Any = None) -> PagePlanes:
    """Open ``source`` and build every plane the measurements need.

    ``dpi`` given by the caller wins; otherwise the file's own resolution tag is
    used; otherwise it stays ``None`` and no resolution advice is produced.

    Raises:
        ValueError: if the image has no pixels, or ``dpi`` is not positive.
    """
    given = coerce_dpi(dpi)
    image = open_image(source)
    image, exif_applied = apply_exif_orientation(image)
    width, height = image.size
    if width < 1 or height < 1:  # pragma: no cover - Pillow rejects these first
        raise ValueError("image has no pixels")

    lum = to_luminance(image)
    work, work_scale = downscale_plane(lum, WORK_LONG_EDGE)
    fine, fine_scale = downscale_plane(lum, SKEW_LONG_EDGE)

    if given is not None:
        resolved, origin = given, "argument"
    else:
        found = dpi_from_image(image)
        resolved, origin = (found, "image metadata") if found else (None, None)

    return PagePlanes(
        width=int(width),
        height=int(height),
        lum=lum,
        work=work,
        work_scale=work_scale,
        fine=fine,
        fine_scale=fine_scale,
        colour=colour_spread(image),
        dpi=resolved,
        dpi_source=origin,
        exif_applied=exif_applied,
    )
