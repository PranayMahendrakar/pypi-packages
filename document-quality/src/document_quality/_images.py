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

import dataclasses
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
#: A dpi read from a file within this of a whole number is taken as that
#: number: PNG's pixels-per-metre field cannot store 300 dpi exactly.
DPI_SNAP = 0.05

_EXIF_ORIENTATION_TAG = 0x0112
_WIDE_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")

# Pillow moved the resampling filters in 9.1; both spellings are supported.
_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")


#: About how many pixels a page-wide percentile is taken over. A level such as
#: "the 99.5th percentile of the page" is a statistic of the whole sheet, and
#: an evenly strided sample of this size gives it to well under a grey level
#: while costing a fraction of a full partition of a megapixel plane.
PERCENTILE_SAMPLE = 200_000


def percentile_sample(plane: np.ndarray, target: int = PERCENTILE_SAMPLE) -> np.ndarray:
    """An evenly strided view of ``plane`` with about ``target`` pixels.

    Only ever read from; a plane already that small comes back as it is.
    """
    stride = int(round(np.sqrt(plane.size / float(max(1, target)))))
    if stride <= 1:
        return plane
    return plane[::stride, ::stride]


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
        if os.path.isdir(path):
            raise IsADirectoryError(
                "{0!r} is a folder, not an image file; pass the files in it, or "
                "use assess_batch".format(path)
            )
        image = Image.open(path)
        try:
            image.load()
        except BaseException:
            # A truncated or corrupt file must not stay open - on Windows an
            # open handle keeps the file locked until garbage collection.
            image.close()
            raise
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
        data = to_8bit(data)
    else:
        data = data.copy()
    # uint8 (h, w), (h, w, 3) and (h, w, 4) arrays open as L, RGB and RGBA.
    return Image.fromarray(np.ascontiguousarray(data))


def to_8bit(data: np.ndarray, sixteen_bit: bool = False) -> np.ndarray:
    """Bring a numeric array onto the 0-255 scale without changing its contrast.

    Four conventions meet here, and each is mapped by its own fixed scale, so
    a faded page stays faded however it arrives: floats in 0-1 are multiplied
    by 255, anything already in 0-255 is kept, and 16-bit data - a ``uint16``
    array, a 16-bit Pillow mode, or any values in 0-65535 - is divided by 257,
    which takes 65535 to exactly 255. Stretching the range actually used onto
    0-255 instead would turn ink at 200 on paper at 245 into black on white,
    and pass a page OCR will struggle with. Only data on no recognisable scale
    at all - negative values, or beyond 16 bits - has its used range
    stretched, because there is no other scale to read it on.

    ``sixteen_bit`` says the data is 16-bit whatever values it holds.

    Raises:
        TypeError: if the data is complex or not numeric at all.
    """
    kind = data.dtype.kind
    if kind == "c":
        raise TypeError("complex arrays are not images; pass the magnitude or real part")
    if kind == "b":
        return data.astype(np.uint8) * np.uint8(255)
    if kind not in "uif":
        raise TypeError(
            "array images must hold numbers, not {0}".format(data.dtype)
        )
    if kind == "u" and data.dtype.itemsize == 2:
        sixteen_bit = True
    floats = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(floats)
    values = floats[finite]
    low = float(values.min()) if values.size else 0.0
    high = float(values.max()) if values.size else 0.0
    floats = np.where(finite, floats, low)
    if sixteen_bit and low >= 0.0 and high <= 65535.0:
        floats = floats / 257.0
    elif low >= 0.0 and high <= 1.0:
        floats = floats * 255.0
    elif low >= 0.0 and high <= 255.0:
        pass
    elif low >= 0.0 and high <= 65535.0:
        floats = floats / 257.0
    else:
        span = high - low
        logger.warning(
            "image values run from %g to %g, which is no standard scale; "
            "stretching them onto 0-255", low, high,
        )
        floats = (floats - low) * (255.0 / span) if span > 1e-12 else floats * 0.0
    return np.clip(np.rint(floats), 0.0, 255.0).astype(np.uint8)


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
    """16-bit, 32-bit and float scans as 8-bit, each on its own fixed scale.

    See :func:`to_8bit`: a 16-bit mode is divided by 257, a float mode is read
    as 0-1 or 0-255, and nothing is stretched unless it sits on no known scale.
    """
    data = np.asarray(image)
    return Image.fromarray(to_8bit(data, sixteen_bit=image.mode.startswith("I;16")))


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
    small = Image.fromarray(np.ascontiguousarray(lum, dtype=np.uint8)).resize(target, _BOX)
    plane = np.asarray(small, dtype=np.float32) / np.float32(255.0)
    return plane, small.size[0] / float(width)


def _shrink_plane(plane: np.ndarray, scale: float, long_edge: int) -> Tuple[np.ndarray, float]:
    """``plane`` (already at ``scale`` of native) box-filtered down to ``long_edge``.

    Shrinking the work plane rather than the native image gives the same
    small plane for a fraction of the reading.
    """
    height, width = plane.shape
    longest = max(height, width)
    if long_edge <= 0 or longest <= long_edge:
        return plane.copy(), scale
    ratio = long_edge / float(longest)
    target = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    small = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32)).resize(target, _BOX)
    return np.asarray(small, dtype=np.float32).copy(), scale * small.size[0] / float(width)


def colour_spread(image: Image.Image) -> float:
    """How far from neutral grey this page is: 0 a grey scan, 1 fully saturated.

    Measured on a small thumbnail as the mean of ``max(r,g,b) - min(r,g,b)``,
    which paper and ink keep near zero even when the scanner was set to colour,
    and a photograph does not.
    """
    if image.mode in ("1", "L", "LA", "I", "F") or image.mode.startswith("I;16"):
        return 0.0                      # no colour channels, nothing to spread
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
    found = float(sum(values) / len(values))
    # PNG stores whole pixels per metre, so 300 dpi comes back as 299.9994 and
    # 200 dpi as 199.9996 - just under the floor it was saved at. Anything
    # within a twentieth of a dot of a whole number is that whole number.
    nearest = float(round(found))
    if abs(found - nearest) <= DPI_SNAP:
        found = nearest
    return found


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

    #: ``(left, top, right, bottom)`` in native pixels when the planes have been
    #: cropped to the inside of a scanner border, else ``None``.
    interior: Optional[Tuple[int, int, int, int]] = None
    #: Share of the image that was black lid outside a crooked sheet, painted
    #: over with paper before measuring. 0 when there was none.
    outside_share: float = 0.0

    @property
    def megapixels(self) -> float:
        """Page area in megapixels."""
        return (self.width * self.height) / 1e6

    def turned(self, quarter_turns: int) -> "PagePlanes":
        """These planes turned ``quarter_turns`` x 90 degrees counter-clockwise.

        Used to measure a page lying on its side as the upright page it will
        be once rotated. Width, height and dpi describe the file and are kept.
        """
        k = int(quarter_turns) % 4
        if k == 0:
            return self
        return dataclasses.replace(
            self,
            lum=np.ascontiguousarray(np.rot90(self.lum, k)),
            work=np.ascontiguousarray(np.rot90(self.work, k)),
            fine=np.ascontiguousarray(np.rot90(self.fine, k)),
        )

    def cropped(self, box: Tuple[int, int, int, int]) -> "PagePlanes":
        """These planes cut down to ``box`` (native ``left, top, right, bottom``).

        Width, height and dpi stay those of the whole image, because they are
        facts about the scan; only the pixels measured change. Every array is a
        fresh copy, never a view.
        """
        left, top, right, bottom = (int(value) for value in box)
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.lum.shape[1], right), min(self.lum.shape[0], bottom)
        if right - left < 2 or bottom - top < 2:
            return self

        def cut(plane: np.ndarray, scale: float) -> np.ndarray:
            rows, columns = plane.shape
            y0 = min(rows - 1, int(np.floor(top * scale)))
            x0 = min(columns - 1, int(np.floor(left * scale)))
            y1 = max(y0 + 1, min(rows, int(np.ceil(bottom * scale))))
            x1 = max(x0 + 1, min(columns, int(np.ceil(right * scale))))
            return plane[y0:y1, x0:x1].copy()

        return PagePlanes(
            width=self.width,
            height=self.height,
            lum=self.lum[top:bottom, left:right].copy(),
            work=cut(self.work, self.work_scale),
            work_scale=self.work_scale,
            fine=cut(self.fine, self.fine_scale),
            fine_scale=self.fine_scale,
            colour=self.colour,
            dpi=self.dpi,
            dpi_source=self.dpi_source,
            exif_applied=self.exif_applied,
            interior=(left, top, right, bottom),
        )


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
    fine, fine_scale = _shrink_plane(work, work_scale, SKEW_LONG_EDGE)

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
