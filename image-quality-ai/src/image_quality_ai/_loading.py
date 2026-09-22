"""Turn whatever the caller hands over into pixels the measures can trust.

A path, a PIL image and a numpy array all end up in the same place: one
full-resolution luminance plane for the statistics that need every pixel, and
one plane resampled to the fixed analysis size for the spatial measures, so a
12-megapixel photo, a thumbnail of it and a 224px crop of it are judged on the
same scale.

Anything that can go wrong with a file goes wrong here, in one place, with the
path in the message.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from ._thresholds import ANALYSIS_LONG_EDGE, ANALYSIS_MIN_EDGE

#: Rec. 601 luma weights, the same ones Pillow uses for ``convert("L")``. Every
#: input path uses these, so a photo measured from disk and the same photo
#: measured as an array give identical numbers.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)

try:                                        # Pillow >= 9.1
    _BOX = Image.Resampling.BOX
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:                      # pragma: no cover - older Pillow
    _BOX = Image.BOX
    _NEAREST = Image.NEAREST

#: Suffixes the CLI picks up when it is pointed at a directory.
IMAGE_SUFFIXES = (
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".gif", ".tif", ".tiff",
    ".webp", ".ppm", ".pgm", ".pbm", ".ico",
)

#: PIL modes that carry more than 8 bits per channel.
_HIGH_DEPTH_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N")


@dataclass
class LoadedImage:
    """One image, ready to measure.

    Attributes:
        luminance: full-resolution luminance, float32 in 0.0 to 1.0.
        analysis: the same plane resampled to the analysis size, down or up. The
            spatial measures work here so their numbers do not depend on how
            many megapixels the camera happened to have.
        width: pixel width after EXIF orientation was applied.
        height: pixel height after EXIF orientation was applied.
        channels: 1 for greyscale, 3 for colour, 4 when alpha was present.
        source: a label for messages, usually the file path.
        bit_depth: bits per channel in the source, 8 or 16 for normal images.
        mode: the PIL mode, or ``array[dtype]`` for a numpy input.
        orientation_applied: whether an EXIF orientation tag was acted on.
        notes: anything the caller should know about how this was read.
    """

    luminance: np.ndarray
    analysis: np.ndarray
    width: int
    height: int
    channels: int
    source: str
    bit_depth: int = 8
    mode: str = "L"
    orientation_applied: bool = False
    notes: Tuple[str, ...] = ()

    @property
    def pixels(self) -> int:
        """Total pixel count of the full-resolution image."""
        return int(self.width) * int(self.height)

    def info(self) -> Dict[str, Any]:
        """JSON-safe description of what was read."""
        return {
            "source": self.source,
            "width": int(self.width),
            "height": int(self.height),
            "pixels": self.pixels,
            "channels": int(self.channels),
            "bit_depth": int(self.bit_depth),
            "mode": self.mode,
            "orientation_applied": bool(self.orientation_applied),
            "analysis_size": [int(self.analysis.shape[1]), int(self.analysis.shape[0])],
            "notes": list(self.notes),
        }


def _analysis_plane(luminance: np.ndarray, long_edge: int = ANALYSIS_LONG_EDGE) -> np.ndarray:
    """Put the luminance on the fixed analysis grid, whichever side it starts on.

    Coming down, the plane is area-averaged. Going up it is pixel-replicated,
    which invents no detail: it restates the same picture on the one grid every
    image is measured on. That matters because second-difference energy is
    counted per pixel, and a small frame spreads the same edges over fewer
    pixels, so the same picture at 128px carried five times the Laplacian
    variance of itself at 512px and flipped ``is_blurry`` on its way down.
    Replication dilutes that energy by the factor the shrinking added, so a
    thumbnail, a 224px crop and the 12-megapixel original all report one number.

    A plane with fewer than :data:`ANALYSIS_MIN_EDGE` pixels on a side has no
    interior to measure, so it is left exactly as it is and the measures report
    that they could not judge it, rather than a replicated blank scoring zero.
    """
    height, width = luminance.shape
    longest = max(width, height)
    if longest == long_edge:
        return luminance
    if height < ANALYSIS_MIN_EDGE or width < ANALYSIS_MIN_EDGE:
        return luminance
    scale = float(long_edge) / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    plane = Image.fromarray(np.ascontiguousarray(luminance, dtype=np.float32), mode="F")
    resample = _BOX if longest > long_edge else _NEAREST
    return np.asarray(plane.resize((new_width, new_height), resample), dtype=np.float32)


def _integer_scale(array: np.ndarray, hint: Optional[float]) -> float:
    """Divisor that puts an integer array on a 0.0 to 1.0 scale.

    A PIL mode knows its own depth, so when the caller passes a hint it wins. A
    bare numpy array does not, so for anything other than uint8 and uint16 the
    largest value present decides, which is documented in the README.
    """
    if hint is not None:
        return float(hint)
    if array.dtype in (np.dtype(np.uint8), np.dtype(np.int8)):
        return 255.0
    if array.dtype in (np.dtype(np.uint16), np.dtype(np.int16)):
        return 65535.0
    peak = float(array.max()) if array.size else 0.0
    if peak <= 1.0:
        return 1.0
    if peak <= 255.0:
        return 255.0
    if peak <= 65535.0:
        return 65535.0
    return peak


def _normalise_float(array: np.ndarray) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Put a float array on a 0.0 to 1.0 scale, reading the convention used.

    Always returns an array this module owns, so nothing downstream can write
    back into the caller's data.
    """
    if array.size and not np.isfinite(array).all():
        raise ValueError(
            "image array contains NaN or infinite values; clean them before assessing"
        )
    data = array.astype(np.float32, copy=False)
    low = float(data.min()) if data.size else 0.0
    high = float(data.max()) if data.size else 0.0
    if low < -0.01 and high <= 1.05:
        # The -1..1 convention that image models are usually fed.
        return (
            (data + np.float32(1.0)) * np.float32(0.5),
            ("float pixels read as the -1..1 convention",),
        )
    if high <= 1.05:
        return np.array(data, dtype=np.float32), ()
    if high <= 255.0:
        return data / np.float32(255.0), ("float pixels read as a 0-255 scale",)
    return (
        data / np.float32(high),
        ("float pixels rescaled by their maximum, {0:g}".format(high),),
    )


def _weighted_rgb(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance of an HxWx3 block, left on the input's own scale."""
    red, green, blue = LUMA_WEIGHTS
    out = rgb[:, :, 0].astype(np.float32) * np.float32(red)
    out += rgb[:, :, 1].astype(np.float32) * np.float32(green)
    out += rgb[:, :, 2].astype(np.float32) * np.float32(blue)
    return out


def _luminance_from_array(
    array: np.ndarray, scale_hint: Optional[float] = None
) -> Tuple[np.ndarray, int, Tuple[str, ...]]:
    """Luminance plane, channel count and notes for a numpy image."""
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in (2, 3):
        raise ValueError(
            "image array must be HxW or HxWx3 (HxWx4 with alpha), got shape {0}".format(
                tuple(array.shape)
            )
        )
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError(
            "image array must have 3 or 4 channels, got {0} (shape {1})".format(
                array.shape[2], tuple(array.shape)
            )
        )
    if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("image has no pixels (shape {0})".format(tuple(array.shape)))

    notes: Tuple[str, ...] = ()
    if array.dtype == np.bool_:
        data: np.ndarray = array.astype(np.float32)
        scale = 1.0
    elif np.issubdtype(array.dtype, np.floating):
        data, notes = _normalise_float(array)
        scale = 1.0
    elif np.issubdtype(array.dtype, np.integer):
        data = array
        scale = _integer_scale(array, scale_hint)
    else:
        raise ValueError(
            "image array has unsupported dtype {0}; use an integer or float array".format(
                array.dtype
            )
        )

    channels = 1 if data.ndim == 2 else int(data.shape[2])
    if data.ndim == 2:
        luminance = data.astype(np.float32, copy=True)
    elif channels == 4:
        # Composite over white, the way a viewer shows it, so a transparent PNG
        # is judged on what a person would actually see.
        alpha = data[:, :, 3].astype(np.float32) / np.float32(scale)
        np.clip(alpha, 0.0, 1.0, out=alpha)
        luminance = _weighted_rgb(data[:, :, :3])
        luminance *= alpha
        luminance += np.float32(scale) * (np.float32(1.0) - alpha)
        if float(alpha.min()) < 1.0:
            notes = notes + ("alpha channel composited over white",)
    else:
        luminance = _weighted_rgb(data)

    if scale != 1.0:
        luminance /= np.float32(scale)
    np.clip(luminance, 0.0, 1.0, out=luminance)
    return luminance, channels, notes


def _bit_depth_of(array: np.ndarray, mode: str) -> int:
    if mode in _HIGH_DEPTH_MODES or array.dtype in (np.dtype(np.uint16), np.dtype(np.int16)):
        return 16
    if mode == "F" or np.issubdtype(array.dtype, np.floating):
        return 32
    if mode == "1" or array.dtype == np.bool_:
        return 1
    return 8


def _prepare_pil(image: Image.Image) -> Tuple[Image.Image, Tuple[str, ...], Optional[float]]:
    """Get an image into a mode numpy can read, and say what scale it is on."""
    mode = image.mode
    if mode in ("P", "PA"):
        return image.convert("RGBA" if mode == "PA" else "RGB"), (
            "palette image expanded to RGB",
        ), None
    if mode == "1":
        return image.convert("L"), ("1-bit image read as black and white",), None
    if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        return image.convert("I"), ("16-bit image read at full depth",), 65535.0
    if mode == "I":
        return image, ("16-bit image read at full depth",), 65535.0
    if mode == "LA":
        return image.convert("RGBA"), (), None
    if mode in ("CMYK", "YCbCr", "LAB", "HSV"):
        return image.convert("RGB"), (
            "{0} image converted to RGB".format(mode),
        ), None
    return image, (), None


def _from_pil(image: Image.Image, source: str) -> LoadedImage:
    """Load an already-open PIL image, applying EXIF orientation first."""
    original_mode = image.mode
    orientation_applied = False
    try:
        exif = image.getexif()
        orientation = exif.get(274) if exif else None
        rotated = ImageOps.exif_transpose(image)
        if rotated is not None:
            orientation_applied = orientation is not None and int(orientation) > 1
            image = rotated
    except Exception:                       # pragma: no cover - malformed EXIF
        orientation_applied = False

    image, notes, scale_hint = _prepare_pil(image)
    array = np.asarray(image)
    luminance, channels, more = _luminance_from_array(array, scale_hint)
    height, width = luminance.shape
    return LoadedImage(
        luminance=luminance,
        analysis=_analysis_plane(luminance),
        width=int(width),
        height=int(height),
        channels=channels,
        source=source,
        bit_depth=_bit_depth_of(array, original_mode),
        mode=original_mode,
        orientation_applied=orientation_applied,
        notes=notes + more,
    )


def _tidy_detail(detail: str, path: str) -> str:
    """Drop the filename Pillow repeats inside its own message.

    The caller already gets the path at the front of our message; seeing it a
    second time, escaped, only makes the error harder to read.
    """
    for form in (" " + repr(path), repr(path), " " + path, path):
        if form and form in detail:
            detail = detail.replace(form, "")
    detail = detail.strip().strip(":").strip()
    return detail or "the file is not a readable image"


def _from_path(path: str) -> LoadedImage:
    """Load from disk, with every failure named after the file."""
    if not os.path.exists(path):
        raise FileNotFoundError("no such image file: {0}".format(path))
    if os.path.isdir(path):
        raise ValueError("{0} is a directory, not an image file".format(path))
    try:
        with Image.open(path) as handle:
            handle.load()
            return _from_pil(handle, path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        detail = _tidy_detail(str(exc).strip() or type(exc).__name__, path)
        raise ValueError("cannot read image {0}: {1}".format(path, detail)) from None


def load_image(image: Any, *, source: Optional[str] = None) -> LoadedImage:
    """Read a path, a PIL image or a numpy array into a :class:`LoadedImage`.

    Args:
        image: a path (``str`` or ``os.PathLike``), a ``PIL.Image.Image``, or a
            numpy array shaped HxW, HxWx3 or HxWx4.
        source: label to use in messages instead of the default.

    Returns:
        The loaded image, with EXIF orientation already applied.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file is corrupt or truncated, the array has a shape or
            dtype that is not an image, or the image has no pixels.
        TypeError: the argument is not a path, a PIL image or an array.
    """
    if isinstance(image, LoadedImage):
        return image
    if isinstance(image, Image.Image):
        return _from_pil(image, source or "<PIL.Image>")
    if isinstance(image, np.ndarray):
        luminance, channels, notes = _luminance_from_array(image)
        height, width = luminance.shape
        return LoadedImage(
            luminance=luminance,
            analysis=_analysis_plane(luminance),
            width=int(width),
            height=int(height),
            channels=channels,
            source=source or "<array>",
            bit_depth=_bit_depth_of(image, ""),
            mode="array[{0}]".format(image.dtype),
            notes=notes,
        )
    if isinstance(image, (str, bytes, os.PathLike)):
        path = os.fspath(image)
        if isinstance(path, bytes):
            path = path.decode("utf-8", errors="replace")
        return _from_path(path)
    raise TypeError(
        "image must be a path, a PIL.Image.Image or a numpy array, got "
        + type(image).__name__
    )


def label_for(image: Any) -> str:
    """The name to show for an input before it has been read."""
    if isinstance(image, LoadedImage):
        return image.source
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    if isinstance(image, bytes):
        return image.decode("utf-8", errors="replace")
    if isinstance(image, Image.Image):
        return "<PIL.Image>"
    if isinstance(image, np.ndarray):
        return "<array {0}>".format("x".join(str(n) for n in image.shape))
    return "<{0}>".format(type(image).__name__)
