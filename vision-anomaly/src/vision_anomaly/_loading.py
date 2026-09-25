"""Turn whatever the caller hands over into one comparable analysis image.

A path, a PIL image and a numpy array all end up in the same place: an RGB
plane and a luminance plane, both float32 on a 0.0 to 1.0 scale, both resampled
onto one fixed square analysis grid. That grid is what makes a 4000x3000 photo
and a 640x480 frame of the same scene comparable at all - every feature in
:mod:`vision_anomaly._features` is counted on it.

Everything that can go wrong with an input goes wrong here, in one place, with
the path or label in the message.

Nothing in this module ever writes into an array the caller owns. Pillow hands
back read-only views and numpy arrays belong to whoever passed them, so every
plane that leaves here is freshly allocated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

#: Edge of the square grid every image is measured on. 256 leaves a 64x64 pixel
#: cell under the default 4x4 grid, which is enough pixels for a gradient
#: histogram to mean anything, and small enough that a thousand images take a
#: second or two.
ANALYSIS_SIZE = 256

#: Rec. 601 luma weights, the same ones Pillow uses for ``convert("L")``, so an
#: image measured from disk and the same image measured as an array agree.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)

#: Suffixes the CLI picks up when it is pointed at a directory.
IMAGE_SUFFIXES = (
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".gif", ".tif", ".tiff",
    ".webp", ".ppm", ".pgm", ".pbm", ".ico",
)

try:                                        # Pillow >= 9.1
    _BOX = Image.Resampling.BOX
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:                      # pragma: no cover - older Pillow
    _BOX = Image.BOX
    _BILINEAR = Image.BILINEAR


@dataclass
class LoadedImage:
    """One image on the analysis grid, ready for feature extraction.

    Attributes:
        rgb: HxWx3 float32 in 0.0 to 1.0, on the analysis grid. Greyscale input
            is carried in all three channels; RGBA is composited over white,
            which is what a viewer shows.
        luminance: HxW float32 in 0.0 to 1.0, the Rec. 601 luma of ``rgb``.
        width: pixel width of the source, after EXIF orientation.
        height: pixel height of the source, after EXIF orientation.
        channels: 1 for greyscale, 3 for colour, 4 when alpha was present.
        source: the label used in messages, usually the file path.
        mode: the PIL mode, or ``array[dtype]`` for a numpy input.
        resized: whether the source had to be resampled onto the analysis grid.
        notes: anything the caller should know about how this was read.
        value_scale: the divisor that put integer pixels on the 0..1 scale, or
            ``None`` when the source was float or boolean. A profile records
            this so later images are read the same way it was fitted.
    """

    rgb: np.ndarray
    luminance: np.ndarray
    width: int
    height: int
    channels: int
    source: str
    mode: str = "L"
    resized: bool = False
    notes: Tuple[str, ...] = ()
    value_scale: Optional[float] = None

    @property
    def size(self) -> Tuple[int, int]:
        """Source ``(width, height)`` in pixels."""
        return (int(self.width), int(self.height))

    def info(self) -> Dict[str, Any]:
        """JSON-safe description of what was read."""
        return {
            "source": self.source,
            "width": int(self.width),
            "height": int(self.height),
            "channels": int(self.channels),
            "mode": self.mode,
            "resized": bool(self.resized),
            "analysis_size": int(self.rgb.shape[0]),
            "notes": list(self.notes),
            "value_scale": None if self.value_scale is None else float(self.value_scale),
        }


#: Full-scale value of the integer containers that carry a picture, by item
#: size. Used only to notice that an array's values do not fill its container,
#: never to decide the divisor - see :func:`_scale_for` for why.
_DTYPE_CAPACITY = {1: 255.0, 2: 65535.0}


def _dtype_capacity(dtype: np.dtype) -> Optional[float]:
    """Full scale of an 8 or 16-bit integer container, else ``None``."""
    if not np.issubdtype(dtype, np.integer):
        return None
    return _DTYPE_CAPACITY.get(int(np.dtype(dtype).itemsize))


def _scale_for(array: np.ndarray) -> float:
    """Divisor that puts an integer array on a 0.0 to 1.0 scale.

    Read from what the values *are*, never from the container they arrived in.
    A uint16 array holding 0..255 is an 8-bit picture someone widened, and
    dividing it by 65535 would land it at a 257th of its real brightness. That
    mistake is invisible: every feature stays a legal 0..1 number, so nothing
    raises, but the whole vector shrinks below
    :data:`vision_anomaly._profile.NOISE_FLOOR` and real faults stop counting.
    The same pixel values must therefore reach the same place whichever integer
    dtype carries them, which is what this function guarantees.
    """
    peak = float(array.max()) if array.size else 0.0
    if peak <= 1.0:
        return 1.0
    if peak <= 255.0:
        return 255.0
    if peak <= 65535.0:
        return 65535.0
    return peak


def _integer_notes(
    array: np.ndarray, scale: float, hinted: bool
) -> Tuple[str, ...]:
    """What the caller should know about how an integer array was read.

    ``hinted`` means the scale came from a fitted profile rather than from these
    values. Then the interesting thing is not which scale was chosen but whether
    this image looks like it belongs on it: a frame whose own values say 0-65535
    read against a profile fitted at 0-255 will saturate, and the user deserves
    to be told that before they act on the score.
    """
    notes: List[str] = []
    if array.size and int(array.min()) < 0:
        notes.append(
            "this {0} array holds negative pixel values; they were clipped to "
            "zero, so whatever produced them has already lost that part of the "
            "picture".format(array.dtype)
        )
    if hinted:
        own = _scale_for(array)
        if own != scale:
            notes.append(
                "this image's values fill 0-{0:g}, but it was read on the 0-{1:g} "
                "scale the known-good set was fitted on, so that the two stay "
                "comparable".format(own, scale)
            )
        return tuple(notes)
    capacity = _dtype_capacity(array.dtype)
    if capacity is not None and scale < capacity:
        notes.append(
            "this {0} array only uses 0-{1:g} of its container, so it was read "
            "on that scale rather than 0-{2:g}".format(array.dtype, scale, capacity)
        )
    return tuple(notes)


def _normalise_float(array: np.ndarray) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Put a float array on a 0.0 to 1.0 scale, reading the convention used.

    Always returns an array this module owns, never a view of the caller's.
    """
    if array.size and not np.isfinite(array).all():
        raise ValueError(
            "image array contains NaN or infinite values; clean them before scoring"
        )
    data = np.array(array, dtype=np.float32)
    low = float(data.min()) if data.size else 0.0
    high = float(data.max()) if data.size else 0.0
    if low < -0.01 and high <= 1.05:
        # The -1..1 convention that image models are usually fed.
        data += np.float32(1.0)
        data *= np.float32(0.5)
        return data, ("float pixels read as the -1..1 convention",)
    if high <= 1.05:
        return data, ()
    if high <= 255.0:
        data /= np.float32(255.0)
        return data, ("float pixels read as a 0-255 scale",)
    data /= np.float32(high)
    return data, ("float pixels rescaled by their maximum, {0:g}".format(high),)


def _rgb_from_array(
    array: np.ndarray, scale_hint: Optional[float] = None
) -> Tuple[np.ndarray, int, Tuple[str, ...], Optional[float]]:
    """HxWx3 float32 in 0..1, the channel count, any notes, and the divisor.

    ``scale_hint`` is authoritative when given: it is the scale a fitted profile
    was built at, and scoring against that profile has to read new images the
    same way or the comparison is between two different pictures. The divisor
    that was actually used comes back as the fourth item so a profile can record
    it; it is ``None`` when the input was not an integer array, where there is
    nothing to guess.

    The returned array is always newly allocated, so the caller's pixels are
    never written into and never aliased.
    """
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in (2, 3):
        raise ValueError(
            "image array must be HxW, HxWx3 or HxWx4, got shape {0}".format(
                tuple(array.shape)
            )
        )
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError(
            "image array must have 1, 3 or 4 channels, got {0} (shape {1})".format(
                array.shape[2], tuple(array.shape)
            )
        )
    if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("image has no pixels (shape {0})".format(tuple(array.shape)))

    notes: Tuple[str, ...] = ()
    used_scale: Optional[float] = None
    if array.dtype == np.bool_:
        data = array.astype(np.float32)
    elif np.issubdtype(array.dtype, np.floating):
        data, notes = _normalise_float(array)
    elif np.issubdtype(array.dtype, np.integer):
        hinted = scale_hint is not None and float(scale_hint) > 0.0
        scale = float(scale_hint) if hinted else _scale_for(array)
        used_scale = float(scale)
        notes = notes + _integer_notes(array, used_scale, hinted)
        data = array.astype(np.float32) / np.float32(scale or 1.0)
    else:
        raise ValueError(
            "image array has unsupported dtype {0}; use an integer or float array".format(
                array.dtype
            )
        )

    channels = 1 if data.ndim == 2 else int(data.shape[2])
    if channels == 1:
        rgb = np.repeat(data[:, :, None], 3, axis=2)
    elif channels == 4:
        alpha = np.clip(data[:, :, 3], 0.0, 1.0)[:, :, None]
        rgb = data[:, :, :3] * alpha + (np.float32(1.0) - alpha)
        if float(alpha.min()) < 1.0:
            notes = notes + ("alpha channel composited over white",)
    else:
        rgb = data

    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb, channels, notes, used_scale


def _to_analysis_grid(rgb: np.ndarray, size: int) -> Tuple[np.ndarray, bool]:
    """Resample an HxWx3 plane onto the square analysis grid.

    Coming down, the plane is area-averaged, which keeps a shrunk photograph's
    statistics close to the original's. Going up it is bilinear. The grid is
    square on purpose: two frames of the same subject at different aspect ratios
    then line up cell for cell, so "the top left of the picture" means one thing
    across a mixed set. The stretch that costs is recorded as a note rather than
    hidden.

    Each channel is resampled on its own through Pillow's single-channel "F"
    mode, so the float precision survives; going through 8-bit RGB would quantise
    every pixel on the way in and out.
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height == size and width == size:
        return rgb, False
    shrinking = width > size or height > size
    resample = _BOX if shrinking else _BILINEAR
    out = np.empty((size, size, 3), dtype=np.float32)
    for index in range(3):
        band = Image.fromarray(
            np.ascontiguousarray(rgb[:, :, index], dtype=np.float32), mode="F"
        )
        out[:, :, index] = np.asarray(
            band.resize((size, size), resample), dtype=np.float32
        )
    np.clip(out, 0.0, 1.0, out=out)
    return out, True


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance of an HxWx3 plane, as a fresh HxW array."""
    red, green, blue = LUMA_WEIGHTS
    out = rgb[:, :, 0] * np.float32(red)
    out = out + rgb[:, :, 1] * np.float32(green)
    out = out + rgb[:, :, 2] * np.float32(blue)
    return np.ascontiguousarray(out, dtype=np.float32)


def _prepare_pil(image: Image.Image) -> Tuple[Image.Image, Tuple[str, ...], Optional[float]]:
    """Get a PIL image into a mode numpy can read, and say what scale it is on."""
    mode = image.mode
    if mode in ("P", "PA"):
        return image.convert("RGBA" if mode == "PA" else "RGB"), (
            "palette image expanded to RGB",
        ), None
    if mode == "1":
        return image.convert("L"), ("1-bit image read as black and white",), None
    if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        return image.convert("I"), ("16-bit image read on the scale its values use",), None
    if mode == "I":
        return image, ("deep image read on the scale its values use",), None
    if mode == "LA":
        return image.convert("RGBA"), (), None
    if mode in ("CMYK", "YCbCr", "LAB", "HSV"):
        return image.convert("RGB"), ("{0} image converted to RGB".format(mode),), None
    return image, (), None


def _from_pil(
    image: Image.Image,
    source: str,
    size: int,
    value_scale: Optional[float] = None,
) -> LoadedImage:
    """Load an already-open PIL image, applying EXIF orientation first."""
    original_mode = image.mode
    try:
        rotated = ImageOps.exif_transpose(image)
        if rotated is not None:
            image = rotated
    except Exception:                       # pragma: no cover - malformed EXIF
        pass

    image, notes, scale_hint = _prepare_pil(image)
    if value_scale is not None:
        scale_hint = value_scale
    array = np.asarray(image)               # may be read-only; never written into
    rgb, channels, more, used_scale = _rgb_from_array(array, scale_hint)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    grid, resized = _to_analysis_grid(rgb, size)
    return LoadedImage(
        rgb=grid,
        luminance=_luma(grid),
        width=width,
        height=height,
        channels=channels,
        source=source,
        mode=original_mode,
        resized=resized,
        notes=notes + more,
        value_scale=used_scale,
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


def _from_path(
    path: str, size: int, value_scale: Optional[float] = None
) -> LoadedImage:
    """Load from disk, with every failure named after the file."""
    if not os.path.exists(path):
        raise ValueError("no such image file: {0}".format(path))
    if os.path.isdir(path):
        raise ValueError("{0} is a directory, not an image file".format(path))
    try:
        with Image.open(path) as handle:
            handle.load()
            return _from_pil(handle, path, size, value_scale)
    except ValueError as exc:
        text = str(exc)
        if text.startswith("cannot read image ") or text.startswith("no such image file"):
            raise
        raise ValueError(
            "cannot read image {0}: {1}".format(path, _tidy_detail(text, path))
        ) from None
    except Exception as exc:
        detail = _tidy_detail(str(exc).strip() or type(exc).__name__, path)
        raise ValueError("cannot read image {0}: {1}".format(path, detail)) from None


def load_image(
    image: Any,
    *,
    source: Optional[str] = None,
    size: int = ANALYSIS_SIZE,
    value_scale: Optional[float] = None,
) -> LoadedImage:
    """Read a path, a PIL image or a numpy array onto the analysis grid.

    Args:
        image: a path (``str`` or ``os.PathLike``), a ``PIL.Image.Image``, or a
            numpy array shaped HxW, HxWx1, HxWx3 or HxWx4.
        source: label to use in messages instead of the default.
        size: edge of the square analysis grid.
        value_scale: divisor to put integer pixels on the 0..1 scale, instead of
            reading it from the values. A fitted profile passes the divisor it
            was built with, so a 16-bit frame and an 8-bit one are never
            compared on two different scales.

    Returns:
        The loaded image, with EXIF orientation applied and pixels this library
        owns. The caller's array or image is never modified.

    Raises:
        ValueError: the path does not exist, the file is corrupt or truncated,
            or the array has a shape or dtype that is not an image. The message
            always names the path.
        TypeError: the argument is not a path, a PIL image or an array.
    """
    if isinstance(image, LoadedImage):
        return image
    if isinstance(image, Image.Image):
        return _from_pil(image, source or "<PIL.Image>", size, value_scale)
    if isinstance(image, np.ndarray):
        rgb, channels, notes, used_scale = _rgb_from_array(image, value_scale)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        grid, resized = _to_analysis_grid(rgb, size)
        return LoadedImage(
            rgb=grid,
            luminance=_luma(grid),
            width=width,
            height=height,
            channels=channels,
            source=source or "<array>",
            mode="array[{0}]".format(image.dtype),
            resized=resized,
            notes=notes,
            value_scale=used_scale,
        )
    if isinstance(image, (str, bytes, os.PathLike)):
        path = os.fspath(image)
        if isinstance(path, bytes):
            path = path.decode("utf-8", errors="replace")
        return _from_path(path, size, value_scale)
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


def iter_images(target: str, *, recursive: bool = False) -> List[str]:
    """Every image file under ``target``, sorted, for the CLI.

    A file path comes back as itself, whatever its suffix, so an explicitly
    named file is never silently skipped.
    """
    if os.path.isfile(target):
        return [target]
    if not os.path.isdir(target):
        raise ValueError("no such file or directory: {0}".format(target))
    found: List[str] = []
    if recursive:
        for root, dirs, names in os.walk(target):
            dirs.sort()
            for name in sorted(names):
                if name.lower().endswith(IMAGE_SUFFIXES):
                    found.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            if os.path.isfile(full) and name.lower().endswith(IMAGE_SUFFIXES):
                found.append(full)
    return found
