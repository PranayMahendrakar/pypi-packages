"""Getting any image into one predictable shape without touching the original.

Everything becomes a float32 array ``(height, width, channels)`` on a 0-255
scale: greyscale has one channel, RGB three. An alpha channel is kept as an
extra channel and the colour channels are premultiplied by it, so fully
transparent pixels all look the same whatever colour an encoder left in them.
The caller's array or image is only ever read: every path below copies.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".pgm", ".tif", ".tiff", ".webp",
}


@dataclass
class Pixels:
    """An image ready for measuring."""

    data: np.ndarray            # (h, w, c) float32, 0-255
    valid: np.ndarray           # (h, w) bool, False where the source held NaN
    kind: str                   # "grey", "grey+alpha", "rgb", "rgba"
    notes: List[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def empty(self) -> bool:
        return self.data.size == 0


def looks_like_image_path(path: Any) -> bool:
    """True when ``path`` has a suffix Pillow normally reads."""
    return os.path.splitext(str(path))[1].lower() in IMAGE_SUFFIXES


def open_source(image: Any) -> Tuple[Any, Optional[Image.Image]]:
    """Return ``(source, opened)``: a path is opened (upright, per EXIF), anything else passes."""
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path!r} does not exist")
        with Image.open(path) as handle:
            handle.load()
            opened = ImageOps.exif_transpose(handle)
            if opened is handle:
                opened = handle.copy()
        return opened, opened
    return image, None


def to_pixels(image: Any) -> Pixels:
    """Convert a PIL image, numpy array or nested list into :class:`Pixels`."""
    if isinstance(image, Image.Image):
        return _from_pil(image)
    if isinstance(image, (str, os.PathLike)):
        opened, _ = open_source(image)
        return _from_pil(opened)
    if isinstance(image, np.ndarray):
        return _from_array(image)
    if isinstance(image, (list, tuple)):
        return _from_array(np.array(image))
    raise TypeError(
        "image must be a PIL.Image, a file path or a numpy array, "
        f"not {type(image).__name__}"
    )


def _empty(kind: str = "grey") -> Pixels:
    return Pixels(np.zeros((0, 0, 1), dtype=np.float32), np.zeros((0, 0), dtype=bool), kind,
                  ["the image has no pixels"])


def _from_pil(image: Image.Image) -> Pixels:
    width, height = image.size
    if width == 0 or height == 0:
        return _empty()
    mode = image.mode
    if mode in ("1", "L", "P", "LA", "PA", "RGB", "RGBA", "RGBa", "La", "CMYK", "YCbCr", "LAB", "HSV"):
        has_alpha = mode in ("LA", "PA", "RGBA", "RGBa", "La") or (
            mode == "P" and "transparency" in image.info
        )
        grey = mode in ("1", "L", "LA", "La") or (
            mode in ("P", "PA") and _palette_is_grey(image)
        )
        target = ("LA" if has_alpha else "L") if grey else ("RGBA" if has_alpha else "RGB")
        array = np.array(image.convert(target), dtype=np.float32, copy=True)
        return _from_array(array, scale_hint="uint8")
    # 16-bit, 32-bit integer and float modes.
    array = np.array(image, copy=True)
    return _from_array(array)


def _palette_is_grey(image: Image.Image) -> bool:
    try:
        rgb = np.array(image.convert("RGB"), dtype=np.int16, copy=True)
    except (OSError, ValueError):
        return False
    return bool(np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]))


def _from_array(array: np.ndarray, scale_hint: Optional[str] = None) -> Pixels:
    if array.size == 0:
        return _empty()
    if array.ndim not in (2, 3):
        raise ValueError(f"array images must be 2-D or 3-D, got shape {array.shape}")
    if array.ndim == 3 and array.shape[2] not in (1, 2, 3, 4) and array.shape[0] not in (1, 2, 3, 4):
        raise ValueError(
            f"array images need 1, 2, 3 or 4 channels (grey, grey+alpha, RGB, RGBA), "
            f"got {array.shape[2]}"
        )
    notes: List[str] = []
    if (array.ndim == 3 and array.shape[0] in (1, 2, 3, 4) and array.shape[2] > 4):
        array = np.moveaxis(array, 0, -1)
        notes.append("the array looked channels-first (c, h, w) and was read that way")
    if array.dtype == object or not (
        np.issubdtype(array.dtype, np.number) or array.dtype == bool
    ):
        raise TypeError(f"array images must be numeric, got dtype {array.dtype}")
    data = np.array(array, dtype=np.float32, copy=True)
    if data.ndim == 2:
        data = data[:, :, None]
    channels = data.shape[2]
    finite = np.isfinite(data)
    valid = finite.all(axis=2)
    if not valid.all():
        notes.append(f"{int((~valid).sum())} pixels held NaN or infinity and were ignored")
        data[~finite] = 0.0

    if scale_hint != "uint8":
        data = _to_0_255(data, array.dtype, valid, notes)

    if channels in (2, 4):
        alpha = data[:, :, -1:] / 255.0
        data[:, :, :-1] *= alpha
        kind = "grey+alpha" if channels == 2 else "rgba"
    else:
        kind = "grey" if channels == 1 else "rgb"
    return Pixels(data, valid, kind, notes)


def _to_0_255(data: np.ndarray, dtype: np.dtype, valid: np.ndarray, notes: List[str]) -> np.ndarray:
    if dtype == bool:
        return data * 255.0
    if dtype == np.uint8:
        return data
    if dtype == np.uint16:
        return data / 257.0
    good = data[valid] if valid.any() else data.reshape(-1)
    low = float(good.min()) if good.size else 0.0
    high = float(good.max()) if good.size else 0.0
    if np.issubdtype(dtype, np.floating) and low >= 0.0 and high <= 1.0:
        return data * 255.0
    if low >= 0.0 and high <= 255.0:
        return data
    if high - low <= 0:
        return np.zeros_like(data)
    notes.append(
        f"pixel values ran from {low:g} to {high:g}; they were stretched to 0-255 before measuring"
    )
    return (data - low) * (255.0 / (high - low))


def copy_for_detector(image: Any) -> Any:
    """What a user detector receives: a copy, so it cannot write into the caller's image."""
    if isinstance(image, np.ndarray):
        return np.array(image, copy=True)
    if isinstance(image, Image.Image):
        return image.copy()
    if isinstance(image, (list, tuple)):
        return copy.deepcopy(image)
    return image
