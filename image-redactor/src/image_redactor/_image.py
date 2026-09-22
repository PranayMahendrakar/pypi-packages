"""Pull any supported image into a plain uint8 array, and hand the same kind back.

Nothing in here ever writes into the object the caller passed in: every path
produces a fresh array, so ``redact`` can promise the original is untouched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

#: The three normalised modes everything downstream has to cope with.
MODES = ("L", "RGB", "RGBA")

#: Pillow mode -> the normalised mode we convert it to.
_PIL_MODE_MAP = {
    "1": "L",
    "L": "L",
    "I": "L",
    "I;16": "L",
    "F": "L",
    "LA": "RGBA",
    "P": "RGB",
    "PA": "RGBA",
    "RGB": "RGB",
    "RGBA": "RGBA",
    "RGBX": "RGB",
    "CMYK": "RGB",
    "YCbCr": "RGB",
    "HSV": "RGB",
    "LAB": "RGB",
}

_JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".jfif"}

_CHANNELS = {"L": 1, "RGB": 3, "RGBA": 4}


@dataclass
class Loaded:
    """An image normalised to ``uint8`` plus enough context to give it back.

    ``array`` is always 3-dimensional ``(height, width, channels)`` with
    ``channels`` of 1, 3 or 4, matching ``mode``.
    """

    array: np.ndarray
    mode: str
    kind: str                      # "pil", "array" or "path"
    source: Optional[str] = None   # the path, when kind == "path"
    was_2d: bool = False           # the caller's array had no channel axis
    float_unit: bool = False       # the caller's array was float in 0..1
    float_dtype: Optional[str] = None

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)``, the order Pillow uses."""
        return (self.width, self.height)

    def to_pil(self, array: Optional[np.ndarray] = None) -> Image.Image:
        """A new Pillow image for ``array`` (the loaded one by default)."""
        data = self.array if array is None else array
        if self.mode == "L":
            return Image.fromarray(data[:, :, 0], mode="L")
        return Image.fromarray(data, mode=self.mode)

    def to_output(self, array: np.ndarray) -> Any:
        """Return ``array`` shaped like whatever the caller handed us."""
        if self.kind == "array":
            out = array[:, :, 0] if self.was_2d else array
            if self.float_unit:
                return (out.astype(np.float32) / 255.0).astype(self.float_dtype or "float32")
            return out.copy()
        return self.to_pil(array)


def _normalise_pil(image: Image.Image) -> Tuple[np.ndarray, str]:
    """Convert a Pillow image to ``(uint8 array, normalised mode)``."""
    target = _PIL_MODE_MAP.get(image.mode)
    if target is None:
        target = "RGBA" if "A" in image.mode else "RGB"
    if image.mode == "P" and "transparency" in image.info:
        target = "RGBA"
    converted = image if image.mode == target else image.convert(target)
    data = np.asarray(converted)
    if data.dtype != np.uint8:  # pragma: no cover - Pillow converts for us
        data = np.clip(data, 0, 255).astype(np.uint8)
    if data.ndim == 2:
        data = data[:, :, None]
    return np.ascontiguousarray(data), target


def _normalise_array(array: np.ndarray) -> Tuple[np.ndarray, str, bool, bool, Optional[str]]:
    """Convert a numpy array to ``(uint8 array, mode, was_2d, float_unit, dtype)``."""
    if array.ndim not in (2, 3):
        raise ValueError(
            "an image array must be 2-dimensional (height, width) or 3-dimensional "
            f"(height, width, channels); got shape {tuple(array.shape)}"
        )
    was_2d = array.ndim == 2
    data = array[:, :, None] if was_2d else array
    channels = int(data.shape[2])
    if channels not in _CHANNELS.values():
        raise ValueError(
            "an image array must have 1 (grey), 3 (RGB) or 4 (RGBA) channels; "
            f"got {channels} from shape {tuple(array.shape)}"
        )
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(
            f"the image is empty: shape {tuple(array.shape)} has a zero height or width"
        )
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]

    float_unit = False
    float_dtype: Optional[str] = None
    if data.dtype == np.uint8:
        out = np.ascontiguousarray(data)
    elif np.issubdtype(data.dtype, np.floating):
        float_dtype = str(data.dtype)
        finite = data[np.isfinite(data)]
        peak = float(finite.max()) if finite.size else 0.0
        float_unit = peak <= 1.0
        scale = 255.0 if float_unit else 1.0
        out = np.clip(np.nan_to_num(data, nan=0.0) * scale, 0, 255).astype(np.uint8)
    else:
        out = np.clip(data, 0, 255).astype(np.uint8)
    return out, mode, was_2d, float_unit, float_dtype


def load_image(image: Any) -> Loaded:
    """Load ``image`` from a path, a Pillow image or a numpy array.

    A ``str`` or ``Path`` is always read from disk, so a typo raises
    ``FileNotFoundError`` naming it instead of being mistaken for pixel data.
    """
    if image is None:
        raise ValueError(
            "image is None; pass a file path, a PIL.Image.Image or a numpy array"
        )
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"{str(image)!r} does not exist")
        with Image.open(path) as handle:
            handle.load()
            data, mode = _normalise_pil(handle)
        return Loaded(array=data, mode=mode, kind="path", source=str(path))
    if isinstance(image, Image.Image):
        data, mode = _normalise_pil(image)
        return Loaded(array=data, mode=mode, kind="pil")
    if isinstance(image, np.ndarray):
        data, mode, was_2d, float_unit, float_dtype = _normalise_array(image)
        return Loaded(
            array=data,
            mode=mode,
            kind="array",
            was_2d=was_2d,
            float_unit=float_unit,
            float_dtype=float_dtype,
        )
    if hasattr(image, "__array__"):
        data, mode, was_2d, float_unit, float_dtype = _normalise_array(np.asarray(image))
        return Loaded(
            array=data,
            mode=mode,
            kind="array",
            was_2d=was_2d,
            float_unit=float_unit,
            float_dtype=float_dtype,
        )
    raise TypeError(
        "image must be a file path, a PIL.Image.Image or a numpy array; "
        f"got {type(image).__name__}"
    )


def save_image(image: Image.Image, dst: Any) -> Tuple[Path, Optional[str]]:
    """Write ``image`` to ``dst``, creating the folder. Returns ``(path, warning)``."""
    path = Path(dst)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    warning: Optional[str] = None
    out = image
    if path.suffix.lower() in _JPEG_SUFFIXES and image.mode == "RGBA":
        out = image.convert("RGB")
        warning = (
            f"JPEG cannot store transparency, so {path.name} was written as RGB; "
            "use .png to keep the alpha channel"
        )
    out.save(path)
    LOGGER.debug("wrote redacted image to %s", path)
    return path, warning
