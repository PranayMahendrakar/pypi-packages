"""Turn whatever the caller passes in into one small, comparable grey image.

Every check in this package runs on a *working image*: the frame decimated by a
whole-pixel stride until it holds at most ``Thresholds.analysis_pixels`` pixels.
Decimation - taking every n-th pixel - is used rather than area averaging on
purpose. Averaging would smooth away exactly the per-pixel sensor noise that
tells a live static scene apart from a frozen feed, and would flatter a
defocused lens. Taking a subset of the real pixels keeps the noise statistics of
the original frame intact while making 1080p cheap to measure.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Pillow moved the resampling constants in 9.1; this spelling works on 9.x-11.x.
_RESAMPLE = getattr(Image, "Resampling", Image)
BILINEAR = _RESAMPLE.BILINEAR

IMAGE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
)

# ITU-R BT.601 luma weights, the ones PIL uses for mode "L".
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

_MONO_MODES = ("L", "1", "I", "I;16", "F")


@dataclass(frozen=True)
class Frame:
    """A frame reduced to what the checks actually read.

    Attributes:
        gray: float32 luminance, 0-255, at working resolution.
        rgb: float32 colour at working resolution, or None for a mono frame.
        width: width of the frame as it was handed in.
        height: height of the frame as it was handed in.
        stride: the decimation factor used to reach working resolution.
        source: a file name when the frame came from disk, else None.
    """

    gray: np.ndarray
    rgb: Optional[np.ndarray]
    width: int
    height: int
    stride: int
    source: Optional[str] = None

    @property
    def size(self) -> Tuple[int, int]:
        """Width and height of the original frame."""
        return (self.width, self.height)

    @property
    def shape(self) -> Tuple[int, int]:
        """Rows and columns of the working image."""
        return (int(self.gray.shape[0]), int(self.gray.shape[1]))

    def describe(self) -> str:
        """A short label: the pixel size, plus the file name when there is one."""
        text = "{}x{}".format(self.width, self.height)
        if self.source:
            text = "{} {}".format(text, self.source)
        return text


def _as_array(frame: Any) -> Tuple[np.ndarray, Optional[str]]:
    """Return the raw pixel array for any supported input, plus its file name."""
    if isinstance(frame, (str, os.PathLike)):
        path = os.fspath(frame)
        source = os.path.basename(path) or path
        if os.path.isdir(path):
            # Windows raises PermissionError here, POSIX IsADirectoryError; neither
            # message tells the caller what they actually did wrong.
            raise ValueError(
                "{} is a directory; pass a single image to check(), or the "
                "directory to check_stream()".format(path)
            )
        try:
            with Image.open(path) as handle:
                mode = "L" if handle.mode in _MONO_MODES else "RGB"
                array = np.asarray(handle.convert(mode))
        except FileNotFoundError:
            raise FileNotFoundError("no such image file: {}".format(path)) from None
        except IsADirectoryError:
            raise ValueError(
                "{} is a directory; pass a single image to check(), or the "
                "directory to check_stream()".format(path)
            ) from None
        except OSError as exc:
            raise ValueError("cannot read {} as an image: {}".format(path, exc)) from None
        return array, source

    if isinstance(frame, Image.Image):
        mode = "L" if frame.mode in _MONO_MODES else "RGB"
        name = os.path.basename(getattr(frame, "filename", "") or "") or None
        return np.asarray(frame.convert(mode)), name

    if isinstance(frame, np.ndarray):
        return frame, None

    if frame is None:
        raise ValueError("the frame is None; pass an array, a PIL image or a path to an image")

    try:
        array = np.asarray(frame)
    except Exception:  # pragma: no cover - np.asarray is very permissive
        array = None
    if array is None or array.dtype == object or array.ndim == 0:
        raise ValueError(
            "cannot read a frame of type {}; pass a numpy array, a PIL image or a "
            "path to an image file".format(type(frame).__name__)
        )
    return array, None


def _to_display_range(array: np.ndarray) -> np.ndarray:
    """Bring any numeric pixel array onto the 0-255 scale the checks assume."""
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        top = float(finite.max()) if finite.size else 0.0
        bottom = float(finite.min()) if finite.size else 0.0
        scaled = array * 255.0 if (top <= 1.0 + 1e-6 and bottom >= -1e-6) else array
        return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.max > 255:
            # 16-bit sensors and PIL mode "I": bring the full range down to 0-255.
            return array.astype(np.float32) * (255.0 / float(info.max))
        return array
    raise ValueError("frames of dtype {} are not supported".format(array.dtype))


def _stride_for(height: int, width: int, budget: int) -> int:
    """The smallest whole-pixel stride that brings the pixel count under budget."""
    pixels = height * width
    if budget <= 0 or pixels <= budget:
        return 1
    return max(1, int(math.ceil(math.sqrt(pixels / float(budget)))))


def load_frame(frame: Any, *, analysis_pixels: int = 409_600) -> Frame:
    """Read any supported frame into a Frame at working resolution.

    Accepts a numpy array (HxW, HxWx1, HxWx3 or HxWx4), a PIL image, or a path
    to an image file. A floating point array whose values all sit in 0..1 is
    taken to be normalised and is scaled up to 0..255.

    Args:
        frame: the frame to read.
        analysis_pixels: pixel budget for the working image; 0 disables decimation.

    Returns:
        The frame reduced to luminance, and to colour when the frame has any.

    Raises:
        ValueError: the object is not a readable frame, or it is empty.
        FileNotFoundError: a path was given and there is no file there.
    """
    array, source = _as_array(frame)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] != 3):
        raise ValueError(
            "a frame must be HxW (grey) or HxWx3 / HxWx4 (colour); got shape {}".format(
                tuple(array.shape)
            )
        )
    height, width = int(array.shape[0]), int(array.shape[1])
    if height == 0 or width == 0:
        raise ValueError("the frame is empty (shape {})".format(tuple(array.shape)))

    stride = _stride_for(height, width, int(analysis_pixels))
    small = _to_display_range(array[::stride, ::stride]).astype(np.float32, copy=False)
    if small.ndim == 3:
        rgb = np.clip(np.ascontiguousarray(small), 0.0, 255.0)
        gray = rgb @ _LUMA
    else:
        rgb = None
        gray = np.clip(np.ascontiguousarray(small), 0.0, 255.0)
    return Frame(
        gray=np.ascontiguousarray(gray),
        rgb=rgb,
        width=width,
        height=height,
        stride=stride,
        source=source,
    )


def as_frame(frame: Any, *, analysis_pixels: int = 409_600) -> Frame:
    """load_frame, except that an already-loaded Frame passes straight through."""
    if isinstance(frame, Frame):
        return frame
    return load_frame(frame, analysis_pixels=analysis_pixels)


def match_shape(gray: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a working-resolution grey image to the given (rows, columns)."""
    rows, cols = int(shape[0]), int(shape[1])
    if gray.shape == (rows, cols):
        return gray
    image = Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8), mode="L")
    return np.asarray(image.resize((cols, rows), BILINEAR)).astype(np.float32)


def natural_key(name: str) -> List[Tuple[int, int, str]]:
    """Sort key that puts frame2.png before frame10.png."""
    parts: List[Tuple[int, int, str]] = []
    digits = ""
    for char in name:
        if char.isdigit():
            digits += char
            continue
        if digits:
            parts.append((1, int(digits), ""))
            digits = ""
        parts.append((0, 0, char.lower()))
    if digits:
        parts.append((1, int(digits), ""))
    return parts


def list_images(directory: Any) -> List[str]:
    """Every image file directly inside a directory, in natural filename order.

    Raises:
        ValueError: the path is not a directory, or holds no readable image files.
    """
    path = os.fspath(directory)
    if not os.path.isdir(path):
        raise ValueError("not a directory: {}".format(path))
    names = [
        name
        for name in os.listdir(path)
        if name.lower().endswith(IMAGE_SUFFIXES) and os.path.isfile(os.path.join(path, name))
    ]
    if not names:
        raise ValueError(
            "no image files in {} (looked for {})".format(path, ", ".join(IMAGE_SUFFIXES))
        )
    names.sort(key=natural_key)
    return [os.path.join(path, name) for name in names]


def iter_frames(frames: Any) -> Sequence[Any]:
    """Normalise the input of check_stream into a concrete sequence of frames.

    A path to a directory becomes its image files in natural order. A single
    array, PIL image or path becomes a one-item sequence. Anything else that is
    iterable is materialised as a list.
    """
    if isinstance(frames, (str, os.PathLike)):
        path = os.fspath(frames)
        return list_images(path) if os.path.isdir(path) else [path]
    if isinstance(frames, np.ndarray) and frames.ndim in (2, 3):
        return [frames]
    if isinstance(frames, (Frame, Image.Image)):
        return [frames]
    try:
        return list(frames)
    except TypeError:
        raise ValueError(
            "frames must be an iterable of frames or a directory of images; got {}".format(
                type(frames).__name__
            )
        ) from None
