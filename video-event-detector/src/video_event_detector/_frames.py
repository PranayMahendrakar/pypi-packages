"""Turn whatever the caller hands in into one small grey working image.

Every rule in this package runs on a *working image*: the frame converted to
luminance on a 0-255 scale and block-averaged by a whole-pixel factor until it
holds at most :data:`WORKING_PIXELS` pixels. Averaging is the right reduction
here (unlike a focus or noise measurement): it lowers sensor noise, which is
exactly what a motion threshold wants, and it makes 1080p cheap to follow.

Nothing handed in is ever written to. Every conversion starts from an explicit
copy, so a caller's read-only array, memory-mapped file or shared buffer stays
exactly as it was.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Pillow moved the resampling constants in 9.1; this spelling works on 9.x-12.x.
_RESAMPLE = getattr(Image, "Resampling", Image)
_BOX = _RESAMPLE.BOX
_BILINEAR = _RESAMPLE.BILINEAR

IMAGE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
)
"""File name endings read from a frame directory. Anything else there is ignored."""

WORKING_PIXELS = 20000
"""Largest working image, in pixels (about 160 x 120)."""

MIN_SIDE = 8
"""Smallest frame side, in pixels, that can carry a motion measurement."""

# ITU-R BT.601 luma weights, the ones PIL itself uses for mode "L".
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

_SIXTEEN_BIT_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I")


def natural_key(name: str) -> List[Any]:
    """Sort key that puts ``frame_2.png`` before ``frame_10.png``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_frame_files(directory: str) -> List[str]:
    """Image files in ``directory``, in natural (numbered) order.

    Raises:
        ValueError: the directory holds no image file this package can read.
    """
    names = [
        entry
        for entry in os.listdir(directory)
        if entry.lower().endswith(IMAGE_SUFFIXES)
        and os.path.isfile(os.path.join(directory, entry))
    ]
    if not names:
        raise ValueError(
            "no image files in {}; expected numbered frames such as frame_0001.png "
            "(this package does not decode video files - extract frames first, for "
            "example: ffmpeg -i clip.mp4 -vf fps=10 frames/%05d.png)".format(directory)
        )
    names.sort(key=natural_key)
    return [os.path.join(directory, name) for name in names]


def _pil_to_array(image: Image.Image) -> np.ndarray:
    """Pixels of a PIL image as a new numpy array, grey or RGB."""
    mode = image.mode
    if mode in ("L", "F"):
        return np.array(image)
    if mode in _SIXTEEN_BIT_MODES:
        return np.array(image.convert("I"))
    if mode == "1":
        return np.array(image.convert("L"))
    return np.array(image.convert("RGB"))


def read_image(path: str) -> np.ndarray:
    """Read one image file into a numpy array.

    Raises:
        FileNotFoundError: there is no such file.
        ValueError: the file exists but is not an image PIL can read.
    """
    if os.path.isdir(path):
        raise ValueError(
            "{} is a directory; pass the directory itself as `frames`, not inside a "
            "list of frames".format(path)
        )
    try:
        with Image.open(path) as handle:
            handle.load()
            return _pil_to_array(handle)
    except FileNotFoundError:
        raise FileNotFoundError("no such image file: {}".format(path)) from None
    except OSError as exc:
        raise ValueError(
            "cannot read {} as an image ({}); this package reads still frames, not "
            "video files - extract frames first, for example with "
            "ffmpeg -i clip.mp4 -vf fps=10 frames/%05d.png".format(path, exc)
        ) from None


def iter_frames(frames: Any) -> Tuple[Iterator[Tuple[Any, Optional[str]]], Optional[str], Optional[int]]:
    """Normalise the ``frames`` argument of :func:`detect_events`.

    Returns:
        An iterator of ``(frame, name)`` pairs, a description of the source for
        the report, and the number of frames when it is known up front.

    Raises:
        ValueError: ``frames`` is not a directory, a sequence or an iterable of frames.
        FileNotFoundError: ``frames`` names a path that does not exist.
    """
    if isinstance(frames, (str, os.PathLike)):
        path = os.fspath(frames)
        if not os.path.exists(path):
            raise FileNotFoundError("no such directory or image file: {}".format(path))
        if os.path.isdir(path):
            files = list_frame_files(path)
            pairs = ((name, os.path.basename(name)) for name in files)
            return pairs, path, len(files)
        # A single image file: a one-frame sequence, honest if not very useful.
        return iter([(path, os.path.basename(path))]), path, 1

    if isinstance(frames, Image.Image):
        raise ValueError(
            "frames is a single PIL image; pass a list of frames, e.g. [image1, image2, ...]"
        )

    if isinstance(frames, np.ndarray):
        if frames.ndim == 4 or frames.ndim == 3:
            # (T, H, W) grey stack or (T, H, W, C) colour stack. Iterating a numpy
            # array yields views; they are only ever read, never written.
            count = int(frames.shape[0])
            return ((frames[i], None) for i in range(count)), None, count
        raise ValueError(
            "a numpy array of frames must be a stack shaped (frames, height, width) or "
            "(frames, height, width, channels); got shape {}. Pass a single frame "
            "inside a list".format(frames.shape)
        )

    if isinstance(frames, (bytes, bytearray)):
        raise ValueError("frames must be a list of images or a directory, not bytes")

    try:
        iterator = iter(frames)
    except TypeError:
        raise ValueError(
            "frames must be a list of numpy arrays or PIL images, a numpy stack, or a "
            "directory of numbered image files; got {}".format(type(frames).__name__)
        ) from None
    count: Optional[int] = None
    if isinstance(frames, Sequence):
        count = len(frames)

    def named() -> Iterator[Tuple[Any, Optional[str]]]:
        for item in iterator:
            if isinstance(item, (str, os.PathLike)):
                yield item, os.path.basename(os.fspath(item))
            else:
                yield item, None

    return named(), None, count


class FramePrep:
    """Convert frames to working images, remembering what had to be adjusted.

    The first frame fixes the reference size. A later frame of a different size
    is resized to it (and noted), so one odd frame cannot break the background
    model. Bit depth and float scaling are decided on the first frame of each
    kind and then held, so the whole sequence stays on one brightness scale.
    """

    def __init__(self, budget: int = WORKING_PIXELS) -> None:
        self.budget = int(budget)
        self.reference: Optional[Tuple[int, int]] = None  # (height, width)
        self.factor = 1
        self.working_shape: Optional[Tuple[int, int]] = None
        self.resized: Dict[Tuple[int, int], List[Any]] = {}
        self.float_unit: Optional[float] = None
        self.int_unit: Optional[float] = None
        self.colour = False

    # -- conversion -------------------------------------------------------
    def _raw(self, frame: Any) -> np.ndarray:
        if frame is None:
            raise ValueError("a frame is None; pass numpy arrays, PIL images or image paths")
        if isinstance(frame, (str, os.PathLike)):
            return read_image(os.fspath(frame))
        if isinstance(frame, Image.Image):
            return _pil_to_array(frame)
        if isinstance(frame, np.ndarray):
            return frame
        try:
            array = np.asarray(frame)
        except Exception:  # pragma: no cover - np.asarray is very permissive
            array = None
        if array is None or array.dtype == object or array.ndim < 2:
            raise ValueError(
                "cannot read a frame of type {}; pass a numpy array, a PIL image or a "
                "path to an image file".format(type(frame).__name__)
            )
        return array

    def _gray(self, array: np.ndarray) -> np.ndarray:
        """A new float32 luminance image on 0-255; ``array`` is never modified."""
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.ndim == 3 and array.shape[2] == 2:  # grey + alpha
            array = array[:, :, 0]
        if array.ndim == 3 and array.shape[2] in (3, 4):
            self.colour = True
            values = self._scaled(np.array(array[:, :, :3], dtype=np.float32, copy=True), array.dtype)
            gray = values @ _LUMA
        elif array.ndim == 2:
            gray = self._scaled(np.array(array, dtype=np.float32, copy=True), array.dtype)
        else:
            raise ValueError(
                "a frame must be height x width (grey) or height x width x 3/4 (colour); "
                "got shape {}".format(array.shape)
            )
        gray = np.ascontiguousarray(gray, dtype=np.float32)
        finite = np.isfinite(gray)
        if not finite.all():
            # NaN and infinity carry no brightness: mark them unknown (NaN) and let
            # the detector treat them as unchanged instead of as black or white.
            gray[~finite] = np.nan
        np.clip(gray, 0.0, 255.0, out=gray)
        return gray

    def _scaled(self, values: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Bring pixel values onto 0-255, choosing the scale once per sequence."""
        if dtype == np.bool_:
            return values * 255.0
        if np.issubdtype(dtype, np.floating):
            if self.float_unit is None:
                finite = values[np.isfinite(values)]
                top = float(finite.max()) if finite.size else 0.0
                bottom = float(finite.min()) if finite.size else 0.0
                self.float_unit = 255.0 if (top <= 1.0 + 1e-6 and bottom >= -1e-6) else 1.0
            return values * self.float_unit if self.float_unit != 1.0 else values
        if np.issubdtype(dtype, np.integer):
            if dtype == np.uint8:
                return values
            if self.int_unit is None:
                top = float(values.max()) if values.size else 0.0
                if top <= 255.0:
                    self.int_unit = 255.0
                else:
                    # 10-, 12-, 14- and 16-bit sensors: scale by the smallest usual
                    # full range that holds the first frame, then keep that scale.
                    bits = int(math.ceil(math.log2(top + 1.0)))
                    bits = next((b for b in (10, 12, 14, 16) if bits <= b), bits)
                    self.int_unit = float(2 ** bits - 1)
            if self.int_unit == 255.0:
                return values
            return values * (255.0 / self.int_unit)
        raise ValueError("frames of dtype {} are not supported".format(dtype))

    def _reduce(self, gray: np.ndarray) -> np.ndarray:
        k = self.factor
        if k == 1:
            return gray
        rows, cols = self.working_shape  # type: ignore[misc]
        block = gray[: rows * k, : cols * k].reshape(rows, k, cols, k)
        return block.mean(axis=(1, 3), dtype=np.float32)

    def __call__(self, frame: Any, index: int, name: Optional[str] = None) -> np.ndarray:
        """Return the working image for one frame.

        Raises:
            ValueError: the frame cannot be read or is too small to measure.
        """
        gray = self._gray(self._raw(frame))
        height, width = gray.shape
        if height < MIN_SIDE or width < MIN_SIDE:
            raise ValueError(
                "frame {} is {}x{} pixels; at least {}x{} is needed to measure motion".format(
                    index, width, height, MIN_SIDE, MIN_SIDE
                )
            )
        if self.reference is None:
            self.reference = (height, width)
            self.factor = max(1, int(math.ceil(math.sqrt(height * width / float(self.budget)))))
            self.working_shape = (max(1, height // self.factor), max(1, width // self.factor))
        elif (height, width) != self.reference:
            ref_h, ref_w = self.reference
            entry = self.resized.setdefault((height, width), [0, index, name])
            entry[0] += 1
            shrink = height >= ref_h and width >= ref_w
            image = Image.fromarray(gray)
            image = image.resize((ref_w, ref_h), resample=_BOX if shrink else _BILINEAR)
            gray = np.array(image, dtype=np.float32, copy=True)
        return self._reduce(gray)

    # -- reporting --------------------------------------------------------
    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        """(width, height) of the reference frame, or None before the first frame."""
        if self.reference is None:
            return None
        return (self.reference[1], self.reference[0])

    @property
    def working_size(self) -> Optional[Tuple[int, int]]:
        """(width, height) of the working image, or None before the first frame."""
        if self.working_shape is None:
            return None
        return (self.working_shape[1], self.working_shape[0])

    def notes(self) -> List[str]:
        """Plain-language notes about adjustments made to the input."""
        out: List[str] = []
        if self.reference is not None:
            ref_h, ref_w = self.reference
            for (height, width), (count, first, name) in sorted(self.resized.items()):
                where = "frame {}".format(first) if not name else "frame {} ({})".format(first, name)
                out.append(
                    "{} frame{} {}x{}, not {}x{} like the first frame (first at {}); "
                    "{} resized to match{}".format(
                        count,
                        "s were" if count != 1 else " was",
                        width,
                        height,
                        ref_w,
                        ref_h,
                        where,
                        "they were" if count != 1 else "it was",
                        "" if abs(width / float(height) - ref_w / float(ref_h)) < 0.01
                        else ", which also stretched the picture (different aspect ratio)",
                    )
                )
        if self.int_unit is not None and self.int_unit > 255.0:
            out.append(
                "frames hold values up to {:.0f} ({}-bit); they were scaled to 0-255".format(
                    self.int_unit, int(round(math.log2(self.int_unit + 1.0)))
                )
            )
        return out
