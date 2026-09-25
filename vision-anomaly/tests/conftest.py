"""Shared image factories.

Every pixel in this suite is generated here with numpy and Pillow. No test reads
a file it did not write, nothing is downloaded, and nothing touches the network.

The scene is a machine-vision cliche on purpose: a textured belt with a lighter
machined part sitting on it. The "normal" images differ the way real ones do -
the part lands a few pixels off, the lamp flickers, the sensor adds grain - and
each fault changes one obvious thing about the picture, so a test that fails
says which kind of departure stopped being detected.

Seeds are fixed, so a failure can always be reproduced.
"""
from __future__ import annotations

import functools
from typing import Callable, List, Tuple

import numpy as np
import pytest
from PIL import Image

#: Base seed. Every factory offsets from it, so no two produce the same noise.
SEED = 20260925

#: Source size of a generated frame, width by height.
FRAME = (512, 384)

#: Where the part sits in an untroubled frame, as fractions of the frame: left,
#: top, right, bottom. Kept as fractions so a frame of any size puts the part in
#: the same place, which is the whole point of the square analysis grid.
#:
#: The values are chosen to keep every edge of the part well away from the grid's
#: cell boundaries at 0.25, 0.50 and 0.75, so the few pixels of jitter below
#: never march an edge across one. A normal set that does straddle a boundary is
#: a real thing that happens, and the package reports it as a departure, which is
#: correct and not what these tests are measuring.
PART = (0.293, 0.292, 0.699, 0.646)


def _cached(factory: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Generate each frame once, and hand it back read-only.

    The caching keeps the suite quick. The read-only flag is the point: every
    frame these tests pass into the library is an array numpy will refuse to let
    anyone write to, so the promise that caller images are never modified is
    enforced by the machine rather than by one test remembering to check.
    """
    memo = functools.lru_cache(maxsize=None)(factory)

    @functools.wraps(factory)
    def wrapper(*args, **kwargs):
        frame = memo(*args, **kwargs)
        frame.flags.writeable = False
        return frame

    return wrapper


def _finish(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lamp flicker and sensor grain, then back to 8-bit."""
    image = image * float(rng.normal(1.0, 0.010))
    image = image + rng.normal(0.0, 0.004, image.shape)
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _belt(width: int, height: int) -> np.ndarray:
    """The woven background the part sits on, as an HxWx3 float plane."""
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    weave = (
        0.30
        + 0.05 * np.sin(cols / 6.0) * np.cos(rows / 7.5)
        + 0.02 * np.sin(rows / 3.0)
    )
    return np.dstack([weave * 1.02, weave, weave * 0.90]).astype(np.float32)


def _part_surface(height: int, width: int) -> np.ndarray:
    """The machined grain on the part: regular, fine, and always the same."""
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    return 0.60 + 0.06 * np.sin(cols / 2.5) + 0.03 * np.cos(rows / 3.0)


def _place_part(image: np.ndarray, box: Tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    surface = _part_surface(bottom - top, right - left)
    image[top:bottom, left:right, 0] = surface * 1.00
    image[top:bottom, left:right, 1] = surface * 0.97
    image[top:bottom, left:right, 2] = surface * 0.88


def _jitter(
    rng: np.random.Generator, size: Tuple[int, int] = FRAME
) -> Tuple[int, int, int, int]:
    """The part's box in pixels, nudged the way a real one never sits still."""
    width, height = size
    left, top, right, bottom = PART
    dx = int(rng.integers(-3, 4))
    dy = int(rng.integers(-3, 4))
    return (
        int(left * width) + dx,
        int(top * height) + dy,
        int(right * width) + dx,
        int(bottom * height) + dy,
    )


@_cached
def normal_scene(index: int = 0, size: Tuple[int, int] = FRAME) -> np.ndarray:
    """One good frame: belt, part, a little jitter and grain. HxWx3 uint8."""
    rng = np.random.default_rng(SEED + index)
    width, height = size
    image = _belt(width, height)
    _place_part(image, _jitter(rng, size))
    return _finish(image, rng)


def normal_set(count: int = 8, size: Tuple[int, int] = FRAME) -> List[np.ndarray]:
    """``count`` good frames, spanning the jitter a later frame may show."""
    return [normal_scene(i, size) for i in range(count)]


@_cached
def held_out_scene(index: int = 0) -> np.ndarray:
    """A good frame the profile was not fitted on."""
    return normal_scene(500 + index)


@_cached
def colour_fault(index: int = 0) -> np.ndarray:
    """A good frame with a large red stain across the part."""
    rng = np.random.default_rng(SEED + 900 + index)
    width, height = FRAME
    image = _belt(width, height)
    _place_part(image, _jitter(rng))
    image[150:230, 190:330, 0] = 0.95
    image[150:230, 190:330, 1] = 0.12
    image[150:230, 190:330, 2] = 0.10
    return _finish(image, rng)


@_cached
def missing_part(index: int = 0) -> np.ndarray:
    """The belt with nothing on it."""
    rng = np.random.default_rng(SEED + 910 + index)
    width, height = FRAME
    return _finish(_belt(width, height), rng)


@_cached
def rough_part(index: int = 0) -> np.ndarray:
    """The part there but its surface torn up - same colour, wrong texture."""
    rng = np.random.default_rng(SEED + 920 + index)
    width, height = FRAME
    image = _belt(width, height)
    box = _jitter(rng)
    _place_part(image, box)
    left, top, right, bottom = box
    grain = rng.normal(0.0, 0.16, (bottom - top, right - left, 1))
    image[top:bottom, left:right, :] += grain
    return _finish(image, rng)


@_cached
def different_scene(index: int = 0) -> np.ndarray:
    """A completely different picture: cool diagonal stripes, no part at all."""
    rng = np.random.default_rng(SEED + 930 + index)
    width, height = FRAME
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    stripes = 0.5 + 0.35 * np.sign(np.sin((cols + rows) / 11.0))
    image = np.dstack([stripes * 0.35, stripes * 0.60, stripes * 1.00]).astype(np.float32)
    return _finish(image, rng)


def as_grey(colour: np.ndarray) -> np.ndarray:
    """An HxWx3 frame as a single-channel HxW one."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.clip(colour.astype(np.float32) @ weights, 0, 255).astype(np.uint8)


def as_rgba(colour: np.ndarray, alpha: int = 255) -> np.ndarray:
    """An HxWx3 frame with an alpha channel bolted on."""
    band = np.full(colour.shape[:2] + (1,), alpha, dtype=np.uint8)
    return np.concatenate([colour, band], axis=2)


def save_png(image: np.ndarray, path) -> str:
    """Write a frame to ``path`` and return the path as a string."""
    Image.fromarray(image).save(path)
    return str(path)


@pytest.fixture
def good_images() -> List[np.ndarray]:
    """Eight known-good frames."""
    return normal_set(8)


@pytest.fixture
def clean_image() -> np.ndarray:
    """A ninth good frame, held out of the fit."""
    return held_out_scene(0)


@pytest.fixture
def odd_image() -> np.ndarray:
    """A frame that is obviously not one of the good ones."""
    return different_scene(0)


@pytest.fixture
def good_paths(tmp_path, good_images) -> List[str]:
    """The good frames written out as PNG files."""
    return [
        save_png(image, tmp_path / "good_{0:02d}.png".format(i))
        for i, image in enumerate(good_images)
    ]
