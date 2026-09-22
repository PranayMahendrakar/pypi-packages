"""Synthetic camera frames for the test suite.

Every frame the tests use is generated here with numpy and Pillow - nothing is
downloaded and nothing is read from disk that a test did not just write.

The generators exist because the package's hardest distinction needs both sides
of it on hand: ``live_frame`` is a still scene re-exposed, so nearly every pixel
moves by a grey level of read noise, while handing the same array twice is a
frozen feed. A scene generator that produced a noise-free image could not tell
those apart, and one that produced a noisy image would look like a failing
sensor to the noise check.
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 20260922
"""Fixed so every frame in the suite is byte-for-byte reproducible."""


def _smooth(field: np.ndarray, rounds: int) -> np.ndarray:
    """Blur a field with a 5-point stencil - the cheap way to band-limit noise."""
    for _ in range(rounds):
        field = (
            field
            + np.roll(field, 1, 0)
            + np.roll(field, -1, 0)
            + np.roll(field, 1, 1)
            + np.roll(field, -1, 1)
        ) / 5.0
    return field


def _band_limited(rows: int, cols: int, rounds: int, seed: int) -> np.ndarray:
    """A smooth random field of unit variance: texture with no per-pixel noise."""
    rng = np.random.default_rng(seed)
    field = _smooth(rng.normal(0.0, 1.0, (rows, cols)).astype(np.float32), rounds)
    return field / float(field.std() + 1e-9)


def scene(rows: int = 240, cols: int = 320, *, seed: int = SEED, layout: int = 0) -> np.ndarray:
    """A sharp, well exposed, locked-off camera view as uint8 luminance.

    Detail sits at several spatial scales, the way a real room does: a slow
    gradient, mid-scale shapes, and fine texture. Crucially the fine texture is
    band-limited, so a healthy frame reads as sharp without reading as noisy.

    Args:
        rows, cols: frame size in pixels.
        seed: changes the texture while keeping the layout.
        layout: changes the large-scale structure, so two layouts are different
            scenes rather than two views of the same one.

    Returns:
        A uint8 array of shape (rows, cols).
    """
    grid_y, grid_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
    if layout == 0:
        image = 118.0 + 26.0 * np.sin(grid_x / 23.0) + 18.0 * np.cos(grid_y / 17.0)
    else:
        image = 132.0 + 24.0 * np.cos(grid_x / 11.0 + layout) - 20.0 * np.sin(grid_y / 29.0)
    image = image + 30.0 * _band_limited(rows, cols, 6, seed)
    image = image + 22.0 * _band_limited(rows, cols, 20, seed + 1)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def live_frame(base: np.ndarray, tick: int) -> np.ndarray:
    """The same still scene, exposed again: a fresh grain of sensor read noise.

    This is what a working camera watching an empty corridor hands back. The
    picture is the same to within a grey level, yet nearly every pixel moved -
    which is exactly what tells it apart from a repeated buffer.
    """
    rng = np.random.default_rng(SEED + 1000 + tick)
    noisy = base.astype(np.float32) + rng.normal(0.0, 1.1, base.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def blurred(base: np.ndarray, rounds: int = 40) -> np.ndarray:
    """The same scene through a lens that is out of focus."""
    return np.clip(_smooth(base.astype(np.float32), rounds), 0.0, 255.0).astype(np.uint8)


def obstructed(base: np.ndarray, share: float = 0.3) -> np.ndarray:
    """The scene with a flat, featureless object held in front of part of the lens."""
    frame = base.copy()
    rows, cols = frame.shape
    height = int(rows * np.sqrt(share))
    width = int(cols * np.sqrt(share))
    frame[2 : 2 + height, 2 : 2 + width] = 38
    return frame


def colourise(base: np.ndarray, gains: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Turn a grey frame into an RGB one, optionally with a channel gain."""
    stacked = np.stack([base, base, base], axis=-1).astype(np.float32)
    for channel, gain in enumerate(gains):
        stacked[:, :, channel] = stacked[:, :, channel] * float(gain)
    return np.clip(stacked, 0.0, 255.0).astype(np.uint8)


def big_scene(rows: int = 1080, cols: int = 1920, *, seed: int = SEED) -> np.ndarray:
    """A 1080p frame, built small and enlarged the way a real lens band-limits.

    Generating band-limited texture directly at 1080p would take longer than the
    test it feeds. Enlarging a small scene and then adding per-pixel noise at
    full resolution gives the same thing a camera gives: optical detail the lens
    could actually resolve, plus grain at the sensor's own pitch.
    """
    from PIL import Image

    small = scene(rows // 3, cols // 3, seed=seed)
    grown = Image.fromarray(small).resize((cols, rows), Image.BICUBIC)
    rng = np.random.default_rng(seed + 77)
    noisy = np.asarray(grown).astype(np.float32) + rng.normal(0.0, 1.1, (rows, cols))
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def save_png(path, frame: np.ndarray) -> str:
    """Write a frame to a PNG and return the path as a string."""
    from PIL import Image

    Image.fromarray(frame).save(str(path))
    return str(path)


@pytest.fixture
def base():
    """One sharp, healthy 320x240 camera view."""
    return scene()


@pytest.fixture
def other_scene():
    """A different view entirely - what the camera sees after it is turned."""
    return scene(layout=3, seed=SEED + 500)


@pytest.fixture
def still_run(base):
    """Twelve frames of a genuinely static scene from a live camera."""
    return [live_frame(base, tick) for tick in range(12)]
