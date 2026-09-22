"""Shared image builders. Everything here is generated with numpy; nothing is downloaded."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

SEED = 20260922


def noise(height: int = 120, width: int = 160, channels: int = 3, seed: int = SEED) -> np.ndarray:
    """A deterministic uint8 noise image, so "did the pixels change" is unambiguous."""
    rng = np.random.default_rng(seed)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def gradient(height: int = 64, width: int = 96) -> np.ndarray:
    """A smooth RGB gradient: every pixel differs from its neighbours."""
    yy, xx = np.mgrid[0:height, 0:width]
    red = (xx * 255 // max(1, width - 1)).astype(np.uint8)
    green = (yy * 255 // max(1, height - 1)).astype(np.uint8)
    blue = ((xx + yy) * 255 // max(1, width + height - 2)).astype(np.uint8)
    return np.dstack([red, green, blue])


def scene() -> np.ndarray:
    """A dark frame with one skin-toned ellipse and one bright wide rectangle.

    Deliberately easy: it is the kind of "obvious case" the built-in heuristics
    are the only thing they are honestly good for.
    """
    canvas = np.full((200, 300, 3), 30, dtype=np.uint8)
    yy, xx = np.mgrid[0:200, 0:300]
    face = ((xx - 80) ** 2 / 900.0 + (yy - 70) ** 2 / 1300.0) <= 1.0
    canvas[face] = (215, 160, 130)
    canvas[140:170, 180:280] = 250
    return canvas


@pytest.fixture
def photo() -> np.ndarray:
    return noise()


@pytest.fixture
def smooth() -> np.ndarray:
    return gradient()


@pytest.fixture
def grey() -> np.ndarray:
    return noise(80, 100, channels=1)


@pytest.fixture
def rgba() -> np.ndarray:
    return noise(80, 100, channels=4)


@pytest.fixture
def pil_photo() -> Image.Image:
    return Image.fromarray(noise(70, 90))


@pytest.fixture
def obvious_scene() -> np.ndarray:
    return scene()
