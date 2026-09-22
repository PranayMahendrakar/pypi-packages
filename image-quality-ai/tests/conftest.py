"""Shared image factories. Everything is generated with numpy and Pillow: no
test in this suite reads a file it did not write, and none of them touch the
network.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

#: Seed for every random fixture, so a failure can always be reproduced.
SEED = 12345


def photo(width: int = 320, height: int = 240) -> np.ndarray:
    """A plausible snapshot: gradient background, textured ground, centred subject.

    Sharp, well exposed, reasonable contrast, clean and centred, so any single
    defect a test introduces is the only thing wrong with it.
    """
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    image = 0.30 + 0.25 * (rows / height)
    image += 0.05 * np.sin(cols / 23.0) * np.cos(rows / 31.0)
    ground = int(height * 0.62)
    image[ground:, :] = 0.28 + 0.10 * np.sin(cols[ground:, :] / 7.0)
    top, bottom = int(height * 0.30), int(height * 0.72)
    left, right = int(width * 0.32), int(width * 0.68)
    inner_rows, inner_cols = np.mgrid[0:bottom - top, 0:right - left]
    subject = 0.62 + 0.18 * (((inner_cols // 9) + (inner_rows // 9)) % 2)
    subject = subject + 0.05 * np.sin(inner_cols / 3.0)
    image[top:bottom, left:right] = subject
    np.clip(image, 0.0, 1.0, out=image)
    return (image * 255).astype(np.uint8)


#: Spectral slope of :func:`detailed_photo`. Natural images fall off at roughly
#: 1/f, which is what keeps detail present at every scale.
SPECTRAL_SLOPE = 1.2


def detailed_photo(width: int = 1200, height: int = 900) -> np.ndarray:
    """A frame that still looks like something after being shrunk ten times.

    :func:`photo` is built from a handful of smooth functions with one period
    each, so a 128px copy of it has lost almost everything and any measurement
    of that copy says more about the resampler than about the picture. This one
    carries detail at every scale - a 1/f spectrum, which is what a photograph
    actually has - with a bold subject laid over it, so a thumbnail of it is
    still a picture of the same thing. Deterministic: same seed, same pixels.
    """
    rng = np.random.default_rng(SEED)
    rows_f = np.fft.fftfreq(height)[:, None]
    cols_f = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(rows_f * rows_f + cols_f * cols_f)
    radius[0, 0] = 1.0                      # leave the DC term alone
    spectrum = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    plane = np.real(np.fft.ifft2(spectrum / radius ** SPECTRAL_SLOPE))
    plane = _unit(plane)

    rows, cols = np.mgrid[0:height, 0:width]
    subject = 0.5 + 0.30 * np.sign(
        np.sin(cols * (9.0 / width) * 2.0 * np.pi)
        * np.cos(rows * (7.0 / height) * 2.0 * np.pi)
    )
    plane = _unit(0.72 * plane + 0.28 * subject)
    return ((0.14 + 0.72 * plane) * 255).astype(np.uint8)


def _unit(plane: np.ndarray) -> np.ndarray:
    """``plane`` rescaled to run 0.0 to 1.0."""
    plane = plane - plane.min()
    return plane / max(float(plane.max()), 1e-9)


def as_rgb(grey: np.ndarray) -> np.ndarray:
    """An HxW plane as an HxWx3 colour image."""
    return np.ascontiguousarray(np.dstack([grey, grey, grey]))


def blurred(grey: np.ndarray, radius: float = 6.0) -> np.ndarray:
    """The same frame, out of focus."""
    return np.asarray(Image.fromarray(grey).filter(ImageFilter.GaussianBlur(radius)))


def noisy(grey: np.ndarray, sigma: float = 14.0) -> np.ndarray:
    """The same frame with sensor grain on it."""
    rng = np.random.default_rng(SEED)
    grain = rng.normal(0.0, sigma, grey.shape)
    return np.clip(grey.astype(np.float32) + grain, 0, 255).astype(np.uint8)


@pytest.fixture
def good() -> np.ndarray:
    """A greyscale photo with nothing wrong with it."""
    return photo()


@pytest.fixture
def good_rgb() -> np.ndarray:
    """The same photo in colour."""
    return as_rgb(photo())


@pytest.fixture
def tmp_photo(tmp_path, good_rgb) -> str:
    """The good photo written to a PNG, returned as a path string."""
    target = tmp_path / "good.png"
    Image.fromarray(good_rgb).save(target)
    return str(target)
