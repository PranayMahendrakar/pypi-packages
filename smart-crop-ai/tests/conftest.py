"""Test images, built with numpy and Pillow. Nothing is downloaded."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from PIL import Image

#: Where the planted subject lives in :func:`subject_array`: (top, bottom, left, right).
SUBJECT_ROWS = (140, 280)
SUBJECT_COLS = (560, 700)

#: Size of the image :func:`subject_array` builds, as (width, height).
SUBJECT_SIZE = (800, 400)


def subject_array(seed: int = 0) -> np.ndarray:
    """A plain wall with one high-detail patch well off to the right.

    The patch is the only thing in the image worth keeping, and a centre crop of
    any square size slices through it, so every strategy has an unambiguous
    right answer and a centre crop has an unambiguously wrong one.
    """
    wall = np.full((SUBJECT_SIZE[1], SUBJECT_SIZE[0], 3), 200, dtype=np.uint8)
    top, bottom = SUBJECT_ROWS
    left, right = SUBJECT_COLS
    patch = np.random.default_rng(seed).integers(
        0, 255, (bottom - top, right - left, 3)
    )
    wall[top:bottom, left:right] = patch.astype(np.uint8)
    return wall


def centred_subject_array(seed: int = 1) -> np.ndarray:
    """The same wall, with the patch sitting dead centre instead."""
    wall = np.full((SUBJECT_SIZE[1], SUBJECT_SIZE[0], 3), 200, dtype=np.uint8)
    height, width = wall.shape[:2]
    patch_h, patch_w = 140, 140
    top = (height - patch_h) // 2
    left = (width - patch_w) // 2
    patch = np.random.default_rng(seed).integers(0, 255, (patch_h, patch_w, 3))
    wall[top : top + patch_h, left : left + patch_w] = patch.astype(np.uint8)
    return wall


def box_contains_subject(box: Tuple[int, int, int, int]) -> bool:
    """True when ``box`` holds the whole planted patch of :func:`subject_array`."""
    left, top, right, bottom = box
    return (
        left <= SUBJECT_COLS[0]
        and right >= SUBJECT_COLS[1]
        and top <= SUBJECT_ROWS[0]
        and bottom >= SUBJECT_ROWS[1]
    )


@pytest.fixture
def subject_image() -> Image.Image:
    """The off-centre-subject wall, as an RGB PIL image."""
    return Image.fromarray(subject_array())


@pytest.fixture
def centred_image() -> Image.Image:
    """The same wall with the subject in the middle."""
    return Image.fromarray(centred_subject_array())


@pytest.fixture
def flat_image() -> Image.Image:
    """One single colour, edge to edge. Nothing to find."""
    return Image.fromarray(np.full((200, 400, 3), 128, dtype=np.uint8))


@pytest.fixture
def exif_rotated(tmp_path) -> str:
    """A landscape JPEG whose EXIF says it should be displayed portrait."""
    array = subject_array()
    image = Image.fromarray(array)
    exif = image.getexif()
    exif[0x0112] = 6            # rotate 90 degrees clockwise to display
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif, quality=95)
    return str(path)
