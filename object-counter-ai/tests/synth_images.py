"""Test images with a known number of things in them, generated with numpy."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def blank(h: int = 120, w: int = 160, value: int = 220, channels: Optional[int] = None) -> np.ndarray:
    shape = (h, w) if channels is None else (h, w, channels)
    return np.full(shape, value, dtype=np.uint8)


def draw_disc(img: np.ndarray, cx: float, cy: float, r: float, value) -> np.ndarray:
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = value
    return img


def draw_square(img: np.ndarray, cx: int, cy: int, half: int, value) -> np.ndarray:
    img[max(0, cy - half): cy + half, max(0, cx - half): cx + half] = value
    return img


def scatter(n: int, h: int, w: int, r: int, seed: int, gap: int = 6,
            margin: Optional[int] = None) -> List[Tuple[int, int]]:
    """``n`` centres that keep shapes of radius ``r`` at least ``gap`` pixels apart."""
    rng = np.random.default_rng(seed)
    margin = r + 3 if margin is None else margin
    centres: List[Tuple[int, int]] = []
    tries = 0
    while len(centres) < n:
        tries += 1
        if tries > 20000:
            raise RuntimeError("could not place shapes; make the image bigger")
        x = int(rng.integers(margin, w - margin))
        y = int(rng.integers(margin, h - margin))
        if all((x - a) ** 2 + (y - b) ** 2 > (2 * r * 1.45 + gap) ** 2 for a, b in centres):
            centres.append((x, y))
    return centres


def discs_image(n: int, seed: int = 0, h: int = 240, w: int = 320, r: int = 9,
                background: int = 220, value: int = 40) -> np.ndarray:
    img = blank(h, w, background)
    for x, y in scatter(n, h, w, r, seed):
        draw_disc(img, x, y, r, value)
    return img


def discs_and_squares(n_discs: int, n_squares: int, seed: int = 0, h: int = 240, w: int = 320,
                      r: int = 9, background: int = 30, value: int = 200) -> np.ndarray:
    img = blank(h, w, background)
    centres = scatter(n_discs + n_squares, h, w, r, seed)
    for i, (x, y) in enumerate(centres):
        if i < n_discs:
            draw_disc(img, x, y, r, value)
        else:
            draw_square(img, x, y, r, value)
    return img


def add_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def with_gradient(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """Add a left-to-right lighting ramp (as a vignette-ish uneven background)."""
    ramp = np.linspace(low, high, img.shape[1])[None, :]
    if img.ndim == 3:
        ramp = ramp[:, :, None]
    return np.clip(img.astype(np.float64) + ramp, 0, 255).astype(np.uint8)


def moving_disc_frames(path: Sequence[Tuple[float, float]], h: int = 120, w: int = 200,
                       r: int = 6, extra: Sequence[Sequence[Tuple[float, float]]] = ()) -> List[np.ndarray]:
    """One frame per position; ``extra[i]`` lists more discs for frame i."""
    frames = []
    for i, (x, y) in enumerate(path):
        img = blank(h, w, 225)
        draw_disc(img, x, y, r, 35)
        if i < len(extra):
            for ex, ey in extra[i]:
                draw_disc(img, ex, ey, r, 35)
        frames.append(img)
    return frames
