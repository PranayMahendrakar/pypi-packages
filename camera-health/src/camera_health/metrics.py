"""The measurements every check is built on. Pure numpy, no convolution library.

Nothing here decides whether a camera is healthy; each function returns a number
and ``checks.py`` compares it with a threshold. Keeping it split that way means
the numbers land in ``FrameHealth.metrics`` whether or not a fault was raised, so
a report can always be argued with.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._frames import BILINEAR, Frame, match_shape

from PIL import Image

# Noise propagation constants for the two kernels below. For white noise of
# standard deviation s: the 4-neighbour Laplacian has sum-of-squares 20, so its
# response has standard deviation sqrt(20) * s and mean |response|
# sqrt(2/pi) * sqrt(20) * s. Subtracting that is what stops heavy noise from
# passing as sharp detail.
_LAP_NOISE_VAR = 20.0
_LAP_NOISE_ABS = 3.5682  # sqrt(2/pi) * sqrt(20)

# The Immerkaer mask [[1,-2,1],[-2,4,-2],[1,-2,1]] has sum-of-squares 36, so its
# response standard deviation is 6s and the median of its absolute value is
# 0.6745 * 6 * s. Dividing by that turns a median into a robust sigma.
_IMMERKAER_MEDIAN = 4.0470


def laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian of the working image, valid pixels only."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros((0, 0), dtype=np.float32)
    middle = gray[1:-1, 1:-1]
    return (
        4.0 * middle
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )


def noise_sigma(gray: np.ndarray) -> float:
    """Robust per-pixel noise estimate in grey levels.

    A median-of-absolute-response variant of the Immerkaer estimator: the median
    ignores the heavy tail that edges and texture put into the response, so a
    detailed scene is not mistaken for a noisy sensor.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    response = (
        4.0 * gray[1:-1, 1:-1]
        + gray[:-2, :-2]
        + gray[:-2, 2:]
        + gray[2:, :-2]
        + gray[2:, 2:]
        - 2.0
        * (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    )
    return float(np.median(np.abs(response)) / _IMMERKAER_MEDIAN)


def exposure(gray: np.ndarray) -> Dict[str, float]:
    """Brightness, contrast, clipping and usable range of one frame."""
    flat = gray.reshape(-1)
    low, high = np.percentile(flat, (1.0, 99.0))
    return {
        "brightness": float(flat.mean()),
        "contrast": float(flat.std()),
        "dark_fraction": float((flat < 16.0).mean()),
        "clipped_high": float((flat >= 250.0).mean()),
        "clipped_low": float((flat <= 5.0).mean()),
        "percentile_1": float(low),
        "percentile_99": float(high),
        "dynamic_range": float(high - low),
    }


def focus(gray: np.ndarray, sigma: float) -> Tuple[float, float, bool]:
    """Return ``(focus_ratio, scene_contrast, noise_limited)``.

    ``focus_ratio`` is edge energy divided by scene contrast, so it does not move
    when the lights are dimmed or the scene is simply low-contrast: it answers
    "of the variation this frame has, how much of it is fine detail?".

    ``noise_limited`` is True when white noise of the estimated strength would on
    its own account for every bit of edge energy in the frame. The measured ratio
    is then a floor, not a reading: a sharp noisy frame and a soft noisy frame
    look the same, and the caller should decline to judge focus rather than guess.
    """
    lap = laplacian(gray)
    if lap.size == 0:
        return (0.0, float(gray.std()), False)
    raw_lap_var = float(lap.var())
    floor = _LAP_NOISE_VAR * sigma * sigma
    lap_var = max(raw_lap_var - floor, 0.0)
    scene_var = max(float(gray.var()) - sigma * sigma, 0.0)
    scene_contrast = float(np.sqrt(scene_var))
    ratio = float(np.sqrt(lap_var) / max(scene_contrast, 1.0))
    # "Swamped" means the noise floor ate all but a tenth of the edge energy.
    noise_limited = bool(raw_lap_var <= floor * 1.10)
    return (ratio, scene_contrast, noise_limited)


def tile_grid(rows: int, cols: int, want: int) -> int:
    """How many tiles per side this working image can carry (at least 2)."""
    side = min(rows, cols)
    if side < 12:
        return 0
    return int(max(2, min(want, side // 6)))


def tile_detail(gray: np.ndarray, sigma: float, grid: int) -> Optional[Dict[str, Any]]:
    """Per-tile detail and brightness, with the noise floor subtracted.

    Returns None when the frame is too small to tile.

    ``raw`` is the mean absolute Laplacian in each tile. ``detail`` is the same
    figure less ``noise_floor``, the amount white noise of the estimated
    strength would have produced on its own, so a noisy blank tile still reads
    as blank. Both are kept: the caller needs ``raw`` and ``noise_floor``
    together to tell "this tile has no detail" from "no tile has detail the
    noise did not put there", which are very different findings.
    """
    lap = laplacian(gray)
    if lap.size == 0 or grid < 2:
        return None
    rows, cols = lap.shape
    tile_rows, tile_cols = rows // grid, cols // grid
    if tile_rows < 1 or tile_cols < 1:
        return None
    used = np.abs(lap[: tile_rows * grid, : tile_cols * grid])
    raw = used.reshape(grid, tile_rows, grid, tile_cols).mean(axis=(1, 3))
    floor = float(_LAP_NOISE_ABS * sigma)
    detail = np.maximum(raw - floor, 0.0)
    inner = gray[1:-1, 1:-1][: tile_rows * grid, : tile_cols * grid]
    brightness = inner.reshape(grid, tile_rows, grid, tile_cols).mean(axis=(1, 3))
    return {"detail": detail, "raw": raw, "brightness": brightness, "noise_floor": floor}


def largest_blob(mask: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
    """Size and cells of the largest 4-connected True region in a small grid."""
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best_size = 0
    best_cells: List[Tuple[int, int]] = []
    for start_row in range(rows):
        for start_col in range(cols):
            if not mask[start_row, start_col] or seen[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            seen[start_row, start_col] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if 0 <= next_row < rows and 0 <= next_col < cols:
                        if mask[next_row, next_col] and not seen[next_row, next_col]:
                            seen[next_row, next_col] = True
                            stack.append((next_row, next_col))
            if len(cells) > best_size:
                best_size = len(cells)
                best_cells = cells
    return best_size, best_cells


def blob_position(cells: List[Tuple[int, int]], grid: int) -> str:
    """Name the part of the view a blob sits in, for the fault message."""
    if not cells:
        return "the view"
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    row_at = sum(rows) / len(rows) / max(grid - 1, 1)
    col_at = sum(cols) / len(cols) / max(grid - 1, 1)
    vertical = "top" if row_at < 0.38 else ("bottom" if row_at > 0.62 else "middle")
    horizontal = "left" if col_at < 0.38 else ("right" if col_at > 0.62 else "centre")
    if vertical == "middle" and horizontal == "centre":
        return "the centre of the view"
    if vertical == "middle":
        return "the {} of the view".format(horizontal)
    if horizontal == "centre":
        return "the {} of the view".format(vertical)
    return "the {} {} of the view".format(vertical, horizontal)


def colour_cast(rgb: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """Grey-world colour balance. None for a mono frame.

    Pixels that are crushed to black or blown to white carry no colour, so they
    are left out before the channel means are taken.
    """
    if rgb is None:
        return None
    luma = rgb.mean(axis=2)
    usable = (luma > 8.0) & (luma < 248.0)
    pixels = rgb[usable] if usable.any() else rgb.reshape(-1, 3)
    means = pixels.mean(axis=0)
    overall = float(means.mean())
    spread = float(means.max() - means.min())
    deviations = means - overall
    index = int(np.argmax(np.abs(deviations)))
    return {
        "red": float(means[0]),
        "green": float(means[1]),
        "blue": float(means[2]),
        "cast": float(np.abs(deviations).max() / max(overall, 1.0)),
        "spread": spread,
        "channel": float(index),
        "direction": float(np.sign(deviations[index])),
        "coverage": float(usable.mean()),
    }


CHANNEL_NAMES = ("red", "green", "blue")


def frame_difference(gray: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    """How far two working images are apart, in three complementary ways.

    ``mean_diff`` says how much changed, ``changed_fraction`` says how much of
    the frame changed at all, and ``max_diff`` catches a single moving object on
    an otherwise still scene. The pair matters: a live sensor watching a still
    room gives a tiny ``mean_diff`` but a large ``changed_fraction``, because
    read noise jitters nearly every pixel. A repeated buffer gives zero of both.
    """
    other = match_shape(other, gray.shape)
    diff = np.abs(gray - other)
    return {
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "changed_fraction": float((diff > 0.5).mean()),
        "identical": float(bool(np.array_equal(gray, other))),
    }


def signature(gray: np.ndarray, size: int) -> Tuple[np.ndarray, float]:
    """A small brightness-invariant thumbnail of the scene, plus its contrast.

    The thumbnail is mean-subtracted and scaled to unit variance, so turning the
    lights up or down does not change it; only the layout of the scene does.
    """
    image = Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8), mode="L")
    small = np.asarray(image.resize((size, size), BILINEAR)).astype(np.float32)
    contrast = float(small.std())
    if contrast < 1e-6:
        return (np.zeros((size, size), dtype=np.float32), contrast)
    return ((small - small.mean()) / contrast, contrast)


def scene_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Correlation of two normalised signatures: 1.0 identical, 0.0 unrelated."""
    if first.shape != second.shape or first.size == 0:
        return 0.0
    return float(np.clip((first * second).mean(), -1.0, 1.0))


def measure(frame: Frame, sigma: Optional[float] = None) -> Dict[str, float]:
    """Every single-frame number, ready to be dropped into a report."""
    gray = frame.gray
    if sigma is None:
        sigma = noise_sigma(gray)
    ratio, contrast, noise_limited = focus(gray, sigma)
    metrics = exposure(gray)
    metrics.update(
        {
            "noise": float(sigma),
            "focus": float(ratio),
            "focus_noise_limited": float(noise_limited),
            "scene_contrast": float(contrast),
            "width": float(frame.width),
            "height": float(frame.height),
        }
    )
    return metrics
