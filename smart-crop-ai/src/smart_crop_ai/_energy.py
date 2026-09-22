"""Energy maps: which pixels of an image are worth keeping.

Pure numpy. Every map returned here is a ``float64`` array in ``[0, 1]`` with the
same shape as the working image, where 1 marks a pixel as interesting as this
package can measure and 0 marks a pixel with nothing in it at all.

The scales are absolute, not per-image min/max: a photograph of a blank wall
produces a map of near-zeros rather than a map of amplified sensor noise. That
is what lets :mod:`smart_crop_ai._core` tell a confident crop from a guess.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Luma weights (Rec. 601), the usual perceptual grey.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# A luma step of this size across one pixel saturates the gradient term.
_GRAD_FULL_SCALE = 2.0
# A local colour standard deviation of this size saturates the variance term.
_STD_FULL_SCALE = 0.25
# Weights of the two saliency terms; they sum to 1.
_GRAD_WEIGHT = 0.65
_STD_WEIGHT = 0.35
# Sobel magnitude below this is texture-free noise, not an edge.
_EDGE_FLOOR = 0.10

#: Energy below this is not detail, it is arithmetic.
#:
#: Downscaling a perfectly uniform image with a resampling filter leaves float
#: residue of order 1e-6 behind, and without this floor the window search would
#: happily maximise that residue and report a confident crop of a blank wall.
#: Real detail sits two or more orders of magnitude above it.
NOISE_FLOOR = 1e-4

_SOBEL_X = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
_SOBEL_Y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


def denoise(energy: np.ndarray) -> np.ndarray:
    """Flatten resampling residue to exact zero, so a blank wall scores nothing.

    Anything under :data:`NOISE_FLOOR` is arithmetic, not detail. Leaving it in
    would let the window search maximise float noise and hand back a confident
    crop of an empty image.
    """
    return np.where(energy < NOISE_FLOOR, 0.0, energy)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Perceptual grey of an ``(h, w, 3)`` float array in ``[0, 1]``."""
    return rgb @ _LUMA


def conv3(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 convolution with edge replication, no scipy."""
    padded = np.pad(a, 1, mode="edge")
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            if weight:
                out += weight * padded[dy : dy + h, dx : dx + w]
    return out


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a 2-D float array."""
    return np.hypot(conv3(gray, _SOBEL_X), conv3(gray, _SOBEL_Y))


def box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a ``(2*radius+1)`` square window, edges replicated."""
    if radius <= 0:
        return a.astype(np.float64, copy=True)
    h, w = a.shape
    padded = np.pad(a.astype(np.float64), radius, mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    k = 2 * radius + 1
    total = (
        integral[k : k + h, k : k + w]
        - integral[0:h, k : k + w]
        - integral[k : k + h, 0:w]
        + integral[0:h, 0:w]
    )
    return total / float(k * k)


def box_std(a: np.ndarray, radius: int) -> np.ndarray:
    """Standard deviation over a square window, computed from box means."""
    mean = box_mean(a, radius)
    mean_sq = box_mean(a * a, radius)
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))


def integral_image(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero first row and column."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return out


def saliency_map(rgb: np.ndarray) -> np.ndarray:
    """Gradient magnitude plus local colour variance: the default workhorse.

    Edges carry most of the weight, local colour spread fills in textured
    regions that a pure edge detector would leave hollow.
    """
    gray = luma(rgb)
    grad = np.clip(sobel_magnitude(gray) / _GRAD_FULL_SCALE, 0.0, 1.0)
    radius = _radius_for(rgb.shape, 2)
    channel_std = np.stack([box_std(rgb[:, :, c], radius) for c in range(rgb.shape[2])], axis=0)
    colour = np.clip(channel_std.mean(axis=0) / _STD_FULL_SCALE, 0.0, 1.0)
    energy = _GRAD_WEIGHT * grad + _STD_WEIGHT * colour
    return denoise(np.clip(box_mean(energy, _radius_for(rgb.shape, 1)), 0.0, 1.0))


def entropy_map(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    """Local Shannon entropy of the luma histogram, scaled to ``[0, 1]``.

    Better than saliency for texture-heavy images, where every pixel has a
    gradient but only some regions actually carry information.
    """
    gray = luma(rgb)
    quantized = np.clip((gray * bins).astype(np.int64), 0, bins - 1)
    radius = _radius_for(rgb.shape, 4)
    entropy = np.zeros(gray.shape, dtype=np.float64)
    for b in range(bins):
        p = box_mean((quantized == b).astype(np.float64), radius)
        np.subtract(entropy, np.where(p > 0.0, p * np.log2(np.where(p > 0.0, p, 1.0)), 0.0), out=entropy)
    return denoise(np.clip(entropy / np.log2(bins), 0.0, 1.0))


def edge_map(rgb: np.ndarray) -> np.ndarray:
    """Sobel edge density: the fraction of nearby pixels sitting on an edge.

    Better than saliency for product shots, where the subject is a clean shape
    on a plain background and gradient strength inside it means nothing.
    """
    magnitude = sobel_magnitude(luma(rgb))
    threshold = max(float(magnitude.mean() + 0.5 * magnitude.std()), _EDGE_FLOOR)
    edges = (magnitude > threshold).astype(np.float64)
    return denoise(np.clip(box_mean(edges, _radius_for(rgb.shape, 3)), 0.0, 1.0))


def _radius_for(shape: Tuple[int, ...], wanted: int) -> int:
    """Shrink a filter radius so it still fits inside a very small image."""
    smallest = min(int(shape[0]), int(shape[1]))
    return max(0, min(wanted, (smallest - 1) // 2))


#: Strategy name -> map builder. ``center`` needs no map and is not listed.
ENERGY_BUILDERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "saliency": saliency_map,
    "entropy": entropy_map,
    "edges": edge_map,
}
