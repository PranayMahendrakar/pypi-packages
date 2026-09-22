"""smart-crop-ai: crop to the interesting part of an image, not the middle of it.

Give it a size or an aspect ratio and it scans every window of that shape for the
one holding the most detail, then tells you how much better that window was than
a plain centre crop. When the answer is "not at all", it says so and takes the
centre instead of inventing a subject.

    >>> import numpy as np, smart_crop_ai
    >>> from PIL import Image
    >>> wall = np.full((200, 400, 3), 210, dtype=np.uint8)
    >>> wall[40:120, 300:380] = 20                      # something, off to the right
    >>> result = smart_crop_ai.crop(Image.fromarray(wall), ratio=1.0)
    >>> result.box[0] > (400 - 200) // 2                # moved right of centre
    True
    >>> result.confidence > 0
    True

Pure numpy and Pillow. No OpenCV, no torch, no model download, nothing touches
the network, and the same image always gives the same box.
"""
from __future__ import annotations

from ._core import (
    ANALYSIS_MAX_SIDE,
    AUTO_STRATEGY,
    DEFAULT_PADDING,
    DEFAULT_STRATEGY,
    MAX_PADDING,
    STRATEGIES,
    crop,
    crop_to_file,
    thumbnail,
)
from ._energy import ENERGY_BUILDERS, edge_map, entropy_map, saliency_map
from ._images import IMAGE_SUFFIXES, open_image
from ._result import CONFIDENCE_LABELS, CropResult, confidence_label

__version__ = "0.1.0"

__all__ = [
    "crop",
    "crop_to_file",
    "thumbnail",
    "CropResult",
    "STRATEGIES",
    "DEFAULT_STRATEGY",
    "DEFAULT_PADDING",
    "MAX_PADDING",
    "AUTO_STRATEGY",
    "ANALYSIS_MAX_SIDE",
    "CONFIDENCE_LABELS",
    "confidence_label",
    "ENERGY_BUILDERS",
    "saliency_map",
    "entropy_map",
    "edge_map",
    "open_image",
    "IMAGE_SUFFIXES",
    "__version__",
]
