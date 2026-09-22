"""image-quality-ai: find the photos that will waste a model's time.

Blur, darkness, blown highlights, grain, flat contrast and a subject falling out
of frame, measured with nothing but numpy and Pillow.

    >>> import numpy as np, image_quality_ai
    >>> report = image_quality_ai.assess(np.zeros((64, 64), dtype=np.uint8))
    >>> report.usable
    False
    >>> report.issues[0][:20]
    'There is no detail i'

Every measure reports the raw number it computed as well as a 0-100 score, and
every threshold behind those scores is a documented constant you can override.
No model is downloaded, no OpenCV, nothing touches the network.
"""
from __future__ import annotations

from ._core import CRITICAL_MEASURES, ImageAssessor, assess, assess_batch, is_blurry
from ._measures import MEASURE_NAMES
from ._report import BatchReport, Metric, QualityReport
from ._thresholds import (
    DEFAULT_THRESHOLDS,
    Thresholds,
    describe_thresholds,
)

__version__ = "0.1.0"

__all__ = [
    "assess",
    "assess_batch",
    "is_blurry",
    "ImageAssessor",
    "QualityReport",
    "BatchReport",
    "Metric",
    "Thresholds",
    "DEFAULT_THRESHOLDS",
    "describe_thresholds",
    "MEASURE_NAMES",
    "CRITICAL_MEASURES",
    "__version__",
]
