"""vision-anomaly: spot unusual images against a set of normal ones.

No labelled defects, no training, no model download. Show it images you are
happy with, then ask it about anything else.

    >>> import numpy as np
    >>> from vision_anomaly import Detector
    >>> good = [np.full((64, 64, 3), 130, dtype=np.uint8) for _ in range(6)]
    >>> detector = Detector().fit(good)
    >>> detector.score(good[0])
    0.0
    >>> result = detector.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    >>> result.anomalous
    True

Every number an image is judged on is hand-built and documented: per-channel
colour histograms, edge density and gradient orientation on a coarse grid, and
block-wise brightness, contrast and texture. Each one is standardised against
the fitted set with its median and MAD, so a score is in robust sigmas, and the
result says which families moved and which grid cells moved most.

What that buys and what it does not: this compares hand-built statistics, so it
catches gross departures - the wrong colour, the wrong texture, a missing part,
a different scene - and will miss a subtle defect that a trained model would
see. The README says so at greater length.
"""
from __future__ import annotations

from ._detector import DEFAULT_SENSITIVITY, Detector, detect_anomalies
from ._features import (
    GROUP_DESCRIPTIONS,
    GROUP_NAMES,
    FeatureConfig,
    describe_features,
)
from ._loading import ANALYSIS_SIZE
from ._profile import INLIER_BAND, WEAK_PROFILE_IMAGES, Profile
from ._result import AnomalyResult, BatchReport, Box

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "detect_anomalies",
    "AnomalyResult",
    "BatchReport",
    "Box",
    "Profile",
    "FeatureConfig",
    "describe_features",
    "GROUP_NAMES",
    "GROUP_DESCRIPTIONS",
    "DEFAULT_SENSITIVITY",
    "ANALYSIS_SIZE",
    "INLIER_BAND",
    "WEAK_PROFILE_IMAGES",
    "__version__",
]
