"""camera-health: is this camera actually seeing anything?

    >>> from camera_health import check
    >>> health = check(frame, previous=last_frame)
    >>> print(health.summary())

One call over a frame reports an obstructed lens, a defocused one, darkness,
overexposure, a frozen feed, a scene that has been tampered with, a colour cast
and heavy sensor noise - with the numbers behind every verdict, and an explicit
"not checkable" for anything a single frame cannot answer.
"""

from ._frames import Frame, list_images, load_frame
from .checks import check, check_stream, score_from
from .faults import (
    COLOUR_CAST,
    CRITICAL,
    DARKNESS,
    DEFOCUS,
    DRIFT,
    FAULT_KINDS,
    FROZEN,
    INFO,
    NOISE,
    OBSTRUCTION,
    OVEREXPOSURE,
    SEVERITIES,
    TAMPERING,
    WARNING,
    Fault,
    canonical_kind,
)
from .monitor import DEGRADED, FAILED, HEALTHY, STATES, Alert, Monitor
from .result import FrameHealth, StreamReport
from .thresholds import DEFAULT_THRESHOLDS, Thresholds

__version__ = "0.1.0"

__all__ = [
    "Alert",
    "COLOUR_CAST",
    "CRITICAL",
    "DARKNESS",
    "DEFAULT_THRESHOLDS",
    "DEFOCUS",
    "DEGRADED",
    "DRIFT",
    "FAILED",
    "FAULT_KINDS",
    "FROZEN",
    "Fault",
    "Frame",
    "FrameHealth",
    "HEALTHY",
    "INFO",
    "Monitor",
    "NOISE",
    "OBSTRUCTION",
    "OVEREXPOSURE",
    "SEVERITIES",
    "STATES",
    "StreamReport",
    "TAMPERING",
    "Thresholds",
    "WARNING",
    "__version__",
    "canonical_kind",
    "check",
    "check_stream",
    "list_images",
    "load_frame",
    "score_from",
]
