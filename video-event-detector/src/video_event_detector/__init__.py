"""Detect falls, crowding, abandoned objects and sudden movement from a sequence of frames.

These are motion heuristics on a running background model, not activity
recognition. Every event is a hypothesis to review, not a verdict.

Quick use::

    from video_event_detector import detect_events
    report = detect_events(frames, fps=10)      # list of arrays / PIL images, or a folder
    print(report.summary())

For live frames, feed a :class:`Detector` one frame at a time with
:meth:`Detector.update`.
"""

from .api import detect_events
from .detector import Detector
from .events import EVENT_KINDS, Event, EventReport

__version__ = "0.1.0"

__all__ = ["Detector", "Event", "EventReport", "EVENT_KINDS", "detect_events", "__version__"]
