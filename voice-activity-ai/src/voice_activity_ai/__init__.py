"""voice-activity-ai: find where someone is actually speaking in a recording.

    from voice_activity_ai import detect
    activity = detect("meeting.wav")
    print(activity.summary())

This is an energy and spectral heuristic, not a neural voice activity detector.
It needs nothing but numpy, downloads no model, and is good at what heuristics
are good at: trimming dead air, splitting a recording into utterances, and
telling you how much of a file is actually someone talking.
"""

from __future__ import annotations

from ._result import Segment, VoiceActivity
from .core import detect, speech_ratio, trim_silence

__version__ = "0.1.0"

__all__ = [
    "detect",
    "speech_ratio",
    "trim_silence",
    "VoiceActivity",
    "Segment",
    "__version__",
]
