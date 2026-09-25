"""call-ai-metrics: talk time, interruptions, silence and pace in a two-party call.

    from call_ai_metrics import analyze
    report = analyze("call.wav", speakers=["agent", "customer"])
    print(report.summary())

Give it a stereo recording with one party per channel, a pair of mono arrays, or
diarized segments ``[(start_s, end_s, speaker), ...]`` from any tool. Every figure
is computed from "who was speaking when", so it is only as good as that
segmentation: on audio the speech is found by a per-channel energy heuristic, not
a neural model, and on segments it is whatever the diarizer said.
"""

from __future__ import annotations

from ._report import (
    BleedCheck,
    CallReport,
    Crosstalk,
    Interruption,
    Segment,
    SpeakerStats,
    Turn,
)
from ._segments import count_words
from .core import CallAnalyzer, analyze

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "CallAnalyzer",
    "CallReport",
    "SpeakerStats",
    "Segment",
    "Turn",
    "Interruption",
    "BleedCheck",
    "Crosstalk",
    "count_words",
    "__version__",
]
