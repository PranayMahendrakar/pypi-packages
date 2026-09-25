"""audio-anomaly: detect unusual sounds in machine or environment recordings without labelled faults.

The 90% case is one line::

    report = audio_anomaly.detect("pump.wav")
    print(report.summary())

Give it a known-good recording and every frame is judged against that instead::

    report = audio_anomaly.detect("pump_today.wav", reference="pump_healthy.wav")

For checking many recordings against one reference, build a :class:`Monitor`
once and call :meth:`Monitor.check` on each. :func:`spectral_profile` gives the
typical spectrum of a recording, to save with ``numpy.save`` and pass back as
``reference=`` later.
"""

from ._detect import Monitor, detect, spectral_profile
from ._report import KIND_MEANINGS, KINDS, AudioAnomalyReport, Event

__version__ = "0.1.0"

__all__ = [
    "detect",
    "Monitor",
    "spectral_profile",
    "AudioAnomalyReport",
    "Event",
    "KINDS",
    "KIND_MEANINGS",
    "__version__",
]
