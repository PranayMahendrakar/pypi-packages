"""sensor-anomaly: find what is wrong across a whole rack of sensors in one call.

Point it at a wide table - one column per channel - and it answers three questions
at once::

    import sensor_anomaly

    report = sensor_anomaly.detect(df)
    print(report.summary())
    print(report.worst_channels)

1. Which channels misbehaved on their own (robust z-score of the rolling residual,
   so drift and daily cycles do not drown the result).
2. Which rows are wrong only *between* channels - flow falling while current rises,
   with both readings still inside their normal ranges. That broken correlation is
   invisible per channel and is exactly what the cross-channel model is for. Its
   cut is corrected for the size of the table, so channels that all track one
   process variable do not read as a plant-wide fault.
3. Which sensors are simply broken: flatlined, stuck at zero, railed at their
   measurement limits, dropping bursts of samples, or stepping to a new level.
   None of those are statistical outliers, and all of them mean a bad sensor.

Same settings across many tables::

    detector = sensor_anomaly.Detector(sensitivity=4.0, time="timestamp")
    report = detector.detect(batch)
"""

from ._core import Detector, detect
from ._faults import FAULT_HELP, FAULT_ORDER
from ._result import ChannelResult, Event, SensorReport

__version__ = "0.1.0"

__all__ = [
    "ChannelResult",
    "Detector",
    "Event",
    "FAULT_HELP",
    "FAULT_ORDER",
    "SensorReport",
    "detect",
    "__version__",
]
