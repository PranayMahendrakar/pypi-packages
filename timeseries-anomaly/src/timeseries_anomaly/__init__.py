"""timeseries-anomaly: find anomalies in any time series or IoT signal with one call.

No training, no model files, no configuration. Point it at numbers and it tells you
which ones do not belong, what they should have been, and how far off they were::

    import timeseries_anomaly

    result = timeseries_anomaly.detect(readings)
    print(result.summary())
    print(result.anomalies)

For data that arrives in batches, learn the baseline once and reuse it::

    detector = timeseries_anomaly.Detector().fit(history)
    result = detector.score(new_batch)
"""
from ._core import Detector, detect
from ._methods import METHODS
from ._result import AnomalyResult

__version__ = "0.1.0"

__all__ = [
    "AnomalyResult",
    "Detector",
    "METHODS",
    "detect",
    "__version__",
]
