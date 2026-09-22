"""model-benchmark: put several models on the same task and see which one wins.

Latency, memory and accuracy for every model, side by side, in three lines::

    from model_benchmark import benchmark

    report = benchmark({"fast": fast_model, "accurate": big_model}, (X, y), metric="accuracy")
    print(report.summary())

A model is anything callable, or anything with a ``.predict`` method, so
scikit-learn estimators, torch modules and plain functions all work without this
package importing any of them. One model raising does not end the run: it lands
in ``report.failures`` and the others are still measured.
"""
from ._core import Benchmark, benchmark, compare
from ._metrics import METRICS
from ._result import BenchmarkReport, ModelResult, Timing

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "BenchmarkReport",
    "METRICS",
    "ModelResult",
    "Timing",
    "benchmark",
    "compare",
    "__version__",
]
