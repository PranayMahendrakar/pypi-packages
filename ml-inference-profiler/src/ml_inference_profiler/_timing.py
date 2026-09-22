"""Clock access and the measured per-stage overhead floor.

Everything in this package times with :func:`time.perf_counter`, the only clock in the
standard library that is monotonic and has the resolution a millisecond-scale pipeline
needs. Durations are kept in seconds internally and converted to milliseconds once, at
the edge, so rounding happens in exactly one place.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

#: The clock used for every measurement in this package.
clock = time.perf_counter

_overhead_seconds: Optional[float] = None


def to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return float(seconds) * 1000.0


def percentile_ms(durations: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of ``durations`` (seconds), in milliseconds.

    Linear interpolation between the two nearest samples, so a single call gives that
    call's duration rather than an error.
    """
    if not durations:
        return 0.0
    if len(durations) == 1:
        return to_ms(durations[0])
    return to_ms(float(np.percentile(np.asarray(durations, dtype=float), q)))


def median_ms(durations: Sequence[float]) -> float:
    """The median of ``durations`` (seconds), in milliseconds."""
    if not durations:
        return 0.0
    return to_ms(float(np.median(np.asarray(durations, dtype=float))))


def measure_stage_overhead(samples: int = 300, rounds: int = 3) -> float:
    """Measure what one ``with profiler.stage(...)`` costs, in seconds.

    This is the floor of the whole measurement: a stage cannot be timed more finely than
    the bookkeeping that times it. The result is measured once per process (it depends on
    the machine, not on the pipeline) and cached.

    Returns:
        The best-of-``rounds`` mean cost of entering and leaving one empty stage, in
        seconds. Best-of, not mean-of, because a scheduler hiccup can only make a
        measurement slower, never faster.
    """
    global _overhead_seconds
    if _overhead_seconds is None:
        _overhead_seconds = _measure_stage_overhead(samples, rounds)
    return _overhead_seconds


def _measure_stage_overhead(samples: int, rounds: int) -> float:
    from .profiler import Profiler  # local import: profiler imports this module

    samples = max(int(samples), 1)
    scratch = Profiler(name="_overhead_calibration")
    best: Optional[float] = None
    for _ in range(max(int(rounds), 1)):
        scratch.reset()
        start = clock()
        for _ in range(samples):
            with scratch.stage("overhead"):
                pass
        per_call = (clock() - start) / samples
        if best is None or per_call < best:
            best = per_call
    scratch.reset()
    return max(best or 0.0, 0.0)


def reset_overhead_cache() -> None:
    """Forget the cached overhead measurement (used by the tests)."""
    global _overhead_seconds
    _overhead_seconds = None


__all__: List[str] = [
    "clock",
    "to_ms",
    "percentile_ms",
    "median_ms",
    "measure_stage_overhead",
    "reset_overhead_cache",
]
