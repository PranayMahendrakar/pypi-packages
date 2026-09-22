"""ml-inference-profiler: find the slow step in an ML inference pipeline.

    import ml_inference_profiler as mip

    steps = [("preprocess", prep), ("model", predict), ("postprocess", decode)]
    report = mip.profile_pipeline(steps, batch)
    print(report.summary())         # the tree, the bottleneck and what to do about it

For finer control, drive the :class:`Profiler` yourself:

    profiler = mip.Profiler("resnet50")
    with profiler.stage("preprocess"):
        with profiler.stage("resize"):     # stages nest to any depth
            image = resize(image)
    report = profiler.report()
"""

from __future__ import annotations

from typing import Any, Sequence

from .profiler import Profiler
from .report import FRAME_COLUMNS, ProfileReport, Stage, load_report
from .suggest import classify

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Profiler",
    "ProfileReport",
    "Stage",
    "profile_pipeline",
    "load_report",
    "classify",
    "FRAME_COLUMNS",
]


def profile_pipeline(
    steps: Sequence[Any],
    data: Any = None,
    *,
    name: str = "pipeline",
    repeats: int = 5,
    warmup: int = 1,
) -> ProfileReport:
    """Time a pipeline in one line and return the report.

    Args:
        steps: ``(label, callable)`` pairs, or bare callables whose ``__name__`` becomes
            the label. Each step receives the previous step's output; a step that returns
            ``None`` passes its input along.
        data: The input given to the first step at the start of every repeat.
        name: Name for the report.
        repeats: Timed passes over the pipeline (at least 1).
        warmup: Untimed passes first (0 or more), so cold caches do not skew the timings.

    Returns:
        A :class:`ProfileReport`; ``print(report.summary())`` explains it.

    Example:
        >>> report = profile_pipeline([("double", lambda x: x * 2)], [1, 2], repeats=2)
        >>> report.find("double").calls
        2
    """
    return Profiler(name=name).run(steps, data, repeats=repeats, warmup=warmup)
