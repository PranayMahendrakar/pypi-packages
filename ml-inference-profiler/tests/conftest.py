"""Shared fixtures: a fake clock for exact timing assertions, and report factories."""

from __future__ import annotations

from typing import List, Optional

import pytest

from ml_inference_profiler import ProfileReport, Stage
from ml_inference_profiler._timing import measure_stage_overhead


class FakeClock:
    """A monotonic clock that advances exactly one tick per call.

    With one tick per ``clock()`` call, a stage that wraps nothing takes one tick and a
    stage that wraps one child takes three (enter parent, enter child, leave child, leave
    parent), which makes self time and cumulative time exact instead of approximate.
    """

    def __init__(self, tick: float = 0.001) -> None:
        self.tick = tick
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.tick
        return self.now


class FrozenClock:
    """A clock that never advances, so every stage measures exactly zero."""

    def __call__(self) -> float:
        return 123.0


@pytest.fixture(scope="session", autouse=True)
def _warm_overhead_cache():
    """Measure the per-stage overhead once, before any test patches the clock."""
    measure_stage_overhead()


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("ml_inference_profiler.profiler.clock", clock)
    return clock


@pytest.fixture
def frozen_clock(monkeypatch):
    clock = FrozenClock()
    monkeypatch.setattr("ml_inference_profiler.profiler.clock", clock)
    return clock


def make_stage(
    label: str,
    total_ms: float,
    *,
    calls: int = 1,
    self_ms: Optional[float] = None,
    p95_ms: Optional[float] = None,
    median_ms: Optional[float] = None,
    first_ms: Optional[float] = None,
    max_ms: Optional[float] = None,
    depth: int = 0,
    parent: Optional[str] = None,
    errors: int = 0,
    run_total: float = 100.0,
) -> Stage:
    """Build a Stage with plausible derived numbers, for testing the advice rules."""
    self_ms = total_ms if self_ms is None else self_ms
    median_ms = total_ms / calls if median_ms is None else median_ms
    p95_ms = median_ms if p95_ms is None else p95_ms
    max_ms = p95_ms if max_ms is None else max_ms
    first_ms = max_ms if first_ms is None else first_ms
    return Stage(
        label=label,
        calls=calls,
        total_ms=total_ms,
        mean_ms=total_ms / calls,
        p95_ms=p95_ms,
        share=100.0,
        depth=depth,
        parent=parent,
        self_ms=self_ms,
        pct_of_total=100.0 * total_ms / run_total,
        self_pct_of_total=100.0 * self_ms / run_total,
        median_ms=median_ms,
        min_ms=median_ms,
        max_ms=max_ms,
        first_ms=first_ms,
        errors=errors,
        path=label if parent is None else f"{parent}/{label}",
    )


def make_report(stages: List[Stage], **kwargs) -> ProfileReport:
    """Build a ProfileReport whose total is the sum of the top-level stages."""
    total = sum(s.total_ms for s in stages if s.depth == 0)
    kwargs.setdefault("total_ms", total)
    return ProfileReport(stages=stages, **kwargs)


@pytest.fixture
def stage_factory():
    return make_stage


@pytest.fixture
def report_factory():
    return make_report
