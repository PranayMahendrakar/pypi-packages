"""The throughput the package promises: 30 frames of 1080p in a couple of seconds.

A camera health check that cannot keep up with the camera is not a health check,
so the budget is part of the contract rather than a nice-to-have. The frames are
generated here with numpy and Pillow like every other frame in the suite.

The budget below is deliberately loose against what the code actually does - the
run measures around 0.35s on an ordinary laptop - because a shared CI box under
load should report a real regression, not a busy neighbour.
"""

from __future__ import annotations

import time

import pytest

from camera_health import Monitor, check, check_stream
from conftest import big_scene, live_frame

BUDGET_SECONDS = 4.0
"""Ample room over the measured ~0.35s, still far under "a couple of seconds"."""


@pytest.fixture(scope="module")
def frames_1080p():
    """Thirty 1080p frames of a live, static scene - built once for this module."""
    base = big_scene()
    return [live_frame(base, tick) for tick in range(30)]


def test_thirty_1080p_frames_run_through_check_stream_inside_the_budget(frames_1080p):
    started = time.perf_counter()
    report = check_stream(frames_1080p)
    elapsed = time.perf_counter() - started

    assert report.frames_checked == 30
    assert elapsed < BUDGET_SECONDS, "30 frames of 1080p took {:.2f}s".format(elapsed)


def test_thirty_1080p_frames_run_through_the_monitor_inside_the_budget(frames_1080p):
    monitor = Monitor()

    started = time.perf_counter()
    for frame in frames_1080p:
        monitor.update(frame)
    elapsed = time.perf_counter() - started

    assert monitor.n_frames == 30
    assert elapsed < BUDGET_SECONDS, "30 frames of 1080p took {:.2f}s".format(elapsed)


def test_a_live_1080p_static_scene_is_healthy_and_not_frozen(frames_1080p):
    """Speed must not be bought by giving up the verdict at full resolution."""
    report = check_stream(frames_1080p, freeze_frames=5)

    assert report.ok
    assert "frozen" not in report.fault_counts


def test_a_1080p_frame_is_checked_the_same_way_twice(frames_1080p):
    first = check(frames_1080p[0])
    second = check(frames_1080p[0])

    assert first.score == second.score
    assert first.to_dict() == second.to_dict()
