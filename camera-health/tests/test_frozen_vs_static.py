"""The distinction this package exists for: a frozen feed is not a static scene.

A camera watching an empty corridor produces frames that look identical. They are
not: read noise moves nearly every pixel by a grey level between exposures. A
frozen feed is a repeated buffer, so no pixel moves at all. Both sides of that
are tested here, because getting either one wrong makes the package useless -
crying wolf on every still room, or missing a camera that has actually stopped.
"""

from __future__ import annotations

import numpy as np

from camera_health import CRITICAL, FROZEN, WARNING, Monitor, check, check_stream
from conftest import live_frame, scene


def test_a_static_scene_is_never_called_frozen(base, still_run):
    """Twelve frames of an empty corridor, well past freeze_frames of 5."""
    monitor = Monitor(freeze_frames=5)

    for frame in still_run:
        health = monitor.update(frame)
        assert not health.has(FROZEN), health.summary()

    assert monitor.state == "healthy"
    assert monitor.uptime == 1.0
    assert monitor.alerts == []


def test_a_static_scene_says_so_in_a_note(base):
    """The report explains why near-identical frames were still accepted."""
    health = check(live_frame(base, 2), previous=live_frame(base, 1))

    assert not health.has(FROZEN)
    assert any("static but the sensor is live" in note for note in health.notes)
    assert health.metrics["mean_diff"] < 2.5
    assert health.metrics["changed_fraction"] > 0.2


def test_mean_change_alone_cannot_tell_them_apart(base):
    """The reason changed_fraction exists: mean_diff is near zero for both."""
    live = check(live_frame(base, 2), previous=live_frame(base, 1))
    frozen = check(base, previous=base)

    assert live.metrics["mean_diff"] < 2.5
    assert frozen.metrics["mean_diff"] == 0.0
    # The share of pixels that moved is what separates them, by a wide margin.
    assert live.metrics["changed_fraction"] > 0.5
    assert frozen.metrics["changed_fraction"] == 0.0


def test_one_repeat_is_only_a_warning_with_the_monitor_default(base):
    """A single dropped frame on a still scene looks exactly like a freeze."""
    monitor = Monitor(freeze_frames=5)
    stuck = live_frame(base, 0)

    first = monitor.update(stuck)
    second = monitor.update(stuck)

    assert not first.has(FROZEN)
    assert second.fault(FROZEN).severity == WARNING
    assert second.ok
    assert "2 identical frames of the 5 needed" in second.fault(FROZEN).message


def test_a_frozen_feed_is_critical_once_freeze_frames_is_reached(base):
    monitor = Monitor(freeze_frames=5)
    stuck = live_frame(base, 0)
    severities = []

    for _ in range(6):
        severities.append(monitor.update(stuck).fault(FROZEN))

    assert severities[0] is None
    assert [fault.severity for fault in severities[1:4]] == [WARNING] * 3
    assert severities[4].severity == CRITICAL
    assert "bit for bit identical" in severities[4].message
    assert monitor.state == "failed"
    assert [alert.kind for alert in monitor.alerts].count(FROZEN) == 2


def test_freeze_frames_is_honoured(base):
    """Raising the count delays the verdict by exactly that many frames."""
    stuck = live_frame(base, 0)
    for wanted in (2, 3, 8):
        monitor = Monitor(freeze_frames=wanted)
        first_critical = None
        for index in range(wanted + 2):
            fault = monitor.update(stuck).fault(FROZEN)
            if fault is not None and fault.severity == CRITICAL and first_critical is None:
                first_critical = index
        # Frame 0 has no predecessor, so the wanted-th identical frame is index wanted - 1.
        assert first_critical == wanted - 1, wanted


def test_a_near_identical_repeat_counts_as_frozen(base):
    """A re-encoded buffer differs by a hair; that is still not a live sensor."""
    nudged = base.astype(np.float32)
    nudged[0, 0] += 3.0
    monitor = Monitor(freeze_frames=3)

    monitor.update(base)
    monitor.update(nudged)
    health = monitor.update(base)

    fault = health.fault(FROZEN)
    assert fault is not None and fault.severity == CRITICAL
    assert "identical to within" in fault.message


def test_a_real_moving_scene_clears_the_repeat_count(base):
    """One genuine change resets the run, so a later repeat starts over."""
    monitor = Monitor(freeze_frames=3)
    stuck = live_frame(base, 0)

    monitor.update(stuck)
    monitor.update(stuck)
    monitor.update(scene(layout=1))
    health = monitor.update(stuck)

    assert not health.has(FROZEN)


def test_a_single_frame_cannot_be_asked_about_freezing(base):
    health = check(base)

    assert not health.was_checked(FROZEN)
    assert "no previous frame" in health.not_checkable[FROZEN]
    assert not health.has(FROZEN)


def test_a_static_run_through_check_stream_stays_healthy(still_run):
    report = check_stream(still_run, freeze_frames=5)

    assert report.ok
    assert report.uptime == 1.0
    assert report.first_failure is None
    assert FROZEN not in report.fault_counts


def test_a_frozen_run_through_check_stream_fails_at_the_right_frame(base):
    stuck = live_frame(base, 0)
    frames = [live_frame(base, tick) for tick in range(3)] + [stuck] * 6

    report = check_stream(frames, freeze_frames=5)

    assert not report.ok
    assert report.first_failure == 7
    assert report.fault_counts[FROZEN] >= 2
    assert report.frames_with(FROZEN) == [4, 5, 6, 7, 8]
