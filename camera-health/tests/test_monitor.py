"""Monitor: the things only a run of frames can show."""

from __future__ import annotations

import json

import numpy as np
import pytest

from camera_health import (
    DEGRADED,
    DRIFT,
    FAILED,
    HEALTHY,
    STATES,
    TAMPERING,
    Alert,
    Monitor,
)
from conftest import live_frame, obstructed, scene


def test_a_healthy_run_stays_healthy(still_run):
    monitor = Monitor()

    for frame in still_run:
        monitor.update(frame)

    assert monitor.state == HEALTHY
    assert monitor.state in STATES
    assert monitor.uptime == 1.0
    assert monitor.alerts == []
    assert monitor.last.ok


def test_uptime_before_any_frame_and_after_a_bad_one(base):
    monitor = Monitor()
    assert monitor.uptime == 1.0
    assert "no frames seen yet" in monitor.summary()

    monitor.update(base)
    monitor.update(np.zeros_like(base))

    assert monitor.uptime == pytest.approx(0.5)
    assert monitor.state == FAILED


def test_the_first_frame_is_adopted_as_the_reference(base):
    monitor = Monitor()

    first = monitor.update(base)

    assert monitor.reference_adopted
    assert not first.was_checked(TAMPERING)
    assert "cannot differ from itself" in first.not_checkable[TAMPERING]
    assert any("frame 0 was adopted" in note for note in monitor.notes)
    assert "reference: frame 0 of this run" in monitor.summary()


def test_a_given_reference_is_used_from_the_first_frame(base, other_scene):
    monitor = Monitor(base)

    health = monitor.update(other_scene)

    assert not monitor.reference_adopted
    assert health.was_checked(TAMPERING)
    assert health.has(TAMPERING)


def test_an_alert_is_raised_once_per_fault_and_again_when_it_worsens(base):
    monitor = Monitor()
    monitor.update(base)
    for _ in range(4):
        monitor.update(np.zeros_like(base))

    kinds = [alert.kind for alert in monitor.alerts]
    assert kinds.count("darkness") == 1
    assert monitor.alerts[0].when_index == 1
    assert monitor.alerts[0].message.startswith("[critical]")
    assert "frame 1" in monitor.alerts[0].line()


def test_a_camera_that_comes_back_raises_a_recovered_alert(base, still_run):
    monitor = Monitor(base, history=4)
    monitor.update(still_run[0])
    monitor.update(np.zeros_like(base))
    assert monitor.state == FAILED

    monitor.update(still_run[1])

    assert monitor.state == DEGRADED  # back, but one clean frame is not yet proof
    assert monitor.alerts[-1].kind == "recovered"
    assert "usable again at frame 2, after 1 unusable frame(s)" in monitor.alerts[-1].message

    for frame in still_run[2:]:
        monitor.update(frame)

    assert monitor.state == HEALTHY  # the bad frame has fallen out of the window
    assert monitor.uptime < 1.0  # but uptime still remembers it


def test_a_warning_only_frame_is_degraded_not_failed(base):
    monitor = Monitor(base)
    monitor.update(live_frame(base, 0))

    health = monitor.update(obstructed(live_frame(base, 1), share=0.2))

    assert health.ok
    assert health.faults
    assert monitor.state == DEGRADED


def test_a_camera_being_nudged_reads_as_drift_not_tampering(base):
    """One pixel of pan per frame: the match falls slowly, never off a cliff."""
    monitor = Monitor(base, freeze_frames=5, history=30)
    drifted = None

    for tick in range(16):
        drifted = monitor.update(live_frame(np.roll(base, tick, axis=1), tick))

    fault = drifted.fault(DRIFT)
    assert fault is not None
    assert not drifted.has(TAMPERING)
    assert "drifted away from the reference" in fault.message
    assert drifted.ok and drifted.score < 100.0


def test_drift_is_not_decided_on_too_few_frames(base):
    monitor = Monitor(base)

    health = monitor.update(live_frame(base, 0))

    assert not health.was_checked(DRIFT)
    assert "at least" in health.not_checkable[DRIFT]


def test_a_steady_camera_never_drifts(base, still_run):
    monitor = Monitor(base, history=30)

    for frame in still_run:
        health = monitor.update(frame)

    assert not health.has(DRIFT)
    assert health.checks[DRIFT] == "pass"


def test_history_is_bounded(base, still_run):
    monitor = Monitor(base, history=4)

    for frame in still_run:
        monitor.update(frame)

    assert len(monitor.history) == 4
    assert monitor.n_frames == len(still_run)


def test_monitor_to_dict_is_json_safe(base, still_run):
    monitor = Monitor(base)
    for frame in still_run[:3]:
        monitor.update(frame)

    payload = monitor.to_dict()
    text = json.dumps(payload, ensure_ascii=False)

    assert json.loads(text)["frames_seen"] == 3
    assert payload["state"] in STATES
    assert payload["last"]["ok"] is True


def test_alert_is_json_safe():
    alert = Alert(3, "darkness", "the lens is capped")

    assert alert.to_dict() == {
        "when_index": 3,
        "kind": "darkness",
        "message": "the lens is capped",
    }


def test_bad_monitor_arguments_are_refused(base):
    with pytest.raises(ValueError, match="freeze_frames"):
        Monitor(base, freeze_frames=1)
    with pytest.raises(ValueError, match="history"):
        Monitor(base, history=1)
    with pytest.raises(TypeError, match="thresholds must be"):
        Monitor(base, thresholds=7)


def test_monitor_accepts_a_pil_image_and_a_path(tmp_path, base):
    from PIL import Image

    from conftest import save_png

    path = save_png(tmp_path / "frame1.png", base)
    monitor = Monitor()

    monitor.update(Image.fromarray(base))
    health = monitor.update(path)

    assert health.source == "frame1.png"
    assert monitor.n_frames == 2


def test_summary_is_plain_ascii(base, still_run):
    monitor = Monitor(base)
    for frame in still_run[:4]:
        monitor.update(frame)
    monitor.update(np.zeros_like(base))

    text = monitor.summary()

    assert text.isascii()
    assert "camera monitor:" in text
    assert "alerts (" in text
