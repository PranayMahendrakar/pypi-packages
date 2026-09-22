"""check_stream: a whole run of frames, rolled up into one report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from camera_health import DARKNESS, OBSTRUCTION, check_stream
from conftest import live_frame, obstructed, save_png, scene


def test_a_clean_run_is_ok(still_run):
    report = check_stream(still_run)

    assert report.ok
    assert report.frames_checked == len(still_run)
    assert report.uptime == 1.0
    assert report.first_failure is None
    assert report.faults == []
    assert report.score == 100.0
    assert "no faults found" in report.summary()


def test_a_run_with_one_bad_frame_names_it(base, still_run):
    frames = list(still_run)
    frames[5] = np.zeros_like(base)

    report = check_stream(frames)

    assert not report.ok
    assert report.first_failure == 5
    assert report.fault_counts[DARKNESS] == 1
    assert report.first_seen[DARKNESS] == 5
    assert report.frames_with(DARKNESS) == [5]
    assert report.worst_frame.index == 5
    assert report.uptime == pytest.approx(11 / 12)
    assert "first failure: frame 5" in report.summary()


def test_faults_are_rolled_up_one_entry_per_kind(base):
    frames = [obstructed(live_frame(base, tick), share=0.6) for tick in range(4)]

    report = check_stream(frames)

    kinds = [fault.kind for fault in report.faults]
    assert kinds.count(OBSTRUCTION) == 1
    assert report.fault_counts[OBSTRUCTION] == 4
    assert "4 frame(s), first at frame 0" in report.summary()


def test_an_empty_run_is_reported_not_crashed():
    report = check_stream([])

    assert not report.ok
    assert report.frames_checked == 0
    assert report.uptime == 0.0
    assert report.by_frame == []
    assert report.worst_frame is None
    assert "no frames were given" in report.summary()
    assert json.loads(json.dumps(report.to_dict()))["frames_checked"] == 0


def test_a_run_of_one_frame_works_and_marks_what_it_cannot_check(base):
    report = check_stream([base])

    assert report.ok
    assert report.frames_checked == 1
    health = report.by_frame[0]
    assert not health.was_checked("frozen")
    assert not health.was_checked("tampering")
    assert not health.was_checked("drift")


def test_a_reference_is_used_when_one_is_given(base, other_scene):
    report = check_stream([other_scene, other_scene], reference=base)

    assert not report.ok
    assert report.fault_counts["tampering"] == 2
    assert report.first_failure == 0


def test_a_directory_is_read_in_natural_order(tmp_path, base):
    for index in (1, 2, 10):
        save_png(tmp_path / "frame{}.png".format(index), live_frame(base, index))

    report = check_stream(str(tmp_path))

    assert [health.source for health in report.by_frame] == [
        "frame1.png",
        "frame2.png",
        "frame10.png",
    ]
    assert report.ok


def test_a_directory_with_no_images_is_a_clear_error(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing to see", encoding="utf-8")

    with pytest.raises(ValueError, match="no image files in"):
        check_stream(str(tmp_path))


def test_thresholds_reach_every_frame_of_a_run(base):
    dim = [(live_frame(base, tick).astype(np.float32) * 0.22).astype(np.uint8) for tick in range(3)]

    assert check_stream(dim).fault_counts.get(DARKNESS) == 3
    assert DARKNESS not in check_stream(dim, thresholds={"dark_mean": 5.0}).fault_counts


def test_stream_to_dict_is_json_safe(still_run):
    report = check_stream(still_run[:3])

    text = json.dumps(report.to_dict(), ensure_ascii=False)
    payload = json.loads(text)

    assert payload["frames_checked"] == 3
    assert len(payload["by_frame"]) == 3
    assert payload["by_frame"][0]["checks"]["darkness"] == "pass"


def test_stream_summary_is_plain_ascii(base, still_run):
    frames = list(still_run)
    frames[2] = np.zeros_like(base)

    assert check_stream(frames).summary().isascii()
