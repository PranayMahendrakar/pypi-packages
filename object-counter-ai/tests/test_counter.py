"""Counter: per-frame counts, tracking, and counting objects crossing a line."""
from __future__ import annotations

import json

import numpy as np
import pytest

import object_counter_ai as oc
from synth_images import moving_disc_frames

LINE = ((0, 60), (200, 60))      # horizontal, across the whole 200-wide frame


def _down(x: float, y0: float, y1: float, step: float):
    ys = np.arange(y0, y1 + 1e-9, step)
    return [(x, float(y)) for y in ys]


def _run(frames, **options):
    counter = oc.Counter(**options).line(*LINE)
    for frame in frames:
        counter.update(frame)
    return counter


def test_one_object_crossing_is_counted_once_forward():
    counter = _run(moving_disc_frames(_down(50, 20, 100, 5)))
    assert counter.totals == {"line 1": 1}
    assert counter.lines["line 1"]["forward"] == 1
    assert counter.lines["line 1"]["backward"] == 0
    assert counter.tracks_started == 1
    assert counter.frames == len(counter.history) == 17


def test_moving_up_is_backward():
    counter = _run(moving_disc_frames(list(reversed(_down(50, 20, 100, 5)))))
    assert counter.lines["line 1"]["backward"] == 1
    assert counter.totals["line 1"] == 1


def test_object_lingering_on_the_line_is_counted_once():
    """It reaches the line, wobbles across it for 12 frames, then moves on: one crossing."""
    path = _down(50, 20, 56, 4)
    path += [(50.0, 60.0 + (4.0 if i % 2 else -4.0)) for i in range(12)]
    path += [(50.0, 60.0)] * 5                                  # sits exactly on the line
    path += _down(50, 64, 104, 4)
    counter = _run(moving_disc_frames(path))
    assert counter.totals == {"line 1": 1}
    assert counter.tracks_started == 1
    assert counter.lines["line 1"]["recrossed"] >= 5
    crossings = [r.crossings["line 1"] for r in counter.history]
    assert sum(crossings) == 1


def test_object_that_stops_short_is_not_counted():
    counter = _run(moving_disc_frames(_down(50, 10, 52, 3) + [(50.0, 52.0)] * 10))
    assert counter.totals == {"line 1": 0}


def test_object_moving_parallel_to_the_line_is_not_counted():
    path = [(float(x), 40.0) for x in range(10, 190, 6)]
    assert _run(moving_disc_frames(path)).totals == {"line 1": 0}


def test_object_passing_beyond_the_end_of_a_short_line_is_not_counted():
    counter = oc.Counter().line((0, 60), (80, 60))
    for frame in moving_disc_frames(_down(150, 20, 100, 5)):
        counter.update(frame)
    assert counter.totals == {"line 1": 0}


def test_two_objects_crossing_count_two_with_stable_track_ids():
    main = _down(40, 20, 100, 5)
    other = [[point] for point in reversed(_down(150, 20, 100, 5))]
    assert len(other) == len(main)
    counter = _run(moving_disc_frames(main, extra=other))
    assert counter.totals == {"line 1": 2}
    assert counter.lines["line 1"]["forward"] == 1
    assert counter.lines["line 1"]["backward"] == 1
    ids = {tuple(sorted(r.track_ids)) for r in counter.history}
    assert ids == {(1, 2)}


def test_object_leaving_and_returning_is_not_counted_twice():
    path = _down(50, 30, 90, 6) + list(reversed(_down(50, 30, 90, 6))) + _down(50, 30, 90, 6)
    counter = _run(moving_disc_frames(path))
    assert counter.totals == {"line 1": 1}
    assert counter.lines["line 1"]["recrossed"] == 2


def test_failed_detector_frame_is_recorded_and_tracking_survives():
    frames = moving_disc_frames(_down(50, 20, 100, 5))
    calls = {"n": 0}

    def flaky(image):
        calls["n"] += 1
        if calls["n"] == 8:                     # the frame where it is crossing
            raise TimeoutError("inference server busy")
        dark = np.argwhere(np.asarray(image) < 128)
        (top, left), (bottom, right) = dark.min(axis=0), dark.max(axis=0) + 1
        return [((left, top, right, bottom), "part", 0.9)]

    counter = _run(frames, detector=flaky)
    assert counter.frames == len(frames) == len(counter.history)
    assert counter.failed_frames == 1
    failed = counter.history[7]
    assert not failed.ok and failed.frame == 7 and "TimeoutError" in failed.error
    assert counter.totals == {"line 1": 1}
    assert counter.tracks_started == 1
    assert "1 frame(s) not counted" in counter.summary()


def test_object_briefly_missing_keeps_its_track():
    frames = moving_disc_frames(_down(50, 20, 100, 5))
    for i in (7, 8):
        frames[i] = np.full_like(frames[i], 225)         # vanishes for two frames
    counter = _run(frames, max_missed=3)
    assert counter.tracks_started == 1
    assert counter.totals == {"line 1": 1}


def test_explicit_max_distance_too_small_breaks_tracks():
    frames = moving_disc_frames(_down(50, 20, 100, 10))
    counter = _run(frames, max_distance=3)
    assert counter.tracks_started == len(frames)
    assert counter.totals == {"line 1": 0}          # never seen on both sides as one object


def test_several_named_lines_and_labels():
    def detector(image):
        dark = np.argwhere(np.asarray(image) < 128)
        (top, left), (bottom, right) = dark.min(axis=0), dark.max(axis=0) + 1
        return [((left, top, right, bottom), "bolt", 0.8)]

    counter = oc.Counter(detector).line(*LINE, name="belt").line((0, 90), (200, 90), name="exit")
    for frame in moving_disc_frames(_down(50, 20, 110, 5)):
        counter.update(frame)
    assert counter.totals == {"belt": 1, "exit": 1}
    assert counter.lines["exit"]["by_label"] == {"bolt": 1}
    assert "bolt 1" in counter.summary()


def test_per_frame_counts_and_peak():
    frames = moving_disc_frames([(40.0, 30.0)] * 3, extra=[[], [(120.0, 30.0)], [(120.0, 30.0), (160.0, 90.0)]])
    counter = oc.Counter()
    results = [counter.update(f) for f in frames]
    assert [r.count for r in results] == [1, 2, 3]
    assert [r.frame for r in results] == [0, 1, 2]
    assert counter.peak == 3
    assert counter.totals == {}
    assert "add one with .line" in counter.summary()


def test_reset_keeps_lines_but_forgets_counts():
    counter = _run(moving_disc_frames(_down(50, 20, 100, 5)))
    counter.reset()
    assert counter.totals == {"line 1": 0}
    assert counter.history == [] and counter.frames == 0


def test_counter_is_deterministic_and_json_safe():
    frames = moving_disc_frames(_down(50, 20, 100, 5))
    one, two = _run(frames), _run(frames)
    assert one.to_dict() == two.to_dict()
    text = json.dumps(one.to_dict(), ensure_ascii=False)
    assert '"line 1"' in text
    assert one.summary().isascii()


def test_bad_options_and_lines_are_rejected():
    with pytest.raises(TypeError):
        oc.Counter(max_speed=5)
    with pytest.raises(ValueError):
        oc.Counter(max_distance=0)
    with pytest.raises(ValueError):
        oc.Counter(max_missed=-1)
    with pytest.raises(ValueError):
        oc.Counter().line((5, 5), (5, 5))
    with pytest.raises(ValueError):
        oc.Counter().line((0, 0), "far away")
    with pytest.raises(ValueError, match="already"):
        oc.Counter().line((0, 0), (1, 1), name="a").line((0, 1), (1, 2), name="a")


def test_unreadable_frame_raises_without_advancing_the_stream():
    counter = oc.Counter()
    counter.update(moving_disc_frames([(50.0, 50.0)])[0])
    with pytest.raises(ValueError):
        counter.update(np.zeros((4, 4, 4, 4)))
    assert counter.frames == len(counter.history) == 1
