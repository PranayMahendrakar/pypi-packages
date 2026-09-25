"""Each event kind: plant it and find it, and give clean input and find nothing.

Every sequence is synthesised by ``synth.py`` with numpy; nothing is downloaded.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import synth
from video_event_detector import Detector, detect_events, motion


def kinds(report):
    return [event.kind for event in report.events]


# -- a still scene ------------------------------------------------------------
@pytest.mark.parametrize("sigma", [2.0, 5.0])
def test_static_scene_produces_no_events(sigma):
    frames = synth.noisy(synth.texture(seed=2), 80, seed=2, sigma=sigma)
    report = detect_events(frames)
    assert report.events == []
    assert report.background["state"] == "learned"
    assert report.analysed_frames == 50
    assert "looked for, not seen: sudden_motion, fall, crowding, abandoned" in report.summary()


# -- sudden_motion --------------------------------------------------------------
def test_sudden_motion_found_when_a_large_block_bursts_in():
    report = detect_events(synth.sudden_sequence())
    assert kinds(report) == ["sudden_motion"]
    event = report.events[0]
    assert 49 <= event.start <= 51
    assert event.end >= event.start
    x, y, w, h = event.region
    assert 25 <= x <= 40 and 20 <= y <= 35 and 50 <= w <= 75 and 45 <= h <= 60
    assert 0.5 <= event.confidence <= 1.0


def test_small_mover_is_not_sudden_motion():
    report = detect_events(synth.sudden_sequence(big=False))
    assert report.events == []
    assert report.stats["peak_moving_share"] > 0.0  # it was seen moving, just not "sudden"


# -- fall -----------------------------------------------------------------------
@pytest.mark.parametrize("textured", [False, True])
def test_fall_found_when_a_tall_region_collapses_and_stays_low(textured):
    frames, fps = synth.fall_sequence(textured=textured)
    report = detect_events(frames, fps=fps)
    assert kinds(report) == ["fall"]
    event = report.events[0]
    assert 5.8 <= event.start <= 6.2          # the collapse began after frame 60
    assert event.end == pytest.approx(9.4)    # still lying there at the last frame
    x, y, w, h = event.region
    assert (x, y, w, h) == (60, 88, 36, 12)   # the lying shape, in frame pixels
    assert "36 px tall dropped to 12 px" in event.message
    assert "review" in event.message


def test_walking_without_falling_is_not_a_fall():
    frames, fps = synth.fall_sequence(fall=False)
    assert detect_events(frames, fps=fps).events == []


def test_brief_crouch_that_recovers_is_not_a_fall():
    assert detect_events(synth.crouch_sequence()).events == []


# -- crowding -------------------------------------------------------------------
def test_crowding_found_when_much_of_the_view_keeps_moving():
    report = detect_events(synth.crowd_sequence())
    assert "crowding" in kinds(report)
    assert "fall" not in kinds(report) and "abandoned" not in kinds(report)
    crowd = report.by_kind["crowding"][0]
    assert crowd.end - crowd.start >= 5
    assert report.stats["peak_moving_share"] >= 0.2


def test_a_few_movers_are_not_crowding():
    report = detect_events(synth.crowd_sequence(people=3))
    assert report.events == []


# -- abandoned ------------------------------------------------------------------
def test_abandoned_object_found_once():
    report = detect_events(synth.abandoned_sequence(), fps=10, dwell=2)
    assert kinds(report) == ["abandoned"]
    event = report.events[0]
    assert 4.6 <= event.start <= 5.05           # it arrived around 5.0 s
    assert event.end == pytest.approx(10.9)     # still there at the end
    x, y, w, h = event.region
    assert (x, y, w, h) == (67, 78, 10, 10)


def test_abandoned_is_reported_only_after_the_dwell_time():
    frames = synth.abandoned_sequence()
    for dwell in (2.0, 4.0):
        detector = Detector(dwell=dwell)
        first = None
        for index, frame in enumerate(frames):
            raised = detector.update(frame, timestamp=index / 10.0)
            if first is None and any(e.kind == "abandoned" for e in raised):
                first = index / 10.0
        assert first is not None
        assert first >= 5.0 + dwell           # put down at 5.0 s, still from 5.1 s
        assert first <= 5.0 + dwell + 1.0
    # A dwell longer than the object stays in view reports nothing.
    assert detect_events(frames, fps=10, dwell=7).events == []


def test_object_picked_up_before_the_dwell_is_not_abandoned():
    report = detect_events(synth.abandoned_sequence(pick_up_at=60), fps=10, dwell=2)
    assert report.events == []


def test_object_taken_away_is_not_reported_as_abandoned():
    report = detect_events(synth.removed_sequence())
    assert report.events == []
    assert any("taken away" in note for note in report.notes)


# -- camera shake -----------------------------------------------------------------
def test_brief_camera_shake_is_camera_motion_not_events():
    report = detect_events(synth.shaken_sequence())
    assert report.events == []
    assert report.camera_motion_frames == 5
    assert "camera shake" in report.summary()


def test_the_shake_would_raise_events_if_not_treated_as_camera_motion(monkeypatch):
    # Proves the shake test is not vacuous: the same frames, with the global-shift
    # check switched off, do produce a (false) event.
    monkeypatch.setattr(motion.MotionModel, "_camera_shift", lambda self, *args: None)
    report = detect_events(synth.shaken_sequence())
    assert report.events != []


def test_camera_bumped_to_a_new_position_is_re_registered():
    report = detect_events(synth.shaken_sequence(bump=True))
    assert report.events == []
    assert any("new position" in note for note in report.notes)


# -- lighting ---------------------------------------------------------------------
@pytest.mark.parametrize("textured", [True, False])
def test_gradual_lighting_change_does_not_trigger_sudden_motion(textured):
    report = detect_events(synth.lighting_sequence(textured=textured))
    assert "sudden_motion" not in kinds(report)
    assert report.events == []


def test_quick_dimming_is_lighting_not_motion():
    background = synth.texture(seed=3, mean=120.0, spread=18.0)
    rng = np.random.default_rng(0)
    frames = []
    for index in range(80):
        k = min(1.0, max(0.0, (index - 50) / 5.0))
        lit = background * (1.0 - 0.3 * k) - 15.0 * k
        frames.append(np.clip(np.round(lit + rng.normal(0, 2, lit.shape)), 0, 255).astype(np.uint8))
    report = detect_events(frames)
    assert report.events == []
    assert any("lighting" in note for note in report.notes)


def test_lighting_change_does_not_hide_a_real_event():
    frames = synth.lighting_sequence(count=90)
    planted = synth.sudden_sequence(count=90)
    mixed = []
    for index, (lit, burst) in enumerate(zip(frames, planted)):
        frame = lit.copy()
        if index >= 50:
            frame[burst >= 225] = 250
        mixed.append(frame)
    assert "sudden_motion" in kinds(detect_events(mixed))


# -- frame sizes and short clips --------------------------------------------------
def test_frames_of_differing_sizes_are_resized_with_a_note():
    frames = synth.noisy(synth.texture(seed=4), 70, seed=4)
    mixed = [np.kron(f, np.ones((2, 2), np.uint8)) if i % 7 == 3 else f for i, f in enumerate(frames)]
    report = detect_events(mixed)
    assert report.events == []
    assert any("resized to match" in note and "320x240" in note for note in report.notes)
    assert report.frame_size == (160, 120)


def test_planted_event_is_still_found_across_resized_frames():
    frames = synth.abandoned_sequence()
    mixed = [np.kron(f, np.ones((2, 2), np.uint8)) if i % 5 == 2 else f for i, f in enumerate(frames)]
    report = detect_events(mixed, fps=10, dwell=2)
    assert kinds(report) == ["abandoned"]
    assert any("resized" in note for note in report.notes)


def test_sequence_shorter_than_the_background_window_says_it_is_learning():
    frames = synth.noisy(synth.texture(seed=5), 10, seed=5)
    report = detect_events(frames)
    assert report.events == []
    assert report.background["state"] == "learning"
    assert report.background["frames_seen"] == 10
    text = report.summary()
    assert "STILL LEARNING" in text and "background_frames=3" in text
    json.dumps(report.to_dict(), allow_nan=False)


def test_a_shorter_background_window_lets_a_short_clip_run():
    frames = synth.abandoned_sequence(count=80)[25:]
    report = detect_events(frames, fps=10, background_frames=8, dwell=2)
    assert report.background["state"] == "learned"
    assert kinds(report) == ["abandoned"]


# -- determinism and caller data ----------------------------------------------------
def test_deterministic():
    frames = synth.crowd_sequence(count=70)
    first = detect_events(frames).to_dict()
    second = detect_events(frames).to_dict()
    assert first == second


def test_caller_frames_are_never_modified():
    frames = synth.abandoned_sequence(count=80)[25:]
    floats = [f.astype(np.float32) for f in frames]  # float32 is the no-copy trap
    before = [f.copy() for f in floats]
    for f in floats:
        f.flags.writeable = False                     # any write would raise
    report = detect_events(floats, fps=10, dwell=2, background_frames=8)
    assert kinds(report) == ["abandoned"]
    for f, b in zip(floats, before):
        assert np.array_equal(f, b)
    uint8 = synth.fall_sequence()[0]
    copies = [f.copy() for f in uint8]
    detect_events(uint8)
    assert all(np.array_equal(a, b) for a, b in zip(uint8, copies))


def test_events_extend_on_the_same_object_while_ongoing():
    detector = Detector(dwell=2.0)
    raised = []
    for index, frame in enumerate(synth.abandoned_sequence()):
        raised += detector.update(frame, timestamp=index / 10.0)
    assert len(raised) == 1
    assert raised[0] is detector.events[0]
    assert raised[0].end == pytest.approx(10.9)
