"""Single-frame checks: every fault kind, and the frames that must not raise one."""

from __future__ import annotations

import numpy as np
import pytest

from camera_health import (
    COLOUR_CAST,
    CRITICAL,
    DARKNESS,
    DEFOCUS,
    FAULT_KINDS,
    NOISE,
    OBSTRUCTION,
    OVEREXPOSURE,
    TAMPERING,
    WARNING,
    check,
)
from conftest import SEED, blurred, colourise, live_frame, obstructed, scene


def test_quickstart_block_from_the_readme():
    """The exact code in README Quickstart, which is what QA runs."""
    y, x = np.indices((240, 320))
    view = (120 + 60 * np.sin(x / 4) * np.cos(y / 5)).astype(np.uint8)
    blocked = view.copy()
    blocked[30:210, 40:280] = 90

    health = check(blocked, previous=view)

    text = health.summary()
    assert health.has(OBSTRUCTION)
    assert not health.ok
    assert text.startswith("camera health: NOT OK")
    assert "obstruction" in text
    assert health.to_dict()["faults"][0]["kind"] == OBSTRUCTION


def test_a_healthy_frame_scores_100_and_names_no_fault(base):
    health = check(base)

    assert health.ok
    assert health.score == 100.0
    assert health.faults == []
    assert health.worst is None
    assert "no faults found" in health.summary()


def test_every_fault_kind_has_a_check_entry(base):
    health = check(base)

    assert set(health.checks) == set(FAULT_KINDS)
    assert all(state in ("pass", "fault", "not_checkable") for state in health.checks.values())


def test_obstruction_is_found_and_placed(base):
    """A corner blob is reported, and the message says which corner."""
    health = check(obstructed(base, share=0.3))

    fault = health.fault(OBSTRUCTION)
    assert fault is not None
    assert fault.severity == WARNING
    assert "of the view has no detail" in fault.message
    assert "top left" in fault.message
    assert health.metrics["obstruction_area"] > 0.15


def test_a_big_obstruction_is_critical_and_its_area_is_close_to_the_truth(base):
    health = check(obstructed(base, share=0.6))

    fault = health.fault(OBSTRUCTION)
    assert fault is not None and fault.severity == CRITICAL
    assert not health.ok
    # Whole tiles only, so the figure is a lower bound - but a tight one.
    assert 0.45 <= health.metrics["obstruction_area"] <= 0.60


def test_a_covered_lens_is_an_obstruction_not_a_defocus():
    """A flat grey field at a normal brightness is a lens cap, not a soft lens."""
    health = check(np.full((240, 320), 128, dtype=np.uint8))

    fault = health.fault(OBSTRUCTION)
    assert fault is not None and fault.severity == CRITICAL
    assert "covered, fogged or painted over" in fault.message
    assert health.metrics["obstruction_area"] == 1.0


def test_defocus_is_found_and_darkness_is_not(base):
    """Softening a frame must not change how bright it is judged to be."""
    health = check(blurred(base, rounds=40))

    assert health.has(DEFOCUS)
    assert not health.has(DARKNESS)
    assert health.metrics["focus"] < 0.085


def test_dimming_a_sharp_frame_does_not_make_it_defocused(base):
    """The focus ratio is scaled by scene contrast, so gain changes do not move it."""
    bright = check(base)
    dim = check((base.astype(np.float32) * 0.45).astype(np.uint8))

    assert not dim.has(DEFOCUS)
    assert abs(dim.metrics["focus"] - bright.metrics["focus"]) < 0.03


def test_a_completely_black_frame_is_darkness_and_never_an_obstruction():
    """The edge case: no detail anywhere is no light, not something over the lens."""
    health = check(np.zeros((240, 320), dtype=np.uint8))

    assert not health.ok
    assert health.fault(DARKNESS).severity == CRITICAL
    assert "entirely black" in health.fault(DARKNESS).message
    assert not health.has(OBSTRUCTION)
    assert not health.was_checked(OBSTRUCTION)
    assert "too dark" in health.not_checkable[OBSTRUCTION]


def test_a_dim_frame_is_a_warning_not_a_failure(base):
    health = check((base.astype(np.float32) * 0.22).astype(np.uint8))

    assert health.has(DARKNESS)
    assert health.fault(DARKNESS).severity == WARNING
    assert health.ok
    assert health.score < 100.0


def test_a_blown_out_frame_is_overexposure():
    health = check(np.full((240, 320), 252, dtype=np.uint8))

    assert not health.ok
    assert health.fault(OVEREXPOSURE).severity == CRITICAL
    assert not health.was_checked(DEFOCUS)


def test_a_colour_cast_is_found_and_the_channel_named(base):
    health = check(colourise(base, gains=(1.6, 1.0, 0.7)))

    fault = health.fault(COLOUR_CAST)
    assert fault is not None
    assert "red" in fault.message
    assert health.metrics["colour_cast"] > 0.18


def test_a_neutral_colour_frame_carries_no_cast(base):
    health = check(colourise(base))

    assert not health.has(COLOUR_CAST)
    assert not health.was_checked(COLOUR_CAST)
    assert "mono feed carried in a colour frame" in health.not_checkable[COLOUR_CAST]


def test_a_greyscale_frame_cannot_be_asked_about_colour(base):
    health = check(base)

    assert not health.was_checked(COLOUR_CAST)
    assert "greyscale" in health.not_checkable[COLOUR_CAST]


def test_heavy_noise_is_reported(base):
    rng = np.random.default_rng(SEED)
    noisy = np.clip(
        base.astype(np.float32) + rng.normal(0.0, 18.0, base.shape), 0.0, 255.0
    ).astype(np.uint8)

    health = check(noisy)

    assert health.fault(NOISE).severity == CRITICAL
    assert health.metrics["noise"] > 12.0


def test_a_clean_frame_is_not_called_noisy(base):
    assert check(base).metrics["noise"] < 6.0


def test_tampering_needs_a_reference_and_finds_a_turned_camera(base, other_scene):
    without = check(base)
    assert not without.was_checked(TAMPERING)
    assert "no reference frame" in without.not_checkable[TAMPERING]

    health = check(other_scene, reference=base)
    assert health.fault(TAMPERING) is not None
    assert health.metrics["scene_correlation"] < 0.55


def test_turning_the_lights_down_is_not_tampering(base):
    """The scene signature is brightness-normalised, so exposure is not layout."""
    darker = (base.astype(np.float32) * 0.55).astype(np.uint8)

    health = check(darker, reference=base)

    assert not health.has(TAMPERING)
    assert health.metrics["scene_correlation"] > 0.9


def test_fault_helpers_accept_common_spellings(base):
    health = check(blurred(base, rounds=40))

    assert health.has("blur")
    assert health.has("out_of_focus")
    assert health.fault("DEFOCUS") is health.fault(DEFOCUS)
    with pytest.raises(ValueError, match="unknown fault kind"):
        health.has("smudge")


def test_thresholds_can_be_overridden_per_call(base):
    dim = (base.astype(np.float32) * 0.22).astype(np.uint8)

    assert check(dim).has(DARKNESS)
    assert not check(dim, thresholds={"dark_mean": 5.0}).has(DARKNESS)

    with pytest.raises(ValueError, match="unknown threshold"):
        check(base, thresholds={"darkness": 5.0})


def test_freeze_frames_below_two_is_refused(base):
    with pytest.raises(ValueError, match="must be 2 or more"):
        check(base, freeze_frames=1)


def test_the_caller_frame_is_never_modified(base):
    reference = scene(layout=2)
    before_frame = base.copy()
    before_reference = reference.copy()

    check(base, reference=reference, previous=live_frame(base, 1))

    assert np.array_equal(base, before_frame)
    assert np.array_equal(reference, before_reference)


def test_the_same_frame_always_gives_the_same_report(base):
    first = check(base, reference=scene(layout=1), previous=live_frame(base, 2))
    second = check(base, reference=scene(layout=1), previous=live_frame(base, 2))

    assert first.to_dict() == second.to_dict()
