"""The container a picture arrives in must not change what the picture is.

The same pixel values carried by uint8, uint16 or int32 are the same picture.
Reading the depth from the dtype instead of the values made a widened copy of a
known-good image score 107 and be reported anomalous with full confidence and an
empty ``.warnings`` - and, worse, made a whole 16-bit pipeline self-consistently
blind, because every feature landed 257 times too small and the profile's
absolute noise floor then swallowed real faults.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image

from conftest import colour_fault, held_out_scene, normal_set, save_png
from vision_anomaly import Detector
from vision_anomaly._loading import load_image


def widen(image: np.ndarray, dtype=np.uint16) -> np.ndarray:
    """The same picture in a bigger integer container. Same values, no rescale."""
    return np.asarray(image).astype(dtype)


def test_widening_the_container_does_not_move_a_pixel():
    """The premise: the arrays really are equal before either is read."""
    image = held_out_scene(0)

    assert np.array_equal(image, widen(image))


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.int32, np.int64, np.uint32])
def test_a_clean_image_scores_the_same_whatever_integer_holds_it(
    good_images, clean_image, dtype
):
    """Blocker: a uint16 copy of a clean image scored 107.83 and was flagged."""
    detector = Detector().fit(good_images)

    narrow = detector.score(clean_image)
    wide = detector.score(widen(clean_image, dtype))

    assert wide == pytest.approx(narrow, abs=1e-6)
    assert not detector.predict(widen(clean_image, dtype)).anomalous


def test_a_copy_of_a_fitted_image_scores_near_zero_in_a_wider_container(good_images):
    detector = Detector().fit(good_images)

    assert detector.score(widen(good_images[0])) <= detector.profile.worst_fit_score


def test_a_sixteen_bit_pipeline_still_catches_an_obvious_fault(good_images):
    """Blocker: fitting and scoring wholly in uint16 missed a blatant stain.

    Self-consistency was not enough. Dividing by 65535 put every feature at a
    257th of its proper level, which is below the profile's absolute noise floor,
    so the spread of "normal" grew to swallow anything.
    """
    eight = Detector().fit(good_images)
    sixteen = Detector().fit([widen(image) for image in good_images])
    fault = colour_fault(0)

    on_eight = eight.score(fault)
    on_sixteen = sixteen.score(widen(fault))

    assert on_eight > eight.sensitivity
    assert on_sixteen > sixteen.sensitivity
    assert on_sixteen == pytest.approx(on_eight, rel=1e-6)


def test_a_sixteen_bit_pipeline_still_passes_a_clean_image(good_images, clean_image):
    sixteen = Detector().fit([widen(image) for image in good_images])

    assert sixteen.score(widen(clean_image)) < sixteen.sensitivity


def test_the_fitted_profile_records_the_scale_it_read_and_reuses_it(good_images):
    """Scoring must not guess the depth again from one image's own contents."""
    detector = Detector().fit(good_images)

    assert detector.profile.value_scale == 255.0
    # A genuinely 16-bit frame against an 8-bit profile saturates rather than
    # being quietly rescaled into looking normal - and it says so.
    deep = widen(held_out_scene(0), np.uint16) * 257
    result = detector.predict(deep)

    assert result.anomalous
    assert any("0-65535" in note for note in result.notes)



def test_a_saved_profile_carries_the_scale_through_json(good_images, tmp_path):
    detector = Detector().fit([widen(image) for image in good_images])
    path = detector.save(tmp_path / "profile.json")

    reloaded = Detector.load(path)

    assert reloaded.profile.value_scale == detector.profile.value_scale
    assert reloaded.score(widen(held_out_scene(0))) == pytest.approx(
        detector.score(widen(held_out_scene(0))), abs=1e-6
    )


def test_a_widened_array_says_which_scale_it_was_read_on():
    """Failing everything else, the reading must at least be visible."""
    loaded = load_image(widen(held_out_scene(0)))

    assert loaded.value_scale == 255.0
    assert any("uint16" in note and "0-255" in note for note in loaded.notes)


def test_a_genuine_sixteen_bit_array_keeps_its_full_range():
    loaded = load_image(np.full((32, 32), 30000, dtype=np.uint16))

    assert loaded.value_scale == 65535.0
    assert float(loaded.luminance.mean()) == pytest.approx(30000 / 65535.0, abs=1e-3)


def test_a_file_and_an_array_of_the_same_deep_picture_agree(tmp_path):
    """A 16-bit PNG must not be read differently from the array it came from."""
    array = widen(held_out_scene(0))
    path = str(tmp_path / "deep.png")
    Image.fromarray(array[:, :, 0].astype(np.uint16)).save(path)

    from_disk = load_image(path)
    from_array = load_image(array[:, :, 0])

    assert from_disk.value_scale == from_array.value_scale
    assert float(np.abs(from_disk.luminance - from_array.luminance).max()) < 0.02


def test_negative_pixels_in_a_signed_container_are_reported():
    """An 8-bit picture cast to int8 loses half its values; say so."""
    array = held_out_scene(0).astype(np.int8)

    loaded = load_image(array)

    assert any("negative" in note for note in loaded.notes)


def test_a_fit_set_that_mixes_bit_depths_is_refused_quietly_no_longer(good_images):
    """Blocker: alternating depths fitted silently and detected nothing again.

    CONVENTIONS requires a degenerate auto-detected result to be detected,
    fallen back from, and recorded - not averaged into a profile that calls
    everything normal.
    """
    mixed = [
        widen(image) * 257 if index % 2 else image
        for index, image in enumerate(good_images)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        detector = Detector().fit(mixed)

    messages = [str(entry.message) for entry in caught]
    assert any("do not agree on how deep" in text for text in messages)
    assert any("do not agree on how deep" in text for text in detector.profile.warnings)
    assert "warning:" in detector.summary()


def test_a_mixed_depth_fit_set_still_produces_a_usable_profile(good_images):
    """Having warned, it must fall back to one scale rather than give up."""
    mixed = [
        widen(image) * 257 if index % 2 else image
        for index, image in enumerate(good_images)
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        detector = Detector().fit(mixed)

    assert detector.profile.value_scale in (255.0, 65535.0)
    assert detector.profile.worst_fit_score >= 0.0


def test_reading_is_still_deterministic_across_containers(good_images, clean_image):
    detector = Detector().fit(good_images)
    wide = widen(clean_image)

    assert detector.score(wide) == detector.score(wide)
    assert wide.dtype == np.uint16            # the caller's array is untouched
    assert np.array_equal(wide, widen(clean_image))
