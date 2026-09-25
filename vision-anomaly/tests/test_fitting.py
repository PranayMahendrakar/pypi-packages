"""What fit() accepts, what it refuses, and what it warns about."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import (
    as_grey,
    as_rgba,
    different_scene,
    held_out_scene,
    normal_scene,
    normal_set,
    save_png,
)
from vision_anomaly import Detector
from vision_anomaly._profile import WEAK_PROFILE_IMAGES


def small(count=6, value=120, size=(48, 48)):
    """A few near-identical tiny frames, for tests the scene does not matter to."""
    rng = np.random.default_rng(7)
    return [
        np.clip(
            rng.normal(value, 3.0, (size[1], size[0], 3)), 0, 255
        ).astype(np.uint8)
        for _ in range(count)
    ]


def test_fitting_on_fewer_than_five_images_warns_but_still_works():
    images = small(3)

    with pytest.warns(UserWarning) as caught:
        detector = Detector().fit(images)

    message = str(caught[0].message)
    assert "only 3 images" in message
    assert str(WEAK_PROFILE_IMAGES) in message
    assert detector.fitted
    assert detector.score(images[0]) == pytest.approx(0.0, abs=1e-9)


def test_the_weak_profile_warning_is_carried_into_every_result():
    """A warning nobody sees is a warning that did not happen."""
    images = small(2)
    with pytest.warns(UserWarning):
        detector = Detector().fit(images)

    result = detector.predict(images[0])

    assert any("only 2 images" in note for note in result.warnings)
    assert "warning:" in result.summary()
    assert result.to_dict()["warnings"]


def test_five_images_is_enough_to_stop_warning(recwarn):
    Detector().fit(small(WEAK_PROFILE_IMAGES))

    assert not [item for item in recwarn if item.category is UserWarning]


def test_fitting_on_a_single_image_raises_a_clear_error():
    single = small(1)

    with pytest.raises(ValueError) as raised:
        Detector().fit(single)

    message = str(raised.value)
    assert "at least 2" in message
    assert "variation" in message


def test_fitting_on_a_bare_image_is_treated_as_one_image():
    """A common slip: passing the array instead of a list of arrays."""
    with pytest.raises(ValueError) as raised:
        Detector().fit(np.zeros((32, 32, 3), dtype=np.uint8))
    assert "at least 2" in str(raised.value)

    with pytest.raises(ValueError):
        Detector().fit(Image.new("RGB", (32, 32)))


def test_fitting_on_nothing_raises_a_clear_error():
    for empty in ([], (), None):
        with pytest.raises(ValueError) as raised:
            Detector().fit(empty)
        assert "none were given" in str(raised.value)


def test_images_of_different_sizes_are_resized_with_a_note():
    images = [
        normal_scene(0, size=(512, 384)),
        normal_scene(1, size=(320, 240)),
        normal_scene(2, size=(640, 640)),
        normal_scene(3, size=(512, 384)),
        normal_scene(4, size=(200, 400)),
    ]

    detector = Detector().fit(images)
    notes = detector.profile.notes

    assert any("resampled onto a 256x256 square" in note for note in notes)
    assert any("4 different sizes" in note for note in notes)
    assert any("proportions, not" in note for note in notes)
    assert len(detector.profile.sizes) == 4


def test_a_test_image_of_an_unseen_size_says_so():
    detector = Detector().fit(normal_set(6, size=(512, 384)))

    result = detector.predict(held_out_scene(0))
    assert not any("this image is" in note for note in result.notes)

    odd_size = detector.predict(normal_scene(99, size=(300, 300)))
    assert any("this image is 300x300" in note for note in odd_size.notes)
    assert any("512x384" in note for note in odd_size.notes)


def test_greyscale_rgb_and_rgba_all_work():
    colour = normal_set(6)
    grey = [as_grey(image) for image in colour]
    rgba = [as_rgba(image) for image in colour]

    for images in (colour, grey, rgba):
        detector = Detector().fit(images)
        assert detector.score(images[0]) == pytest.approx(0.0, abs=1e-9)
        assert detector.predict(images[1]).channels in (1, 3, 4)

    grey_detector = Detector().fit(grey)
    assert grey_detector.predict(as_grey(different_scene(0))).anomalous


def test_a_greyscale_profile_reads_an_rgb_copy_of_the_same_frame_as_normal():
    """Grey carried in three channels, so the two are the same picture."""
    colour = normal_set(6)
    detector = Detector().fit([as_grey(image) for image in colour])

    grey_as_rgb = np.dstack([as_grey(colour[0])] * 3)

    assert detector.score(grey_as_rgb) == pytest.approx(0.0, abs=1e-9)


def test_an_opaque_alpha_channel_changes_nothing():
    colour = normal_set(6)
    plain = Detector().fit(colour)
    with_alpha = Detector().fit([as_rgba(image) for image in colour])

    assert plain.profile.median.tolist() == pytest.approx(
        with_alpha.profile.median.tolist()
    )


def test_a_mixed_channel_set_is_noted():
    images = normal_set(4) + [as_grey(normal_scene(9))]

    detector = Detector().fit(images)

    assert any("mix" in note for note in detector.profile.notes)


def test_fitting_from_a_directory_path(tmp_path):
    for index, image in enumerate(normal_set(6)):
        save_png(image, tmp_path / "good_{0}.png".format(index))

    detector = Detector().fit(str(tmp_path))

    assert detector.profile.n_images == 6
    assert detector.score(held_out_scene(0)) < detector.sensitivity


def test_an_empty_directory_says_so(tmp_path):
    with pytest.raises(ValueError) as raised:
        Detector().fit(str(tmp_path))
    assert "no image files found" in str(raised.value)


def test_fitting_from_a_list_of_paths(good_paths):
    detector = Detector().fit(good_paths)

    assert detector.profile.n_images == len(good_paths)
    assert detector.predict(good_paths[0]).source == good_paths[0]
    assert detector.score(good_paths[0]) == pytest.approx(0.0, abs=1e-9)


def test_fitting_from_pil_images():
    images = [Image.fromarray(image) for image in normal_set(6)]

    detector = Detector().fit(images)

    assert detector.predict(images[0]).source == "<PIL.Image>"
    assert detector.score(images[0]) == pytest.approx(0.0, abs=1e-9)


def test_a_detector_can_be_refitted():
    first = small(6, value=80)
    second = small(6, value=200)
    detector = Detector().fit(first)
    assert detector.predict(second[0]).anomalous

    detector.fit(second)

    assert not detector.predict(second[0]).anomalous
    assert detector.predict(first[0]).anomalous


def test_an_unusable_grid_is_refused():
    with pytest.raises(ValueError) as raised:
        Detector(grid=0)
    assert "grid must be at least 1" in str(raised.value)

    with pytest.raises(ValueError) as raised:
        Detector(grid=5, analysis_size=256)
    assert "divide evenly" in str(raised.value)

    with pytest.raises(ValueError) as raised:
        Detector(grid=8, analysis_size=16)
    assert "too small" in str(raised.value)


def test_a_mistyped_directory_says_the_path_is_missing(tmp_path):
    """A typo is far likelier than a genuine one-image fit set.

    Being lectured about needing more than one image sends the reader looking at
    the wrong thing entirely when the real problem is a name that is not there.
    """
    missing = str(tmp_path / "no_such_dir")

    with pytest.raises(ValueError) as raised:
        Detector().fit(missing)

    text = str(raised.value)
    assert "no such image file" in text
    assert missing in text
    assert "at least 2" not in text


def test_one_real_image_still_gets_the_explanation(tmp_path):
    """The count message is right when the path really is one real image."""
    only = save_png(normal_scene(0), tmp_path / "only.png")

    with pytest.raises(ValueError) as raised:
        Detector().fit(only)

    assert "at least 2 known-good images" in str(raised.value)
