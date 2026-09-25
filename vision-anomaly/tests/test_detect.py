"""The test this package exists to pass, plus the promises around it."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    colour_fault,
    different_scene,
    held_out_scene,
    missing_part,
    normal_scene,
    normal_set,
    rough_part,
)
from vision_anomaly import Detector, detect_anomalies


def test_a_different_image_scores_far_higher_than_a_held_out_clean_one(
    good_images, clean_image, odd_image
):
    """The proof. Fit on similar clean images; the odd one out must stand out.

    Not "scores higher" - far higher, and on the correct side of the default
    threshold in both directions. A detector that flagged everything would pass
    a one-sided version of this test.
    """
    detector = Detector().fit(good_images)

    clean = detector.score(clean_image)
    odd = detector.score(odd_image)

    assert clean < detector.sensitivity
    assert odd > detector.sensitivity
    assert odd > 10 * max(clean, 1.0)


@pytest.mark.parametrize(
    "make_fault", [colour_fault, missing_part, rough_part, different_scene]
)
def test_every_kind_of_gross_fault_is_caught(good_images, clean_image, make_fault):
    """Wrong colour, missing part, wrong texture and a different scene entirely."""
    detector = Detector().fit(good_images)

    assert not detector.predict(clean_image).anomalous
    assert detector.predict(make_fault(0)).anomalous


def test_an_identical_copy_of_a_fitted_image_scores_zero(good_images):
    """The floor of the scale. Anything else makes "how far" meaningless."""
    detector = Detector().fit(good_images)

    for image in good_images:
        assert detector.score(image) == pytest.approx(0.0, abs=1e-9)


def test_a_held_out_clean_image_scores_near_zero(good_images):
    """Not fitted on, but from the same production run."""
    detector = Detector().fit(good_images)

    scores = [detector.score(held_out_scene(i)) for i in range(5)]

    assert max(scores) < 1.0


def test_scoring_is_deterministic(good_images, odd_image):
    """Same images in, same numbers out - twice, from two detectors."""
    first = Detector().fit(good_images)
    second = Detector().fit(list(good_images))

    assert first.score(odd_image) == second.score(odd_image)
    assert first.score(odd_image) == first.score(odd_image)
    assert first.profile.median.tolist() == second.profile.median.tolist()


def test_the_caller_images_are_never_modified(good_images, odd_image):
    """Arrays handed in belong to the caller, on the way in and on the way out."""
    before = [image.copy() for image in good_images]
    odd_before = odd_image.copy()

    detector = Detector().fit(good_images)
    detector.predict(odd_image)
    detector.predict_batch(good_images)

    for original, given in zip(before, good_images):
        assert np.array_equal(original, given)
    assert np.array_equal(odd_before, odd_image)


def test_scoring_before_fit_raises_a_clear_error(clean_image):
    """The commonest mistake deserves the commonest fix in the message."""
    detector = Detector()

    for call in (detector.score, detector.predict, detector.predict_batch):
        with pytest.raises(RuntimeError) as raised:
            call(clean_image)
        assert "fit" in str(raised.value)
        assert "known-good" in str(raised.value)

    with pytest.raises(RuntimeError):
        detector.save("unused.json")
    assert detector.fitted is False


def test_sensitivity_moves_the_line_and_nothing_else(good_images):
    """A different threshold changes the verdict, never the measurement."""
    borderline = missing_part(0)
    strict = Detector(sensitivity=3.0).fit(good_images)
    relaxed = Detector(sensitivity=500.0).fit(good_images)

    assert strict.score(borderline) == relaxed.score(borderline)
    assert strict.predict(borderline).anomalous
    assert not relaxed.predict(borderline).anomalous


def test_sensitivity_must_be_a_positive_number():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError) as raised:
            Detector(sensitivity=bad)
        assert "sensitivity" in str(raised.value)


def test_detect_anomalies_does_the_whole_job_in_one_line(
    good_images, clean_image, odd_image
):
    report = detect_anomalies(good_images, [clean_image, odd_image])

    assert len(report) == 2
    assert report.n_anomalous == 1
    assert report.anomalous[0].score == max(report.scores)
    assert report[1].anomalous


def test_detect_anomalies_passes_options_through(good_images, odd_image):
    report = detect_anomalies(
        good_images, [odd_image], sensitivity=4.0, grid=2, max_regions=1
    )

    assert report.threshold == 4.0
    assert len(report[0].regions) <= 1
    assert report[0].regions[0].grid == 2


def test_a_finer_grid_localises_more_tightly(good_images):
    """More cells means a smaller box around the same fault."""
    fault = colour_fault(0)
    coarse = Detector(grid=2).fit(good_images).predict(fault)
    fine = Detector(grid=8).fit(good_images).predict(fault)

    assert coarse.anomalous and fine.anomalous
    assert fine.regions[0].width < coarse.regions[0].width


def test_predict_batch_keeps_the_order_it_was_given(good_images):
    images = [held_out_scene(0), different_scene(0), held_out_scene(1)]
    report = Detector().fit(good_images).predict_batch(images)

    assert [result.anomalous for result in report] == [False, True, False]
    assert report.worst is report[1]
    assert len(report.normal) == 2


def test_a_stacked_array_of_frames_is_accepted():
    """NxHxWx3 is how frames arrive from a camera library, not a list."""
    stack = np.stack(normal_set(6))
    detector = Detector().fit(stack)

    assert detector.profile.n_images == 6
    assert detector.score(normal_scene(0)) == pytest.approx(0.0, abs=1e-9)


def test_repr_says_whether_it_is_fitted(good_images):
    detector = Detector()
    assert "unfitted" in repr(detector)
    assert "not fitted" in detector.summary()

    detector.fit(good_images)
    assert "fitted on 8 images" in repr(detector)
    assert "216 features" in repr(detector)
