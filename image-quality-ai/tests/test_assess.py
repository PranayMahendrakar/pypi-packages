"""The headline behaviour: a good photo passes, each kind of bad photo is caught
by the right measure and explained in words."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest
from PIL import Image, ImageFilter

import image_quality_ai
from conftest import blurred, detailed_photo, noisy, photo


def test_quickstart_from_readme_runs_and_explains_itself():
    # The exact block from README.md's Quickstart.
    photo = np.random.default_rng(0).integers(0, 60, (240, 320, 3), dtype=np.uint8)
    report = image_quality_ai.assess(photo)
    text = report.summary()

    assert isinstance(text, str) and text
    assert report.usable is False
    assert report.issues, "a dark, grainy snap must have something to say for itself"
    assert isinstance(report.issues[0], str)


def test_good_photo_passes(good_rgb):
    report = image_quality_ai.assess(good_rgb)

    assert report.usable is True
    assert report.score >= 80.0
    assert report.grade in ("A", "B")
    assert report.failed_gates == []
    assert all(metric.ok for metric in report.metrics.values())


def test_every_measure_is_reported_with_a_value_a_score_and_words(good_rgb):
    report = image_quality_ai.assess(good_rgb)

    assert set(report.metrics) == set(image_quality_ai.MEASURE_NAMES)
    for name, metric in report.metrics.items():
        assert metric.name == name
        assert isinstance(metric.value, float)
        assert 0.0 <= metric.score <= 100.0
        assert isinstance(metric.ok, bool)
        assert metric.message
        # the sentence must name a number, not just hand down a verdict
        assert any(character.isdigit() for character in metric.message)


def test_blur_is_caught_and_named(good):
    report = image_quality_ai.assess(blurred(good, 6.0))

    assert report.metrics["sharpness"].ok is False
    assert report.usable is False
    assert "sharpness" in report.failed_gates
    assert "focus" in report.issues[0] or "blur" in report.issues[0]


def test_darkness_is_caught_and_named(good):
    dark = (good.astype(np.float32) * 0.13).astype(np.uint8)
    report = image_quality_ai.assess(dark)

    assert report.metrics["exposure"].ok is False
    assert report.usable is False
    assert "exposure" in report.failed_gates
    assert any("underexposed" in issue for issue in report.issues)


def test_overexposure_is_caught_and_named(good):
    bright = np.clip(good.astype(np.float32) * 1.9 + 60, 0, 255).astype(np.uint8)
    report = image_quality_ai.assess(bright)

    assert report.metrics["exposure"].ok is False
    assert any("overexposed" in issue for issue in report.issues)


def test_clipping_is_reported_even_when_the_mean_looks_fine():
    # Half the frame crushed to black, half blown to white: the mean lands mid-grey.
    frame = np.zeros((200, 200), dtype=np.uint8)
    frame[:, 100:] = 255
    report = image_quality_ai.assess(frame)

    exposure = report.metrics["exposure"]
    assert exposure.details["clipped_black"] == pytest.approx(0.5, abs=0.01)
    assert exposure.details["clipped_white"] == pytest.approx(0.5, abs=0.01)
    assert any("black" in issue and "white" in issue for issue in report.issues)


def test_low_contrast_is_caught(good):
    flat = (118 + (good.astype(np.float32) - good.mean()) * 0.10).astype(np.uint8)
    report = image_quality_ai.assess(flat)

    assert report.metrics["contrast"].ok is False
    assert any("flat" in issue for issue in report.issues)


def test_noise_is_caught_and_named(good):
    report = image_quality_ai.assess(noisy(good, 14.0))

    assert report.metrics["noise"].ok is False
    assert report.metrics["noise"].value > 3.0
    assert any("grainy" in issue for issue in report.issues)


def test_noise_does_not_depend_on_how_many_megapixels_the_camera_had(good):
    """A shrunken copy averages grain away; tiles at native resolution do not."""
    small = noisy(good, 14.0)
    enlarged = np.asarray(
        Image.fromarray(good).resize((1600, 1200), Image.Resampling.BICUBIC)
    )
    large = noisy(enlarged, 14.0)

    small_sigma = image_quality_ai.assess(small).metrics["noise"].value
    large_report = image_quality_ai.assess(large)

    assert small_sigma == pytest.approx(large_report.metrics["noise"].value, rel=0.25)
    assert large_report.metrics["noise"].ok is False


def test_sharpness_does_not_depend_on_how_many_megapixels_the_camera_had():
    """The point of the fixed analysis size: one photo, one number.

    The same frame downscaled across a 4x range of sizes must score the same.
    The source is softened first so the score lands mid-range: a photo pinned at
    100 would pass this test without the normalisation doing any work.
    """
    source = Image.fromarray(photo(2400, 1800)).filter(ImageFilter.GaussianBlur(2.2))
    scores = [
        image_quality_ai.assess(
            np.asarray(source.resize(size, Image.Resampling.BOX))
        ).metrics["sharpness"].score
        for size in ((2400, 1800), (1600, 1200), (1000, 750), (600, 450))
    ]

    assert 20.0 < min(scores) < 90.0, "the source must be mid-range for this to mean anything"
    assert max(scores) - min(scores) < 5.0, "sharpness drifted with size: {0}".format(scores)


#: The sizes that actually reach a vision model. Every one is below
#: ANALYSIS_LONG_EDGE, which is 512, so none of them was covered by the test
#: above - and measuring them at their own resolution was the bug below.
THUMBNAIL_SIZES = ((1200, 900), (512, 384), (384, 288), (256, 192), (224, 168), (128, 96))


def test_sharpness_holds_steady_below_the_fixed_analysis_size():
    """One picture, one number, on the small side of the analysis grid too.

    The analysis plane was area-averaged down to 512 but never replicated up to
    it, so anything already smaller was measured at its own resolution, where
    the same scene carries far more Laplacian energy per pixel: one slightly
    soft photo read about five times sharper at 224px than at 1200px, and both
    is_blurry and usable flipped on the way down. 224, 256 and 384 are exactly
    the sizes that reach a vision model, so the headline call was wrong for
    them.
    """
    source = Image.fromarray(detailed_photo()).filter(ImageFilter.GaussianBlur(0.9))
    copies = [
        np.asarray(source.resize(size, Image.Resampling.LANCZOS))
        for size in THUMBNAIL_SIZES
    ]
    reports = [image_quality_ai.assess(copy) for copy in copies]

    measured = {tuple(report.image["analysis_size"]) for report in reports}
    assert measured == {(512, 384)}, "every copy must be judged on one grid, got {0}".format(
        measured
    )
    values = [report.metrics["sharpness"].value for report in reports]
    assert max(values) / min(values) < 2.5, "sharpness drifted with size: {0}".format(values)
    assert [image_quality_ai.is_blurry(copy) for copy in copies] == [False] * len(copies)
    assert [report.usable for report in reports] == [True] * len(copies)


def test_a_thumbnail_of_a_sharp_photo_is_still_sharp():
    """The README's own claim: a photo and a thumbnail of it get one number."""
    source = Image.fromarray(detailed_photo())
    full = image_quality_ai.assess(np.asarray(source))
    thumbnail = image_quality_ai.assess(
        np.asarray(source.resize((224, 168), Image.Resampling.LANCZOS))
    )

    assert thumbnail.metrics["sharpness"].value == pytest.approx(
        full.metrics["sharpness"].value, rel=0.5
    )
    assert thumbnail.usable is full.usable is True


def test_an_upscaled_photo_is_correctly_reported_as_softer(good):
    """Enlarging a frame adds pixels, not detail, and the score must say so."""
    upscaled = np.asarray(Image.fromarray(good).resize((1280, 960), Image.Resampling.BICUBIC))

    assert (
        image_quality_ai.assess(upscaled).metrics["sharpness"].score
        < image_quality_ai.assess(good).metrics["sharpness"].score
    )


def test_off_centre_subject_lowers_framing(good):
    centred = image_quality_ai.assess(good).metrics["framing"]
    pushed = np.roll(np.roll(good, -170, axis=1), -120, axis=0)
    shifted = image_quality_ai.assess(pushed).metrics["framing"]

    assert shifted.score < centred.score
    assert shifted.details["subject_offset"] > centred.details["subject_offset"]


def test_framing_steps_aside_when_there_is_no_subject():
    blank = np.full((120, 160), 128, dtype=np.uint8)
    report = image_quality_ai.assess(blank)
    framing = report.metrics["framing"]

    assert framing.applies is False
    assert "not judged" in framing.message
    # a measure that could not be judged is left out instead of dragging the score down
    assert not any("subject" in issue for issue in report.issues)


def test_a_failed_gate_holds_the_score_below_usable(good):
    """A blurred frame with perfect exposure must not average its way to a pass."""
    report = image_quality_ai.assess(blurred(good, 8.0))

    assert report.failed_gates
    assert report.score < report.thresholds.usable_score
    assert report.usable is False
    assert report.grade == "F"


def test_the_gate_sentence_reads_correctly_for_one_gate_and_for_two(good):
    """The report's whole job is explaining itself, so the prose has to agree
    with itself when one gate fails and when both do."""
    one = image_quality_ai.assess(blurred(good, 8.0))          # sharp only
    both = image_quality_ai.assess((good.astype(np.float32) * 0.05).astype(np.uint8))

    assert one.failed_gates == ["sharpness"]
    assert "sharpness is a pass/fail gate" in one.summary()
    assert both.failed_gates == ["sharpness", "exposure"]
    assert "sharpness and exposure are pass/fail gates" in both.summary()


def test_score_grade_and_usable_never_disagree(good):
    for candidate in (good, blurred(good, 5.0), noisy(good, 20.0), good // 8):
        report = image_quality_ai.assess(candidate)
        assert report.usable == (
            report.score >= report.thresholds.usable_score and not report.failed_gates
        )
        if report.score < 60.0:
            assert report.grade == "F"


def test_issues_are_ordered_worst_first(good):
    dark_and_grainy = noisy((good.astype(np.float32) * 0.2).astype(np.uint8), 12.0)
    report = image_quality_ai.assess(dark_and_grainy)

    assert len(report.issues) >= 2
    # every issue names the score behind it, so the ordering can be read back
    scored = [
        float(match.group(1))
        for match in (re.search(r"scored (\d+(?:\.\d+)?) of 100", issue) for issue in report.issues)
        if match is not None
    ]
    assert len(scored) == len(report.issues), "every issue must name its score"
    assert scored == sorted(scored), "issues must run worst first, got {0}".format(scored)
    assert scored[0] <= min(
        metric.score for metric in report.metrics.values() if metric.applies and not metric.ok
    )


def test_a_threshold_override_that_crosses_a_floor_still_scores_sanely(good):
    """Overrides are unconstrained, so a borderline can be pushed past the floor
    underneath it. The ramp must give way rather than double back on itself and
    hand out a score that means nothing."""
    soft = blurred(good, 6.0)
    ordinary = image_quality_ai.assess(soft).metrics["sharpness"].score
    crossed = image_quality_ai.assess(
        soft, thresholds={"sharpness_blurry": 0.5}   # below sharpness_floor, which is 2.0
    ).metrics["sharpness"]

    assert 0.0 <= crossed.score <= 100.0
    assert crossed.score >= ordinary, "a more forgiving limit must never score worse"


def test_unknown_and_unusable_thresholds_are_rejected_clearly(good):
    with pytest.raises(ValueError, match="unknown threshold"):
        image_quality_ai.assess(good, thresholds={"sharpnes_blurry": 40.0})
    with pytest.raises(ValueError, match="must be finite"):
        image_quality_ai.assess(good, thresholds={"sharpness_blurry": float("inf")})
    with pytest.raises(TypeError, match="thresholds must be"):
        image_quality_ai.assess(good, thresholds=40.0)


def test_report_round_trips_to_json(good_rgb):
    report = image_quality_ai.assess(good_rgb)
    data = report.to_dict()

    assert set(data) >= {"score", "grade", "usable", "metrics", "issues", "image"}
    restored = json.loads(report.to_json())
    assert restored == json.loads(json.dumps(data, ensure_ascii=False))
    assert restored["metrics"]["noise"]["value"] >= 0.0


def test_explain_and_scores_helpers(good_rgb):
    report = image_quality_ai.assess(good_rgb)

    assert report.explain("noise") == report.metrics["noise"].message
    assert set(report.scores()) == set(image_quality_ai.MEASURE_NAMES)
    with pytest.raises(KeyError, match="no measure named"):
        report.explain("colour")


def test_is_blurry_answers_the_one_question(good):
    assert image_quality_ai.is_blurry(blurred(good, 8.0)) is True
    assert image_quality_ai.is_blurry(good) is False
    assert image_quality_ai.is_blurry(good, threshold=10_000_000.0) is True


def test_is_blurry_rejects_a_nonsense_threshold(good):
    with pytest.raises(ValueError, match="must be a number"):
        image_quality_ai.is_blurry(good, threshold="sharp")
    with pytest.raises(ValueError, match="finite"):
        image_quality_ai.is_blurry(good, threshold=float("nan"))


def test_image_assessor_fixes_the_thresholds_once(good):
    strict = image_quality_ai.ImageAssessor({"usable_score": 99.0})

    assert strict.assess(good).usable is False
    assert image_quality_ai.assess(good).usable is True
    assert strict.measures() == image_quality_ai.MEASURE_NAMES
    assert strict.is_blurry(good, threshold=10_000_000.0) is True
