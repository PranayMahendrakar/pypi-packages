"""Everything that can be handed to :func:`assess`, and everything that can go
wrong with it: odd shapes, odd depths, EXIF, and files that are not images."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PIL import Image

import image_quality_ai
from conftest import as_rgb, photo


# ---------------------------------------------------------------------------
# degenerate images still produce a report
# ---------------------------------------------------------------------------
def test_one_by_one_image_reports_instead_of_dividing_by_zero():
    report = image_quality_ai.assess(np.array([[128]], dtype=np.uint8))

    assert 0.0 <= report.score <= 100.0
    assert report.usable is False
    assert report.metrics["sharpness"].value == 0.0
    assert "too small" in report.metrics["sharpness"].message
    assert report.metrics["noise"].details["measured"] is False
    assert report.summary()


def test_noise_steps_aside_when_the_frame_is_too_small_to_filter():
    """A measure that could not be judged must not hand out marks for it.

    Noise needs a 3x3 median, which a 1x1 frame cannot supply. It used to say so
    in its own message and then report applies=True with a score of 100, which
    put about eight points of overall score on an image that earned none of it,
    and contradicted framing, which steps aside in exactly the same situation.
    """
    report = image_quality_ai.assess(np.full((1, 1), 128, dtype=np.uint8))
    noise = report.metrics["noise"]

    assert noise.applies is False
    assert noise.details["measured"] is False
    assert "not judged" in noise.message
    assert report.metrics["framing"].applies is False, "framing handles this the same way"
    assert not any("grainy" in issue for issue in report.issues)

    # the overall score is the weighted mean of the measures that did apply, so
    # a measure that stepped aside is out of both the top and the bottom of it
    weights = report.thresholds.weights()
    judged = {name: m for name, m in report.metrics.items() if m.applies}
    assert set(judged) == {"sharpness", "exposure", "contrast"}
    mean = sum(weights[name] * m.score for name, m in judged.items()) / sum(
        weights[name] for name in judged
    )
    gate = report.thresholds.usable_score / 100.0    # sharpness fails on a 1x1 frame
    assert report.failed_gates == ["sharpness"]
    assert report.score == pytest.approx(mean * gate, abs=0.05)


@pytest.mark.parametrize("level", [0, 255])
def test_a_blank_frame_is_not_diagnosed_as_a_focus_problem(level):
    """A frame with nothing in it has nothing to be in or out of focus.

    The verdict was always right - 7.1, F, not usable - but the plain-language
    issue is what this package sells, and it used to blame an all-white or
    all-black frame on focus or motion blur, naming a cause that never happened.
    """
    report = image_quality_ai.assess(np.full((64, 64), level, dtype=np.uint8))

    assert report.usable is False
    assert report.metrics["sharpness"].value == 0.0
    top = report.issues[0]
    assert "no detail in this frame" in top
    assert "blank or featureless" in top
    assert "nothing to be in or out of focus" in top
    assert not any(
        "badly out of focus" in issue or "motion blurred" in issue
        for issue in report.issues
    ), report.issues


@pytest.mark.parametrize("size", [(1, 1), (1, 40), (40, 1), (2, 2), (3, 3)])
def test_tiny_images_of_every_shape_survive(size):
    report = image_quality_ai.assess(np.full(size, 200, dtype=np.uint8))

    assert 0.0 <= report.score <= 100.0
    assert report.to_dict()["score"] == pytest.approx(report.score, abs=0.05)


@pytest.mark.parametrize("level", [0, 255])
def test_fully_black_and_fully_white_frames_report_instead_of_dividing_by_zero(level):
    report = image_quality_ai.assess(np.full((64, 64), level, dtype=np.uint8))

    assert 0.0 <= report.score <= 100.0
    assert report.usable is False
    assert report.metrics["contrast"].value == pytest.approx(0.0, abs=0.01)
    assert report.metrics["exposure"].ok is False
    assert report.issues
    assert report.summary()


def test_an_empty_array_is_rejected_clearly():
    with pytest.raises(ValueError, match="no pixels"):
        image_quality_ai.assess(np.zeros((0, 10), dtype=np.uint8))


# ---------------------------------------------------------------------------
# every input type
# ---------------------------------------------------------------------------
def test_greyscale_rgb_and_rgba_arrays_all_work(good):
    grey = image_quality_ai.assess(good)
    colour = image_quality_ai.assess(as_rgb(good))
    alpha = image_quality_ai.assess(
        np.dstack([good, good, good, np.full_like(good, 255)])
    )

    assert grey.image["channels"] == 1
    assert colour.image["channels"] == 3
    assert alpha.image["channels"] == 4
    # the same pixels, so the same verdict
    assert grey.score == pytest.approx(colour.score, abs=0.01)
    assert grey.score == pytest.approx(alpha.score, abs=0.01)


def test_a_transparent_png_is_judged_on_what_a_viewer_would_show(tmp_path, good):
    target = tmp_path / "transparent.png"
    rgba = np.dstack([good, good, good, np.full_like(good, 60)])
    Image.fromarray(rgba, "RGBA").save(target)

    report = image_quality_ai.assess(str(target))

    assert report.image["channels"] == 4
    assert any("alpha" in note for note in report.image["notes"])
    assert 0.0 <= report.score <= 100.0


def test_sixteen_bit_images_are_read_at_full_depth(tmp_path, good):
    target = tmp_path / "depth16.tif"
    Image.fromarray(good.astype(np.uint16) * 257).save(target)

    report = image_quality_ai.assess(str(target))

    assert report.image["bit_depth"] == 16
    assert any("16-bit" in note for note in report.image["notes"])
    # the same scene at 8 bits scores the same
    assert report.score == pytest.approx(image_quality_ai.assess(good).score, abs=3.0)


def test_sixteen_bit_arrays_are_scaled_by_their_dtype(good):
    report = image_quality_ai.assess(good.astype(np.uint16) * 257)

    assert report.metrics["exposure"].value == pytest.approx(
        image_quality_ai.assess(good).metrics["exposure"].value, abs=0.01
    )


def test_float_arrays_on_the_zero_to_one_scale_work(good):
    report = image_quality_ai.assess(good.astype(np.float32) / 255.0)

    assert report.score == pytest.approx(image_quality_ai.assess(good).score, abs=1.0)


def test_the_caller_image_object_is_never_modified(good, good_rgb):
    """Assessing is a read. Whatever was handed over comes back untouched,
    whether it was an array of any dtype or a live PIL image."""
    for original in (
        good,                                   # uint8 greyscale
        good_rgb,                               # uint8 RGB
        np.dstack([good, good, good, np.full_like(good, 90)]),   # uint8 RGBA
        good.astype(np.uint16) * 257,           # 16-bit
        good.astype(np.float32) / 255.0,        # float 0..1
    ):
        before = original.copy()
        image_quality_ai.assess(original)
        assert np.array_equal(original, before), "assess wrote back into a {0} array".format(
            original.dtype
        )

    live = Image.fromarray(good_rgb)
    pixels_before = np.asarray(live).copy()
    size_before, mode_before = live.size, live.mode
    image_quality_ai.assess(live)

    assert (live.size, live.mode) == (size_before, mode_before)
    assert np.array_equal(np.asarray(live), pixels_before)


def test_a_float_array_full_of_nan_is_rejected_clearly():
    with pytest.raises(ValueError, match="NaN or infinite"):
        image_quality_ai.assess(np.full((8, 8), np.nan, dtype=np.float32))


def test_a_pil_image_can_be_passed_straight_in(good_rgb):
    report = image_quality_ai.assess(Image.fromarray(good_rgb))

    assert report.source == "<PIL.Image>"
    assert report.score == pytest.approx(image_quality_ai.assess(good_rgb).score, abs=0.01)


def test_a_palette_image_is_expanded(tmp_path, good_rgb):
    target = tmp_path / "palette.png"
    Image.fromarray(good_rgb).convert("P").save(target)

    report = image_quality_ai.assess(str(target))

    assert any("palette" in note for note in report.image["notes"])
    assert 0.0 <= report.score <= 100.0


def test_a_one_bit_image_is_read_as_black_and_white(tmp_path, good):
    target = tmp_path / "bilevel.png"
    Image.fromarray(good).convert("1").save(target)

    report = image_quality_ai.assess(str(target))

    assert any("1-bit" in note for note in report.image["notes"])
    assert 0.0 <= report.score <= 100.0


def test_a_shape_that_is_not_an_image_is_rejected_clearly():
    with pytest.raises(ValueError, match="HxW"):
        image_quality_ai.assess(np.zeros((4, 4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="3 or 4 channels"):
        image_quality_ai.assess(np.zeros((4, 4, 2), dtype=np.uint8))


def test_something_that_is_not_an_image_at_all_is_rejected_clearly():
    with pytest.raises(TypeError, match="path, a PIL.Image.Image or a numpy array"):
        image_quality_ai.assess(42)


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------
def test_exif_orientation_is_applied_before_anything_is_measured(tmp_path, good_rgb):
    upright = Image.fromarray(good_rgb)              # 320 wide, 240 tall
    rotated = tmp_path / "rotated.jpg"
    exif = upright.getexif()
    exif[274] = 6                                    # "rotate 90 degrees clockwise"
    upright.save(rotated, exif=exif, quality=95)

    report = image_quality_ai.assess(str(rotated))

    assert (report.image["width"], report.image["height"]) == (240, 320)
    assert report.image["orientation_applied"] is True
    assert "EXIF rotation applied" in report.summary()


def test_a_photo_without_an_orientation_tag_is_left_alone(tmp_path, good_rgb):
    plain = tmp_path / "plain.jpg"
    Image.fromarray(good_rgb).save(plain, quality=95)

    report = image_quality_ai.assess(str(plain))

    assert (report.image["width"], report.image["height"]) == (320, 240)
    assert report.image["orientation_applied"] is False


def test_orientation_does_not_change_the_verdict(tmp_path, good_rgb):
    """Rotating a frame moves the pixels, not the quality of them."""
    plain = tmp_path / "plain.png"
    Image.fromarray(good_rgb).save(plain)
    turned = tmp_path / "turned.png"
    Image.fromarray(np.rot90(good_rgb).copy()).save(turned)

    straight = image_quality_ai.assess(str(plain))
    sideways = image_quality_ai.assess(str(turned))

    assert straight.metrics["exposure"].value == pytest.approx(
        sideways.metrics["exposure"].value, abs=0.001
    )


# ---------------------------------------------------------------------------
# files that will not open
# ---------------------------------------------------------------------------
def test_a_missing_path_raises_file_not_found(tmp_path):
    missing = tmp_path / "not-here.jpg"

    with pytest.raises(FileNotFoundError) as caught:
        image_quality_ai.assess(str(missing))

    assert str(missing) in str(caught.value)


def test_a_truncated_file_raises_a_clear_value_error_naming_the_path(tmp_path, good_rgb):
    whole = tmp_path / "whole.jpg"
    Image.fromarray(good_rgb).save(whole, quality=95)
    broken = tmp_path / "truncated.jpg"
    broken.write_bytes(whole.read_bytes()[: len(whole.read_bytes()) // 3])

    with pytest.raises(ValueError) as caught:
        image_quality_ai.assess(str(broken))

    message = str(caught.value)
    assert str(broken) in message
    assert message.startswith("cannot read image")
    assert "truncated" in message


def test_a_file_that_is_not_an_image_raises_a_clear_value_error(tmp_path):
    fake = tmp_path / "notreally.png"
    fake.write_bytes(b"this is plain text wearing a .png suffix")

    with pytest.raises(ValueError) as caught:
        image_quality_ai.assess(str(fake))

    assert str(fake) in str(caught.value)


def test_an_empty_file_raises_a_clear_value_error(tmp_path):
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")

    with pytest.raises(ValueError) as caught:
        image_quality_ai.assess(str(empty))

    assert str(empty) in str(caught.value)


def test_a_directory_is_not_mistaken_for_an_image(tmp_path):
    with pytest.raises(ValueError, match="is a directory"):
        image_quality_ai.assess(str(tmp_path))


def test_a_path_object_works_as_well_as_a_string(tmp_path, good_rgb):
    target = tmp_path / "pathlib.png"
    Image.fromarray(good_rgb).save(target)

    assert image_quality_ai.assess(target).usable is True


def test_a_unicode_filename_survives_the_whole_round_trip(tmp_path, good_rgb):
    target = tmp_path / "写真-café-★.png"
    Image.fromarray(good_rgb).save(target)

    report = image_quality_ai.assess(str(target))

    assert report.source == str(target)
    assert "café" in report.summary()
    assert "café" in report.to_json()


# ---------------------------------------------------------------------------
# promises the README makes
# ---------------------------------------------------------------------------
def test_scores_are_deterministic(good_rgb):
    first = image_quality_ai.assess(good_rgb).to_dict()
    second = image_quality_ai.assess(good_rgb).to_dict()
    third = image_quality_ai.assess(np.array(good_rgb)).to_dict()

    assert first == second == third


def test_a_four_thousand_by_three_thousand_photo_is_assessed_in_well_under_a_second():
    big = np.asarray(
        Image.fromarray(photo(800, 600)).resize((4000, 3000), Image.Resampling.BICUBIC)
    )
    big = as_rgb(big)

    started = time.perf_counter()
    report = image_quality_ai.assess(big)
    elapsed = time.perf_counter() - started

    assert report.image["pixels"] == 12_000_000
    assert elapsed < 1.0, "12 megapixels took {0:.2f}s".format(elapsed)
