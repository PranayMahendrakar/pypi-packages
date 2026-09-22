"""The edge cases the package promises to handle, one test each.

Every case in this file is something a redaction tool gets wrong quietly if
nobody pins it down: silently failing on an empty run, redacting the caller's
own pixels, leaving a "redacted" region recoverable, or losing a whole batch of
detections because one detector blew up.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from image_redactor import Redactor, detect_faces, detect_plates, redact

from conftest import gradient, noise


# --------------------------------------------------------------------------- #
# nothing to redact is not a failure
# --------------------------------------------------------------------------- #

def test_no_regions_and_no_detections_returns_the_image_unchanged(photo):
    result = redact(photo, detector=lambda image: [])
    assert result.count == 0
    assert result.boxes == []
    assert result.changed is False
    assert np.array_equal(np.asarray(result.image), photo)


def test_an_empty_run_says_so_instead_of_failing(photo):
    result = redact(photo, detector=lambda image: [])
    text = result.summary()
    assert "nothing to redact" in text
    assert "0 regions found" in text
    assert "the image is returned unchanged" in text
    assert any("no regions were redacted" in warning for warning in result.warnings)
    assert result.to_dict()["count"] == 0


def test_an_empty_run_on_a_flat_image_is_still_not_an_error():
    flat = np.zeros((40, 40, 3), dtype=np.uint8)
    result = redact(flat, regions=[], detector=lambda image: [])
    assert result.count == 0 and result.changed is False


# --------------------------------------------------------------------------- #
# boxes outside the image are clipped, not rejected
# --------------------------------------------------------------------------- #

def test_a_box_overhanging_the_edge_is_clipped_to_bounds(photo):
    result = redact(photo, regions=[(-40, -30, 50, 40)], method="blackout", expand=0.0)
    assert result.boxes == [(0, 0, 50, 40)]
    assert np.asarray(result.image)[0:40, 0:50].max() == 0
    assert any("clipped" in warning for warning in result.warnings)


def test_a_box_past_the_far_edge_is_clipped_too(photo):
    result = redact(photo, regions=[(140, 100, 9999, 9999)], method="blackout", expand=0.0)
    assert result.boxes == [(140, 100, 160, 120)]
    assert result.changed is True


def test_a_box_entirely_outside_is_dropped_and_reported(photo):
    result = redact(photo, regions=[(900, 900, 1000, 1000)], method="blackout")
    assert result.count == 0
    assert result.changed is False
    assert any("entirely outside" in warning for warning in result.warnings)


def test_expanding_never_pushes_a_box_out_of_the_image(photo):
    result = redact(photo, regions=[(0, 0, 160, 120)], expand=0.5)
    left, top, right, bottom = result.boxes[0]
    assert (left, top, right, bottom) == (0, 0, 160, 120)


def test_a_zero_area_box_is_dropped_not_redacted(photo):
    result = redact(photo, regions=[(30, 30, 30, 60)], method="blackout")
    assert result.count == 0
    assert result.changed is False
    assert any("zero width or height" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# an inverted box is a bug in the caller's coordinates, and says so
# --------------------------------------------------------------------------- #

def test_right_less_than_left_raises_a_clear_value_error(photo):
    with pytest.raises(ValueError) as caught:
        redact(photo, regions=[(90, 10, 20, 60)])
    message = str(caught.value)
    assert "inverted" in message
    assert "right (20) is less than left (90)" in message
    assert "left <= right" in message


def test_bottom_less_than_top_raises_a_clear_value_error(photo):
    with pytest.raises(ValueError) as caught:
        redact(photo, regions=[(10, 90, 60, 20)])
    message = str(caught.value)
    assert "inverted" in message
    assert "bottom (20) is less than top (90)" in message


def test_the_inverted_box_error_names_which_box_it_was(photo):
    with pytest.raises(ValueError, match=r"region 2 is inverted"):
        redact(photo, regions=[(1, 1, 9, 9), (2, 2, 8, 8), (90, 10, 20, 60)])


def test_a_malformed_box_is_rejected_with_a_useful_message(photo):
    with pytest.raises(ValueError, match="a box is exactly 4"):
        redact(photo, regions=[(1, 2, 3)])
    with pytest.raises(ValueError, match="non-finite"):
        redact(photo, regions=[(1, 2, float("nan"), 4)])
    with pytest.raises(ValueError, match="is a string"):
        redact(photo, regions=["10,10,40,40"])


# --------------------------------------------------------------------------- #
# irreversibility: the pixels have to actually be gone
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["fill", "blackout", "pixelate"])
def test_irreversible_methods_genuinely_destroy_the_pixels(method):
    original = noise(60, 60, seed=7)
    box = (10, 10, 50, 50)
    result = redact(original, regions=[box], method=method, strength=1.0, expand=0.0)

    assert result.irreversible is True
    assert result.changed is True

    before = original[10:50, 10:50]
    after = np.asarray(result.image)[10:50, 10:50]

    # not merely different - the per-pixel detail is gone
    assert not np.array_equal(before, after)
    assert float(np.abs(after.astype(int) - before.astype(int)).mean()) > 5.0
    assert np.unique(after.reshape(-1, 3), axis=0).shape[0] < 0.2 * before[:, :, 0].size


@pytest.mark.parametrize("method", ["fill", "blackout", "pixelate"])
def test_a_redacted_region_cannot_be_recovered_by_resizing_back(method):
    """The naive attack: resize the image around and hope the detail comes back.

    It cannot, because these three methods threw the bits away - there is
    nothing left to interpolate from. This is exactly what separates them from
    ``blur``, which is a linear filter and is not claimed to be irreversible.
    """
    original = noise(64, 64, seed=21)
    box = (8, 8, 56, 56)
    result = redact(original, regions=[box], method=method, strength=1.0, expand=0.0)
    redacted = np.asarray(result.image)

    # Measure well inside the box: a resize pulls untouched neighbouring pixels
    # across the boundary, and that bleed is not recovery.
    inner = (slice(16, 48), slice(16, 48))
    before = original[inner].astype(int)
    hidden_error = float(np.abs(redacted[inner].astype(int) - before).mean())
    assert hidden_error > 5.0

    image = Image.fromarray(redacted)
    for scale, resample in (
        (4, Image.LANCZOS),
        (4, Image.BICUBIC),
        (8, Image.BILINEAR),
        (2, Image.NEAREST),
    ):
        up = image.resize((image.width * scale, image.height * scale), resample)
        back = np.asarray(up.resize(image.size, resample))
        down = image.resize((image.width // scale, image.height // scale), resample)
        forth = np.asarray(down.resize(image.size, resample))

        for recovered in (back[inner].astype(int), forth[inner].astype(int)):
            assert not np.array_equal(recovered, before)
            # still nowhere near the original...
            assert float(np.abs(recovered - before).mean()) > 5.0
            # ...and no closer to it than the redacted image already was, so the
            # round trip recovered no information at all.
            assert float(np.abs(recovered - before).mean()) >= 0.75 * hidden_error


@pytest.mark.parametrize("method", ["fill", "blackout"])
def test_fill_and_blackout_leave_a_single_flat_colour(method):
    original = noise(40, 40, seed=8)
    result = redact(original, regions=[(5, 5, 35, 35)], method=method, expand=0.0)
    patch = np.asarray(result.image)[5:35, 5:35]
    assert np.unique(patch.reshape(-1, 3), axis=0).shape[0] == 1
    assert patch.std() == 0.0


def test_pixelate_replaces_each_block_with_one_colour():
    original = noise(48, 48, seed=9)
    result = redact(original, regions=[(0, 0, 48, 48)], method="pixelate",
                    strength=1.0, expand=0.0)
    patch = np.asarray(result.image)
    # every pixel inside one block must be the same colour as its neighbours
    # (compare whole RGB triples: a per-channel std would also see R != G != B)
    corner = np.unique(patch[0:4, 0:4].reshape(-1, 3), axis=0)
    assert corner.shape[0] == 1
    # 48 pixels at strength 1.0 is a 16-pixel block, so 3x3 = 9 colours, no more
    assert np.unique(patch.reshape(-1, 3), axis=0).shape[0] <= 9


def test_blur_is_honestly_not_marked_irreversible(photo):
    result = redact(photo, regions=[(10, 10, 60, 60)], method="blur")
    assert result.irreversible is False
    assert result.to_dict()["irreversible"] is False


# --------------------------------------------------------------------------- #
# the caller's image is never touched
# --------------------------------------------------------------------------- #

def test_the_original_numpy_array_is_never_modified_in_place():
    original = noise(50, 60, seed=11)
    keep = original.copy()
    result = redact(original, regions=[(5, 5, 45, 45)], method="blackout")
    assert np.array_equal(original, keep)
    assert result.image is not original
    assert not np.shares_memory(np.asarray(result.image), original)


def test_the_original_pil_image_is_never_modified_in_place():
    source = Image.fromarray(noise(50, 60, seed=12))
    keep = np.asarray(source).copy()
    result = redact(source, regions=[(5, 5, 45, 45)], method="blackout")
    assert np.array_equal(np.asarray(source), keep)
    assert result.image is not source


def test_redacting_twice_from_the_same_source_gives_the_same_answer():
    original = noise(40, 40, seed=13)
    first = redact(original, regions=[(5, 5, 30, 30)], method="pixelate")
    second = redact(original, regions=[(5, 5, 30, 30)], method="pixelate")
    assert np.array_equal(np.asarray(first.image), np.asarray(second.image))


def test_a_reused_redactor_does_not_leak_between_images():
    redactor = Redactor(method="blackout", expand=0.0)
    first = noise(30, 30, seed=14)
    second = noise(30, 30, seed=15)
    redactor.redact(first, regions=[(0, 0, 30, 30)])
    result = redactor.redact(second, regions=[(0, 0, 10, 10)])
    assert np.asarray(result.image)[20:, 20:].max() > 0     # untouched area survived
    assert first.max() > 0 and second.max() > 0


# --------------------------------------------------------------------------- #
# greyscale and RGBA
# --------------------------------------------------------------------------- #

def test_greyscale_arrays_work_and_keep_their_shape(grey):
    result = redact(grey, regions=[(10, 10, 60, 50)], method="blackout", expand=0.0)
    assert result.mode == "L"
    assert result.image.shape == grey.shape == (80, 100)
    assert result.image[10:50, 10:60].max() == 0
    assert result.changed is True
    assert np.array_equal(grey[:10], result.image[:10])


def test_greyscale_fill_uses_a_single_grey_level(grey):
    result = redact(grey, regions=[(5, 5, 40, 40)], method="fill", expand=0.0)
    patch = result.image[5:40, 5:40]
    assert patch.std() == 0.0
    assert patch.flat[0] == 128


def test_rgba_works_and_the_alpha_channel_survives(rgba):
    result = redact(rgba, regions=[(10, 10, 60, 50)], method="blackout", expand=0.0)
    assert result.mode == "RGBA"
    assert result.image.shape == rgba.shape == (80, 100, 4)
    assert np.array_equal(result.image[:, :, 3], rgba[:, :, 3])   # transparency intact
    assert result.image[10:50, 10:60, :3].max() == 0              # colour gone


def test_a_single_channel_3d_array_is_treated_as_greyscale():
    image = noise(40, 40, channels=1, seed=16)[:, :, None]
    result = redact(image, regions=[(5, 5, 30, 30)], method="blackout", expand=0.0)
    assert result.mode == "L"
    assert result.image.shape == (40, 40, 1)


def test_a_greyscale_pil_image_comes_back_greyscale():
    source = Image.fromarray(noise(40, 40, channels=1, seed=17), mode="L")
    result = redact(source, regions=[(5, 5, 30, 30)], method="blackout")
    assert result.image.mode == "L"
    assert result.mode == "L"


def test_a_palette_pil_image_is_handled():
    source = Image.fromarray(noise(40, 40, seed=18)).convert("P")
    result = redact(source, regions=[(5, 5, 30, 30)], method="blackout")
    assert result.mode == "RGB"
    assert result.changed is True


# --------------------------------------------------------------------------- #
# one broken detector must not cost you the others
# --------------------------------------------------------------------------- #

def _boom(image):
    raise RuntimeError("the model file is missing")


def _finds_one(image):
    return [(10, 10, 40, 40)]


def _finds_another(image):
    return [(60, 60, 90, 90)]


def test_a_detector_that_raises_is_caught_and_reported(photo):
    result = redact(photo, detector=_boom, expand=0.0)
    assert result.count == 0
    assert result.changed is False                       # and it did not half-redact
    assert len(result.errors) == 1
    assert "RuntimeError: the model file is missing" in result.errors[0]
    report = result.detections[0]
    assert report.name == "_boom"
    assert report.ok is False
    assert report.boxes == 0


def test_a_failing_detector_does_not_lose_the_others(photo):
    result = redact(photo, detector=[_finds_one, _boom, _finds_another],
                    method="blackout", expand=0.0)
    assert sorted(result.boxes) == [(10, 10, 40, 40), (60, 60, 90, 90)]
    assert result.count == 2
    assert result.changed is True
    assert [report.ok for report in result.detections] == [True, False, True]
    assert len(result.errors) == 1


def test_a_detector_returning_nonsense_is_treated_as_a_failure(photo):
    def _junk(image):
        return {"left": 1}

    result = redact(photo, detector=[_junk, _finds_one], method="blackout", expand=0.0)
    assert result.boxes == [(10, 10, 40, 40)]
    assert len(result.errors) == 1
    assert result.detections[0].ok is False


def test_a_detector_failure_shows_up_in_the_summary_and_the_dict(photo):
    result = redact(photo, detector=[_boom, _finds_one])
    text = result.summary()
    assert "_boom: FAILED" in text
    assert "the model file is missing" in text
    data = result.to_dict()
    assert data["detections"][0]["error"].startswith("RuntimeError")
    assert data["detections"][1]["boxes"] == 1


def test_every_detector_failing_still_returns_the_image_unchanged(photo):
    result = redact(photo, detector=[_boom, _boom])
    assert result.count == 0
    assert np.array_equal(np.asarray(result.image), photo)
    assert len(result.errors) == 2


def test_a_non_callable_detector_is_rejected_up_front(photo):
    with pytest.raises(ValueError, match="detector must be a callable"):
        redact(photo, detector="my_model")
    with pytest.raises(ValueError, match="in the list"):
        redact(photo, detector=[_finds_one, "my_model"])


def test_detectors_are_handed_a_pil_image(photo):
    seen = {}

    def _inspect(image):
        seen["type"] = type(image)
        seen["mode"] = image.mode
        seen["size"] = image.size
        return []

    redact(photo, detector=_inspect)
    assert seen["type"] is Image.Image or issubclass(seen["type"], Image.Image)
    assert seen["mode"] == "RGB"
    assert seen["size"] == (160, 120)


def test_detector_boxes_are_clipped_like_your_own(photo):
    result = redact(photo, detector=lambda image: [(-10, -10, 40, 40)],
                    method="blackout", expand=0.0)
    assert result.boxes == [(0, 0, 40, 40)]


def test_an_inverted_box_from_a_detector_is_that_detectors_failure(photo):
    """Your own inverted box raises; a detector's is caught like any other fault.

    The asymmetry is deliberate. An inverted box in ``regions`` is a bug in the
    caller's own code and must be loud. An inverted box out of a detector is the
    detector misbehaving, and the whole point of the detector contract is that
    one misbehaving detector never costs you the others or the run.
    """
    result = redact(photo, detector=[lambda image: [(90, 10, 20, 60)], _finds_one],
                    method="blackout", expand=0.0)
    assert result.boxes == [(10, 10, 40, 40)]          # the good detector still ran
    assert result.detections[0].ok is False
    assert "inverted" in result.detections[0].error
    assert len(result.errors) == 1

    # but the same box passed as regions= is the caller's bug, and raises
    with pytest.raises(ValueError, match="inverted"):
        redact(photo, regions=[(90, 10, 20, 60)])


def test_regions_and_a_detector_can_be_combined(photo):
    result = redact(photo, regions=[(100, 80, 140, 110)], detector=_finds_one,
                    method="blackout", expand=0.0)
    assert sorted(result.boxes) == [(10, 10, 40, 40), (100, 80, 140, 110)]
    assert result.detector_used == "_finds_one"


# --------------------------------------------------------------------------- #
# the built-in fallback is honest about itself
# --------------------------------------------------------------------------- #

def test_the_builtin_fallback_runs_only_when_nothing_else_was_given(obvious_scene):
    fallback = redact(obvious_scene)
    assert fallback.detector_used is not None
    assert "built-in" in fallback.detector_used

    explicit = redact(obvious_scene, regions=[(10, 10, 40, 40)])
    assert explicit.detector_used is None


def test_the_builtin_fallback_attaches_the_caveat(obvious_scene):
    result = redact(obvious_scene)
    assert any("weak pixel heuristics" in warning for warning in result.warnings)
    assert "weak pixel heuristics" in result.summary()
    assert any("weak pixel heuristics" in w for w in result.to_dict()["warnings"])


def test_the_builtin_fallback_finds_the_obvious_shapes(obvious_scene):
    result = redact(obvious_scene, method="blackout")
    assert result.count >= 2
    assert result.changed is True


def test_the_greyscale_face_caveat_is_recorded(grey):
    result = redact(grey)
    assert any("greyscale" in warning for warning in result.warnings)


def test_detect_faces_returns_nothing_for_greyscale(grey):
    assert detect_faces(grey) == []


def test_detect_faces_and_detect_plates_find_the_planted_shapes(obvious_scene):
    faces = detect_faces(obvious_scene)
    plates = detect_plates(obvious_scene)
    assert len(faces) == 1
    assert len(plates) == 1
    face_left, face_top, face_right, face_bottom = faces[0]
    assert face_left < 80 < face_right and face_top < 70 < face_bottom
    plate_left, plate_top, plate_right, plate_bottom = plates[0]
    assert plate_left <= 185 and plate_right >= 275
    assert plate_top <= 145 and plate_bottom >= 165


def test_the_builtin_detectors_return_nothing_on_a_blank_frame():
    blank = np.zeros((80, 80, 3), dtype=np.uint8)
    assert detect_faces(blank) == []
    assert detect_plates(blank) == []
    white = np.full((80, 80, 3), 255, dtype=np.uint8)
    assert detect_plates(white) == []          # everything bright means nothing found


def test_a_run_that_detects_nothing_is_reported_as_such_not_as_clean():
    blank = np.zeros((80, 80, 3), dtype=np.uint8)
    result = redact(blank)
    assert result.count == 0
    assert result.changed is False
    assert "nothing to redact" in result.summary()
    assert any("weak pixel heuristics" in w for w in result.warnings)


def test_a_smooth_gradient_does_not_crash_the_detectors():
    image = gradient(120, 160)
    detect_faces(image)
    detect_plates(image)
    redact(image, method="blackout")            # must not raise whatever it finds
