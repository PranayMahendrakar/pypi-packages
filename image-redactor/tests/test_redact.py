"""The happy paths: the quickstart, the four methods, and the result object."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

import image_redactor
from image_redactor import METHODS, RedactResult, Redactor, redact

from conftest import gradient, noise


def test_readme_quickstart_runs_verbatim():
    photo = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    result = image_redactor.redact(photo, regions=[(40, 20, 100, 90)], method="pixelate")
    assert isinstance(result.summary(), str)
    assert result.count == 1
    assert result.changed is True
    # the promise the quickstart makes about the caller's array
    assert np.array_equal(
        photo, np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    )


def test_version_is_exported():
    assert image_redactor.__version__ == "0.1.0"


def test_one_line_call_returns_a_usable_result(photo):
    result = redact(photo, regions=[(10, 10, 60, 50)])
    assert isinstance(result, RedactResult)
    assert result.count == 1 == len(result.boxes)
    assert result.width == 160 and result.height == 120
    assert result.size == (160, 120)
    assert result.mode == "RGB"
    assert result.method == "blur"
    assert result.detector_used is None       # our regions were used as given
    assert result.changed is True


@pytest.mark.parametrize("method", METHODS)
def test_every_method_hides_the_region(photo, method):
    result = redact(photo, regions=[(20, 20, 90, 80)], method=method, expand=0.0)
    assert result.method == method
    assert result.count == 1
    assert result.changed is True
    before = photo[20:80, 20:90]
    after = np.asarray(result.image)[20:80, 20:90]
    assert not np.array_equal(before, after)
    # and nothing outside the box moved
    assert np.array_equal(photo[:20], np.asarray(result.image)[:20])


def test_blur_smooths_and_pixelate_flattens(smooth):
    blurred = redact(smooth, regions=[(10, 10, 80, 50)], method="blur", expand=0.0)
    patch = np.asarray(blurred.image)[10:50, 10:80].astype(float)
    assert patch.std() < smooth[10:50, 10:80].astype(float).std()

    blocky = redact(smooth, regions=[(10, 10, 80, 50)], method="pixelate", expand=0.0)
    patch = np.asarray(blocky.image)[10:50, 10:80]
    colours = np.unique(patch.reshape(-1, 3), axis=0)
    assert len(colours) < 40      # flat blocks, not 2800 distinct pixels


def test_fill_and_blackout_paint_flat(photo):
    black = redact(photo, regions=[(10, 10, 50, 40)], method="blackout", expand=0.0)
    assert np.asarray(black.image)[10:40, 10:50].max() == 0

    filled = redact(photo, regions=[(10, 10, 50, 40)], method="fill", expand=0.0)
    patch = np.asarray(filled.image)[10:40, 10:50]
    assert np.unique(patch.reshape(-1, 3), axis=0).shape[0] == 1


def test_strength_zero_is_the_lightest_setting_not_a_no_op(smooth):
    light = redact(smooth, regions=[(5, 5, 90, 60)], method="blur", strength=0.0, expand=0.0)
    heavy = redact(smooth, regions=[(5, 5, 90, 60)], method="blur", strength=1.0, expand=0.0)
    assert light.changed is True and heavy.changed is True
    light_std = np.asarray(light.image)[5:60, 5:90].astype(float).std()
    heavy_std = np.asarray(heavy.image)[5:60, 5:90].astype(float).std()
    assert heavy_std < light_std


def test_expand_grows_the_box_by_the_right_fraction(photo):
    tight = redact(photo, regions=[(50, 30, 150, 80)], expand=0.0)
    loose = redact(photo, regions=[(50, 30, 150, 80)], expand=0.1)
    assert tight.boxes == [(50, 30, 150, 80)]
    # 10 percent of a 100-wide, 50-tall box is 10 and 5 on every edge
    assert loose.boxes == [(40, 25, 160, 85)]


def test_a_single_box_may_be_passed_unwrapped(photo):
    one = redact(photo, regions=(10, 10, 40, 40), expand=0.0)
    wrapped = redact(photo, regions=[(10, 10, 40, 40)], expand=0.0)
    assert one.boxes == wrapped.boxes == [(10, 10, 40, 40)]


def test_overlapping_boxes_are_deduplicated(photo):
    result = redact(
        photo,
        regions=[(10, 10, 60, 60), (10, 10, 60, 60), (20, 20, 40, 40)],
        expand=0.0,
    )
    assert result.boxes == [(10, 10, 60, 60)]
    assert result.count == 1


def test_result_summary_is_plain_ascii_and_explains_itself(photo):
    result = redact(photo, regions=[(10, 10, 60, 60)], method="blackout")
    text = result.summary()
    text.encode("ascii")                     # no arrows, bullets or box characters
    assert "1 region(s) redacted with blackout" in text
    assert "pixels changed: yes" in text
    assert "the original pixels are gone" in text


def test_summary_warns_that_blur_is_not_destruction(photo):
    text = redact(photo, regions=[(10, 10, 60, 60)], method="blur").summary()
    assert "obscured, not destroyed" in text


def test_to_dict_is_json_safe_and_to_json_round_trips(photo):
    result = redact(photo, regions=[(10, 10, 60, 60)], method="pixelate")
    data = result.to_dict()
    assert data["count"] == 1
    assert data["boxes"] == [list(result.boxes[0])]
    assert data["irreversible"] is True
    assert data["method"] == "pixelate"
    assert json.loads(result.to_json()) == data
    json.dumps(data)                          # must not raise on numpy scalars


def test_irreversible_flag_matches_the_method(photo):
    for method in ("pixelate", "fill", "blackout"):
        assert redact(photo, regions=[(1, 1, 30, 30)], method=method).irreversible is True
    assert redact(photo, regions=[(1, 1, 30, 30)], method="blur").irreversible is False


def test_redactor_class_keeps_settings_across_images():
    redactor = Redactor(method="fill", strength=1.0, expand=0.0, fill_color=(10, 20, 30))
    for seed in (1, 2):
        image = noise(40, 50, seed=seed)
        result = redactor.redact(image, regions=[(5, 5, 25, 25)])
        assert result.method == "fill"
        patch = np.asarray(result.image)[5:25, 5:25]
        assert (patch == np.array([10, 20, 30], dtype=np.uint8)).all()


def test_redactor_rejects_a_bad_fill_colour():
    with pytest.raises(ValueError, match="fill_color"):
        Redactor(fill_color=(10, 20))
    with pytest.raises(ValueError, match="between 0 and 255"):
        Redactor(fill_color=(10, 20, 900))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "smudge"}, "method must be one of"),
        ({"strength": 4}, "strength must be between 0 and 1"),
        ({"strength": "hard"}, "strength must be a number"),
        ({"expand": -0.5}, "expand must be 0 or more"),
        ({"expand": "lots"}, "expand must be a number"),
    ],
)
def test_bad_settings_raise_a_clear_value_error(photo, kwargs, match):
    with pytest.raises(ValueError, match=match):
        redact(photo, regions=[(1, 1, 20, 20)], **kwargs)


def test_pil_image_in_pil_image_out(pil_photo):
    result = redact(pil_photo, regions=[(5, 5, 40, 40)], method="blackout")
    assert isinstance(result.image, Image.Image)
    assert result.image.mode == "RGB"
    assert result.image.size == pil_photo.size


def test_a_missing_path_names_itself():
    with pytest.raises(FileNotFoundError, match="no-such-image.png"):
        redact("no-such-image.png")


def test_an_unsupported_input_type_says_what_is_accepted():
    with pytest.raises(TypeError, match="file path, a PIL.Image.Image or a numpy array"):
        redact(42)
    with pytest.raises(ValueError, match="image is None"):
        redact(None)


def test_odd_array_shapes_are_rejected_clearly():
    with pytest.raises(ValueError, match="1 .grey., 3 .RGB. or 4 .RGBA. channels"):
        redact(np.zeros((10, 10, 2), dtype=np.uint8), regions=[(1, 1, 5, 5)])
    with pytest.raises(ValueError, match="2-dimensional"):
        redact(np.zeros((4, 4, 3, 2), dtype=np.uint8), regions=[(1, 1, 3, 3)])
    with pytest.raises(ValueError, match="zero height or width"):
        redact(np.zeros((0, 10, 3), dtype=np.uint8), regions=[(1, 1, 5, 5)])


def test_a_float_image_comes_back_as_a_float_image():
    unit = gradient(40, 40).astype(np.float32) / 255.0
    result = redact(unit, regions=[(5, 5, 30, 30)], method="blackout", expand=0.0)
    assert result.image.dtype == np.float32
    assert result.image.max() <= 1.0
    assert result.image[5:30, 5:30].max() == 0.0
