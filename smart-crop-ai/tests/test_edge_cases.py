"""The awkward inputs. Every one of these is a promise the README makes."""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

import smart_crop_ai
from conftest import SUBJECT_SIZE, box_contains_subject, subject_array


# ------------------------------------------------- a target bigger than the source


def test_a_target_larger_than_the_source_returns_the_whole_image(subject_image):
    """No silent upscaling, confidence 0, and a note saying what happened."""
    result = smart_crop_ai.crop(subject_image, 2000, 2000)

    assert result.box == (0, 0, SUBJECT_SIZE[0], SUBJECT_SIZE[1])
    assert result.image.size == SUBJECT_SIZE
    assert result.strategy_used == "whole_image"
    assert result.confidence == 0.0
    assert result.notes
    assert any(
        "larger than" in note and "Nothing was upscaled" in note
        for note in result.notes
    )


@pytest.mark.parametrize("width, height", [(2000, 100), (100, 2000), (801, 401)])
def test_one_oversized_side_is_enough_to_refuse(subject_image, width, height):
    result = smart_crop_ai.crop(subject_image, width, height)

    assert result.strategy_used == "whole_image"
    assert result.confidence == 0.0
    assert result.image.size == SUBJECT_SIZE


def test_an_oversized_target_never_upscales_the_pixels(subject_image):
    result = smart_crop_ai.crop(subject_image, 5000, 5000)
    assert result.image.size == subject_image.size


# ------------------------------------------------------------ a flat image


def test_a_flat_single_colour_image_falls_back_to_centre_and_says_so(flat_image):
    result = smart_crop_ai.crop(flat_image, ratio=1.0)

    assert result.strategy_used == "center"
    assert result.confidence == 0.0
    assert not result.moved
    assert any("flat" in note for note in result.notes)
    assert result.confidence_label == "none"


@pytest.mark.parametrize("strategy", ["auto", "saliency", "entropy", "edges"])
def test_no_strategy_invents_a_subject_in_a_blank_image(flat_image, strategy):
    """A blank wall must not be sold as a confident crop by any strategy."""
    result = smart_crop_ai.crop(flat_image, ratio=1.0, strategy=strategy)

    assert result.strategy_used == "center"
    assert result.confidence == 0.0


def test_a_smooth_gradient_is_flat_enough_for_auto_to_stand_down():
    """A sky: energy everywhere, but no subject anywhere."""
    gradient = np.tile(
        np.linspace(0, 255, 400, dtype=np.uint8)[None, :, None], (200, 1, 3)
    )
    result = smart_crop_ai.crop(Image.fromarray(gradient), ratio=1.0)

    assert result.strategy_used == "center"
    assert result.confidence == 0.0
    assert any("nearly flat" in note for note in result.notes)


def test_a_flat_image_survives_being_a_large_one():
    """The noise floor has to hold after downscaling, not just at small sizes."""
    big = Image.fromarray(np.full((1400, 2000, 3), 77, dtype=np.uint8))
    result = smart_crop_ai.crop(big, ratio=1.0)

    assert result.strategy_used == "center"
    assert result.confidence == 0.0


def test_a_crop_too_large_to_move_blames_the_crop_not_the_image():
    """A real subject plus a window that covers almost everything.

    Every placement of a 711 wide window in an 800 wide image contains the whole
    subject, so the search has nothing to choose between. The note must say that
    rather than claim the image has no detail: telling someone their obvious
    subject produced a "flat energy map" is how a tool loses their trust.
    """
    result = smart_crop_ai.crop(Image.fromarray(subject_array()), ratio=(16, 9))

    assert result.strategy_used == "center"
    assert result.confidence == 0.0
    note = " ".join(result.notes)
    assert "This image has detail" in note
    assert "the crop covers" in note
    assert "smaller crop" in note
    assert "flat" not in note
    # ...and the energy map really was not flat, which is the whole point.
    assert result.scores["energy_peak"] > 0.5


def test_a_blank_image_blames_the_image_not_the_crop(flat_image):
    note = " ".join(smart_crop_ai.crop(flat_image, ratio=1.0).notes)

    assert "no detail anywhere" in note
    assert "flat" in note
    assert "smaller crop" not in note


def test_the_two_fallbacks_are_told_apart_by_the_energy_peak(flat_image):
    """Same outcome, different reason; the reason must be visible in scores."""
    empty = smart_crop_ai.crop(flat_image, ratio=1.0)
    crowded = smart_crop_ai.crop(Image.fromarray(subject_array()), ratio=(16, 9))

    assert empty.strategy_used == crowded.strategy_used == "center"
    assert empty.scores["energy_peak"] < 0.03
    assert crowded.scores["energy_peak"] > 0.5
    assert empty.notes != crowded.notes


# --------------------------------------------------- ratio together with a size


def test_ratio_with_both_width_and_height_raises_a_clear_error(subject_image):
    with pytest.raises(ValueError) as caught:
        smart_crop_ai.crop(subject_image, 100, 100, ratio=(16, 9))

    message = str(caught.value)
    assert "not both" in message
    assert "width=100" in message and "height=100" in message
    assert "ratio" in message


@pytest.mark.parametrize("kwargs", [{"width": 100}, {"height": 100}])
def test_ratio_with_either_side_alone_also_raises(subject_image, kwargs):
    with pytest.raises(ValueError, match="not both"):
        smart_crop_ai.crop(subject_image, ratio=1.0, **kwargs)


def test_asking_for_nothing_at_all_raises(subject_image):
    with pytest.raises(ValueError, match="nothing to crop to"):
        smart_crop_ai.crop(subject_image)


# -------------------------------------------------------------- image modes


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "P", "CMYK", "I;16", "1"])
def test_every_pillow_mode_works(mode):
    """Greyscale, alpha, palette and the exotic ones all produce a real crop."""
    converted = Image.fromarray(subject_array()).convert(mode)
    result = smart_crop_ai.crop(converted, ratio=1.0)

    assert result.size == (400, 400)
    assert result.image.size == (400, 400)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_greyscale_rgba_and_palette_still_find_the_subject(mode):
    converted = Image.fromarray(subject_array()).convert(mode)
    result = smart_crop_ai.crop(converted, ratio=1.0)

    assert box_contains_subject(result.box)
    assert result.confidence > 0.0


def test_a_palette_image_with_transparency_keeps_its_alpha():
    array = np.zeros((120, 240, 4), dtype=np.uint8)
    array[..., :3] = 200
    array[..., 3] = 255
    array[30:90, 20:80, 3] = 0
    palette = Image.fromarray(array, "RGBA").convert("P", palette=Image.ADAPTIVE)
    palette.info["transparency"] = 0

    result = smart_crop_ai.crop(palette, ratio=1.0)
    assert result.image.size == (120, 120)


def test_fully_transparent_detail_does_not_count_as_a_subject():
    """Two identical patches; only the opaque one may win the crop."""
    canvas = np.full((200, 600, 4), 255, dtype=np.uint8)
    canvas[..., :3] = 200
    patch = np.random.default_rng(5).integers(0, 255, (100, 100, 3)).astype(np.uint8)
    canvas[50:150, 30:130, :3] = patch          # left, invisible
    canvas[50:150, 30:130, 3] = 0
    canvas[50:150, 470:570, :3] = patch         # right, visible

    result = smart_crop_ai.crop(Image.fromarray(canvas, "RGBA"), 200, 200)

    assert result.box[0] <= 470 and result.box[2] >= 570
    assert any("alpha" in note for note in result.notes)


# ------------------------------------------------------------ EXIF orientation


def test_exif_orientation_is_applied_before_anything_is_measured(exif_rotated):
    """A landscape JPEG tagged 'display portrait' must be measured upright."""
    result = smart_crop_ai.crop(exif_rotated, ratio=1.0)

    # The stored pixels are 800x400; the tag says rotate, so the source is 400x800.
    assert result.source_size == (400, 800)
    assert result.image.size == (400, 400)
    assert any("EXIF orientation was applied" in note for note in result.notes)


def test_the_box_refers_to_the_upright_image(exif_rotated):
    result = smart_crop_ai.crop(exif_rotated, ratio=1.0)
    left, top, right, bottom = result.box

    assert 0 <= left < right <= 400
    assert 0 <= top < bottom <= 800


def test_an_untagged_image_is_left_alone(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.fromarray(subject_array()).save(path, quality=95)

    result = smart_crop_ai.crop(str(path), ratio=1.0)

    assert result.source_size == SUBJECT_SIZE
    assert not any("EXIF" in note for note in result.notes)


# ---------------------------------------------------- the box is always in bounds


def test_the_box_is_inside_the_image_across_many_random_sizes():
    """300 random source and target sizes; not one box may leave the image."""
    rng = random.Random(20260922)

    for index in range(300):
        width = rng.randint(3, 180)
        height = rng.randint(3, 180)
        array = (
            np.random.default_rng(index)
            .integers(0, 255, (height, width, 3))
            .astype(np.uint8)
        )
        image = Image.fromarray(array)
        target_w = rng.randint(1, width + 25)
        target_h = rng.randint(1, height + 25)
        strategy = rng.choice(list(smart_crop_ai.STRATEGIES))

        result = smart_crop_ai.crop(image, target_w, target_h, strategy=strategy)
        left, top, right, bottom = result.box

        assert 0 <= left < right <= width, (index, width, height, result.box)
        assert 0 <= top < bottom <= height, (index, width, height, result.box)
        assert result.image.size == (right - left, bottom - top)
        assert 0.0 <= result.confidence <= 1.0


def test_the_box_is_inside_the_image_for_random_ratios():
    rng = random.Random(7)

    for index in range(120):
        width = rng.randint(4, 150)
        height = rng.randint(4, 150)
        array = (
            np.random.default_rng(1000 + index)
            .integers(0, 255, (height, width, 3))
            .astype(np.uint8)
        )
        ratio = rng.uniform(0.15, 6.0)

        result = smart_crop_ai.crop(Image.fromarray(array), ratio=ratio)
        left, top, right, bottom = result.box

        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height


@pytest.mark.parametrize("size", [(1, 1), (1, 5), (5, 1), (2, 2), (3, 7)])
def test_tiny_images_do_not_fall_over(size):
    array = (
        np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3)).astype(np.uint8)
    )

    result = smart_crop_ai.crop(Image.fromarray(array), ratio=1.0)
    left, top, right, bottom = result.box

    assert 0 <= left < right <= size[0]
    assert 0 <= top < bottom <= size[1]


# ------------------------------------------------ the caller image is untouched


def test_the_caller_image_is_never_modified(subject_image):
    before = np.asarray(subject_image).copy()
    mode_before, size_before = subject_image.mode, subject_image.size

    for strategy in smart_crop_ai.STRATEGIES:
        smart_crop_ai.crop(subject_image, ratio=1.0, strategy=strategy)
    smart_crop_ai.thumbnail(subject_image, 50)

    assert np.array_equal(before, np.asarray(subject_image))
    assert (subject_image.mode, subject_image.size) == (mode_before, size_before)


def test_the_returned_image_is_not_a_view_of_the_source(subject_image):
    """Writing on the crop must not reach back into the original image."""
    before = np.asarray(subject_image).copy()

    result = smart_crop_ai.crop(subject_image, 100, 100)
    result.image.paste(Image.new("RGB", (100, 100), (255, 0, 0)), (0, 0))

    assert np.array_equal(before, np.asarray(subject_image))


def test_a_numpy_input_array_is_not_modified():
    array = subject_array()
    before = array.copy()

    smart_crop_ai.crop(array, ratio=1.0)

    assert np.array_equal(before, array)


# ----------------------------------------------------------------- determinism


def test_the_same_image_always_gives_the_same_box(subject_image):
    runs = [smart_crop_ai.crop(subject_image, ratio=1.0) for _ in range(5)]

    assert len({result.box for result in runs}) == 1
    assert len({round(result.confidence, 12) for result in runs}) == 1


@pytest.mark.parametrize("strategy", list(smart_crop_ai.STRATEGIES))
def test_every_strategy_is_deterministic(subject_image, strategy):
    first = smart_crop_ai.crop(subject_image, 300, 250, strategy=strategy)
    second = smart_crop_ai.crop(subject_image, 300, 250, strategy=strategy)

    assert first.box == second.box
    assert first.to_dict() == second.to_dict()


def test_a_path_and_an_image_of_the_same_pixels_agree(tmp_path, subject_image):
    path = tmp_path / "same.png"
    subject_image.save(path)

    assert (
        smart_crop_ai.crop(str(path), ratio=1.0).box
        == smart_crop_ai.crop(subject_image, ratio=1.0).box
    )


# ------------------------------------------------------------- bad arguments


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"ratio": 1.0, "strategy": "nope"}, "unknown strategy"),
        ({"ratio": 1.0, "padding": 0.9}, "padding must be"),
        ({"ratio": 1.0, "padding": -0.1}, "padding must be"),
        ({"ratio": (1, 0)}, "ratio must be positive"),
        ({"ratio": (0, 1)}, "ratio must be positive"),
        ({"ratio": "x:y"}, "ratio string"),
        ({"ratio": (1, 2, 3)}, "ratio pair"),
        ({"ratio": float("nan")}, "ratio must be finite"),
        ({"width": 0, "height": 10}, "at least 1 pixel"),
        ({"width": 10, "height": -5}, "at least 1 pixel"),
    ],
)
def test_bad_arguments_raise_readable_value_errors(subject_image, kwargs, message):
    with pytest.raises(ValueError, match=message):
        smart_crop_ai.crop(subject_image, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ratio": 1.0, "strategy": 5},
        {"ratio": object()},
        {"width": "wide", "height": 10},
    ],
)
def test_wrong_types_raise_type_errors(subject_image, kwargs):
    with pytest.raises(TypeError):
        smart_crop_ai.crop(subject_image, **kwargs)


def test_an_unreadable_input_type_is_named_in_the_error():
    with pytest.raises(TypeError, match="not int"):
        smart_crop_ai.crop(42, ratio=1.0)


@pytest.mark.parametrize(
    "array", [np.zeros((4, 4, 2), np.uint8), np.zeros((2, 2, 2, 2), np.uint8)]
)
def test_an_array_of_the_wrong_shape_is_rejected(array):
    with pytest.raises(ValueError):
        smart_crop_ai.crop(array, ratio=1.0)


def test_an_empty_array_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        smart_crop_ai.crop(np.zeros((0, 5, 3), np.uint8), ratio=1.0)


@pytest.mark.parametrize("size", [0, -4, (10, 0), (1, 2, 3)])
def test_a_bad_thumbnail_size_is_rejected(subject_image, size):
    with pytest.raises((ValueError, TypeError)):
        smart_crop_ai.thumbnail(subject_image, size)
