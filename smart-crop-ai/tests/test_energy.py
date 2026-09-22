"""The energy maps and the image plumbing underneath the decision."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import smart_crop_ai
from smart_crop_ai import ENERGY_BUILDERS, edge_map, entropy_map, open_image, saliency_map
from smart_crop_ai._energy import (
    NOISE_FLOOR,
    box_mean,
    box_std,
    denoise,
    integral_image,
    luma,
    sobel_magnitude,
)
from smart_crop_ai._images import (
    apply_exif_orientation,
    flatten_for_format,
    normalize_mode,
    working_arrays,
)
from conftest import subject_array


def rgb_floats(array: np.ndarray) -> np.ndarray:
    """A uint8 image array as the float RGB the energy maps expect."""
    return array.astype(np.float64) / 255.0


@pytest.fixture
def detailed() -> np.ndarray:
    return rgb_floats(subject_array())


@pytest.fixture
def blank() -> np.ndarray:
    return np.full((60, 90, 3), 0.5, dtype=np.float64)


# ----------------------------------------------------------------- the maps


@pytest.mark.parametrize("builder", list(ENERGY_BUILDERS.values()), ids=list(ENERGY_BUILDERS))
def test_every_map_is_the_right_shape_and_range(builder, detailed):
    energy = builder(detailed)

    assert energy.shape == detailed.shape[:2]
    assert energy.dtype == np.float64
    assert energy.min() >= 0.0 and energy.max() <= 1.0


@pytest.mark.parametrize("builder", list(ENERGY_BUILDERS.values()), ids=list(ENERGY_BUILDERS))
def test_every_map_is_flat_on_a_blank_image(builder, blank):
    """The scales are absolute, so a blank wall must not be amplified."""
    energy = builder(blank)
    assert energy.max() <= NOISE_FLOOR * 10


@pytest.mark.parametrize("builder", list(ENERGY_BUILDERS.values()), ids=list(ENERGY_BUILDERS))
def test_every_map_scores_the_subject_above_the_wall(builder, detailed):
    """Energy inside the planted patch must beat energy out on the plain wall."""
    energy = builder(detailed)
    inside = energy[150:270, 570:690].mean()
    outside = energy[10:110, 10:110].mean()

    assert inside > outside


@pytest.mark.parametrize("builder", list(ENERGY_BUILDERS.values()), ids=list(ENERGY_BUILDERS))
def test_every_map_is_deterministic(builder, detailed):
    assert np.array_equal(builder(detailed), builder(detailed))


def test_the_named_maps_are_the_ones_in_the_registry():
    assert ENERGY_BUILDERS["saliency"] is saliency_map
    assert ENERGY_BUILDERS["entropy"] is entropy_map
    assert ENERGY_BUILDERS["edges"] is edge_map
    assert set(ENERGY_BUILDERS) == set(smart_crop_ai.STRATEGIES) - {"auto", "center"}


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (2, 3), (5, 5)])
@pytest.mark.parametrize("builder", list(ENERGY_BUILDERS.values()), ids=list(ENERGY_BUILDERS))
def test_maps_survive_tiny_images(builder, shape):
    tiny = np.random.default_rng(0).random((shape[0], shape[1], 3))
    energy = builder(tiny)

    assert energy.shape == shape
    assert np.isfinite(energy).all()


def test_an_edge_scores_higher_than_the_flat_side_of_it():
    array = np.zeros((40, 40, 3), dtype=np.float64)
    array[:, 20:] = 1.0

    energy = saliency_map(array)
    assert energy[:, 18:22].mean() > energy[:, :10].mean()


# ------------------------------------------------------------- the numerics


def test_denoise_flattens_resampling_residue_but_keeps_real_detail():
    values = np.array([[0.0, NOISE_FLOOR / 2.0], [NOISE_FLOOR * 100, 1.0]])
    cleaned = denoise(values)

    assert cleaned[0, 1] == 0.0
    assert cleaned[1, 0] == pytest.approx(NOISE_FLOOR * 100)
    assert cleaned[1, 1] == 1.0


def test_luma_weights_sum_to_one():
    white = np.ones((3, 3, 3), dtype=np.float64)
    assert luma(white) == pytest.approx(np.ones((3, 3)))


def test_box_mean_of_a_constant_is_that_constant():
    assert box_mean(np.full((10, 10), 4.0), 2) == pytest.approx(np.full((10, 10), 4.0))


def test_box_mean_with_no_radius_is_the_input():
    array = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(box_mean(array, 0), array)


def test_box_std_of_a_constant_is_zero():
    assert box_std(np.full((8, 8), 2.0), 1) == pytest.approx(np.zeros((8, 8)), abs=1e-12)


def test_box_std_is_never_negative_from_float_error():
    array = np.random.default_rng(0).random((30, 30)) * 1e-8
    assert (box_std(array, 3) >= 0.0).all()


def test_integral_image_matches_a_slow_cumulative_sum():
    array = np.random.default_rng(1).random((6, 7))
    table = integral_image(array)

    assert table.shape == (7, 8)
    assert table[0].sum() == 0.0 and table[:, 0].sum() == 0.0
    assert table[6, 7] == pytest.approx(array.sum())
    assert table[3, 4] == pytest.approx(array[:3, :4].sum())


def test_sobel_finds_a_vertical_edge():
    array = np.zeros((20, 20))
    array[:, 10:] = 1.0
    magnitude = sobel_magnitude(array)

    assert magnitude[:, 9:11].max() > 1.0
    assert magnitude[:, :5].max() == pytest.approx(0.0)


# -------------------------------------------------------- opening an image


def test_open_image_accepts_a_pil_image(subject_image):
    assert open_image(subject_image) is subject_image


def test_open_image_accepts_a_path(tmp_path, subject_image):
    path = tmp_path / "x.png"
    subject_image.save(path)

    assert open_image(str(path)).size == subject_image.size
    assert open_image(path).size == subject_image.size          # os.PathLike


@pytest.mark.parametrize(
    "array, mode",
    [
        (np.zeros((4, 5), np.uint8), "L"),
        (np.zeros((4, 5, 1), np.uint8), "L"),
        (np.zeros((4, 5, 3), np.uint8), "RGB"),
        (np.zeros((4, 5, 4), np.uint8), "RGBA"),
    ],
)
def test_open_image_accepts_arrays_of_every_channel_count(array, mode):
    image = open_image(array)
    assert image.mode == mode
    assert image.size == (5, 4)


def test_open_image_reads_float_arrays_in_zero_to_one():
    floats = np.linspace(0.0, 1.0, 12).reshape(2, 2, 3)
    image = open_image(floats)

    assert image.mode == "RGB"
    assert np.asarray(image).max() == 255


def test_open_image_names_the_type_it_cannot_read():
    with pytest.raises(TypeError, match="not dict"):
        open_image({})


def test_open_image_says_which_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.png"):
        open_image(str(tmp_path / "missing.png"))


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "I;16", "1", "RGB", "RGBA"])
def test_normalize_mode_always_lands_on_rgb_or_rgba(mode):
    converted = Image.fromarray(subject_array()).convert(mode)
    normalized = normalize_mode(converted)

    assert normalized.mode in ("RGB", "RGBA")
    assert normalized.size == converted.size


def test_working_arrays_downscales_and_reports_the_scale(subject_image):
    rgb, alpha, scale = working_arrays(subject_image, 100)

    assert max(rgb.shape[:2]) == 100
    assert rgb.shape[2] == 3
    assert alpha is None
    assert scale == pytest.approx(100.0 / 800.0)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_working_arrays_leaves_a_small_image_alone(flat_image):
    rgb, _, scale = working_arrays(flat_image, 4000)

    assert rgb.shape[:2] == (200, 400)
    assert scale == 1.0


def test_working_arrays_splits_out_alpha():
    array = np.zeros((20, 30, 4), dtype=np.uint8)
    array[..., 3] = 128
    rgb, alpha, _ = working_arrays(Image.fromarray(array, "RGBA"), 0)

    assert rgb.shape == (20, 30, 3)
    assert alpha is not None
    assert alpha == pytest.approx(np.full((20, 30), 128.0 / 255.0))


def test_apply_exif_orientation_rotates_only_when_told(tmp_path, subject_image):
    plain, rotated_flag = apply_exif_orientation(subject_image)
    assert plain is subject_image and not rotated_flag

    exif = subject_image.getexif()
    exif[0x0112] = 6
    path = tmp_path / "r.jpg"
    subject_image.save(path, exif=exif, quality=95)

    turned, flag = apply_exif_orientation(Image.open(path))
    assert flag
    assert turned.size == (400, 800)


@pytest.mark.parametrize("suffix, expected", [(".jpg", True), (".bmp", True), (".png", False), (".webp", False)])
def test_flatten_for_format_only_flattens_where_it_must(suffix, expected):
    rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    flattened, did = flatten_for_format(rgba, "out" + suffix)

    assert did is expected
    assert (flattened.mode == "RGB") is expected


def test_flatten_for_format_leaves_an_rgb_image_alone():
    rgb = Image.new("RGB", (8, 8))
    same, did = flatten_for_format(rgb, "out.jpg")

    assert same is rgb and not did


def test_image_suffixes_are_lowercase_with_a_dot():
    for suffix in smart_crop_ai.IMAGE_SUFFIXES:
        assert suffix.startswith(".") and suffix == suffix.lower()
