"""The hand-built features: the properties the rest of the package relies on."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import different_scene, normal_scene
from vision_anomaly import FeatureConfig, describe_features
from vision_anomaly._features import (
    GROUP_DESCRIPTIONS,
    GROUP_NAMES,
    build_space,
    extract,
    extract_many,
)
from vision_anomaly._loading import load_image


@pytest.fixture(scope="module")
def space():
    return build_space(FeatureConfig())


def test_the_layout_is_the_documented_216_numbers(space):
    assert len(space) == 216
    assert len(space.names) == len(space.groups) == len(space.cell_of)
    assert len(set(space.names)) == len(space.names)     # no duplicate names
    assert set(space.groups) == set(GROUP_NAMES)
    assert space.cells == 16


def test_every_feature_belongs_to_a_family_and_most_to_a_cell(space):
    counted = sum(len(space.group_index(group)) for group in GROUP_NAMES)
    assert counted == len(space)

    owned = sum(len(space.cell_index(cell)) for cell in range(space.cells))
    unowned = sum(1 for cell in space.cell_of if cell < 0)
    assert owned + unowned == len(space)
    assert unowned == 3 * 16 + 8            # colour histograms and the frame one


def test_every_feature_is_a_zero_to_one_quantity(space):
    """The single absolute noise floor in the profile depends on this."""
    for factory in (normal_scene, different_scene):
        vector = extract(load_image(factory(0)), space)
        assert float(vector.min()) >= 0.0
        assert float(vector.max()) <= 1.0
        assert np.isfinite(vector).all()
        assert vector.dtype == np.float32


def test_extraction_is_deterministic(space):
    image = load_image(normal_scene(0))

    assert np.array_equal(extract(image, space), extract(image, space))


def test_a_cell_box_tiles_the_analysis_grid(space):
    covered = np.zeros((256, 256), dtype=int)
    for cell in range(space.cells):
        x, y, width, height = space.cell_box(cell)
        covered[y : y + height, x : x + width] += 1

    assert covered.min() == covered.max() == 1


def test_histograms_move_smoothly_rather_than_in_jumps(space):
    """The reason bins are interpolated instead of hard.

    Brightening a flat frame one grey level at a time must move the colour
    histogram a little each time. With hard bins one of these steps moves it by
    a whole bin, which is what made ordinary lighting drift look like an anomaly.
    """
    index = space.group_index("colour")[:48]      # the three channel histograms
    steps = [
        extract(load_image(np.full((64, 64, 3), level, dtype=np.uint8)), space)[index]
        for level in range(120, 136)
    ]
    moves = [float(np.abs(later - earlier).sum()) for earlier, later in zip(steps, steps[1:])]

    assert max(moves) < 4 * (sum(moves) / len(moves))
    assert min(moves) > 0.0                        # it does move


def test_orientation_bins_wrap_round(space):
    """179 degrees and 1 degree are nearly the same edge, not opposite ones."""
    index = space.group_index("orientation")
    rows, cols = np.mgrid[0:256, 0:256]

    def striped(angle_degrees):
        radians = np.deg2rad(angle_degrees)
        wave = np.sin((cols * np.cos(radians) + rows * np.sin(radians)) / 3.0)
        return load_image(((wave * 0.5 + 0.5) * 255).astype(np.uint8))

    near_zero = extract(striped(1.0), space)[index]
    near_180 = extract(striped(179.0), space)[index]
    right_angle = extract(striped(90.0), space)[index]

    assert np.abs(near_zero - near_180).sum() < np.abs(near_zero - right_angle).sum()


def test_a_flat_cell_reports_no_preferred_direction_rather_than_none(space):
    """A blank patch has no direction; zeros would read as one that vanished."""
    index = space.group_index("orientation")
    flat = extract(load_image(np.full((128, 128, 3), 90, dtype=np.uint8)), space)[index]

    per_cell = flat[:64].reshape(16, 4)
    assert np.allclose(per_cell, 0.25)
    assert np.allclose(flat[64:], 1.0 / 8.0)


def test_texture_separates_rough_from_smooth_at_the_same_brightness(space):
    """The feature that catches a torn surface when the colour has not changed."""
    index = space.group_index("texture")
    rng = np.random.default_rng(3)
    smooth = np.full((256, 256, 3), 128, dtype=np.uint8)
    rough = np.clip(rng.normal(128, 40, (256, 256, 3)), 0, 255).astype(np.uint8)

    smooth_texture = extract(load_image(smooth), space)[index]
    rough_texture = extract(load_image(rough), space)[index]

    assert float(rough_texture.mean()) > 10 * float(smooth_texture.mean() + 1e-6)


def test_brightness_tracks_the_picture(space):
    index = space.group_index("brightness")
    dark = extract(load_image(np.full((64, 64, 3), 20, dtype=np.uint8)), space)[index]
    bright = extract(load_image(np.full((64, 64, 3), 220, dtype=np.uint8)), space)[index]

    assert float(dark.mean()) == pytest.approx(20 / 255.0, abs=0.01)
    assert float(bright.mean()) == pytest.approx(220 / 255.0, abs=0.01)


def test_the_same_picture_at_two_sizes_gives_nearly_the_same_features(space):
    """A thumbnail and the original must not be two different pictures."""
    big = load_image(normal_scene(0, size=(1024, 768)))
    small = load_image(normal_scene(0, size=(256, 192)))

    difference = np.abs(extract(big, space) - extract(small, space))

    assert float(difference.max()) < 0.06


def test_extract_refuses_an_image_from_the_wrong_grid(space):
    wrong = load_image(normal_scene(0), size=128)

    with pytest.raises(ValueError) as raised:
        extract(wrong, space)
    assert "128x128" in str(raised.value)
    assert "256" in str(raised.value)


def test_extract_many_stacks_rows(space):
    images = [load_image(normal_scene(i)) for i in range(3)]

    matrix = extract_many(images, space)

    assert matrix.shape == (3, len(space))
    assert extract_many([], space).shape == (0, len(space))


def test_a_finer_grid_produces_more_features():
    coarse = build_space(FeatureConfig(grid=2))
    fine = build_space(FeatureConfig(grid=8))

    assert len(fine) > len(coarse)
    assert fine.cells == 64 and coarse.cells == 4


def test_describe_features_prints_the_whole_table():
    text = describe_features()

    assert text.isascii()
    assert "216 numbers per image" in text
    for group in GROUP_NAMES:
        assert group in text
        assert GROUP_DESCRIPTIONS[group] in text


def test_an_unusable_layout_is_refused():
    with pytest.raises(ValueError) as raised:
        FeatureConfig(colour_bins=1).validate()
    assert "colour_bins must be at least 2" in str(raised.value)

    with pytest.raises(ValueError) as raised:
        FeatureConfig(edge_threshold=0.0).validate()
    assert "edge_threshold" in str(raised.value)
