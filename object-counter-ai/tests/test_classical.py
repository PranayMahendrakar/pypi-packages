"""The built-in classical counter: planted counts, clean inputs, and its limits."""
from __future__ import annotations

import numpy as np
import pytest

import object_counter_ai as oc
from synth_images import (add_noise, blank, discs_and_squares, discs_image, draw_disc,
                          draw_square, with_gradient)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 7, 18])
def test_counts_planted_dark_discs_on_light(seed, n):
    result = oc.count(discs_image(n, seed=seed))
    assert result.count == n
    assert result.method == "classical"
    assert result.by_label == {"blob": n}
    assert result.details["background"]["polarity"] == "light"


def test_counts_light_things_on_dark():
    img = discs_image(11, seed=5, background=25, value=210)
    result = oc.count(img)
    assert result.count == 11
    assert result.details["background"]["polarity"] == "dark"


@pytest.mark.parametrize("seed", [0, 4])
def test_counts_mixed_discs_and_squares(seed):
    result = oc.count(discs_and_squares(6, 5, seed=seed))
    assert result.count == 11


def test_boxes_areas_and_centroids_match_the_planted_square():
    img = blank(100, 100, 230)
    img[20:40, 50:80] = 20
    result = oc.count(img)
    assert result.count == 1
    assert result.boxes == [(50, 20, 80, 40)]
    assert result.areas == [600.0]
    assert result.centroids[0] == pytest.approx((65.0, 30.0))
    assert len(result.labels) == len(result.scores) == len(result.boxes) == 1


def test_noisy_image_still_counts_exactly_and_reports_noise():
    img = add_noise(discs_image(12, seed=7), sigma=9.0, seed=1)
    result = oc.count(img)
    assert result.count == 12
    assert result.details["noise"] > 5


def test_uneven_lighting_is_modelled_not_counted():
    img = with_gradient(discs_image(9, seed=3, background=150, value=20), -60, 80)
    result = oc.count(img)
    assert result.count == 9
    assert result.details["background"]["model"] == "surface"


# -- clean inputs must not raise false alarms ------------------------------------------

def test_plain_image_counts_zero_with_high_confidence():
    result = oc.count(blank(80, 90, 128))
    assert result.count == 0
    assert result.confidence == 1.0
    assert result.boxes == []


def test_pure_noise_counts_zero():
    img = add_noise(blank(200, 200, 128), sigma=12.0, seed=3)
    assert oc.count(img).count == 0


@pytest.mark.parametrize("low, high", [(-80, 80), (0, 120), (60, -60)])
def test_lighting_gradient_alone_counts_zero(low, high):
    img = with_gradient(blank(150, 220, 110), low, high)
    assert oc.count(img).count == 0


def test_radial_vignette_alone_counts_zero():
    yy, xx = np.mgrid[:160, :200]
    r2 = ((xx - 100) / 100.0) ** 2 + ((yy - 80) / 80.0) ** 2
    img = np.clip(210 - 70 * r2, 0, 255).astype(np.uint8)
    assert oc.count(img).count == 0
    draw_disc(img, 60, 60, 8, 10)
    draw_disc(img, 140, 100, 8, 10)
    assert oc.count(img).count == 2


def test_faint_texture_below_min_contrast_is_not_counted():
    img = blank(100, 100, 128)
    img[30:60, 30:60] = 134          # 6 levels: below the minimum contrast
    assert oc.count(img).count == 0


# -- colour ---------------------------------------------------------------------------

def test_colour_only_contrast_is_seen_in_rgb():
    img = np.zeros((100, 140, 3), dtype=np.uint8)
    img[:, :] = (40, 160, 40)                        # green belt
    for x in (30, 70, 110):
        draw_disc(img, x, 50, 10, (180, 60, 180))    # magenta parts, same brightness
    grey = img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    assert grey.max() - grey.min() < 10              # nearly invisible in greyscale
    assert oc.count(img).count == 3


# -- area filtering -------------------------------------------------------------------

def test_specks_under_min_area_are_ignored_and_reported():
    img = discs_image(5, seed=2, h=200, w=260, r=10)
    img[5, 5] = 0
    img[190:192, 250:252] = 0
    result = oc.count(img, min_area=20)
    assert result.count == 5
    assert result.details["specks"] == 2
    assert "under min_area" in result.summary()


def test_default_min_area_ignores_single_pixels():
    img = discs_image(4, seed=1)
    img[3, 3] = 0
    result = oc.count(img)
    assert result.count == 4
    assert result.details["min_area_was_default"] is True
    assert result.details["min_area"] >= 9


def test_max_area_drops_large_blobs():
    img = blank(120, 200, 230)
    draw_square(img, 40, 60, 25, 20)   # 2500 px
    draw_disc(img, 140, 30, 6, 20)
    draw_disc(img, 150, 90, 6, 20)
    result = oc.count(img, max_area=1000)
    assert result.count == 2
    assert result.details["too_large"] == 1
    assert oc.count(img).count == 3


def test_min_area_larger_than_max_area_is_rejected():
    with pytest.raises(ValueError, match="min_area"):
        oc.count(blank(), min_area=100, max_area=10)
    with pytest.raises(ValueError, match="min_area"):
        oc.count(blank(), min_area=-1)


# -- touching objects -------------------------------------------------------------------

def _pair(distance: int, r: int = 12, r2: int = None) -> np.ndarray:
    img = blank(90, 140, 230)
    draw_disc(img, 40, 45, r, 30)
    draw_disc(img, 40 + distance, 45, r if r2 is None else r2, 30)
    return img


@pytest.mark.parametrize("distance", [20, 19, 18])
def test_touching_discs_are_split_at_a_narrow_neck(distance):
    result = oc.count(_pair(distance))
    assert result.count == 2
    assert result.details["split_blobs"] == 1
    assert all(result.details["split"])
    left, right = sorted(result.boxes)
    neck = 40 + distance / 2.0
    assert abs(left[2] - neck) <= 1.5 and abs(right[0] - neck) <= 1.5   # cut at the neck


@pytest.mark.parametrize("distance", [22, 21])
def test_unequal_touching_discs_are_split(distance):
    result = oc.count(_pair(distance, r=14, r2=8))
    assert result.count == 2
    areas = sorted(result.areas)
    assert areas[0] < 0.5 * areas[1]


def test_a_chain_of_three_touching_discs_counts_three():
    img = blank(80, 160, 230)
    for x in (30, 52, 74):
        draw_disc(img, x, 40, 12, 30)
    assert oc.count(img).count == 3


@pytest.mark.parametrize("distance", [14, 10, 6])
def test_limit_heavily_overlapping_discs_count_as_one(distance):
    """Documented limit: with no narrow neck the counter sees one blob, not two."""
    result = oc.count(_pair(distance))
    assert result.count == 1
    assert result.details["split_blobs"] == 0


def test_limit_squares_sharing_a_full_edge_count_as_one():
    """Documented limit: two parts pressed edge to edge are one rectangle to a blob counter."""
    img = blank(80, 120, 230)
    img[20:50, 20:50] = 30
    img[20:50, 50:80] = 30
    assert oc.count(img).count == 1


@pytest.mark.parametrize("shape", ["bolt", "washer", "ellipse", "bracket"])
def test_single_irregular_parts_are_not_split(shape):
    img = blank(120, 180, 230)
    if shape == "bolt":
        draw_disc(img, 30, 60, 14, 30)
        img[55:66, 30:160] = 30
    elif shape == "washer":
        draw_disc(img, 90, 60, 32, 30)
        draw_disc(img, 90, 60, 18, 230)
    elif shape == "ellipse":
        yy, xx = np.ogrid[:120, :180]
        img[((xx - 90) / 70.0) ** 2 + ((yy - 60) / 22.0) ** 2 <= 1] = 30
    else:
        img[20:100, 30:44] = 30
        img[86:100, 30:150] = 30
    result = oc.count(img)
    assert result.count == 1
    assert result.details["split_blobs"] == 0


def test_blobs_cut_by_the_image_edge_are_flagged():
    img = blank(100, 100, 230)
    draw_disc(img, 0, 50, 12, 30)
    draw_disc(img, 60, 50, 12, 30)
    result = oc.count(img)
    assert result.count == 2
    assert any("image edge" in note for note in result.notes)
    assert result.confidence < 1.0


def test_crowded_picture_lowers_confidence_and_says_why():
    img = blank(100, 100, 230)
    img[5:95, 5:55] = 30          # half the picture is "object"
    result = oc.count(img)
    assert result.confidence < 1.0
    assert result.details["confidence_factors"]


def test_deterministic():
    img = add_noise(discs_image(10, seed=9), 6.0, seed=2)
    first = oc.count(img).to_dict()
    for _ in range(3):
        assert oc.count(img).to_dict() == first


def test_large_frame_is_counted_quickly():
    import time

    img = np.full((800, 1000), 200, dtype=np.uint8)
    yy, xx = np.ogrid[:800, :1000]
    mask = np.zeros((800, 1000), dtype=bool)
    for cy in range(40, 800, 80):
        for cx in range(40, 1000, 80):
            mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= 22 ** 2
    img[mask] = 50
    img = add_noise(img, 4.0, seed=0)
    start = time.perf_counter()
    result = oc.count(img)
    elapsed = time.perf_counter() - start
    assert result.count == 10 * 12
    assert elapsed < 5.0


def test_large_touching_parts_are_split_on_the_reduced_copy():
    """Blobs wider than the analysis size are examined reduced, then cut at full size."""
    img = blank(420, 700, 225)
    draw_disc(img, 220, 210, 100, 35)
    draw_disc(img, 400, 210, 100, 35)       # centres 180 apart: overlapping, with a neck
    draw_disc(img, 620, 80, 20, 35)
    result = oc.count(img)
    assert result.count == 3
    assert result.details["split_blobs"] == 1
    halves = sorted(b for b, a in zip(result.boxes, result.areas) if a > 5000)
    assert abs(halves[0][2] - 310) <= 3 and abs(halves[1][0] - 310) <= 3


def test_things_covering_about_half_the_picture_are_still_counted():
    """The median would call the parts the background; the border says otherwise."""
    img = blank(160, 560, 225)
    for left in (16, 124, 232, 340, 448):
        img[25:135, left:left + 96] = 35          # five parts with 12-pixel gaps
    covered = float((img < 128).mean())
    assert covered > 0.55
    result = oc.count(img)
    assert result.count == 5
    assert any("border" in n for n in result.notes)
    assert result.confidence < 1.0


def test_clean_background_is_not_second_guessed():
    result = oc.count(discs_image(8, seed=6))
    assert not any("border" in n for n in result.notes)
