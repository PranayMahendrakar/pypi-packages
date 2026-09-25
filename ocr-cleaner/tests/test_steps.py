"""The individual operations, tested on their own rather than through the pipeline.

These are cheap and exact: no page goes through six steps here, so a failure
points at one function instead of at "cleaning got worse".
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ocr_cleaner import _analysis, _images, _steps

from conftest import PAPER, add_grain, add_specks, blank_page, text_page


# --- thresholds ------------------------------------------------------------ #

def test_otsu_finds_the_valley_between_two_peaks():
    plane = np.concatenate([
        np.full(5000, 40, dtype=np.uint8), np.full(5000, 220, dtype=np.uint8)
    ]).reshape(100, 100)
    level = _steps.otsu_level(plane)
    assert 40 <= level < 220


def test_otsu_on_a_flat_plane_does_not_crash():
    assert 0 <= _steps.otsu_level(np.full((20, 20), 128, dtype=np.uint8)) <= 255


def test_global_threshold_splits_at_the_level():
    plane = np.array([[0, 100, 200, 255]], dtype=np.uint8)
    assert list(_steps.global_threshold(plane, 100)[0]) == [0, 0, 255, 255]


def test_adaptive_threshold_finds_ink_under_a_lighting_ramp():
    plane = np.tile(np.linspace(250, 90, 400), (200, 1))
    plane[90:110, 50:350] -= 45.0            # a stroke, always 45 below its paper
    plane = np.clip(plane, 0, 255).astype(np.uint8)
    binary = _steps.adaptive_threshold(plane, 61, 12.0)
    assert (binary[95:105, 100:300] == 0).mean() > 0.95
    assert (binary[10:40, :] == 0).mean() < 0.02


def test_local_mean_tracks_a_ramp():
    plane = np.tile(np.linspace(0, 255, 200), (60, 1)).astype(np.uint8)
    means = _steps.local_mean(plane, 31)
    assert means.shape == plane.shape
    assert means[30, 10] < means[30, 100] < means[30, 190]


def test_is_binary():
    assert _steps.is_binary(np.array([0, 255, 0], dtype=np.uint8))
    assert not _steps.is_binary(np.array([0, 128, 255], dtype=np.uint8))


@pytest.mark.parametrize(
    "height,expected", [(None, 31), (4.0, 15), (11.0, 33), (30.0, 91), (900.0, 151)]
)
def test_the_adaptive_window_is_odd_and_bounded(height, expected):
    window = _steps.adaptive_window_for(height)
    assert window == expected
    assert window % 2 == 1


# --- denoise --------------------------------------------------------------- #

@pytest.mark.parametrize("height,size", [(None, 3), (8.0, 3), (39.0, 3), (40.0, 5), (90.0, 5)])
def test_the_median_window_follows_the_text_height(height, size):
    assert _steps.median_window_for(height) == size


def test_noise_is_measured_as_nothing_on_clean_paper():
    page = text_page()
    levels, specks = _steps.estimate_noise(page, float(PAPER))
    assert levels < _steps.NOISE_FLOOR
    assert not _steps.is_noisy(levels, specks)


def test_grain_is_measured_as_noise():
    page = add_grain(text_page(), 8.0, seed=31)
    levels, specks = _steps.estimate_noise(page, float(PAPER))
    assert levels >= _steps.NOISE_FLOOR
    assert _steps.is_noisy(levels, specks)


def test_specks_are_noticed_even_when_the_grain_is_not():
    page = add_specks(text_page(), 0.01, seed=32)
    levels, specks = _steps.estimate_noise(page, float(PAPER))
    assert specks >= _steps.SPECK_SHARE_FLOOR
    assert _steps.is_noisy(levels, specks)


def test_the_median_filter_removes_single_pixel_specks():
    page = text_page()
    speckled = add_specks(page, 0.01, seed=33)
    filtered = _images.to_array(
        _steps.median_filter(_images.from_array(speckled), 3)
    )
    before = int((speckled[5:60, :] < 100).sum())
    after = int((filtered[5:60, :] < 100).sum())
    assert before > 100
    assert after < before / 10


# --- border ---------------------------------------------------------------- #

def test_find_content_box_finds_a_black_band():
    page = text_page()
    page[:, :30] = 5
    box = _steps.find_content_box(page, float(PAPER))
    assert box is not None
    assert box[0] >= 30
    assert box[2] == page.shape[1]


def test_find_content_box_returns_none_on_a_clean_page():
    assert _steps.find_content_box(text_page(), float(PAPER)) is None


def test_find_content_box_refuses_to_eat_the_page():
    """A page that is mostly dark is not a page with a huge border."""
    mostly_dark = np.full((300, 300), 10, dtype=np.uint8)
    mostly_dark[140:160, 140:160] = 250
    assert _steps.find_content_box(mostly_dark, 250.0) is None


def test_a_row_of_text_is_never_mistaken_for_a_border():
    """Text at the very top edge: heavy, but not solid, so it stays."""
    page = np.full((200, 400), PAPER, dtype=np.uint8)
    for column in range(0, 400, 12):
        page[0:14, column: column + 8] = 30    # a dense line, 2/3 covered
    assert _steps.find_content_box(page, float(PAPER)) is None


def test_the_gap_lets_the_scan_reach_a_band_pushed_inward():
    darkness = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert _steps._trim_one_side(darkness, 8, gap=0) == 0
    assert _steps._trim_one_side(darkness, 8, gap=4) == 5 + _steps.BORDER_FEATHER


def test_widen_for_rotation_only_grows_the_sides_that_had_a_border():
    box = (30, 0, 400, 500)                    # a band on the left only
    widened = _steps.widen_for_rotation(box, (400, 500), (440, 540), 4.0)
    assert widened is not None
    assert widened[0] > 30                     # the left grew
    assert widened[1] == 0                     # the top did not
    assert widened[3] == 540


def test_widen_for_rotation_refuses_an_absurd_result():
    assert _steps.widen_for_rotation((100, 100, 120, 120), (400, 500), (440, 540), 14.0) is None


def test_union_boxes_takes_the_tighter_side():
    assert _steps.union_boxes((10, 0, 90, 100), (0, 5, 100, 95), (100, 100)) == (10, 5, 90, 95)
    assert _steps.union_boxes(None, None, (100, 100)) is None
    assert _steps.union_boxes((0, 0, 100, 100), None, (100, 100)) is None


def test_describe_border_names_each_side():
    text = _steps.describe_border((10, 4, 90, 96), (100, 100))
    assert "10 px off the left" in text
    assert "4 px off the top" in text
    assert "leaving 80 x 92" in text


# --- deskew and upscale ---------------------------------------------------- #

def test_deskew_fills_with_the_paper_level_it_is_given():
    image = Image.new("L", (100, 100), 200)
    turned, detail = _steps.deskew(image, 5.0, 200.0)
    assert turned.size[0] > 100
    assert np.array(turned)[0, 0] == 200
    assert "-5.00 degrees" in detail


def test_upscale_keeps_a_binary_page_binary():
    plane = np.where(np.indices((40, 40))[0] % 4 == 0, 0, 255).astype(np.uint8)
    bigger, detail = _steps.upscale(_images.from_array(plane), 2.0, binary=True)
    assert bigger.size == (80, 80)
    assert set(np.unique(np.array(bigger))) <= {0, 255}
    assert "black and white" in detail


def test_upscale_leaves_a_greyscale_page_grey():
    plane = np.tile(np.linspace(0, 255, 40), (40, 1)).astype(np.uint8)
    bigger, detail = _steps.upscale(_images.from_array(plane), 1.5, binary=False)
    assert bigger.size == (60, 60)
    assert "black and white" not in detail


# --- the measurements ------------------------------------------------------ #

def test_the_ink_plane_ignores_a_lighting_ramp():
    flat = text_page()
    lit = np.clip(flat.astype(np.float64) + np.linspace(0, -80, flat.shape[1])[None, :], 0, 255)
    ink_flat, depth_flat = _analysis.ink_plane(flat)
    ink_lit, depth_lit = _analysis.ink_plane(lit.astype(np.uint8))
    assert abs(depth_flat - depth_lit) < 0.25 * depth_flat
    assert abs(float(ink_flat.mean()) - float(ink_lit.mean())) < 0.05


def test_the_ink_plane_reports_no_depth_on_blank_paper():
    _ink, depth = _analysis.ink_plane(blank_page(sigma=3.0))
    assert depth < _analysis.MIN_INK_DEPTH


def test_the_comb_score_of_a_flat_profile_is_zero():
    assert _analysis.comb_score(np.ones(50)) == 0.0
    assert _analysis.comb_score(np.array([1.0])) == 0.0


def test_line_geometry_measures_the_bars_it_is_given():
    profile = np.zeros(300)
    for start in range(10, 290, 20):
        profile[start: start + 8] = 5.0
    geometry = _analysis.line_geometry(profile)
    assert geometry.line_count == 14
    assert geometry.pitch == pytest.approx(20.0, abs=1.0)
    assert geometry.text_height == pytest.approx(8.0, abs=2.0)


def test_line_geometry_measures_a_line_from_its_ascender_to_its_descender():
    """A line of text is three bands, not one, and all three are the line.

    A tall x-height core, a thin ascender band above it and a thinner descender
    band below - and because only a few glyphs in a line carry either, those two
    sum to about a tenth of what a core row does. A cut taken from the whole
    profile swing lands between them and the core, so every line comes apart
    into three runs and the median of the pieces reports a sliver. Growing each
    line outward from its own core is what keeps it whole.
    """
    profile = np.zeros(400)
    for start in range(20, 380, 40):
        profile[start - 6: start] = 0.9
        profile[start: start + 12] = 9.0
        profile[start + 12: start + 17] = 0.8
    geometry = _analysis.line_geometry(profile)
    assert geometry.line_count == 9
    assert geometry.pitch == pytest.approx(40.0, abs=1.0)
    assert geometry.text_height == pytest.approx(23.0, abs=1.0)


def test_line_geometry_ignores_a_speck_that_belongs_to_no_line():
    """Ink in the gap is not a line, however far the cut is lowered for tails."""
    profile = np.zeros(400)
    for start in range(20, 380, 40):
        profile[start: start + 12] = 9.0
    profile[36:38] = 1.5
    geometry = _analysis.line_geometry(profile)
    assert geometry.line_count == 9
    assert geometry.text_height == pytest.approx(12.0, abs=1.0)


def test_line_geometry_does_not_swallow_the_gap_to_the_next_line():
    """A line may grow at most to the middle of the gap beside it."""
    profile = np.full(400, 2.0)
    for start in range(20, 380, 40):
        profile[start: start + 12] = 9.0
    geometry = _analysis.line_geometry(profile)
    assert geometry.line_count == 9
    assert geometry.text_height <= 40.0


def test_paper_between_is_the_level_the_gaps_sit_at():
    profile = np.full(200, 1.0)
    profile[0:3] = 0.0                       # one darker strip, not a gap
    cores = [(20, 30), (60, 70), (100, 110)]
    for start, stop in cores:
        profile[start:stop] = 8.0
    assert _analysis.paper_between(profile, cores) == pytest.approx(1.0)
    assert _analysis.paper_between(profile, []) == pytest.approx(0.0)


def test_line_geometry_on_nothing():
    assert _analysis.line_geometry(np.zeros(100)).line_count == 0
    assert _analysis.line_geometry(np.zeros(5)).text_height is None


def test_paper_level_ignores_one_blown_highlight():
    plane = np.full((100, 100), 200, dtype=np.uint8)
    plane[0, 0] = 255
    assert _images.paper_level(plane) == pytest.approx(200, abs=1)


def test_analysis_plane_reports_the_scale_it_used():
    plane = np.zeros((2000, 1000), dtype=np.uint8)
    small, scale = _images.analysis_plane(plane, 500)
    assert max(small.shape) <= 500
    assert scale == pytest.approx(0.25, abs=0.01)
    same, one = _images.analysis_plane(plane, 0)
    assert one == 1.0
    assert same.shape == plane.shape


def test_to_array_is_always_writable_and_never_shared():
    """``np.asarray`` of a Pillow image can be read-only; this must not be."""
    image = Image.new("L", (8, 8), 100)
    array = _images.to_array(image)
    assert array.flags.writeable
    array[0, 0] = 7
    assert image.getpixel((0, 0)) == 100
