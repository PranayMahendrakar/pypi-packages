"""The pages that are not ordinary scans, and the promises that hold for all of them."""
from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

import ocr_cleaner

from conftest import (
    PAPER,
    add_grain,
    blank_page,
    colour_page,
    photograph,
    text_page,
)


# --- a blank page ---------------------------------------------------------- #

@pytest.mark.parametrize("sigma", (0.0, 3.0, 6.0))
def test_a_blank_page_is_reported_blank_and_left_alone(sigma):
    sheet = blank_page(sigma=sigma)
    result = ocr_cleaner.clean(sheet)
    assert result.page_kind == "blank"
    assert result.is_blank
    assert result.applied == []
    assert np.array_equal(np.array(result.image), sheet)


def test_a_blank_page_is_not_thresholded_into_speckles():
    """The failure this guards against: grain scaled up until it reads as text."""
    sheet = blank_page(sigma=4.0)
    result = ocr_cleaner.clean(sheet, threshold="adaptive")
    assert not result.binary
    assert set(np.unique(np.array(result.image))) != {0, 255}
    assert "blank" in result.step("threshold").detail


def test_a_blank_page_says_so_in_its_notes_and_steps():
    result = ocr_cleaner.clean(blank_page())
    assert result.skipped == list(ocr_cleaner.STEP_NAMES)
    assert all(step.detail for step in result.steps)
    assert any("blank" in note for note in result.notes)


def test_dark_paper_with_nothing_on_it_is_still_blank():
    assert ocr_cleaner.clean(blank_page(paper=120)).page_kind == "blank"


# --- a photograph ---------------------------------------------------------- #

def test_a_photograph_is_reported_as_one_and_left_alone():
    frame = photograph()
    result = ocr_cleaner.clean(frame)
    assert result.page_kind == "photograph"
    assert result.is_photograph
    assert result.applied == []
    assert np.array_equal(np.array(result.image), frame)
    assert "photograph" in result.page_kind_detail


def test_a_photograph_is_not_binarised_even_when_asked():
    result = ocr_cleaner.clean(photograph(), threshold="otsu")
    assert not result.binary
    assert any("photograph" in note for note in result.notes)


def test_a_photograph_reports_no_skew_rather_than_a_guess():
    assert ocr_cleaner.clean(photograph()).skew_corrected_degrees == 0.0


# --- a page that is already clean ------------------------------------------ #

def test_an_already_clean_page_passes_through_and_names_what_it_skipped(page):
    result = ocr_cleaner.clean(page, threshold="none")
    assert result.page_kind == "document"
    assert result.applied == []
    assert set(result.skipped) == set(ocr_cleaner.STEP_NAMES)
    assert np.array_equal(np.array(result.image), page)
    for name in ("deskew", "border", "denoise"):
        assert len(result.step(name).detail) > 20


def test_an_already_clean_page_only_gets_the_threshold(page):
    result = ocr_cleaner.clean(page)
    assert result.applied == ["threshold"]
    assert result.changed
    assert result.skipped == ["grayscale", "deskew", "border", "denoise", "upscale"]


# --- both kinds of input --------------------------------------------------- #

def test_greyscale_and_colour_reach_the_same_verdict():
    grey = ocr_cleaner.clean(text_page(900, 700, pitch=26, bar=9, left=70, right=630, top=90))
    colour = ocr_cleaner.clean(colour_page(900, 700))
    assert grey.page_kind == colour.page_kind == "document"
    assert grey.output_size == colour.output_size
    assert colour.image.mode == "L"


def test_an_alpha_channel_is_composited_not_dropped(page):
    rgba = Image.fromarray(page, mode="L").convert("RGBA")
    result = ocr_cleaner.clean(rgba)
    assert "transparency" in result.step("grayscale").detail
    assert result.page_kind == "document"


# --- the caller's image is never modified ---------------------------------- #

def test_an_array_passed_in_is_not_modified(bad_scan):
    before = bad_scan.copy()
    ocr_cleaner.clean(bad_scan, dpi=150, upscale_to_dpi=300)
    assert np.array_equal(bad_scan, before)


def test_a_pil_image_passed_in_is_not_modified(bad_scan):
    source = Image.fromarray(bad_scan, mode="L")
    before = np.array(source).copy()
    result = ocr_cleaner.clean(source)
    assert np.array_equal(np.array(source), before)
    assert source.size == (bad_scan.shape[1], bad_scan.shape[0])
    assert result.image is not source


def test_the_result_image_is_a_new_object_even_when_nothing_was_applied(page):
    source = Image.fromarray(page, mode="L")
    result = ocr_cleaner.clean(source, threshold="none")
    assert result.image is not source
    result.image.putpixel((0, 0), 0)
    assert source.getpixel((0, 0)) == PAPER


def test_a_file_on_disk_is_not_rewritten(page_file):
    before = open(page_file, "rb").read()
    ocr_cleaner.clean(page_file)
    assert open(page_file, "rb").read() == before


# --- sizes at both extremes ------------------------------------------------ #

@pytest.mark.parametrize("size", ((1, 1), (4, 9), (30, 30)))
def test_a_tiny_image_is_handled_rather_than_crashing(size):
    tiny = np.full(size, 200, dtype=np.uint8)
    result = ocr_cleaner.clean(tiny)
    assert result.output_size == (size[1], size[0])
    assert all(step.detail for step in result.steps)


def test_an_empty_array_is_rejected():
    with pytest.raises(ValueError):
        ocr_cleaner.clean(np.zeros((0, 5), dtype=np.uint8))


@pytest.mark.parametrize("shape", ((5,), (4, 4, 2), (2, 2, 2, 2)))
def test_an_array_of_the_wrong_shape_is_rejected(shape):
    with pytest.raises(ValueError):
        ocr_cleaner.clean(np.zeros(shape, dtype=np.uint8))


def test_a_one_line_page_is_still_a_document():
    result = ocr_cleaner.clean(text_page(lines=1))
    assert result.page_kind == "document"


def test_a_four_thousand_by_three_thousand_scan_cleans_in_a_couple_of_seconds():
    """The size promise. A 12 megapixel scan is an ordinary input, not a stress test."""
    big = text_page(3000, 4000, pitch=96, bar=30, left=300, right=3700, top=320)
    big = add_grain(big, 7.0, seed=21)
    started = time.perf_counter()
    result = ocr_cleaner.clean(big)
    elapsed = time.perf_counter() - started
    assert result.page_kind == "document"
    assert result.binary
    assert elapsed < 6.0, "12 megapixels took {0:.1f}s".format(elapsed)


# --- odd but legal inputs -------------------------------------------------- #

def test_a_page_of_solid_ink_does_not_crash():
    result = ocr_cleaner.clean(np.zeros((300, 240), dtype=np.uint8))
    assert result.page_kind in ocr_cleaner.PAGE_KINDS
    assert all(step.detail for step in result.steps)


def test_a_float_array_in_zero_to_one_is_read_as_an_image():
    result = ocr_cleaner.clean(text_page().astype(np.float64) / 255.0)
    assert result.page_kind == "document"


def test_a_single_channel_third_dimension_is_accepted(page):
    result = ocr_cleaner.clean(page[:, :, None])
    assert result.page_kind == "document"


def test_border_never_eats_most_of_the_page():
    """A page that is mostly dark must not be trimmed down to a sliver."""
    mostly_dark = np.full((600, 480), 20, dtype=np.uint8)
    mostly_dark[250:350, 200:280] = 240
    result = ocr_cleaner.clean(mostly_dark)
    assert result.output_size[0] >= 480 * 0.4
    assert result.output_size[1] >= 600 * 0.4
