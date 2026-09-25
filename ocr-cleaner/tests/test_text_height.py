"""``estimated_text_height_px`` against lines whose true extent is known.

The field is documented, everywhere it appears, as the height of an inked line
from ascender top to descender foot. That claim is only worth anything if
something checks it against a line that really has an ascender and a descender,
so every page here is built to a height chosen in advance and the reported
number is compared to it.

A line of text is not one band in a projection profile, it is three: a tall
x-height core with a thin ascender band above it and a thinner descender band
below. Only a few glyphs in a line carry either, so those two bands sum to
around a tenth of what a core row does - far enough down that a cut taken from
the whole profile swing lands between them and the core, splits every line into
three, and reports the pieces. The pages below are the shape that shows it:
``ascender_page`` draws the three bands by hand, and ``glyph_page`` renders real
text. Under the paired-extent measurement both report the height they were
drawn to; under a shared cut across the whole profile both came back at about
0.5x.
"""
from __future__ import annotations

import numpy as np
import pytest

import ocr_cleaner

from conftest import ascender_page, glyph_page, scalable_font

#: How far off the drawn extent the estimate may be. The measurement lands
#: within a few percent on every page here; this is the margin that keeps
#: anti-aliasing and the analysis downscale from making it brittle, and it is
#: still nowhere near the 0.5x a broken measurement gives.
TOLERANCE = 0.15


@pytest.mark.parametrize(
    "shape",
    [
        {},
        {"core": 20, "ascender": 12, "descender": 9, "pitch": 70},
        {"core": 6, "ascender": 4, "descender": 3, "stem": 3, "pitch": 24},
        {"core": 14, "ascender": 10, "descender": 8, "pitch": 80},
    ],
)
def test_the_text_height_is_the_whole_line_not_its_x_height(shape):
    page, extent = ascender_page(**shape)
    result = ocr_cleaner.clean(page)
    assert result.estimated_text_height_px == pytest.approx(extent, rel=TOLERANCE)
    # and specifically not the core band, which is what a shared cut reports
    assert result.estimated_text_height_px > shape.get("core", 12) * 1.2


@pytest.mark.parametrize("em", [14, 20, 30, 48])
def test_really_rendered_glyphs_report_their_measured_ink_extent(em):
    """Not drawn bars: text put on the page by a font, measured off the pixels."""
    if scalable_font(em) is None:
        pytest.skip("no scalable font on this machine")
    page, extent = glyph_page(em)
    result = ocr_cleaner.clean(page)
    assert result.estimated_text_height_px == pytest.approx(extent, rel=TOLERANCE)


def test_the_summary_prints_the_extent_it_measured():
    page, extent = ascender_page()
    result = ocr_cleaner.clean(page)
    line = [
        row for row in result.summary().splitlines()
        if row.strip().startswith("text")
    ]
    assert line, "the summary says nothing about the text height"
    assert "about {0:.0f} px tall".format(round(result.estimated_text_height_px)) in line[0]
    assert str(int(round(extent))) in line[0] or str(extent - 1) in line[0]


def test_to_dict_carries_the_same_extent():
    page, extent = ascender_page()
    result = ocr_cleaner.clean(page)
    assert result.to_dict()["estimated_text_height_px"] == pytest.approx(
        extent, rel=TOLERANCE
    )


def test_the_adaptive_window_is_sized_from_the_whole_line():
    """The window the threshold uses follows the extent, not a floor it fell to."""
    page, extent = ascender_page(core=20, ascender=12, descender=9, pitch=70)
    result = ocr_cleaner.clean(page)
    detail = result.step("threshold").detail
    assert "3x the {0:.0f} px text height".format(result.estimated_text_height_px) in detail
    window = int(detail.split(" x ")[0].split()[-1])
    assert window == pytest.approx(3 * extent, rel=TOLERANCE + 0.05)
    assert window > 15, "the window fell back to its floor"


def test_a_taller_page_of_text_reports_a_taller_line():
    small, small_extent = ascender_page(core=6, ascender=4, descender=3, stem=3, pitch=24)
    large, large_extent = ascender_page(core=20, ascender=12, descender=9, pitch=70)
    assert small_extent < large_extent
    small_height = ocr_cleaner.clean(small).estimated_text_height_px
    large_height = ocr_cleaner.clean(large).estimated_text_height_px
    assert small_height < large_height
    assert large_height / small_height == pytest.approx(
        large_extent / small_extent, rel=0.25
    )


def test_the_extent_survives_grain_and_a_turned_page():
    from conftest import add_grain, rotate

    page, extent = ascender_page()
    scanned = add_grain(rotate(page, -2.4), 7.0, seed=4)
    result = ocr_cleaner.clean(scanned)
    assert result.estimated_text_height_px == pytest.approx(extent, rel=0.25)
