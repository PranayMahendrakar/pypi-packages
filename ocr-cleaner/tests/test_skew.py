"""Skew estimation against rotations of a known angle.

This is the test that decides whether the package works. Everything else it
does is sizing and bookkeeping; the angle is the one number it has to get right,
and the only way to know it does is to turn a page by an angle we chose and ask
for it back.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import ocr_cleaner

from conftest import PAPER, rotate, text_page

#: One page, built once and turned many ways. Smaller than a real scan on
#: purpose: the angle search runs on a plane of at most 800 pixels whatever it
#: is given, so a bigger page here would cost seconds and measure nothing extra.
#: At this size the estimate still lands within a hundredth of a degree.
SKEW_PAGE = text_page(560, 430, pitch=18, bar=6, left=45, right=385, top=60)

#: How close the estimate has to be, in degrees. The package claims half a
#: degree; it measures an order of magnitude better than that, and the gap is
#: the margin that keeps this test from being brittle.
TOLERANCE = 0.5

#: Angles to check, both ways round. Negative angles are here because a sign
#: error is the single most likely mistake in the whole package, and it would
#: pass a test that only ever turned pages one way.
ANGLES = (
    0.0, 0.4, -0.4, 1.0, -1.0, 2.5, -2.5, 3.7, -3.7,
    5.0, -5.0, 8.25, -8.25, 12.0, -12.0,
)


@pytest.mark.parametrize("angle", ANGLES)
def test_estimate_skew_recovers_a_known_rotation(angle):
    turned = rotate(SKEW_PAGE, angle)
    assert abs(ocr_cleaner.estimate_skew(turned) - angle) <= TOLERANCE


@pytest.mark.parametrize("angle", (-3.0, 1.6))
def test_estimate_skew_agrees_across_input_kinds(angle):
    turned = rotate(SKEW_PAGE, angle)
    from_array = ocr_cleaner.estimate_skew(turned)
    from_image = ocr_cleaner.estimate_skew(Image.fromarray(turned, mode="L"))
    assert from_array == pytest.approx(from_image, abs=1e-9)


def test_estimate_skew_sign_matches_pil_rotate():
    """The sign is the contract: ``rotate(-estimate_skew(page))`` straightens it."""
    turned = rotate(SKEW_PAGE, 3.0)
    measured = ocr_cleaner.estimate_skew(turned)
    assert measured > 0
    straightened = Image.fromarray(turned, mode="L").rotate(
        -measured, resample=Image.BICUBIC, expand=True, fillcolor=PAPER
    )
    assert abs(ocr_cleaner.estimate_skew(straightened)) <= 0.1


def test_estimate_skew_is_zero_on_a_page_with_no_text():
    assert ocr_cleaner.estimate_skew(np.full((400, 300), 250, dtype=np.uint8)) == 0.0


def test_estimate_skew_survives_grain_and_a_black_margin():
    from conftest import add_grain, add_scanner_border

    turned = add_scanner_border(add_grain(rotate(SKEW_PAGE, -2.4), 8.0, seed=5))
    assert abs(ocr_cleaner.estimate_skew(turned) + 2.4) <= TOLERANCE


def test_estimate_skew_survives_uneven_lighting():
    from conftest import light_unevenly

    turned = light_unevenly(rotate(SKEW_PAGE, 1.9), -90.0)
    assert abs(ocr_cleaner.estimate_skew(turned) - 1.9) <= TOLERANCE


@pytest.mark.parametrize("angle", (-2.2, 4.4))
def test_clean_corrects_the_angle_it_measured(angle):
    result = ocr_cleaner.clean(rotate(SKEW_PAGE, angle), threshold="none")
    assert result.step("deskew").applied
    assert result.skew_corrected_degrees == pytest.approx(angle, abs=TOLERANCE)
    assert abs(ocr_cleaner.estimate_skew(result.image)) <= TOLERANCE


def test_a_straight_page_is_not_rotated(page):
    result = ocr_cleaner.clean(page)
    assert not result.step("deskew").applied
    assert result.skew_corrected_degrees == 0.0
    assert "straight" in result.step("deskew").detail


def test_estimate_skew_is_deterministic():
    turned = rotate(SKEW_PAGE, -1.3)
    assert ocr_cleaner.estimate_skew(turned) == ocr_cleaner.estimate_skew(turned)


@pytest.mark.parametrize("quarter", (90.0, -90.0))
def test_skew_beyond_the_search_range_is_not_invented(quarter):
    """A page turned a quarter turn is not skewed, and is reported as straight.

    Not simply "inside the range": ``_search`` clamps every candidate to
    ``MAX_SKEW_DEGREES`` itself, so that much is true of any input at all and
    would pin nothing. What is worth pinning is where inside the range it lands.
    A sideways page has no comb in its profile at any angle, so the search has
    to settle at upright rather than drift to the edge of its own window, and
    ``clean`` has to leave the page alone rather than interpolate it for
    nothing.
    """
    sideways = rotate(text_page(380, 380, pitch=16, bar=5, left=40, right=340, top=50), quarter)
    measured = ocr_cleaner.estimate_skew(sideways)
    assert abs(measured) <= TOLERANCE
    assert abs(measured) < ocr_cleaner.MAX_SKEW_DEGREES / 2.0
    result = ocr_cleaner.clean(sideways, threshold="none")
    assert not result.step("deskew").applied
    assert result.skew_corrected_degrees == 0.0
