"""A lighting shadow is uneven lighting with a lighting fix; only black is a border.

This is the mistake the sibling package ocr-cleaner had to fix: judge darkness
against one page-wide paper level and a soft shadow down one edge reads as a
scanner border. Here lighting is a gradient of the paper surface and a border
has to be genuinely black.
"""
from __future__ import annotations

import numpy as np
import pytest

import _synthetic as S
import document_quality
from conftest import kinds


@pytest.fixture(scope="module")
def soft(page):
    """The report for a page with a soft shadow down its left edge."""
    return document_quality.assess(S.shadow_left(page, depth=0.3), dpi=300)


def turn_shadow(page, quarter, depth=0.3):
    """``page`` with a shadow falling on the side that is left after ``quarter`` turns."""
    turned = np.ascontiguousarray(np.rot90(page, quarter))
    return np.ascontiguousarray(np.rot90(S.shadow_left(turned, depth=depth), -quarter))


def test_soft_shadow_is_uneven_lighting_with_a_lighting_fix(soft):
    report = soft
    assert report.kind == "document"
    assert kinds(report) == {"lighting"}
    issue = report.issues[0]
    assert issue.severity == "warning"
    assert issue.fix.startswith("increase lighting on the left edge")
    assert "towards the left edge" in issue.message
    # Soft enough to read through: still worth sending.
    assert report.ocr_ready


def test_soft_shadow_is_not_a_scanner_border(soft):
    report = soft
    assert "border" not in kinds(report)
    assert report.measures["border"].ok
    assert report.measures["border"].value == 0.0
    assert not any("border" in note for note in report.notes)


def test_soft_shadow_is_not_unreadable(soft, clean_report):
    report = soft
    # The ink under the shadow is as legible as the rest of the page.
    for name in ("contrast", "sharpness", "text_size", "show_through", "clipping"):
        assert report.measures[name].ok, name
    assert report.estimated_text_height_px == pytest.approx(
        clean_report.estimated_text_height_px, abs=2.0
    )
    assert report.measures["contrast"].value == pytest.approx(
        clean_report.measures["contrast"].value, abs=0.05
    )


def test_deep_shadow_is_still_lighting_and_never_a_border(page):
    report = document_quality.assess(S.shadow_left(page, depth=0.55), dpi=300,
                                     check_orientation=False)
    assert report.kind == "document"
    assert kinds(report) == {"lighting"}
    assert report.issues[0].severity == "failure"
    assert report.issues[0].fix.startswith("increase lighting on the left edge")
    assert "held back by lighting" in report.summary().splitlines()[0]


@pytest.mark.parametrize("quarter,where", [
    (0, "left edge"), (1, "top edge"), (2, "right edge"), (3, "bottom edge"),
])
def test_lighting_fix_names_the_dark_side(page, quarter, where):
    report = document_quality.assess(turn_shadow(page, quarter), check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "lighting")
    assert issue.fix.startswith("increase lighting on the " + where)


def test_corner_shadow_names_the_corner(page):
    left = S.shadow_left(page, depth=0.35)
    top = np.ascontiguousarray(np.rot90(
        S.shadow_left(np.ascontiguousarray(np.rot90(left, 1)), depth=0.35), -1))
    report = document_quality.assess(top, check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "lighting")
    assert issue.fix.startswith("increase lighting on the top-left corner")


def test_evenly_lit_page_has_no_lighting_issue(clean_report):
    assert "lighting" not in kinds(clean_report)
    assert clean_report.measures["lighting"].value < 0.04


def test_black_border_is_a_border_with_a_crop_fix(page, clean_report):
    report = document_quality.assess(S.black_border_left(page, width_px=70), dpi=300,
                                     check_orientation=False)
    assert kinds(report) == {"border"}
    issue = report.issues[0]
    assert issue.severity == "warning"
    assert issue.fix.startswith("crop the black border before OCR")
    assert "off the left edge" in issue.fix
    details = report.measures["border"].to_dict()["details"]
    assert 70 <= details["left"] <= 76
    assert details["right"] == details["top"] == details["bottom"] == 0
    # The border was cropped before measuring, so it is not a shadow either.
    assert "lighting" not in kinds(report)
    assert report.measures["clipping"].ok
    assert report.score == pytest.approx(clean_report.score, abs=2.0)


def test_border_and_shadow_together_are_told_apart(page):
    image = S.black_border_left(S.shadow_left(page, depth=0.3), width_px=60)
    report = document_quality.assess(image, dpi=300, check_orientation=False)
    assert kinds(report) == {"border", "lighting"}
    lighting = next(item for item in report.issues if item.kind == "lighting")
    assert lighting.fix.startswith("increase lighting on the left edge")


def test_borders_on_several_sides(page):
    image = page.copy()
    image[:40] = 8
    image[:, -55:] = 8
    report = document_quality.assess(image, check_orientation=False)
    border = report.measures["border"].details
    assert set(border["sides"]) == {"top", "right"}
    assert 40 <= border["top"] <= 46 and 55 <= border["right"] <= 61


def test_dark_picture_is_not_cropped_as_a_border():
    dark = np.full((800, 600), 15, dtype=np.uint8)
    dark[300:320, 200:400] = 200
    report = document_quality.assess(dark)
    assert "border" not in kinds(report)
    assert report.measures["border"].value == 0.0


@pytest.fixture(scope="module")
def typed_page():
    """Real type, 36 px, with a 125 px margin before the text starts."""
    try:
        return S.typed_page(1275, 1650, size=36, margin=125)
    except RuntimeError:                                    # pragma: no cover
        pytest.skip("no scalable TrueType face is available")


@pytest.mark.parametrize("depth,reach", [(0.7, 0.6), (0.8, 0.35)])
def test_deep_shadow_is_lighting_and_is_never_cropped_as_a_border(typed_page, depth, reach):
    # The paper at the edge is down to 50-75 grey and the ink is still on it.
    # Only a genuinely black, empty band ending in the edge of the sheet is a
    # border; cropping this one would cut into every line of text.
    report = document_quality.assess(S.shadow_left(typed_page, depth, reach), dpi=300)
    assert report.kind == "document"
    assert "border" not in kinds(report)
    assert report.measures["border"].value == 0.0
    assert not any("border" in note for note in report.notes)
    lighting = next(item for item in report.issues if item.kind == "lighting")
    assert lighting.fix.startswith("increase lighting on the left edge")


def test_strong_light_ramp_across_the_page_is_uneven_lighting_not_a_photograph(typed_page):
    ramp = np.linspace(0.4, 1.0, typed_page.shape[1])[None, :]
    report = document_quality.assess((typed_page * ramp).astype(np.uint8), dpi=300)
    assert report.kind == "document"
    assert report.estimated_text_height_px is not None
    lighting = next(item for item in report.issues if item.kind == "lighting")
    assert lighting.fix.startswith("increase lighting on the left edge")
