"""Orientation, skew, line geometry and sparse pages, on real TrueType type.

The hand-drawn glyphs in ``_synthetic.text_page`` exercise exactly what they
were drawn to exercise. These pages are set in a real face through Pillow,
which is where the up-versus-down test and the line measures have to hold up:
a face whose descender zone is as deep as its ascender zone, a sentence set the
same way line after line so its letters stand in columns, a shadow across the
top of the sheet, a cover page with one line on it.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

import _synthetic as S
import document_quality
from conftest import kinds

_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def typed(**options) -> np.ndarray:
    """A page of real type, or a skip when no scalable face is installed."""
    try:
        return S.typed_page(**options)
    except RuntimeError:                                    # pragma: no cover
        pytest.skip("no scalable TrueType face is available")


def turned(page: np.ndarray, quarter: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(page, quarter))


@pytest.fixture(scope="module")
def prose() -> np.ndarray:
    """Running prose, 30 px type on a 1275 x 1650 page."""
    return typed(width=1275, height=1650, size=30)


@pytest.fixture(scope="module")
def pangram() -> np.ndarray:
    """One sentence over and over: every line alike, letters in columns."""
    return typed(width=1275, height=1650, size=30, words=S.PANGRAM)


@pytest.fixture(scope="module")
def prose_report(prose):
    return document_quality.assess(prose, dpi=300)


def test_detect_orientation_on_real_type(prose):
    found = [document_quality.detect_orientation(turned(prose, q)) for q in range(4)]
    assert found == [0, 270, 180, 90]
    # Every answer, applied as documented, puts the page back as it was.
    for quarter, correction in enumerate(found):
        assert np.array_equal(np.rot90(turned(prose, quarter), correction // 90), prose)


def test_detect_orientation_when_letters_stand_in_columns(pangram):
    # One sentence line after line combs almost as well sideways as upright.
    assert document_quality.detect_orientation(pangram) == 0
    assert document_quality.detect_orientation(turned(pangram, 3)) == 90


def test_small_type_on_a_large_page_is_read_the_right_way_up():
    page = typed(width=1700, height=2200, size=20, words=S.PANGRAM)
    assert document_quality.detect_orientation(turned(page, 2)) == 180


def test_upside_down_page_is_held_back_with_a_half_turn(pangram):
    report = document_quality.assess(turned(pangram, 2), dpi=300)
    assert not report.ocr_ready
    assert report.issues[0].kind == "orientation"
    assert report.issues[0].severity == "failure"
    assert report.issues[0].fix == "rotate the page 180 degrees before OCR"
    assert "held back by orientation" in report.headline()


def test_page_on_its_side_gets_the_quarter_turn_that_sets_it_upright(pangram):
    report = document_quality.assess(turned(pangram, 1), dpi=300)
    assert kinds(report) == {"orientation"}
    assert report.issues[0].fix == "rotate the page 90 degrees clockwise before OCR"


def test_upright_real_type_is_ready_with_nothing_to_say(prose_report):
    assert prose_report.ocr_ready
    assert prose_report.issues == []
    assert not any("upside" in note for note in prose_report.notes)


@pytest.mark.parametrize("side", ["top", "bottom"])
def test_shadow_on_the_top_or_bottom_edge_is_only_uneven_lighting(prose, prose_report, side):
    shaded = S.shadow_side(prose, side, depth=0.35, reach=0.5)
    report = document_quality.assess(shaded, dpi=300)
    assert kinds(report) == {"lighting"}
    assert report.issues[0].fix.startswith("increase lighting on the {0} edge".format(side))
    assert report.ocr_ready
    assert document_quality.detect_orientation(shaded) == 0
    # Measured against the local paper, the shadow does not shorten the text.
    assert report.estimated_text_height_px == pytest.approx(
        prose_report.estimated_text_height_px, abs=1.5
    )


@pytest.mark.parametrize("ink", [80])
def test_grey_ink_upright_pages_stay_upright(ink):
    page = typed(width=1275, height=1650, size=24, ink=ink)
    assert document_quality.detect_orientation(page) == 0


def test_soft_scan_is_never_called_upside_down(prose):
    soft = np.asarray(Image.fromarray(prose).filter(ImageFilter.GaussianBlur(1.5)), dtype=np.uint8)
    assert document_quality.detect_orientation(soft) == 0


@pytest.mark.parametrize("lines", [1, 4])
def test_page_with_a_few_lines_is_ready(lines):
    page = typed(width=1275, height=1650, size=36, lines=lines)
    report = document_quality.assess(page, dpi=300)
    assert report.kind == "document"
    assert report.issues == [], [str(item) for item in report.issues]
    assert report.ocr_ready
    assert abs(report.skew_degrees) < 0.2
    assert report.measures["sharpness"].ok


@pytest.mark.parametrize("angle", [-0.25, 2.3])
def test_skew_of_real_type_has_no_dead_zone(prose, angle):
    crooked = np.asarray(
        Image.fromarray(prose).rotate(angle, resample=_BICUBIC, fillcolor=245), dtype=np.uint8
    )
    assert document_quality.estimate_skew(crooked) == pytest.approx(angle, abs=0.1)
