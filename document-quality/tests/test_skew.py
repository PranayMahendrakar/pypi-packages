"""Skew to about half a degree either way round, and which way up the page is."""
from __future__ import annotations

import numpy as np
import pytest

import _synthetic as S
import document_quality
from conftest import kinds

#: Known rotations, counter-clockwise positive, including negative ones.
ANGLES = [-9.0, -4.2, -2.3, -1.1, -0.6, -0.3, 0.0, 0.25, 0.7, 1.5, 3.8, 8.5]


@pytest.mark.parametrize("angle", ANGLES)
def test_estimate_skew_within_half_a_degree(page, angle):
    turned = S.rotate(page, angle)
    found = document_quality.estimate_skew(turned)
    assert abs(found - angle) <= 0.5, (angle, found)


def test_negative_and_positive_skew_have_opposite_signs(page):
    left = document_quality.estimate_skew(S.rotate(page, -3.0))
    right = document_quality.estimate_skew(S.rotate(page, 3.0))
    assert left < -2.5 and right > 2.5


@pytest.mark.parametrize("angle,direction,remedy", [
    (2.3, "counter-clockwise", "clockwise"),
    (-2.3, "clockwise", "counter-clockwise"),
])
def test_skewed_page_gets_a_deskew_fix(page, angle, direction, remedy):
    report = document_quality.assess(S.rotate(page, angle), dpi=300,
                                     check_orientation=False)
    assert abs(report.skew_degrees - angle) <= 0.5
    issue = next(item for item in report.issues if item.kind == "skew")
    assert issue.fix == "deskew by 2.3 degrees {0}".format(remedy)
    assert "degrees {0} of horizontal".format(direction) in issue.message
    assert not report.ocr_ready
    assert "held back by skew" in report.summary().splitlines()[0]


def test_small_skew_is_a_warning_not_a_failure(page):
    report = document_quality.assess(S.rotate(page, 0.8), check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "skew")
    assert issue.severity == "warning"


def test_straight_page_has_no_skew_issue(clean_report):
    assert abs(clean_report.skew_degrees) <= 0.1
    assert "skew" not in kinds(clean_report)


def test_scan_with_black_corner_wedges_still_reads_the_skew(page):
    # A crooked scan of a sheet on a black lid: black triangles in the corners.
    from PIL import Image
    turned = np.asarray(Image.fromarray(page).rotate(-3.0, fillcolor=0), dtype=np.uint8)
    report = document_quality.assess(turned, dpi=300)
    assert abs(report.skew_degrees + 3.0) <= 0.5
    # Corner wedges are neither a border along a whole side nor a shadow.
    assert "border" not in kinds(report)
    assert "lighting" not in kinds(report)


@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
def test_detect_orientation(page, quarter):
    turned = np.ascontiguousarray(np.rot90(page, quarter))
    correction = document_quality.detect_orientation(turned)
    assert correction == (360 - 90 * quarter) % 360
    upright = np.rot90(turned, correction // 90)
    assert np.array_equal(upright, page)


def test_sideways_page_is_measured_upright(page, clean_report):
    report = document_quality.assess(np.ascontiguousarray(np.rot90(page, 1)), dpi=300)
    assert kinds(report) == {"orientation"}
    issue = report.issues[0]
    assert issue.fix == "rotate the page 90 degrees clockwise before OCR"
    # Text height is the upright page's, not the height of a sideways column.
    assert report.estimated_text_height_px == pytest.approx(
        clean_report.estimated_text_height_px, abs=2.0
    )
    assert any("turned upright" in note for note in report.notes)


def test_upside_down_page(page):
    report = document_quality.assess(np.ascontiguousarray(page[::-1, ::-1]))
    assert report.issues[0].kind == "orientation"
    assert report.issues[0].fix == "rotate the page 180 degrees before OCR"


def test_no_lines_means_no_skew_and_no_orientation(blank, photo):
    assert document_quality.estimate_skew(blank) == 0.0
    assert document_quality.estimate_skew(photo) == 0.0
    assert document_quality.estimate_skew(S.photograph(colour=False)) == 0.0
    assert document_quality.detect_orientation(blank) == 0
    assert document_quality.detect_orientation(photo) == 0
