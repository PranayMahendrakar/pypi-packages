"""Resolution, text height, sharpness, show-through, clipping and thresholds."""
from __future__ import annotations

import numpy as np
import pytest

import _synthetic as S
import document_quality
from conftest import kinds
from document_quality import Thresholds, describe_thresholds


def test_dpi_none_leaves_resolution_advice_out(page):
    report = document_quality.assess(page)
    assert report.dpi is None and report.dpi_source is None
    resolution = report.measures["resolution"]
    assert resolution.applies is False and resolution.score is None
    assert "resolution" not in kinds(report)
    assert report.page_inches is None
    assert report.to_dict()["dpi"] is None
    assert report.to_dict()["page_inches"] is None
    assert "dpi unknown, so resolution advice is left out" in report.summary()
    assert "Scanned at" not in report.summary()


def test_dpi_none_never_guesses_a_dpi_for_small_text():
    small = S.text_page(850, 1100, x_height=5, margin=80)
    report = document_quality.assess(small, check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "text_size")
    assert "dpi" not in issue.fix
    assert "higher resolution" in issue.fix
    assert all("dpi" not in fix for fix in report.fixes)


def test_low_dpi_gets_rescan_at_300(page):
    report = document_quality.assess(page, dpi=150, check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "resolution")
    assert issue.fix == "rescan at 300 dpi"
    assert issue.severity == "failure"
    assert not report.ocr_ready


def test_implausible_dpi_tag_is_questioned(page):
    report = document_quality.assess(page, dpi=4000, check_orientation=False)
    assert "dpi_tag" in kinds(report)


def test_dpi_is_read_from_file_metadata(page, tmp_path):
    from PIL import Image
    path = tmp_path / "tagged.png"
    Image.fromarray(page).save(path, dpi=(300, 300))
    report = document_quality.assess(path)
    assert report.dpi == pytest.approx(300, abs=1)
    assert report.dpi_source == "image metadata"
    given = document_quality.assess(path, dpi=200, check_orientation=False)
    assert given.dpi_source == "argument"


@pytest.mark.parametrize("x_height", [7, 10, 16])
def test_text_height_matches_what_was_drawn(x_height):
    drawn = S.inked_line_height(x_height)
    page = S.text_page(850, 1100, x_height=x_height, margin=80)
    report = document_quality.assess(page, check_orientation=False)
    assert report.estimated_text_height_px == pytest.approx(drawn, abs=max(3.0, 0.15 * drawn))


def test_small_text_gets_a_dpi_to_rescan_at():
    small = S.text_page(850, 1100, x_height=5, margin=80)
    report = document_quality.assess(small, dpi=300, check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "text_size")
    assert issue.severity == "failure"
    dpi = int(issue.fix.split()[2])
    assert issue.fix == "rescan at {0} dpi".format(dpi)
    assert 550 <= dpi <= 800


def test_blurred_page_fails_sharpness_only(page):
    report = document_quality.assess(S.blurred(page, 4.0), dpi=300, check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "sharpness")
    assert issue.severity == "failure"
    assert "refocus" in issue.fix
    # The grey halo round a soft stroke is not print from the reverse side.
    assert "show_through" not in kinds(report)


def test_show_through_is_found(page, clean_report):
    report = document_quality.assess(S.show_through(page, strength=0.65), dpi=300,
                                     check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "show_through")
    assert issue.fix == "put a sheet of black paper behind the page and rescan"
    assert "show_through" not in kinds(clean_report)


def test_crushed_blacks_are_clipping(page):
    crushed = page.copy()
    crushed[page < 140] = 0
    crushed[300:620, 150:700] = 0
    report = document_quality.assess(crushed, check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "clipping")
    assert "lower the scanner contrast" in issue.fix


def test_blown_white_page_with_little_on_it_warns_about_the_white_point():
    page = S.text_page(850, 1100, x_height=10, margin=80, paper=255, ink=0, blur=0.0)
    page[110:] = 255                                  # one line of text, nothing else
    report = document_quality.assess(page, check_orientation=False)
    assert report.kind == "document"
    issue = next(item for item in report.issues if item.kind == "clipping")
    assert issue.severity == "warning"
    assert "brightness" in issue.fix


def test_thresholds_can_be_overridden(page, clean_report):
    strict = document_quality.assess(page, dpi=300, thresholds={"target_dpi": 600},
                                     check_orientation=False)
    assert strict.measures["resolution"].score < clean_report.measures["resolution"].score
    same = document_quality.assess(page, dpi=300, thresholds=Thresholds(),
                                   check_orientation=False)
    assert same.score == pytest.approx(clean_report.score)
    with pytest.raises(ValueError):
        document_quality.assess(page, thresholds={"no_such_threshold": 1})
    with pytest.raises(ValueError):
        Thresholds().replace(target_dpi="high")
    with pytest.raises(TypeError):
        document_quality.assess(page, thresholds=[1, 2])


def test_describe_thresholds_lists_every_field():
    text = describe_thresholds()
    for name in Thresholds.field_names():
        assert name in text
    text.encode("ascii")
    assert "600" in describe_thresholds({"target_dpi": 600})


def test_score_weights_and_severities_are_published():
    assert set(document_quality.SEVERITIES) == {"failure", "warning", "info"}
    assert "lighting" in document_quality.SCORE_WEIGHTS


@pytest.mark.parametrize("dpi", [200, 300])
def test_png_dpi_reads_back_as_saved(page, tmp_path, dpi):
    # PNG stores whole pixels per metre, so 200 dpi comes back as 199.9996:
    # just under the floor it was saved at, unless it is read as meant.
    from PIL import Image
    path = tmp_path / "scan_{0}.png".format(dpi)
    Image.fromarray(page).save(path, dpi=(dpi, dpi))
    report = document_quality.assess(path, check_orientation=False)
    assert report.dpi == dpi
    assert "resolution" not in kinds(report)
    measure = report.measures["resolution"]
    assert measure.ok and measure.score >= document_quality.PASS_SCORE
    assert "{0} dpi (from image metadata)".format(dpi) in report.summary().splitlines()[1]


def test_failing_score_never_shows_as_the_passing_mark(page):
    report = document_quality.assess(page, dpi=199.9996, check_orientation=False)
    measure = report.measures["resolution"]
    assert not measure.ok
    assert measure.to_dict()["score"] < document_quality.PASS_SCORE


def test_show_through_from_a_blurred_mirror_page_is_reported():
    # A mirrored copy of another page, blurred and at half strength: plainly
    # visible to anyone holding the sheet, and a tenth of the paper darkened.
    try:
        front = S.typed_page(1700, 2200, size=36)
        back = S.typed_page(1700, 2200, size=36, words=S.PANGRAM)
    except RuntimeError:                                    # pragma: no cover
        pytest.skip("no scalable TrueType face is available")
    report = document_quality.assess(S.ghost_of(front, back, 0.5, 2.0), dpi=300,
                                     check_orientation=False)
    issue = next(item for item in report.issues if item.kind == "show_through")
    assert issue.fix == "put a sheet of black paper behind the page and rescan"
    clean = document_quality.assess(front, dpi=300, check_orientation=False)
    assert "show_through" not in kinds(clean)
