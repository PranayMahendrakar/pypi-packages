"""Blank, faint, photograph and degenerate pages each get the right single answer."""
from __future__ import annotations

import numpy as np
import pytest

import _synthetic as S
import document_quality
from conftest import kinds


def test_blank_page_is_blank_not_a_failure(blank):
    report = document_quality.assess(blank, dpi=300)
    assert report.kind == "blank"
    assert report.is_blank
    assert not report.ocr_ready
    assert report.failures == []
    assert [item.kind for item in report.issues] == ["blank"]
    assert report.issues[0].severity == "info"
    assert "skip this page" in report.issues[0].fix
    assert report.estimated_text_height_px is None
    assert report.skew_degrees == 0.0
    assert report.summary().splitlines()[0].endswith("blank, nothing to OCR - score 0 of 100")


def test_blank_page_under_a_shadow_is_still_blank(blank):
    report = document_quality.assess(S.shadow_left(blank, depth=0.45))
    assert report.kind == "blank"
    assert report.failures == []


def test_perfectly_flat_page_is_blank():
    report = document_quality.assess(np.full((600, 450), 250, dtype=np.uint8))
    assert report.kind == "blank"


def test_faint_page_with_text_is_low_contrast_not_blank(page):
    faint = (page.astype(np.float64) * 0.15 + 200).astype(np.uint8)
    report = document_quality.assess(faint, dpi=300)
    assert report.kind == "document"
    issue = next(item for item in report.issues if item.kind == "contrast")
    assert issue.severity == "failure"
    assert "raise the scanner contrast" in issue.fix


@pytest.mark.parametrize("colour", [True, False])
def test_photograph_is_not_a_document_page(colour):
    report = document_quality.assess(S.photograph(colour=colour), dpi=300)
    assert report.kind == "photograph"
    assert not report.is_document
    assert not report.ocr_ready
    assert [item.kind for item in report.issues] == ["not_a_document"]
    assert "image pipeline" in report.issues[0].fix
    # No text-shaped numbers about a picture.
    assert report.estimated_text_height_px is None
    assert report.skew_degrees == 0.0
    assert "not a document page" in report.summary().splitlines()[0]
    assert report.evidence["paper_share"] < 0.4


def test_page_with_little_text_is_still_a_document(page):
    sparse = np.full_like(page, S.PAPER)
    sparse[80:400] = page[80:400]
    report = document_quality.assess(sparse, dpi=300)
    assert report.kind == "document"
    # A short page is a good page: nothing about it is wrong.
    assert report.ocr_ready, [str(item) for item in report.issues]
    assert report.issues == []


def _grainy_sheet(seed: int, shape=(1650, 1275)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(245 + rng.normal(0.0, 2.0, size=shape), 0, 255).astype(np.uint8)


def assert_blank(report):
    assert report.kind == "blank", report.summary()
    assert [item.kind for item in report.issues] == ["blank"]
    assert report.issues[0].severity == "info"
    assert "skip this page" in report.issues[0].fix


@pytest.mark.parametrize("specks,size", [(20, 3), (12, 5)])
def test_blank_sheet_with_dust_is_still_blank(specks, size):
    # Twenty specks of dust are a comb of spikes down the profile; they are
    # not rows of text, because none of them runs across the page.
    sheet = _grainy_sheet(specks + size)
    rng = np.random.default_rng(size)
    for _ in range(specks):
        top, left = rng.integers(0, 1640), rng.integers(0, 1265)
        sheet[top:top + size, left:left + size] = 50
    assert_blank(document_quality.assess(sheet, dpi=300))


@pytest.mark.parametrize("angle", [3.0, 1.2])
def test_blank_sheet_scanned_crooked_on_a_black_lid_is_blank(angle):
    from PIL import Image
    crooked = np.asarray(Image.fromarray(_grainy_sheet(7)).rotate(angle, fillcolor=0))
    report = document_quality.assess(crooked, dpi=300)
    assert_blank(report)
    assert report.skew_degrees == 0.0


def test_page_scanned_crooked_on_a_black_lid_reads_the_page_not_the_lid(page):
    from PIL import Image
    crooked = np.asarray(Image.fromarray(page).rotate(-2.0, fillcolor=0), dtype=np.uint8)
    report = document_quality.assess(crooked, dpi=300)
    assert report.kind == "document"
    assert report.skew_degrees == pytest.approx(-2.0, abs=0.3)
    assert "border" not in kinds(report) and "clipping" not in kinds(report)
    assert any("dark lid" in note for note in report.notes)


@pytest.mark.parametrize("shape", [(3, 3), (8, 8), (15, 15), (10, 3000)])
def test_image_with_no_rows_of_text_is_never_ready(shape):
    image = np.full(shape, 245, dtype=np.uint8)
    middle = shape[0] // 2
    image[max(0, middle - 1):middle + 1, 1:-1] = 20
    report = document_quality.assess(image)
    assert not report.ocr_ready
    assert report.score < 100
    if report.kind == "document":
        assert "text_size" in {item.kind for item in report.failures}


def test_one_solid_shape_is_not_a_document_page():
    sheet = np.full((1650, 1275), 245, dtype=np.uint8)
    sheet[600:900, 300:950] = 10
    report = document_quality.assess(sheet, dpi=300)
    assert report.kind == "photograph"
    assert not report.ocr_ready


def test_photograph_of_window_blinds_is_not_a_document_page():
    from PIL import Image, ImageFilter
    rows = np.arange(750)[:, None]
    blinds = np.where((rows % 45) < 15, 210.0, 60.0) + np.zeros((750, 1000))
    blinds = np.asarray(Image.fromarray(blinds.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(3)), dtype=np.float64)
    noisy = blinds + np.random.default_rng(1).normal(0.0, 8.0, blinds.shape)
    report = document_quality.assess(np.clip(noisy, 0, 255).astype(np.uint8), dpi=300)
    assert report.kind == "photograph"


@pytest.mark.parametrize("shape", [(1, 400), (400, 1), (3, 3), (12, 900)])
def test_degenerate_sizes_do_not_crash(shape):
    image = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    report = document_quality.assess(image)
    assert report.kind in document_quality.PAGE_KINDS
    report.summary()
    report.to_dict()


def test_all_nan_float_image_is_blank():
    report = document_quality.assess(np.full((300, 200), np.nan))
    assert report.kind == "blank"


def test_bad_inputs_raise_clear_errors(tmp_path):
    with pytest.raises(ValueError):
        document_quality.assess(np.zeros((0, 0), dtype=np.uint8))
    with pytest.raises(ValueError):
        document_quality.assess(np.zeros(100, dtype=np.uint8))
    with pytest.raises(ValueError):
        document_quality.assess(np.zeros((20, 20, 2), dtype=np.uint8))
    with pytest.raises(TypeError):
        document_quality.assess([[0, 1], [1, 0]])
    with pytest.raises(FileNotFoundError):
        document_quality.assess(tmp_path / "missing.png")
    with pytest.raises(ValueError):
        document_quality.assess(np.zeros((50, 50), dtype=np.uint8), dpi=0)
    with pytest.raises(ValueError):
        document_quality.assess(np.zeros((50, 50), dtype=np.uint8), dpi="abc")
