"""The result objects: what they say, and that they can be read back as data."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

import ocr_cleaner
from ocr_cleaner import CleanResult, Step

from conftest import blank_page, photograph, text_page


@pytest.fixture(scope="module")
def report():
    """One clean page's result, shared by every test that only reads it."""
    return ocr_cleaner.clean(text_page())


def test_step_always_carries_a_reason(report):
    for step in report.steps:
        assert step.name in ocr_cleaner.STEP_NAMES
        assert isinstance(step.applied, bool)
        assert step.detail.strip(), "{0} gave no reason".format(step.name)
        assert len(step.detail) > 10


def test_step_str_reads_as_a_line():
    text = str(Step("denoise", False, "already clean"))
    assert "denoise" in text and "skipped" in text and "already clean" in text


def test_step_to_dict_is_json_safe():
    payload = Step("deskew", True, "rotated +2.00 degrees").to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_lookup_helpers(report):
    assert report.step("threshold") is not None
    assert report.step("nonesuch") is None
    assert set(report.applied) | set(report.skipped) == set(ocr_cleaner.STEP_NAMES)
    assert not set(report.applied) & set(report.skipped)


def test_summary_is_plain_ascii(report):
    text = report.summary()
    text.encode("ascii")           # raises if a stray arrow or bullet crept in
    for line in ("page", "size", "text", "skew", "steps"):
        assert line in text
    assert all(step.name in text for step in report.steps)


def test_summary_of_a_blank_page_says_blank():
    assert "blank" in ocr_cleaner.clean(blank_page()).summary()


def test_summary_of_a_photograph_says_photograph():
    assert "photograph" in ocr_cleaner.clean(photograph()).summary()


def test_headline_is_one_line(report):
    line = report.headline()
    assert "\n" not in line
    assert "document" in line


def test_headline_says_when_nothing_was_done():
    assert "left as it came" in ocr_cleaner.clean(blank_page()).headline()


def test_to_dict_is_json_safe_and_complete(report):
    payload = report.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    for key in (
        "page_kind", "steps", "applied", "skipped", "estimated_skew_degrees",
        "skew_corrected_degrees", "estimated_text_height_px", "threshold_mode",
        "binary", "source_size", "output_size", "notes",
    ):
        assert key in payload
    assert len(payload["steps"]) == len(ocr_cleaner.STEP_NAMES)
    assert "image" not in payload


def test_to_json_keeps_non_ascii_as_it_is(report):
    report.source = "ページ-1.png"
    try:
        text = report.to_json()
        assert "ページ" in text
        assert json.loads(text)["source"] == "ページ-1.png"
    finally:
        report.source = "<image>"


def test_to_dict_rounds_but_does_not_lose_the_numbers():
    result = ocr_cleaner.clean(text_page(), dpi=150, upscale_to_dpi=300)
    payload = result.to_dict()
    assert payload["dpi"] == 150.0
    assert payload["output_dpi"] == 300.0
    assert payload["estimated_text_height_px"] > 0


def test_none_dpi_stays_none(report):
    assert report.to_dict()["dpi"] is None
    assert report.to_dict()["output_dpi"] is None


def test_save_writes_and_makes_parent_directories(tmp_path, report):
    destination = tmp_path / "deep" / "down" / "page.png"
    written = report.save(destination)
    assert destination.exists()
    assert written == str(destination)
    assert Image.open(destination).size == report.output_size
    report.destination = None


def test_save_accepts_format_options(tmp_path, report):
    destination = tmp_path / "page.jpg"
    report.save(destination, quality=60)
    assert destination.exists()
    report.destination = None


def test_a_saved_page_says_where_it_went_in_the_summary(tmp_path):
    result = ocr_cleaner.clean(text_page())
    result.save(tmp_path / "out.png")
    assert "written to" in result.summary()


def test_the_result_can_be_built_by_hand():
    """It is a plain dataclass, so a caller can construct one in a test of their own."""
    blank = CleanResult(image=Image.new("L", (4, 4), 255))
    assert blank.applied == []
    assert blank.skipped == []
    assert blank.step("deskew") is None
    assert blank.to_dict()["page_kind"] == "document"
    assert blank.headline()


def test_binary_and_changed_follow_the_steps():
    grey = ocr_cleaner.clean(text_page(), threshold="none")
    assert not grey.binary
    assert not grey.changed
    binary = ocr_cleaner.clean(text_page())
    assert binary.binary
    assert binary.changed


def test_sizes_are_reported_as_width_then_height():
    page = text_page(500, 300, pitch=18, bar=6, left=30, right=270, top=60)
    result = ocr_cleaner.clean(page, threshold="none")
    assert result.source_size == (300, 500)
    assert result.output_size == (300, 500)
    assert result.image.size == result.output_size


def test_repr_says_the_useful_things(report):
    text = repr(report)
    assert "CleanResult" in text and "document" in text
