"""The README quickstart, run verbatim, and the shape of the result it prints."""
from __future__ import annotations

import io
import json
import os
import re
from contextlib import redirect_stdout

import document_quality
from document_quality import BatchReport, Issue, PageReport

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def quickstart_blocks():
    """``(language, body)`` for each fenced block in the README's Quickstart."""
    with open(README, encoding="utf-8") as handle:
        text = handle.read()
    section = text.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"```(\w*)\n(.*?)\n```", section, re.S)


def quickstart_code() -> str:
    """The first python block under '## Quickstart' in the README."""
    code = [body for language, body in quickstart_blocks() if language == "python"]
    assert code, "README quickstart has no python block"
    return code[0] + "\n"


def test_readme_quickstart_runs_verbatim():
    buffer = io.StringIO()
    namespace: dict = {}
    with redirect_stdout(buffer):
        exec(compile(quickstart_code(), "README-quickstart", "exec"), namespace)
    printed = buffer.getvalue()
    report = namespace["report"]
    assert isinstance(report, PageReport)
    first = printed.splitlines()[0]
    assert "not ready to OCR" in first and "held back by skew" in first
    assert "deskew by 2.3 degrees clockwise" in printed
    assert abs(report.skew_degrees - 2.3) <= 0.5
    assert report.kind == "document"
    assert not report.ocr_ready


def test_readme_output_block_matches_what_the_quickstart_prints():
    shown = [body for language, body in quickstart_blocks() if not language][0].splitlines()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(quickstart_code(), "README-quickstart", "exec"), {})
    printed = buffer.getvalue().splitlines()
    # The first three lines and the issue block are shown in full.
    assert printed[:3] == shown[:3]
    assert printed[4:7] == shown[4:7]
    # The measure rows are shown cut short with "..."; what is shown must match.
    for line in shown[7:]:
        if line.strip() == "...":
            continue
        if line.endswith("..."):
            assert any(row.startswith(line[:-3]) for row in printed), line
        else:
            assert line in printed, line


def test_answer_in_three_lines(clean_report):
    lines = clean_report.summary().splitlines()
    assert lines[0].endswith("ready to OCR - score {0:.0f} of 100".format(clean_report.score))
    assert lines[1].startswith("Document page, 850 x 1100 px, 300 dpi")
    assert lines[2].startswith("Text lines about")
    assert clean_report.ocr_ready is True
    assert 0 <= clean_report.score <= 100
    assert clean_report.issues == []
    assert "Nothing to fix." in clean_report.summary()


def test_result_explains_itself(clean_report):
    for name, measure in clean_report.measures.items():
        assert measure.message, name
        if measure.score is not None:
            assert 0 <= measure.score <= 100
            assert measure.ok == (measure.score >= document_quality.PASS_SCORE - 1e-9)
    assert clean_report.explain("skew") == clean_report.measures["skew"].message
    assert set(clean_report.measures) >= {
        "resolution", "text_size", "skew", "contrast", "sharpness", "lighting",
        "show_through", "clipping", "border", "text_coverage",
    }


def test_to_dict_is_json_safe(clean_report):
    data = clean_report.to_dict()
    text = json.dumps(data, allow_nan=False)
    assert json.loads(text)["ocr_ready"] is True
    assert data["kind"] == "document"
    assert data["dpi"] == 300
    assert json.loads(clean_report.to_json())["score"] == data["score"]


def test_summary_is_plain_ascii(clean_report):
    clean_report.summary().encode("ascii")


def test_public_api_is_exported():
    for name in ("assess", "assess_batch", "estimate_skew", "detect_orientation",
                 "PageReport", "BatchReport", "Issue", "Thresholds",
                 "describe_thresholds", "__version__"):
        assert hasattr(document_quality, name), name
    assert document_quality.__version__ == "0.1.0"
    assert Issue("skew", "warning", "m", "f").to_dict()["fix"] == "f"
    assert isinstance(document_quality.assess_batch([]), BatchReport)
