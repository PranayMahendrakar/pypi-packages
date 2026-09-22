"""The result object: risk grading, and the three output formats staying JSON-safe."""
import json

import pandas as pd
import pytest

from privacy_scan_ml import ColumnFinding, Finding, PIIReport, scan
from privacy_scan_ml.report import risk_level


# --------------------------------------------------------------------------- risk


@pytest.mark.parametrize(
    "types, expected",
    [
        ([], "none"),
        (["aadhaar"], "high"),
        (["credit_card"], "high"),
        (["health_condition"], "high"),
        (["email", "phone"], "high"),
        (["date_of_birth", "postal_code", "gender"], "high"),
        (["email"], "medium"),
        (["ipv4", "url"], "medium"),
        (["url"], "low"),
    ],
)
def test_risk_grades(types, expected):
    assert risk_level(types) == expected


def test_risk_reads_off_the_scanned_frame():
    assert scan(pd.DataFrame({"sku": ["A-1"]})).risk == "none"
    assert scan(pd.DataFrame({"website": ["https://example.com"] * 3})).risk == "low"
    assert scan(pd.DataFrame({"aadhaar": ["234567890124"] * 3})).risk == "high"


# --------------------------------------------------------------------------- to_dict


def test_to_dict_is_json_serialisable(customers):
    payload = scan(customers).to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text) == payload
    for key in ["has_pii", "risk", "mode", "rows", "rows_scanned", "n_columns",
                "min_share", "types", "columns", "findings", "warnings"]:
        assert key in payload


def test_column_finding_to_dict_shape(customers):
    finding = scan(customers).columns["email"]
    payload = finding.to_dict()
    assert payload["column"] == "email" and payload["primary"] == "email"
    assert payload["types"]["email"] == 1.0
    assert len(payload["samples"]) <= 3
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["rows_checked"] == 3 and payload["hinted"] is True


def test_finding_to_dict_shape():
    payload = Finding("email", "a***@***.com", (5, 21), 2).to_dict()
    assert payload == {"type": "email", "value_masked": "a***@***.com",
                       "span": [5, 21], "doc": 2}


def test_types_lists_everything_found_once(customers):
    report = scan(customers)
    assert report.types == sorted(set(report.types))
    assert "email" in report.types and "aadhaar" in report.types


# --------------------------------------------------------------------------- text outputs


def test_summary_and_markdown_are_plain_ascii_punctuation(customers):
    report = scan(customers)
    for text in (report.summary(), report.to_markdown()):
        for banned in ("→", "•", "—", "┌", "‘", "“"):
            assert banned not in text, f"{banned!r} is not console-safe"


def test_summary_names_every_flagged_column(customers):
    summary = scan(customers).summary()
    for name in scan(customers).pii_columns:
        assert name in summary


def test_markdown_has_one_row_per_flagged_column(customers):
    report = scan(customers)
    md = report.to_markdown()
    body = [line for line in md.splitlines() if line.startswith("| ")]
    # one header row, one separator row, then the columns
    assert len(body) == len(report.columns) + 2


def test_an_empty_report_still_renders():
    report = PIIReport()
    assert report.has_pii is False and report.risk == "none"
    assert isinstance(report.summary(), str)
    assert isinstance(report.to_markdown(), str)
    assert json.dumps(report.to_dict())


def test_text_mode_markdown_lists_findings():
    report = scan(["mail asha@example.com", "call 9876543210"])
    md = report.to_markdown()
    assert "| document | type | masked value | span |" in md
    assert "email" in md and "phone" in md
    assert "asha@example.com" not in md and "9876543210" not in md


def test_warnings_reach_both_the_summary_and_the_markdown():
    report = scan(pd.DataFrame({"email": ["not-an-email"] * 3}))
    assert report.warnings
    assert "warning:" in report.summary()
    assert "## Warnings" in report.to_markdown()


def test_column_finding_can_be_built_directly():
    finding = ColumnFinding(column="x", types={"email": 1.0}, samples=["a***@***.com"],
                            confidence=0.9)
    assert finding.to_dict()["types"] == {"email": 1.0}
