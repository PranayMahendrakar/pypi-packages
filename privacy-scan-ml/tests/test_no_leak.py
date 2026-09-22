"""The hard promise: a report describes personal data, it never prints it back."""
import json
import re

import pandas as pd

from privacy_scan_ml import scan

# Every value below is invented, and every one of them must stay out of the report.
LEAKY = pd.DataFrame(
    {
        "full_name": ["Asha Ramanathan", "Vikram Nair", "Meera Iyer"],
        "email": ["asha.ramanathan@example.com", "vikram@example.org", "meera@example.net"],
        "phone": ["9876543210", "9123456780", "9988776655"],
        "aadhaar": ["234567890124", "491039587666", "876543210988"],
        "pan": ["ABCPD1234E", "XYZPK9876F", "AAACT1234Z"],
        "card_number": ["4111111111111111", "5555555555554444", "6011000990139424"],
        "ip_address": ["192.168.101.77", "203.0.113.45", "198.51.100.22"],
        "website": ["https://ashastudio.example.com", "www.nairworks.example.org",
                    "http://iyer.example.net/profile"],
        "dob": ["1990-05-14", "1985-02-14", "1977-03-03"],
        "pincode": ["560087", "110011", "400051"],
        "address": ["12 Brigade Road, Bengaluru", "Flat 4B, 22 Park Street",
                    "742 Evergreen Terrace"],
        "notes": ["reach Asha on 9876543210", "billed to 4111111111111111", "no contact given"],
    }
)

DIGIT_RUN = 5  # a masked preview may reveal the last four characters, never more


def _report_text(report) -> str:
    """Everything the report can show a human, concatenated."""
    return "\n".join(
        [report.summary(), json.dumps(report.to_dict(), ensure_ascii=False), report.to_markdown()]
    )


def _digit_runs(value: str, size: int = DIGIT_RUN):
    """Every run of `size` consecutive digits inside a value."""
    for block in re.findall(r"\d+", value):
        for i in range(len(block) - size + 1):
            yield block[i : i + size]


def test_no_input_value_appears_anywhere_in_the_report():
    report = scan(LEAKY)
    text = _report_text(report)
    assert report.has_pii and report.columns, "the fixture must actually be flagged"
    for column in LEAKY.columns:
        for value in LEAKY[column]:
            assert value not in text, f"{column}: the whole value leaked into the report"
            for run in _digit_runs(value):
                assert run not in text, f"{column}: {DIGIT_RUN} digits of a value leaked"


def test_no_input_value_appears_in_a_free_text_report():
    docs = list(LEAKY["notes"]) + list(LEAKY["email"]) + list(LEAKY["address"])
    report = scan(docs)
    text = _report_text(report)
    for doc in docs:
        assert doc not in text
        for run in _digit_runs(doc):
            assert run not in text


def test_samples_are_masked_not_raw():
    report = scan(LEAKY)
    for name, finding in report.columns.items():
        raw = set(LEAKY[name].astype(str))
        for sample in finding.samples:
            assert sample not in raw, f"{name}: a raw value was used as a sample"
            assert "*" in sample or "[" in sample, f"{name}: sample {sample!r} looks unmasked"


def test_markdown_is_a_table_and_escapes_pipes():
    df = pd.DataFrame({"weird|name": ["asha@example.com", "vikram@example.org"]})
    md = scan(df).to_markdown()
    assert md.startswith("# privacy-scan-ml report")
    assert "| --- |" in md
    assert r"weird\|name" in md, "an unescaped pipe would break the table"
    assert "asha@example.com" not in md


def test_markdown_and_summary_survive_an_empty_report():
    report = scan(pd.DataFrame({"sku": ["A-1", "A-2"], "units": [3, 9]}))
    assert not report.has_pii and report.risk == "none"
    assert "No personal data found." in report.to_markdown()
    assert "no personal data found" in report.summary()
