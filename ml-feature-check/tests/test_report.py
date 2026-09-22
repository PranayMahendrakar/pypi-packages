"""The FeatureReport surface: ordering, apply, summary, to_dict, to_markdown."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from ml_feature_check import Finding, check


def test_drop_recommended_is_ordered_by_severity():
    df = pd.DataFrame(
        {
            "near_flat": [1] * 199 + [0],       # medium
            "identifier": list(range(200)),      # high
            "flat": ["x"] * 200,                 # high
            "fine": np.cos(np.linspace(0.0, 12.0, 200)),
        }
    )
    report = check(df)

    severities = [
        min(f.severity for f in report.features[c] if f.severity in ("high", "medium"))
        for c in report.drop_recommended
    ]
    assert severities == sorted(severities, key=["high", "medium"].index)
    assert report.drop_recommended[-1] == "near_flat"
    assert report.keep == ["fine"]


def test_findings_within_a_column_are_worst_first():
    df = pd.DataFrame({"row_id": list(range(50)), "v": [1.0, 2.0] * 25})
    report = check(df)
    order = [f.severity for f in report.features["row_id"]]
    assert order == sorted(order, key=["high", "medium", "low"].index)
    assert order[0] == "high"


def test_apply_removes_only_the_recommended_columns(quickstart_frame):
    report = check(quickstart_frame, target="churned")
    clean = report.apply(quickstart_frame)

    assert list(clean.columns) == report.keep + ["churned"]
    assert len(clean) == len(quickstart_frame)
    assert clean is not quickstart_frame


def test_apply_tolerates_a_frame_that_already_lost_a_column(quickstart_frame):
    report = check(quickstart_frame, target="churned")
    smaller = quickstart_frame.drop(columns=["country"])
    clean = report.apply(smaller)
    assert "country" not in clean.columns
    assert "age" in clean.columns


def test_apply_accepts_a_csv_path(tmp_path, quickstart_frame):
    path = tmp_path / "q.csv"
    quickstart_frame.to_csv(path, index=False, encoding="utf-8")
    report = check(quickstart_frame, target="churned")
    clean = report.apply(str(path))
    assert "row_id" not in clean.columns


def test_summary_is_plain_ascii_punctuation(quickstart_frame):
    report = check(quickstart_frame, target="churned")
    text = report.summary()

    assert "Drop recommended (4):" in text
    assert "Keep (2): age, signed_up" in text
    assert "Notes:" in text
    for bad in ("→", "•", "—", "┌"):
        assert bad not in text
    assert str(report) == text


def test_summary_when_nothing_is_wrong():
    df = pd.DataFrame({"a": np.linspace(0, 1, 50), "b": np.cos(np.linspace(0, 6, 50))})
    report = check(df)
    text = report.summary()
    assert "Drop recommended (0): none" in text
    assert report.drop_recommended == []
    assert report.flagged == []
    assert report.review == []


def test_to_dict_is_json_safe(quickstart_frame):
    report = check(quickstart_frame, target="churned")
    payload = report.to_dict()
    round_tripped = json.loads(json.dumps(payload, ensure_ascii=False))

    assert round_tripped["target"] == "churned"
    assert round_tripped["n_rows"] == 60
    assert round_tripped["n_features"] == 6
    assert round_tripped["drop_recommended"] == [str(c) for c in report.drop_recommended]
    assert set(round_tripped["features"]) == {str(c) for c in report.features}
    assert round_tripped["params"]["corr_threshold"] == 0.95
    finding = round_tripped["features"]["country"][0]
    assert set(finding) == {"kind", "severity", "message", "detail"}


def test_to_markdown_has_a_row_per_feature(quickstart_frame):
    report = check(quickstart_frame, target="churned")
    text = report.to_markdown()

    assert text.startswith("# ml-feature-check report")
    assert "| Column | Verdict | Findings |" in text
    body = [line for line in text.splitlines() if line.startswith("| `")]
    assert len(body) == report.n_features
    assert any("drop" in line and "row_id" in line for line in body)
    assert any("review" in line and "signed_up" in line for line in body)


def test_to_markdown_escapes_pipes():
    df = pd.DataFrame({"a|b": ["x"] * 30, "v": list(range(30))})
    report = check(df)
    line = [ln for ln in report.to_markdown().splitlines() if ln.startswith("| `a")][0]
    assert "a\\|b" in line


def test_accessors():
    df = pd.DataFrame(
        {
            "row_id": list(range(60)),
            "flat": ["x"] * 60,
            "good": np.cos(np.linspace(0.0, 12.0, 60)),
        }
    )
    report = check(df)

    assert report.n_features == 3
    assert set(report.flagged) == {"row_id", "flat"}
    assert report.review == []
    assert report.columns_with("constant") == ["flat"]
    assert report.columns_with("id_like") == ["row_id"]
    assert report.columns_with("duplicate_of") == []

    pairs = list(report.iter_findings())
    assert all(isinstance(f, Finding) for _, f in pairs)
    order = [f.severity for _, f in pairs]
    assert order == sorted(order, key=["high", "medium", "low"].index)

    assert "FeatureReport(features=3" in repr(report)


def test_finding_to_dict():
    finding = Finding("constant", "high", "one value", {"value": "x"})
    assert finding.to_dict() == {
        "kind": "constant",
        "severity": "high",
        "message": "one value",
        "detail": {"value": "x"},
    }


def test_review_lists_kept_columns_that_still_have_a_note():
    df = pd.DataFrame(
        {
            "signed_up": ["2024-01-05", "2024-02-11", "2024-03-18"] * 20,
            "v": np.linspace(0.0, 1.0, 60),
        }
    )
    report = check(df)
    assert report.review == ["signed_up"]
    assert "signed_up" in report.keep
    assert "Worth a look (1):" in report.summary()


# --------------------------------------------------------------------- QA round 1


def test_summary_keeps_a_whitespace_only_column_name_visible():
    """A blank-looking name must not read as the previous column's finding."""
    df = pd.DataFrame(
        {"col with spaces": [1, 2, 3, 4], "u": ["a", "b", "a", "b"], "  ": [3, 3, 3, 3], "y": [0, 1, 0, 1]}
    )
    report = check(df, target="y")
    text = report.summary()

    assert "'  '" in text
    constant = [line for line in text.splitlines() if "constant [high]" in line][0]
    assert constant.lstrip().startswith("'  '")
    assert not text.rstrip().endswith(",")
    assert "Keep (1): col with spaces" in text


def test_summary_keeps_an_empty_column_name_visible():
    df = pd.DataFrame({"": [1, 1, 1, 1], "x": [1.0, 2.0, 3.0, 4.0]})
    report = check(df)
    assert "''" in report.summary()


def test_apply_warns_when_the_frame_shares_no_column(caplog):
    report = check(
        pd.DataFrame({"id": [f"c{i}" for i in range(40)], "k": [1] * 40, "y": [0, 1] * 20}),
        target="y",
    )
    assert report.drop_recommended == ["id", "k"]

    with caplog.at_level(logging.WARNING, logger="ml_feature_check"):
        out = report.apply(pd.DataFrame({"zzz": [1, 2]}))

    assert list(out.columns) == ["zzz"]
    assert "shares no column" in caplog.text


def test_apply_warns_when_a_recommended_column_is_absent(caplog):
    report = check(
        pd.DataFrame({"id": [f"c{i}" for i in range(40)], "k": [1] * 40, "y": [0, 1] * 20}),
        target="y",
    )
    frame = pd.DataFrame({"id": [f"c{i}" for i in range(4)], "y": [0, 1, 0, 1]})

    with caplog.at_level(logging.WARNING, logger="ml_feature_check"):
        out = report.apply(frame)

    assert list(out.columns) == ["y"]
    assert "could not drop" in caplog.text
    assert "k" in caplog.text


def test_apply_on_the_checked_frame_is_silent(caplog, quickstart_frame):
    report = check(quickstart_frame, target="churned")
    with caplog.at_level(logging.WARNING, logger="ml_feature_check"):
        report.apply(quickstart_frame)
    assert caplog.text == ""


# --------------------------------------------------------------------- QA round 2


@pytest.mark.parametrize("bad", ["nope", 42, ["a", "b"], {"a": 1}])
def test_apply_on_a_non_frame_names_what_apply_wants(bad):
    """apply() is documented as taking a frame, so it must not talk about file types.

    A bare string used to be run through the file loader, which told the caller
    their file type was '' when they never thought they were naming a file.
    """
    report = check(pd.DataFrame({"a": [1, 2, 3] * 10, "t": [0, 1, 0] * 10}))

    with pytest.raises(TypeError) as excinfo:
        report.apply(bad)
    message = str(excinfo.value)
    assert "apply() expects a pandas DataFrame" in message
    assert type(bad).__name__ in message
    assert "unsupported file type" not in message


def test_apply_still_reports_a_genuinely_bad_path():
    report = check(pd.DataFrame({"a": [1, 2, 3] * 10, "t": [0, 1, 0] * 10}))
    with pytest.raises(TypeError, match="unsupported file type"):
        report.apply("data.xlsx")
