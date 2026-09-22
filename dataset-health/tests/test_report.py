"""Issue, HealthReport, the scoring table and the three renderers."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from dataset_health import KINDS, SEVERITIES, HealthReport, Issue, diagnose
from dataset_health._report import compute_score, grade_for, json_safe, penalty_for, sort_issues


def make_issue(severity="warning", kind="missing", columns=("a",), message="something"):
    return Issue(severity, kind, list(columns), message, {"share": 0.5})


def test_issue_basics():
    issue = make_issue()
    assert issue.to_dict() == {
        "severity": "warning",
        "kind": "missing",
        "columns": ["a"],
        "message": "something",
        "detail": {"share": 0.5},
    }
    assert str(issue) == "[warning] missing: something"
    assert issue.penalty == penalty_for("missing", "warning")


def test_issue_validates_severity_and_kind():
    with pytest.raises(ValueError, match="severity"):
        Issue("fatal", "missing", [], "no such severity")
    with pytest.raises(ValueError, match="kind"):
        Issue("warning", "", [], "no kind")


def test_issue_columns_and_detail_are_normalised():
    issue = Issue("info", "missing", [0, "b"], "m", {"n": np.int64(3), "share": np.float64(0.25)})
    assert issue.columns == ["0", "b"]
    assert issue.detail == {"n": 3, "share": 0.25}
    assert isinstance(issue.detail["n"], int)


def test_json_safe_handles_numpy_pandas_and_nan():
    value = json_safe(
        {
            np.int64(1): np.float64(2.5),
            "nan": np.float64("nan"),
            "inf": float("inf"),
            "when": pd.Timestamp("2024-01-02T03:04:05"),
            "nat": pd.NaT,
            "flag": np.bool_(True),
            "arr": np.array([1, 2]),
            "raw": b"bytes",
            "delta": pd.Timedelta(days=1),
        }
    )
    assert value["1"] == 2.5
    assert value["nan"] is None
    assert value["inf"] is None
    assert value["when"].startswith("2024-01-02")
    assert value["nat"] is None
    assert value["flag"] is True
    assert value["arr"] == [1, 2]
    assert value["raw"] == "bytes"
    assert json.dumps(value)  # round-trips


def test_json_safe_keeps_plain_dates():
    assert json_safe(dt.date(2024, 5, 6)) == "2024-05-06"


def test_score_starts_at_100_and_floors_at_0():
    assert compute_score([]) == 100
    assert compute_score([make_issue()]) == 100 - int(penalty_for("missing", "warning"))
    many = [Issue("critical", kind, [], "m") for kind in KINDS for _ in range(5)]
    assert compute_score(many) == 0


def test_repeats_of_one_kind_cost_less_each_time():
    one = compute_score([make_issue(columns=["a"])])
    four = compute_score([make_issue(columns=[c]) for c in "abcd"])
    assert four < one
    # base * (1 + log2(4)) == 3 * base, not 4 * base
    assert (100 - four) == pytest.approx(3 * (100 - one), abs=1)


def test_grades():
    assert grade_for(100) == "healthy"
    assert grade_for(90) == "healthy"
    assert grade_for(80) == "mostly healthy"
    assert grade_for(60) == "needs attention"
    assert grade_for(0) == "unhealthy"


def test_sort_issues_puts_critical_first_then_kind_order():
    issues = [
        make_issue("info", "skew", ["z"]),
        make_issue("critical", "missing", ["b"]),
        make_issue("warning", "duplicates", []),
        make_issue("critical", "target_leakage", ["a"]),
    ]
    ordered = [(i.severity, i.kind) for i in sort_issues(issues)]
    assert ordered == [
        ("critical", "target_leakage"),
        ("critical", "missing"),
        ("warning", "duplicates"),
        ("info", "skew"),
    ]


def build_report(issues=None, **kwargs):
    defaults = dict(
        issues=list(issues or []),
        n_rows=10,
        n_columns=2,
        n_rows_analyzed=10,
        sampled=False,
        target=None,
        task=None,
    )
    defaults.update(kwargs)
    return HealthReport(**defaults)


def test_report_computes_its_own_score_and_views():
    report = build_report(
        [
            make_issue("critical", "target_leakage", ["a"]),
            make_issue("warning", "missing", ["b"]),
            make_issue("info", "skew", ["c"]),
        ]
    )
    assert report.score == 100 - 30 - 4 - 1
    assert [i.kind for i in report.critical] == ["target_leakage"]
    assert [i.kind for i in report.warnings] == ["missing"]
    assert [i.kind for i in report.info] == ["skew"]
    assert report.counts == {"critical": 1, "warning": 1, "info": 1}
    assert report.kinds == ["target_leakage", "missing", "skew"]
    assert report.by_kind("missing")[0].columns == ["b"]
    assert report.by_column("c")[0].kind == "skew"
    assert report.by_column("nope") == []


def test_report_accepts_an_explicit_score_and_clamps_it():
    assert build_report(score=42).score == 42
    assert build_report(score=-5).score == 0
    assert build_report(score=500).score == 100


def test_summary_groups_by_severity_and_is_ascii(kitchen_sink):
    text = diagnose(kitchen_sink, target="label").summary()
    assert "CRITICAL" in text and "WARNING" in text and "INFO" in text
    assert text.index("CRITICAL") < text.index("WARNING") < text.index("INFO")
    assert text.isascii()
    assert str(diagnose(kitchen_sink, target="label")) == text


def test_summary_of_a_healthy_frame(clean):
    text = diagnose(clean).summary()
    assert "rows: 200 | columns: 4" in text


def test_to_dict_is_json_serialisable(kitchen_sink):
    payload = diagnose(kitchen_sink, target="label").to_dict()
    assert set(payload) == {
        "score",
        "grade",
        "n_rows",
        "n_columns",
        "n_rows_analyzed",
        "sampled",
        "target",
        "task",
        "counts",
        "issues",
        "columns",
        "notes",
    }
    assert payload["target"] == "label"
    assert payload["columns"]["age"]["role"] == "feature"
    assert payload["columns"]["label"]["role"] == "target"
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text) == payload


def test_to_markdown_renders_the_tables(kitchen_sink):
    md = diagnose(kitchen_sink, target="label").to_markdown()
    assert md.startswith("# dataset-health report")
    assert "| severity | kind | columns | message |" in md
    assert "## Columns" in md
    assert "`label`" in md


def test_to_markdown_says_so_when_there_is_nothing_to_report():
    md = build_report().to_markdown()
    assert "No issues found." in md
    assert "**Score: 100/100 (healthy)**" in md


def test_markdown_escapes_pipes_in_column_names():
    df = pd.DataFrame({"a|b": ["x"] * 20, "n": range(20)})
    md = diagnose(df).to_markdown()
    assert "a\\|b" in md


def test_notes_reach_every_renderer():
    df = pd.DataFrame({"a": range(300), "b": [i % 3 for i in range(300)]})
    report = diagnose(df, sample=50)
    assert report.notes
    assert "notes:" in report.summary()
    assert "## Notes" in report.to_markdown()
    assert report.to_dict()["notes"] == report.notes


def test_every_severity_and_kind_is_spelled_the_same_everywhere():
    assert len(set(KINDS)) == len(KINDS)
    for kind in KINDS:
        for severity in SEVERITIES:
            assert penalty_for(kind, severity) > 0
