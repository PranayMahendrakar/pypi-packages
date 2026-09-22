"""The one-call path: diagnose(), the report it returns, targets and sampling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import dataset_health
from dataset_health import KINDS, SEVERITIES, HealthChecker, HealthReport, Issue, diagnose


def test_quickstart_from_the_readme(messy):
    report = diagnose(messy, target="churn")

    assert isinstance(report, HealthReport)
    assert report.score == 56
    assert report.grade == "needs attention"
    assert report.n_rows == 20
    assert report.n_columns == 5
    assert report.n_rows_analyzed == 20
    assert report.sampled is False
    assert report.target == "churn"
    assert report.task == "classification"

    text = report.summary()
    assert "dataset-health: score 56/100 (needs attention)" in text
    assert "1 critical, 4 warning, 0 info" in text
    assert "will_churn" in text

    kinds = {issue.kind for issue in report.issues}
    assert kinds == {"target_leakage", "missing", "id_like", "constant", "dates_as_strings"}


def test_package_surface():
    assert dataset_health.__version__ == "0.1.0"
    assert set(dataset_health.__all__) <= set(dir(dataset_health))
    assert SEVERITIES == ("critical", "warning", "info")
    assert {"missing", "duplicates", "target_leakage", "class_imbalance"} <= set(KINDS)


def test_clean_frame_scores_high(clean):
    report = diagnose(clean, target="label")
    assert report.score >= 90
    assert report.grade == "healthy"
    assert report.critical == []


def test_a_frame_with_no_findings_scores_100():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=300).round(3), "b": rng.normal(size=300).round(3)})
    report = diagnose(df)
    assert report.issues == []
    assert report.score == 100
    assert report.summary().startswith("dataset-health: score 100/100 (healthy) - no issues found")


def test_input_is_never_modified(messy):
    before = messy.copy(deep=True)
    diagnose(messy, target="churn")
    pd.testing.assert_frame_equal(messy, before)


def test_target_must_be_a_column(messy):
    with pytest.raises(ValueError) as excinfo:
        diagnose(messy, target="not_here")
    message = str(excinfo.value)
    assert "not_here" in message
    assert "churn" in message  # the message lists the columns that do exist


def test_target_is_optional(messy):
    report = diagnose(messy)
    assert report.target is None
    assert report.task is None
    assert report.by_kind("class_imbalance") == []
    assert report.by_kind("target_leakage") == []


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1] * 50, "classification"),          # two numeric values
        (list(range(20)) * 5, "classification"),  # exactly 20 distinct values
        (list(range(21)) * 5, "regression"),      # one too many
        (["yes", "no"] * 50, "classification"),   # non-numeric
        ([True, False] * 50, "classification"),   # boolean
        ([i * 0.5 for i in range(100)], "regression"),
    ],
)
def test_task_is_auto_detected(values, expected):
    df = pd.DataFrame({"x": np.arange(len(values), dtype=float), "y": values})
    assert diagnose(df, target="y").task == expected


def test_sampling_only_above_the_sample_size():
    df = pd.DataFrame({"a": range(500), "b": [i % 7 for i in range(500)]})

    whole = diagnose(df, sample=500)
    assert whole.sampled is False
    assert whole.n_rows_analyzed == 500
    assert not any("sample" in note for note in whole.notes)

    sampled = diagnose(df, sample=100)
    assert sampled.sampled is True
    assert sampled.n_rows == 500
    assert sampled.n_rows_analyzed == 100
    assert any("random sample of 100 of 500 rows" in note for note in sampled.notes)
    assert "analyzed: 100 rows (random sample)" in sampled.summary()


def test_sampling_is_deterministic():
    df = pd.DataFrame({"a": range(400), "b": np.random.default_rng(5).normal(size=400)})
    first = diagnose(df, sample=120, random_state=11).to_dict()
    again = diagnose(df, sample=120, random_state=11).to_dict()
    assert first == again
    other = diagnose(df, sample=120, random_state=12).to_dict()
    assert other["n_rows_analyzed"] == 120


def test_sample_none_and_zero_read_every_row():
    df = pd.DataFrame({"a": range(300)})
    for sample in (None, 0):
        report = diagnose(df, sample=sample)
        assert report.sampled is False
        assert report.n_rows_analyzed == 300


def test_sample_must_be_a_non_negative_int():
    with pytest.raises(ValueError, match="non-negative"):
        diagnose(pd.DataFrame({"a": [1, 2]}), sample=-5)


def test_health_checker_matches_diagnose(messy):
    checker = HealthChecker()
    assert checker.check(messy, "churn").to_dict() == diagnose(messy, target="churn").to_dict()
    assert checker(messy, "churn").score == diagnose(messy, target="churn").score  # callable


def test_health_checker_thresholds_move_the_findings():
    df = pd.DataFrame({"a": [1.0] * 96 + [2.0] * 4, "b": range(100)})
    assert HealthChecker(near_constant=0.99).check(df).by_kind("near_constant") == []
    loose = HealthChecker(near_constant=0.90).check(df).by_kind("near_constant")
    assert [issue.columns for issue in loose] == [["a"]]


def test_health_checker_can_run_a_subset_of_checks(kitchen_sink):
    report = HealthChecker(checks=["missing", "duplicates"]).check(kitchen_sink, "label")
    assert set(report.kinds) <= {"missing", "duplicates"}
    assert report.by_kind("missing")


def test_health_checker_rejects_unknown_checks():
    with pytest.raises(ValueError, match="unknown check kinds"):
        HealthChecker(checks=["missing", "not_a_check"])


def test_duplicate_column_names_raise_a_clear_error():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
    with pytest.raises(ValueError) as excinfo:
        diagnose(df)
    assert "duplicate column names" in str(excinfo.value)
    assert "'a'" in str(excinfo.value)


def test_non_text_column_names_are_converted_and_noted():
    df = pd.DataFrame({0: [1, 2, 3], 1: ["a", "b", "c"]})
    report = diagnose(df)
    assert set(report.columns) == {"0", "1"}
    assert any("not text" in note for note in report.notes)


def test_a_bad_input_type_is_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        diagnose(42)


def test_the_library_never_prints(capsys, kitchen_sink, caplog):
    with caplog.at_level(logging.DEBUG, logger="dataset_health"):
        report = diagnose(kitchen_sink, target="label", sample=50)
    report.summary()
    report.to_dict()
    report.to_markdown()
    assert capsys.readouterr().out == ""
    assert any("sample" in record.message for record in caplog.records)


def test_issues_are_sorted_critical_first(kitchen_sink):
    report = diagnose(kitchen_sink, target="label")
    ranks = [SEVERITIES.index(issue.severity) for issue in report.issues]
    assert ranks == sorted(ranks)
    assert all(isinstance(issue, Issue) for issue in report.issues)
