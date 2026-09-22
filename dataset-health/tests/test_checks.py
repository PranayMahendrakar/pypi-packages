"""One test per check kind: it fires when it should and stays quiet when it should not."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_health import KINDS, HealthChecker, diagnose


def kinds_of(df, target=None, **kwargs):
    return set(diagnose(df, target=target, **kwargs).kinds)


def test_every_column_level_kind_can_fire(kitchen_sink):
    found = set(diagnose(kitchen_sink, target="label").kinds)
    expected = {
        "missing",
        "empty_column",
        "constant",
        "near_constant",
        "id_like",
        "high_cardinality",
        "mixed_types",
        "numeric_as_strings",
        "dates_as_strings",
        "class_imbalance",
        "target_leakage",
        "high_correlation",
        "outliers",
        "skew",
    }
    assert expected <= found
    assert found <= set(KINDS)


def test_missing_severity_climbs_with_the_share():
    df = pd.DataFrame(
        {
            "a_little": [None] + list(range(99)),          # 1%  -> info
            "a_lot": [None] * 10 + list(range(90)),        # 10% -> warning
            "mostly_gone": [None] * 80 + list(range(20)),  # 80% -> critical
        }
    )
    by_column = {i.columns[0]: i for i in diagnose(df).by_kind("missing") if i.columns}
    assert by_column["a_little"].severity == "info"
    assert by_column["a_lot"].severity == "warning"
    assert by_column["mostly_gone"].severity == "critical"
    assert by_column["a_lot"].detail["count"] == 10
    assert by_column["a_lot"].detail["share"] == pytest.approx(0.10)


def test_missing_reports_rows_that_are_mostly_empty():
    rows = [{"a": i, "b": i, "c": i, "d": i} for i in range(90)]
    rows += [{"a": i, "b": None, "c": None, "d": None} for i in range(10)]
    report = diagnose(pd.DataFrame(rows))
    row_issue = [i for i in report.by_kind("missing") if not i.columns]
    assert len(row_issue) == 1
    assert row_issue[0].detail["rows"] == 10
    assert "more than half" in row_issue[0].message


def test_duplicates():
    unique = pd.DataFrame({"a": range(100), "b": [f"v{i}" for i in range(100)]})
    assert "duplicates" not in kinds_of(unique)

    doubled = pd.concat([unique, unique.head(30)], ignore_index=True)
    issue = diagnose(doubled).by_kind("duplicates")[0]
    assert issue.detail["count"] == 30
    assert issue.severity == "critical"  # 30 of 130 rows is over 20%


def test_constant_and_near_constant():
    df = pd.DataFrame(
        {
            "one_value": ["x"] * 200,
            "almost_one": ["x"] * 199 + ["y"],
            "varied": [f"v{i % 5}" for i in range(200)],
        }
    )
    report = diagnose(df)
    assert [i.columns for i in report.by_kind("constant")] == [["one_value"]]
    assert [i.columns for i in report.by_kind("near_constant")] == [["almost_one"]]
    assert report.by_column("varied") == []


def test_id_like_text_integers_and_named_columns():
    n = 120
    df = pd.DataFrame(
        {
            "uuid": [f"u-{i:06d}" for i in range(n)],         # unique text
            "row_no": range(n),                               # consecutive integers
            "ticket_id": [10 * i + 3 for i in range(n)],      # increasing, named like an id
            "measurement": np.linspace(0.0, 1.0, n).round(6),  # unique floats, not an id
            "label": [i % 2 for i in range(n)],
        }
    )
    flagged = {issue.columns[0] for issue in diagnose(df, target="label").by_kind("id_like")}
    assert flagged == {"uuid", "row_no", "ticket_id"}


def test_high_cardinality_only_for_categoricals_that_are_not_ids():
    n = 400
    df = pd.DataFrame({"city": [f"c{i % 60}" for i in range(n)], "small": [f"s{i % 5}" for i in range(n)]})
    flagged = diagnose(df).by_kind("high_cardinality")
    assert [i.columns for i in flagged] == [["city"]]
    assert flagged[0].detail["n_unique"] == 60


def test_mixed_types_in_an_object_column():
    df = pd.DataFrame({"m": [1, "two", 3.0, "four"] * 25, "n": range(100)})
    issue = diagnose(df).by_kind("mixed_types")[0]
    assert issue.columns == ["m"]
    assert set(issue.detail["types"]) == {"int", "float", "str"}


def test_numbers_stored_as_text():
    df = pd.DataFrame({"amount": [str(i % 30) for i in range(100)], "n": range(100)})
    issue = diagnose(df).by_kind("numeric_as_strings")[0]
    assert issue.columns == ["amount"]
    assert issue.severity == "info"


def test_a_text_column_that_is_mostly_numbers_is_mixed_types():
    values = [str(i) for i in range(90)] + ["n/a"] * 10
    issue = diagnose(pd.DataFrame({"amount": values})).by_kind("mixed_types")[0]
    assert issue.detail["non_numeric_count"] == 10
    assert "n/a" in issue.detail["non_numeric_examples"]


def test_dates_as_strings():
    dates = pd.date_range("2020-01-01", periods=150).strftime("%Y-%m-%d").tolist()
    report = diagnose(pd.DataFrame({"signed_on": dates, "n": range(150)}))
    issue = report.by_kind("dates_as_strings")[0]
    assert issue.columns == ["signed_on"]
    assert issue.detail["parsed_share"] >= 0.9
    # real datetimes are not reported
    parsed = pd.DataFrame({"signed_on": pd.to_datetime(dates), "n": range(150)})
    assert "dates_as_strings" not in kinds_of(parsed)


def test_class_imbalance():
    balanced = pd.DataFrame({"x": range(200), "y": [0, 1] * 100})
    assert "class_imbalance" not in kinds_of(balanced, target="y")

    skewed = pd.DataFrame({"x": range(200), "y": [0] * 190 + [1] * 10})
    issue = diagnose(skewed, target="y").by_kind("class_imbalance")[0]
    assert issue.severity == "warning"
    assert issue.detail["minority_share"] == pytest.approx(0.05)
    assert issue.detail["n_classes"] == 2

    rare = pd.DataFrame({"x": range(1000), "y": [0] * 997 + [1] * 3})
    assert diagnose(rare, target="y").by_kind("class_imbalance")[0].severity == "critical"


def test_a_single_class_target_is_critical():
    df = pd.DataFrame({"x": np.linspace(0, 1, 50), "y": ["yes"] * 50})
    issue = diagnose(df, target="y").by_kind("class_imbalance")[0]
    assert issue.severity == "critical"
    assert "single class" in issue.message


def test_a_regression_target_has_no_imbalance_check():
    df = pd.DataFrame({"x": range(200), "y": np.linspace(0, 1000, 200)})
    assert "class_imbalance" not in kinds_of(df, target="y")


def test_leakage_by_correlation():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    df = pd.DataFrame({"copy": y * 3 + 2, "noise": rng.normal(size=200), "y": y})
    issue = diagnose(df, target="y").by_kind("target_leakage")[0]
    assert issue.columns == ["copy"]
    assert issue.severity == "critical"
    assert issue.detail["rule"] == "correlation"
    assert abs(issue.detail["correlation"]) > 0.95


def test_leakage_by_one_to_one_categorical_mapping():
    n = 200
    labels = ["spam", "ham"] * (n // 2)
    df = pd.DataFrame(
        {
            "folder": ["junk" if lab == "spam" else "inbox" for lab in labels],
            "noise": [f"n{i % 4}" for i in range(n)],
            "y": labels,
        }
    )
    issue = diagnose(df, target="y").by_kind("target_leakage")[0]
    assert issue.columns == ["folder"]
    assert issue.detail["rule"] == "mapping"


def test_leakage_by_name_is_only_a_warning():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"churn_reason_code": rng.normal(size=200), "churn": [0, 1] * 100})
    issue = diagnose(df, target="churn").by_kind("target_leakage")[0]
    assert issue.columns == ["churn_reason_code"]
    assert issue.severity == "warning"
    assert issue.detail["rule"] == "name"


def test_leakage_by_correlation_ratio_for_a_numeric_target():
    n = 300
    rng = np.random.default_rng(8)
    groups = [f"g{i % 6}" for i in range(n)]
    # continuous target (so the task is regression) that the group all but determines
    y = np.array([float(g[1]) * 100 for g in groups]) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"group": groups, "y": y})
    report = diagnose(df, target="y")
    assert report.task == "regression"
    issue = report.by_kind("target_leakage")[0]
    assert issue.columns == ["group"]
    assert issue.detail["rule"] == "correlation_ratio"
    assert issue.detail["correlation_ratio"] > 0.95


def test_no_leakage_without_a_target(clean):
    assert "target_leakage" not in kinds_of(clean)


def test_high_correlation_pairs():
    rng = np.random.default_rng(4)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "a_scaled": a * 10 + 1, "independent": rng.normal(size=300)})
    issue = diagnose(df).by_kind("high_correlation")[0]
    assert set(issue.columns) == {"a", "a_scaled"}
    assert issue.detail["correlation"] == pytest.approx(1.0, abs=1e-6)


def test_outliers_by_the_iqr_rule():
    rng = np.random.default_rng(6)
    values = np.concatenate([rng.normal(0, 1, 90), np.full(10, 50.0)])
    issue = diagnose(pd.DataFrame({"v": values})).by_kind("outliers")[0]
    assert issue.detail["count"] == 10
    assert issue.detail["share"] == pytest.approx(0.10)
    assert issue.detail["upper_fence"] < 50.0


def test_skew():
    values = list(np.linspace(0.0, 1.0, 190)) + [500.0] * 10
    issue = diagnose(pd.DataFrame({"v": values})).by_kind("skew")[0]
    assert issue.detail["skew"] > 3
    assert issue.severity == "info"


def test_empty_columns_and_rows():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": [None] * 4, "c": ["x", None, "y", "z"]})
    report = diagnose(df)
    assert [i.columns for i in report.by_kind("empty_column")] == [["b"]]
    row_issue = report.by_kind("empty_row")[0]
    assert row_issue.detail["count"] == 1
    assert row_issue.columns == []


def test_an_empty_target_column_is_critical():
    df = pd.DataFrame({"x": range(10), "y": [None] * 10})
    issue = diagnose(df, target="y").by_kind("empty_column")[0]
    assert issue.severity == "critical"
    assert issue.detail["role"] == "target"


def test_checks_are_stable_across_runs(kitchen_sink):
    first = diagnose(kitchen_sink, target="label").to_dict()
    second = HealthChecker().check(kitchen_sink, "label").to_dict()
    assert first == second
