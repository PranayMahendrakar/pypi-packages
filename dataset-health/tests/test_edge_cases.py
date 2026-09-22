"""Degenerate and awkward inputs: empty frames, one row, one column, odd dtypes, unicode."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from dataset_health import HealthChecker, diagnose


def render_all(report):
    """Every renderer must survive whatever the report holds."""
    assert isinstance(report.summary(), str)
    assert isinstance(report.to_markdown(), str)
    assert isinstance(report.to_dict(), dict)
    return report


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[0, 1, 2]),          # rows but no columns
        pd.DataFrame({"a": [], "b": []}),        # columns but no rows
    ],
    ids=["nothing", "no-columns", "no-rows"],
)
def test_empty_datasets_report_one_critical_issue(df):
    report = render_all(diagnose(df))
    assert [i.kind for i in report.issues] == ["empty_dataset"]
    assert report.score == 0
    assert report.critical


def test_a_single_row():
    report = render_all(diagnose(pd.DataFrame({"a": [1], "b": ["x"], "c": [None]})))
    assert report.n_rows == 1
    # "every column has one value" is vacuous on one row, so it is a note, not an issue
    assert report.by_kind("constant") == []
    assert report.by_kind("near_constant") == []
    assert any("single row" in note for note in report.notes)


def test_a_single_column():
    report = render_all(diagnose(pd.DataFrame({"only": list(range(60))})))
    assert report.n_columns == 1
    assert report.by_kind("high_correlation") == []
    assert report.by_kind("id_like")[0].columns == ["only"]


def test_a_single_column_that_is_also_the_target():
    report = render_all(diagnose(pd.DataFrame({"y": [0, 1] * 30}), target="y"))
    assert report.target == "y"
    assert report.task == "classification"
    assert report.by_kind("target_leakage") == []
    assert any("only column" in note for note in report.notes)


def test_an_all_nan_column():
    df = pd.DataFrame({"blank": [np.nan] * 30, "n": range(30)})
    report = render_all(diagnose(df))
    assert [i.columns for i in report.by_kind("empty_column")] == [["blank"]]
    assert report.by_kind("missing") == []  # reported once, as an empty column
    assert report.columns["blank"]["missing_share"] == 1.0


def test_an_all_none_object_column():
    df = pd.DataFrame({"o": pd.Series([None] * 20, dtype=object), "n": range(20)})
    assert render_all(diagnose(df)).by_kind("empty_column")[0].columns == ["o"]


def test_an_all_numeric_dataset():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(150, 5)).round(4), columns=[f"f{i}" for i in range(5)])
    report = render_all(diagnose(df, target="f0"))
    assert report.task == "regression"
    assert all(profile["type"] == "numeric" for profile in report.columns.values())


def test_an_all_categorical_dataset():
    df = pd.DataFrame({f"c{j}": [f"v{(i + j) % 4}" for i in range(120)] for j in range(4)})
    report = render_all(diagnose(df, target="c0"))
    assert report.task == "classification"
    assert all(profile["type"] == "categorical" for profile in report.columns.values())
    assert report.by_kind("high_correlation") == []  # nothing numeric to correlate


def test_mixed_dtypes_in_one_frame():
    n = 60
    df = pd.DataFrame(
        {
            "i": pd.array(range(n), dtype="Int64"),
            "f": np.linspace(0, 1, n),
            "b": [True, False] * (n // 2),
            "s": pd.array([f"t{i % 3}" for i in range(n)], dtype="string"),
            "cat": pd.Categorical([f"c{i % 4}" for i in range(n)]),
            "when": pd.date_range("2024-01-01", periods=n),
            "span": pd.to_timedelta(np.arange(n), unit="h"),
            "obj": [{"k": i} for i in range(n)],
        }
    )
    report = render_all(diagnose(df))
    types = {name: profile["type"] for name, profile in report.columns.items()}
    assert types["i"] == "numeric"
    assert types["b"] == "boolean"
    assert types["cat"] == "categorical"
    assert types["when"] == "datetime"
    assert types["span"] == "datetime"


def test_unhashable_cells_do_not_break_anything():
    df = pd.DataFrame({"lists": [[1, 2], [3], [1, 2]] * 10, "n": range(30)})
    report = render_all(diagnose(df))
    assert report.columns["lists"]["n_unique"] == 2


def test_unicode_column_names_and_values():
    df = pd.DataFrame(
        {
            "ciudad": ["München", "北京", "São Paulo", "Köln"] * 15,
            "café": [1.5, 2.5, 3.5, 4.5] * 15,
            "étiquette": [0, 1] * 30,
        }
    )
    report = render_all(diagnose(df, target="étiquette"))
    assert "ciudad" in report.columns
    text = report.summary()
    assert "ciudad" in text
    # the report's own punctuation stays plain ASCII; only the data carries non-Latin text
    assert not set(text) & set("→•─—‘’“”")
    assert report.to_dict()["target"] == "étiquette"
    assert "étiquette" in report.to_markdown()


def test_infinities_are_tolerated():
    values = [np.inf, -np.inf] + list(np.linspace(0, 1, 48))
    report = render_all(diagnose(pd.DataFrame({"v": values})))
    detail = report.to_dict()
    assert detail["score"] <= 100
    for issue in report.issues:
        for value in issue.detail.values():
            assert value != float("inf")


def test_a_constant_target_column_of_one_value():
    df = pd.DataFrame({"x": range(40), "y": [7] * 40})
    report = render_all(diagnose(df, target="y"))
    assert report.by_kind("class_imbalance")[0].severity == "critical"


def test_two_rows_is_enough_for_the_constant_check():
    report = diagnose(pd.DataFrame({"a": ["x", "x"], "b": [1, 2]}))
    assert [i.columns for i in report.by_kind("constant")] == [["a"]]


def test_duplicate_rows_of_unhashable_cells():
    df = pd.DataFrame({"l": [[1], [1], [2]] * 10})
    assert render_all(diagnose(df)).by_kind("duplicates")


def test_a_series_is_accepted():
    report = render_all(diagnose(pd.Series(range(40), name="s")))
    assert report.n_columns == 1
    assert "s" in report.columns


def test_no_warnings_are_raised_on_awkward_input(kitchen_sink):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diagnose(kitchen_sink, target="label")
        diagnose(pd.DataFrame({"a": [1]}))
        diagnose(pd.DataFrame())


def test_wide_frames_stay_fast():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(200, 120)).round(3), columns=[f"f{i}" for i in range(120)])
    report = render_all(HealthChecker(high_corr=0.99).check(df))
    assert report.n_columns == 120


def test_sampling_keeps_the_original_row_order():
    df = pd.DataFrame({"n": range(1000), "grp": [i % 4 for i in range(1000)]})
    report = diagnose(df, sample=200, random_state=3)
    assert report.sampled is True
    # a monotonically increasing column still reads as increasing after sampling
    assert report.by_kind("id_like")[0].columns == ["n"]
