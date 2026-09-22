"""Degenerate frames, odd dtypes and the input validation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import all_kinds, kinds, messages
from ml_feature_check import FeatureChecker, check


def test_completely_empty_frame():
    report = check(pd.DataFrame())
    assert report.features == {}
    assert report.drop_recommended == []
    assert report.keep == []
    assert report.n_rows == 0
    assert any("no feature columns" in note for note in report.notes)
    assert report.summary()
    assert report.to_dict()["features"] == {}
    assert report.to_markdown().startswith("# ml-feature-check report")


def test_columns_but_no_rows():
    df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
    report = check(df)
    assert report.keep == ["a", "b"]
    assert report.drop_recommended == []
    assert all(report.features[c] == [] for c in ("a", "b"))
    assert any("no rows" in note for note in report.notes)
    assert report.apply(df).shape == (0, 2)


def test_single_row():
    df = pd.DataFrame({"a": [1], "b": ["x"], "c": [np.nan]})
    report = check(df)
    assert report.n_rows == 1
    # with one row every column has exactly one distinct value
    assert all("constant" in kinds(report, c) for c in ("a", "b", "c"))
    assert report.apply(df).shape == (1, 0)


def test_single_column_frame():
    df = pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0] * 10})
    report = check(df)
    assert report.n_features == 1
    assert report.features["only"] == []
    assert report.keep == ["only"]
    assert list(report.apply(df).columns) == ["only"]


def test_single_column_frame_that_is_also_the_target():
    df = pd.DataFrame({"y": [0, 1] * 20})
    report = check(df, target="y")
    assert report.features == {}
    assert report.n_features == 0
    assert any("no feature columns" in note for note in report.notes)
    assert list(report.apply(df).columns) == ["y"]


def test_target_is_never_a_feature():
    df = pd.DataFrame({"x": [1.0, 2.0] * 20, "y": [0, 1] * 20})
    report = check(df, target="y")
    assert "y" not in report.features
    assert "y" not in report.drop_recommended
    assert "y" not in report.keep
    assert "y" in report.apply(df).columns  # apply never removes the target


def test_without_a_target_the_label_checks_are_skipped_and_noted():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 20, "b": ["x", "y", "z"] * 20})
    report = check(df)

    assert report.target is None
    assert "leakage_suspect" not in all_kinds(report)
    assert "zero_importance" not in all_kinds(report)
    note = [n for n in report.notes if "no target given" in n]
    assert note and "leakage_suspect" in note[0] and "zero_importance" in note[0]
    assert "no target" in report.summary()


def test_datetime_columns_of_every_flavour():
    n = 40
    df = pd.DataFrame(
        {
            "naive": pd.date_range("2024-01-01", periods=n, freq="D"),
            "aware": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "repeated_day": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 8
            ),
            "gap": pd.to_timedelta(np.arange(n), unit="D"),
            "y": [0, 1] * (n // 2),
        }
    )
    report = check(df, target="y")

    assert "id_like" in kinds(report, "naive")
    assert "duplicate_of" in kinds(report, "aware")  # same instants as "naive"
    assert "leakage_suspect" not in kinds(report, "repeated_day")
    assert report.to_dict()  # timestamps in details stay JSON-safe


def test_datetime_with_missing_values():
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-01-01", None, "2024-01-03", "2024-01-04"] * 10),
            "v": [1.0, 2.0, 3.0, 4.0] * 10,
        }
    )
    report = check(df)
    assert "constant" not in kinds(report, "when")
    assert report.summary()


def test_object_column_with_nan_and_mixed_types():
    df = pd.DataFrame(
        {
            "messy": ["a", np.nan, 3, None, "a", pd.NA, "b", 3, "a", np.nan] * 4,
            "notes": ["one", "two", None, "three"] * 10,
            "v": list(range(40)),
        }
    )
    report = check(df)
    assert report.summary()
    payload = report.to_dict()
    import json

    json.dumps(payload)  # must be JSON-safe


def test_zero_variance_columns_emit_no_warning_and_no_nan_finding():
    df = pd.DataFrame(
        {
            "flat": [7.0] * 60,
            "flat_too": [2.0] * 60,
            "real": np.linspace(0.0, 1.0, 60),
            "almost_flat": [0.0] * 59 + [1.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any numpy divide/invalid warning fails the test
        report = check(df)

    for column, findings in report.features.items():
        for finding in findings:
            assert finding.kind != "highly_correlated_with", (column, finding.message)
            for value in finding.detail.values():
                if isinstance(value, float):
                    assert np.isfinite(value), (column, finding.detail)
    assert "constant" in kinds(report, "flat")


def test_correlation_against_an_all_nan_column_is_silent():
    df = pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, 40),
            "b": np.linspace(0.0, 1.0, 40) + 1.0,
            "nothing": [np.nan] * 40,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = check(df)
    assert "highly_correlated_with" in kinds(report, "b")
    assert kinds(report, "nothing") == {"constant"}


def test_unicode_column_names_and_values():
    df = pd.DataFrame(
        {
            "ciudad": ["Sao Paulo", "Munchen", "Zurich"] * 20,
            "説明": ["日本語"] * 60,
            "値": [1.0, 2.0, 3.0, 4.0] * 15,
        }
    )
    report = check(df)
    text = report.summary()
    assert "説明" in text
    assert report.to_markdown()
    import json

    assert "日本語" in json.dumps(report.to_dict(), ensure_ascii=False)


def test_mixed_dtypes_together():
    n = 40
    df = pd.DataFrame(
        {
            "i": np.arange(n, dtype="int64") % 7,
            "f": np.linspace(-1.0, 1.0, n),
            "b": np.tile([True, False], n // 2),
            "s": ["a", "b", "c", "d"] * (n // 4),
            "cat": pd.Categorical(["low", "high"] * (n // 2)),
            "dt": pd.date_range("2024-01-01", periods=n, freq="D"),
            "nullable": pd.array([1, None] * (n // 2), dtype="Int64"),
            "y": [0, 1] * (n // 2),
        }
    )
    report = check(df, target="y")
    assert report.n_features == 7
    assert set(report.features) == {"i", "f", "b", "s", "cat", "dt", "nullable"}
    assert report.summary()


def test_categorical_with_unused_categories():
    values = pd.Categorical(["a"] * 30, categories=["a", "b", "c"])
    df = pd.DataFrame({"only_a": values, "v": list(range(30))})
    report = check(df)
    assert kinds(report, "only_a") == {"constant"}


def test_string_and_boolean_extension_dtypes():
    n = 40
    df = pd.DataFrame(
        {
            "txt": pd.array(["x", "y", None, "x"] * (n // 4), dtype="string"),
            "flag": pd.array([True, False, None, True] * (n // 4), dtype="boolean"),
            "v": np.linspace(0, 1, n),
        }
    )
    report = check(df)
    assert report.summary()
    assert set(report.features) == {"txt", "flag", "v"}


def test_duplicate_column_names_raise_a_clear_error():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column names"):
        check(df)


def test_unknown_target_raises_and_lists_the_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="is not a column"):
        check(df, target="nope")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"corr_threshold": 0.0},
        {"corr_threshold": 1.5},
        {"missing_threshold": float("nan")},
        {"cardinality_threshold": "big"},
        {"sample": 0},
        {"sample": -5},
    ],
)
def test_bad_parameters_raise_value_error(kwargs):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        check(df, **kwargs)


def test_non_tabular_input_raises_type_error():
    with pytest.raises(TypeError, match="DataFrame"):
        check(42)


def test_csv_path_input(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 10, "b": ["x", "y", "z"] * 10})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    report = check(str(path))
    assert set(report.features) == {"a", "b"}
    assert list(report.apply(str(path)).columns) == ["a", "b"]


def test_unsupported_file_type(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("not really a spreadsheet", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        check(str(path))


def test_sampling_is_reported():
    df = pd.DataFrame({"a": np.arange(500) % 5, "b": np.arange(500) % 7})
    report = check(df, sample=100)
    assert report.n_rows == 500
    assert report.n_rows_checked == 100
    assert any("random sample of 100" in note for note in report.notes)
    assert "(100 checked)" in report.summary()


def test_target_with_missing_values_is_noted():
    df = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 40),
            "y": [0, 1, np.nan, 1] * 10,
        }
    )
    report = check(df, target="y")
    assert any("missing target" in note for note in report.notes)


def test_constant_target_skips_the_label_checks():
    df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 40), "y": [1] * 40})
    report = check(df, target="y")
    assert "leakage_suspect" not in all_kinds(report)
    assert any("fewer than two usable values" in note for note in report.notes)


def test_tiny_frame_with_a_target_does_not_crash():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    report = check(df, target="y")
    assert report.summary()
    assert any("skipped" in note for note in report.notes)


def test_index_is_ignored():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0] * 10, "b": ["x", "y", "z"] * 10},
        index=pd.RangeIndex(100, 130),
    )
    report = check(df)
    assert set(report.features) == {"a", "b"}
    assert list(report.apply(df).index) == list(range(100, 130))


def test_checker_is_reusable_across_frames():
    checker = FeatureChecker(missing_threshold=0.5)
    first = checker.check(pd.DataFrame({"a": [1.0, np.nan, 2.0] * 10, "b": list(range(30))}))
    second = checker.check(pd.DataFrame({"a": [1.0, 2.0, 3.0] * 10, "b": list(range(30))}))
    assert "high_missing" not in kinds(first, "a")  # 33% missing, under the threshold
    assert kinds(second, "a") == set()
    assert first.params["missing_threshold"] == 0.5


def test_loo_mapping_accuracy_matches_a_brute_force_leave_one_out():
    """Post-removal ties must not be handed to the majority class (QA round 1)."""
    from ml_feature_check._stats import loo_mapping_accuracy

    def brute(f, y):
        n = len(y)
        majority = int(np.bincount(y).argmax())
        hits = 0
        for i in range(n):
            mask = np.ones(n, bool)
            mask[i] = False
            group = mask & (f == f[i])
            pred = majority if group.sum() == 0 else int(np.bincount(y[group], minlength=int(y.max()) + 1).argmax())
            hits += pred == y[i]
        return hits / n

    rng = np.random.default_rng(3)
    for _ in range(12):
        groups = int(rng.integers(2, 10))
        classes = int(rng.integers(2, 5))
        rows = int(rng.integers(20, 160))
        f = rng.integers(0, groups, rows).astype(np.int64)
        y = rng.integers(0, classes, rows).astype(np.int64)
        got = loo_mapping_accuracy(f, y, classes)
        assert got is not None
        assert got[0] == pytest.approx(brute(f, y))


def test_loo_mapping_accuracy_still_catches_a_perfect_mapping():
    from ml_feature_check._stats import loo_mapping_accuracy

    f = np.repeat(np.arange(10), 6).astype(np.int64)
    y = (f % 2).astype(np.int64)
    score, majority, n, groups = loo_mapping_accuracy(f, y, 2)
    assert score == 1.0
    assert (n, groups) == (60, 10)
    assert majority == 0.5


# --------------------------------------------------------------------- QA round 2


def test_no_numpy_repr_leaks_into_a_user_facing_message():
    """numpy 2 renders a float as "np.float64(2.5)"; no report may show that (QA round 2)."""
    import json

    n = 400
    df = pd.DataFrame(
        {
            "const_float": [2.5] * n,
            "const_int": [7] * n,
            "const_text": ["z"] * n,
            "near_const": [1.0] * (n - 1) + [9.0],
            "mostly_missing": [1.0] + [np.nan] * (n - 1),
        }
    )
    report = check(df)

    rendered = report.summary() + report.to_markdown() + json.dumps(report.to_dict())
    assert "np.float64" not in rendered
    assert "np.int64" not in rendered
    assert "np.str_" not in rendered
    assert "only one distinct value (2.5)" in report.summary()
    assert "99.8% of the non-missing values are 1.0" in report.summary()
    # the combined constant-plus-missing message is the other place a float shows
    assert "only one distinct value (1.0)" in messages(report, "mostly_missing")[0]


def test_a_missing_path_is_reported_by_this_package_not_by_pandas():
    """Every bad input speaks in the package's own voice (QA round 2)."""
    import ml_feature_check._io as io_module

    with pytest.raises(FileNotFoundError, match="no such file") as excinfo:
        check("D:/no/such/file.csv")
    frames = excinfo.traceback
    assert str(frames[-1].path).endswith("_io.py")
    assert io_module  # the module that raised, not pandas.io.common


def test_a_directory_in_place_of_a_file_is_also_ours(tmp_path):
    folder = tmp_path / "data.csv"
    folder.mkdir()
    with pytest.raises(IsADirectoryError, match="is a directory"):
        check(str(folder))


def test_one_row_notes_that_the_constant_verdicts_are_an_artifact():
    """A 1-row frame condemning every column is degenerate, and must say so (QA round 2)."""
    df = pd.DataFrame(
        {"a": [1.0], "b": ["x"], "c": [pd.Timestamp("2024-01-01")], "t": [0]}
    )
    report = check(df, target="t")

    assert report.drop_recommended == ["a", "b", "c"]
    assert any("only 1 row" in note and "artifact" in note for note in report.notes)


def test_a_three_row_frame_says_the_row_count_drove_the_verdict():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": ["x", "x", "x"], "c": [1, 2, 3]})
    report = check(df)
    assert any("only 3 rows" in note for note in report.notes)


def test_a_roomy_frame_is_not_told_its_row_count_is_the_problem():
    df = pd.DataFrame({"a": [1.0] * 50, "b": np.arange(50) % 7})
    report = check(df)
    assert not any("artifact" in note or "re-run on more rows" in note for note in report.notes)


def test_extension_dtype_columns_produce_the_findings_they_should():
    """The dtype tests above were smoke tests; pin the actual verdicts (QA round 2)."""
    n = 40
    df = pd.DataFrame(
        {
            "txt_constant": pd.array(["x"] * n, dtype="string"),
            "flag_constant": pd.array([True] * n, dtype="boolean"),
            "flag_real": pd.array([True, False] * (n // 2), dtype="boolean"),
            "flag_copy": pd.array([True, False] * (n // 2), dtype="boolean"),
            "v": np.linspace(0, 1, n),
        }
    )
    report = check(df)

    assert kinds(report, "txt_constant") == {"constant"}
    assert kinds(report, "flag_constant") == {"constant"}
    assert "duplicate_of" in kinds(report, "flag_copy")
    assert kinds(report, "flag_real") == set()
    assert report.keep == ["flag_real", "v"]
