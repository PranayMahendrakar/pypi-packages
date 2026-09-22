"""Every pipeline step of clean(), its options, and the edge cases that must hold."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.api import types as ptypes

from smartclean_df import clean

# --------------------------------------------------------------------------- step 1: column names


def test_column_names_are_normalized_and_logged():
    df = pd.DataFrame({" Order  ID ": [1], "Total Amount": [2], "ok": [3]})
    result = clean(df)
    assert list(result.df.columns) == ["order_id", "total_amount", "ok"]
    renames = [a for a in result.actions if a.kind == "rename_column"]
    assert [(a.column, a.rows_affected) for a in renames] == [("order_id", 0), ("total_amount", 0)]
    assert "' Order  ID '" in renames[0].detail


def test_column_name_clashes_get_suffixes():
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "a", "A "])
    result = clean(df)
    assert list(result.df.columns) == ["a", "a_2", "a_3"]


def test_normalize_columns_false_keeps_names_as_they_are():
    df = pd.DataFrame([[1, 2]], columns=["X Y", "x  y"])
    result = clean(df, normalize_columns=False)
    assert list(result.df.columns) == ["X Y", "x  y"]
    assert list(clean(df).df.columns) == ["x_y", "x_y_2"]


def test_duplicate_column_labels_raise_a_clear_value_error():
    df = pd.DataFrame([[1, 2]], columns=["X Y", "X Y"])
    with pytest.raises(ValueError, match="duplicate column names"):
        clean(df)
    with pytest.raises(ValueError, match="'X Y'"):
        clean(df, normalize_columns=False)


def test_non_string_column_names_are_left_alone():
    df = pd.DataFrame([[1, 2]])
    result = clean(df)
    assert list(result.df.columns) == [0, 1]
    assert not any(a.kind == "rename_column" for a in result.actions)


# --------------------------------------------------------------------------- step 2: text cells


def test_whitespace_is_stripped_and_missing_tokens_become_nan():
    df = pd.DataFrame(
        {
            "t": ["  a ", "b", "", "NA", "n/a", "NULL", "None", "-", "?", "NaN", " na ", "c"],
            "k": list(range(12)),
        }
    )
    result = clean(df, missing="none", outliers="none")
    assert result.df["t"].tolist()[:2] == ["a", "b"]
    assert result.df["t"].isna().sum() == 9
    kinds = {a.kind: a for a in result.actions}
    assert kinds["strip_whitespace"].rows_affected == 2  # "  a " and " na "
    assert kinds["missing_tokens"].rows_affected == 9


def test_pandas_string_dtype_column_is_cleaned_and_stays_string():
    df = pd.DataFrame({"s": pd.Series([" x", "NA", "y", "y"], dtype="string"), "k": [1, 2, 3, 4]})
    result = clean(df)
    assert isinstance(result.df["s"].dtype, pd.StringDtype)
    assert result.df["s"].tolist() == ["x", "y", "y", "y"]
    assert [a.kind for a in result.actions] == ["strip_whitespace", "missing_tokens", "impute"]


# --------------------------------------------------------------------------- step 3: numbers


def test_numeric_looking_text_is_parsed():
    df = pd.DataFrame(
        {
            "amount": ["1,234", "$12.50", "  7 ", "(300)", "1e3", "-2.5"],
            "pct": ["45%", "5%", "12.5%", "0%", "100%", "7%"],
            "whole": ["1", "2", "3", "4", "5", "6"],
        }
    )
    result = clean(df, outliers="none")
    assert result.df["amount"].tolist() == [1234.0, 12.5, 7.0, -300.0, 1000.0, -2.5]
    assert result.df["pct"].tolist() == [45.0, 5.0, 12.5, 0.0, 100.0, 7.0]
    assert result.df["whole"].dtype == "int64"
    assert result.df["whole"].tolist() == [1, 2, 3, 4, 5, 6]
    assert {a.column for a in result.actions if a.kind == "parse_numeric"} == {"amount", "pct", "whole"}


def test_mixed_numbers_and_text_stay_object():
    df = pd.DataFrame({"m": ["1", "2", "three", "4"]})
    result = clean(df)
    assert result.df["m"].dtype == object
    assert result.df["m"].tolist() == ["1", "2", "three", "4"]
    assert not any(a.kind.startswith("parse_") for a in result.actions)


def test_ninety_percent_rule_coerces_the_rest_to_nan():
    values = [str(i) for i in range(19)] + ["oops"]
    result = clean(pd.DataFrame({"n": values, "k": list(range(20))}), missing="none", outliers="none")
    assert ptypes.is_float_dtype(result.df["n"])
    assert result.df["n"].isna().sum() == 1
    action = next(a for a in result.actions if a.kind == "parse_numeric")
    assert "1 unparseable value(s) set to NaN" in action.detail
    assert action.rows_affected == 20
    below = [str(i) for i in range(17)] + ["a", "b", "c"]
    assert clean(pd.DataFrame({"n": below})).df["n"].dtype == object


def test_codes_with_leading_zeros_are_not_numbers():
    df = pd.DataFrame({"zip": ["00123", "04567", "00001"]})
    assert clean(df).df["zip"].tolist() == ["00123", "04567", "00001"]


def test_parse_numbers_can_be_disabled():
    df = pd.DataFrame({"n": ["1", "2", "3"]})
    assert clean(df, parse_numbers=False).df["n"].dtype == object


# --------------------------------------------------------------------------- step 4: dates


def test_iso_dates_are_parsed():
    df = pd.DataFrame({"d": ["2024-01-05", "2024-02-10", "2024-03-15"]})
    result = clean(df)
    assert ptypes.is_datetime64_any_dtype(result.df["d"])
    assert result.df["d"].iloc[0] == pd.Timestamp("2024-01-05")
    assert next(a for a in result.actions if a.kind == "parse_datetime").rows_affected == 3


def test_day_first_is_inferred_from_the_data():
    df = pd.DataFrame({"d": ["05/01/2024", "13/02/2024", "20/03/2024"]})
    result = clean(df)
    assert result.df["d"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-02-13"),
        pd.Timestamp("2024-03-20"),
    ]
    assert "day-first" in next(a for a in result.actions if a.kind == "parse_datetime").detail
    month_first = clean(pd.DataFrame({"d": ["05/01/2024", "02/13/2024", "03/20/2024"]}))
    assert month_first.df["d"].tolist()[0] == pd.Timestamp("2024-05-01")


def test_mixed_date_formats_parse_without_any_warning():
    dates = [
        "05/01/2024",
        "13/02/2024",
        "20/03/2024",
        "2024-04-01",
        "May 5 2024",
        "06/06/2024",
        "07/07/2024",
        "08/08/2024",
        "09/09/2024",
        "10/10/2024",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = clean(pd.DataFrame({"d": dates}))
    assert ptypes.is_datetime64_any_dtype(result.df["d"])
    assert result.df["d"].isna().sum() == 0
    assert result.df["d"].iloc[0] == pd.Timestamp("2024-01-05")
    assert result.df["d"].iloc[3] == pd.Timestamp("2024-04-01")
    assert result.df["d"].iloc[4] == pd.Timestamp("2024-05-05")


def test_text_that_is_not_dates_is_not_parsed():
    df = pd.DataFrame({"t": ["apple", "banana", "cherry", "2024-01-01"]})
    assert clean(df).df["t"].dtype == object
    assert clean(pd.DataFrame({"d": ["2024-01-05", "2024-02-10"]}), parse_dates=False).df["d"].dtype == object


# --------------------------------------------------------------------------- step 5: booleans


@pytest.mark.parametrize(
    "values",
    [
        ["yes", "no", "Yes", "NO"],
        ["true", "false", "TRUE", "False"],
        ["y", "n", "Y", "N"],
        ["1", "0", "1", "0"],
        ["t", "f", "t", "f"],
    ],
)
def test_boolean_looking_text_is_parsed(values):
    result = clean(pd.DataFrame({"b": values, "k": [1, 2, 3, 4]}))
    assert result.df["b"].dtype == bool
    assert result.df["b"].tolist() == [True, False, True, False]
    assert any(a.kind == "parse_boolean" for a in result.actions)


def test_numeric_zero_one_columns_are_never_touched_by_boolean_parsing():
    df = pd.DataFrame({"flag": [0, 1, 2, 1, 0], "bin": [0, 1, 0, 1, 1], "k": [1, 2, 3, 4, 5]})
    result = clean(df)
    assert result.df["flag"].dtype == "int64" and result.df["flag"].tolist() == [0, 1, 2, 1, 0]
    assert result.df["bin"].dtype == "int64" and result.df["bin"].tolist() == [0, 1, 0, 1, 1]
    assert not any(a.kind == "parse_boolean" for a in result.actions)
    strings = clean(pd.DataFrame({"s": ["0", "1", "2", "1"], "k": [1, 2, 3, 4]}))
    assert strings.df["s"].dtype == "int64"


def test_boolean_with_missing_is_imputed_with_the_mode_and_ends_up_bool():
    df = pd.DataFrame({"b": ["yes", "no", "?", "yes"], "k": [1, 2, 3, 4]})
    result = clean(df)
    assert result.df["b"].dtype == bool
    assert result.df["b"].tolist() == [True, False, True, True]
    none = clean(df, missing="none")
    assert isinstance(none.df["b"].dtype, pd.BooleanDtype)
    assert none.df["b"].isna().sum() == 1


def test_boolean_column_with_a_stray_value_stays_text():
    df = pd.DataFrame({"b": ["yes", "no", "maybe", "yes"]})
    assert clean(df).df["b"].dtype == object
    assert clean(pd.DataFrame({"b": ["yes", "no"]}), parse_booleans=False).df["b"].dtype == object


# --------------------------------------------------------------------------- step 6: duplicates and empties


def test_duplicates_and_empty_rows_and_columns_are_dropped():
    df = pd.DataFrame(
        {
            "a": [1, 1, np.nan, 2],
            "b": ["x", "x", None, "y"],
            "empty": [np.nan, np.nan, np.nan, np.nan],
            "blank": ["", "NA", "-", " "],
        }
    )
    result = clean(df)
    assert list(result.df.columns) == ["a", "b"]
    assert len(result.df) == 2
    kinds = [a.kind for a in result.actions]
    assert kinds.count("drop_duplicates") == 1
    assert kinds.count("drop_empty_rows") == 1
    empties = [a for a in result.actions if a.kind == "drop_empty_column"]
    assert [a.column for a in empties] == ["empty", "blank"]
    assert list(result.df.index) == [0, 1]


def test_duplicates_can_be_kept():
    df = pd.DataFrame({"a": [1, 1]})
    assert len(clean(df, duplicates=False).df) == 2


def test_unhashable_cells_do_not_crash():
    df = pd.DataFrame({"tags": [[1, 2], [3], [1, 2], None], "x": [1, 2, 1, 3]})
    result = clean(df)
    assert len(result.df) == 3
    assert any(a.kind == "drop_duplicates" for a in result.actions)


# --------------------------------------------------------------------------- step 7: missing values


def test_impute_median_mode_and_forward_fill():
    df = pd.DataFrame(
        {
            "num": [1.5, np.nan, 2.5, 10.0],
            "cat": ["a", "b", None, "b"],
            "when": pd.to_datetime([None, "2024-01-02", None, "2024-01-04"]),
        }
    )
    result = clean(df, outliers="none")
    assert result.df["num"].tolist() == [1.5, 2.5, 2.5, 10.0]
    assert result.df["cat"].tolist() == ["a", "b", "b", "b"]
    assert result.df["when"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-04"),
    ]
    details = {a.column: a.detail for a in result.actions if a.kind == "impute"}
    assert details == {
        "num": "filled 1 missing value(s) with median 2.5",
        "cat": "filled 1 missing value(s) with mode 'b'",
        "when": "filled 2 missing value(s) with forward-fill",
    }


def test_integer_columns_stay_integer_after_imputation_when_possible():
    # a float column the caller handed us stays float, whatever its values are:
    # its dtype must not depend on whether this batch happened to hold a NaN
    whole = clean(pd.DataFrame({"n": [1.0, 2.0, np.nan, 4.0, 5.0], "k": [1, 2, 3, 4, 5]}))
    assert ptypes.is_float_dtype(whole.df["n"]) and whole.df["n"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    no_gap = clean(pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0, 5.0], "k": [1, 2, 3, 4, 5]}))
    assert no_gap.df["n"].dtype == whole.df["n"].dtype
    text = clean(pd.DataFrame({"n": ["10", "NA", "30", "20"], "k": [1, 2, 3, 4]}))
    assert text.df["n"].dtype == "int64" and text.df["n"].tolist() == [10, 20, 30, 20]
    fractional = clean(pd.DataFrame({"n": [1.0, 2.5, np.nan, 4.0], "k": [1, 2, 3, 4]}))
    assert ptypes.is_float_dtype(fractional.df["n"])
    assert fractional.df["n"].tolist() == [1.0, 2.5, 2.5, 4.0]
    nullable = clean(pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64"), "k": [1, 2, 3]}))
    assert str(nullable.df["n"].dtype) == "Int64" and nullable.df["n"].tolist() == [1, 2, 3]


def test_mode_ties_go_to_the_first_value_seen():
    df = pd.DataFrame({"c": ["zeta", "alpha", None, "zeta", "alpha"], "k": [1, 2, 3, 4, 5]})
    assert clean(df).df["c"].tolist()[2] == "zeta"


def test_categorical_columns_are_imputed_with_their_mode():
    df = pd.DataFrame({"c": pd.Categorical(["a", "b", None, "b"]), "k": [1, 2, 3, 4]})
    result = clean(df)
    assert isinstance(result.df["c"].dtype, pd.CategoricalDtype)
    assert result.df["c"].tolist() == ["a", "b", "b", "b"]


def test_missing_drop_and_none():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
    dropped = clean(df, missing="drop")
    assert len(dropped.df) == 1 and dropped.df["a"].tolist() == [1]
    action = next(a for a in dropped.actions if a.kind == "drop_missing_rows")
    assert action.column is None and action.rows_affected == 2
    left = clean(df, missing="none")
    assert left.df.isna().sum().sum() == 2
    assert not any(a.kind == "impute" for a in left.actions)


# --------------------------------------------------------------------------- step 8: outliers


def test_outliers_are_clipped_and_integer_columns_stay_integer():
    df = pd.DataFrame({"v": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
    result = clean(df)
    assert result.df["v"].dtype == "int64"
    assert result.df["v"].tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 30]
    action = next(a for a in result.actions if a.kind == "clip_outliers")
    assert action.column == "v" and action.rows_affected == 1
    assert "[-1, 30]" in action.detail


def test_outlier_modes_flag_drop_none_and_factor():
    df = pd.DataFrame({"v": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 1000.0], "k": list("abcdefghij")})
    flagged = clean(df, outliers="flag")
    assert flagged.df["v"].tolist() == df["v"].tolist()
    assert next(a for a in flagged.actions if a.kind == "flag_outliers").rows_affected == 1
    dropped = clean(df, outliers="drop")
    assert len(dropped.df) == 9 and dropped.df["k"].tolist() == list("abcdefghi")
    assert next(a for a in dropped.actions if a.kind == "drop_outliers").rows_affected == 1
    untouched = clean(df, outliers="none")
    assert untouched.df["v"].tolist() == df["v"].tolist() and not any("outlier" in a.kind for a in untouched.actions)
    tight = clean(pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0]}), iqr_factor=1.5)
    assert tight.df["v"].max() < 20


def test_zero_iqr_and_boolean_columns_are_never_clipped():
    df = pd.DataFrame(
        {"const": [5, 5, 5, 5, 100], "flag": [True, True, True, True, False], "k": [1, 2, 3, 4, 5]}
    )
    result = clean(df)
    assert result.df["const"].tolist() == [5, 5, 5, 5, 100]
    assert result.df["flag"].dtype == bool
    assert not any("outlier" in a.kind for a in result.actions)


# --------------------------------------------------------------------------- whole-table behaviour


def test_input_is_never_modified():
    df = pd.DataFrame({"Name ": [" a", "NA", " a", None], "V": ["1", "x", "1", "2"], "d": ["2024-01-01"] * 4})
    before = df.copy(deep=True)
    columns = list(df.columns)
    clean(df)
    clean(df, dry_run=True)
    pd.testing.assert_frame_equal(df, before)
    assert list(df.columns) == columns


def test_dry_run_reports_without_changing():
    df = pd.DataFrame({"A ": ["10", "10", "NA"], "b": ["x", "x", "y"]})
    result = clean(df, dry_run=True)
    pd.testing.assert_frame_equal(result.df, df)
    assert result.dry_run is True
    assert [a.kind for a in result.actions] == ["rename_column", "missing_tokens", "parse_numeric", "drop_duplicates", "impute"]
    assert result.input_shape == (3, 2) and result.output_shape == (2, 2)
    assert "dry run" in result.summary()
    assert result.to_dict()["dry_run"] is True


def test_empty_dataframes():
    nothing = clean(pd.DataFrame())
    assert nothing.df.shape == (0, 0) and nothing.actions == []
    assert "no changes needed" in nothing.summary()
    json.dumps(nothing.to_dict())
    no_rows = clean(pd.DataFrame({"A B": pd.Series([], dtype="float64"), "c": pd.Series([], dtype="object")}))
    assert list(no_rows.df.columns) == ["a_b", "c"] and len(no_rows.df) == 0
    assert [a.kind for a in no_rows.actions] == ["rename_column"]


def test_single_row():
    df = pd.DataFrame({"A": ["1,000"], "B": ["yes"], "C": ["2024-01-01"], "D": [np.nan]})
    result = clean(df)
    assert result.df.shape == (1, 3)
    assert result.df["a"].tolist() == [1000] and result.df["a"].dtype == "int64"
    assert result.df["b"].tolist() == [True]
    assert result.df["c"].iloc[0] == pd.Timestamp("2024-01-01")
    assert any(a.kind == "drop_empty_column" and a.column == "d" for a in result.actions)


def test_all_nan_column_is_dropped_and_logged():
    df = pd.DataFrame({"keep": [1, 2, 3], "gone": [np.nan] * 3, "also": [None, None, None]})
    result = clean(df)
    assert list(result.df.columns) == ["keep"]
    dropped = [a for a in result.actions if a.kind == "drop_empty_column"]
    assert [a.column for a in dropped] == ["gone", "also"]
    assert all("every value missing" in a.detail for a in dropped)


def test_mixed_dtypes_survive():
    df = pd.DataFrame(
        {
            "i": [1, 2, 3, 4],
            "f": [0.5, 1.5, np.nan, 2.5],
            "s": ["a", " b", "NA", "d"],
            "b": [True, False, True, True],
            "t": pd.to_datetime(["2024-01-01", "2024-01-02", None, "2024-01-04"]),
            "c": pd.Categorical(["x", "y", "x", None]),
            "td": pd.to_timedelta([1, 2, None, 4], unit="D"),
        }
    )
    result = clean(df)
    out = result.df
    assert out["i"].dtype == "int64"
    assert ptypes.is_float_dtype(out["f"]) and out["f"].tolist() == [0.5, 1.5, 1.5, 2.5]
    assert out["s"].tolist() == ["a", "b", "a", "d"]
    assert out["b"].dtype == bool
    assert out["t"].isna().sum() == 0
    assert isinstance(out["c"].dtype, pd.CategoricalDtype) and out["c"].tolist() == ["x", "y", "x", "x"]
    assert out["td"].isna().sum() == 0 and out["td"].iloc[2] == pd.Timedelta(days=2)
    json.dumps(result.to_dict())


def test_unicode_text():
    df = pd.DataFrame(
        {
            "Ville ": [" café", "naïve", "東京", "東京", None, "Ærø"],
            "n": ["１", "2", "3", "4", "5", "6"],
        }
    )
    result = clean(df)
    assert list(result.df.columns) == ["ville", "n"]
    assert result.df["ville"].tolist() == [
        "café",
        "naïve",
        "東京",
        "東京",
        "東京",
        "Ærø",
    ]
    assert "東京" in result.summary()
    json.dumps(result.to_dict(), ensure_ascii=False)


def test_custom_index_is_kept_and_range_index_is_reset():
    labelled = pd.DataFrame({"a": [1, 1, 2]}, index=["r1", "r2", "r3"])
    assert clean(labelled).df.index.tolist() == ["r1", "r3"]
    ranged = pd.DataFrame({"a": [1, 1, 2]})
    assert clean(ranged).df.index.tolist() == [0, 1]
    untouched = pd.DataFrame({"a": [1, 2, 3]}, index=[5, 6, 7])
    assert clean(untouched).df.index.tolist() == [5, 6, 7]


def test_csv_path_input(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"Score ": ["1,200", "1,300", "NA"], "when": ["2024-01-01", "2024-01-02", "2024-01-03"]}).to_csv(path, index=False)
    result = clean(path)
    assert list(result.df.columns) == ["score", "when"]
    assert result.df["score"].tolist() == [1200, 1300, 1250]
    assert ptypes.is_datetime64_any_dtype(result.df["when"])
    assert clean(str(path)).df.shape == (3, 2)


def test_parquet_path_input(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    pd.DataFrame({"A": [1, 1, 2], "b": ["x", "x", "y"]}).to_parquet(path, index=False)
    assert clean(path).df.shape == (2, 2)


def test_bad_inputs_raise_builtin_errors():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        clean(df, missing="sometimes")
    with pytest.raises(ValueError):
        clean(df, outliers="maybe")
    with pytest.raises(ValueError):
        clean(df, iqr_factor=-1)
    with pytest.raises(TypeError):
        clean(42)
    with pytest.raises(ValueError):
        clean("table.xlsx")
    with pytest.raises(FileNotFoundError):
        clean("does-not-exist.csv")


def test_summary_and_to_dict_shapes():
    df = pd.DataFrame({"A": ["1", "1", "x"]})
    result = clean(df)
    text = result.summary()
    assert text.splitlines()[0] == "smartclean-df: 3 rows x 1 columns -> 2 rows x 1 columns"
    assert "1. [a] rename_column" in text
    payload = result.to_dict()
    assert payload["input_shape"] == [3, 1] and payload["output_shape"] == [2, 1]
    assert payload["n_actions"] == len(payload["actions"]) == 2
    assert payload["actions"][0] == {"column": "a", "kind": "rename_column", "detail": "renamed from 'A'", "rows_affected": 0}
    assert payload["dtypes"] == {"a": "object"}
    json.dumps(payload)


def test_pipeline_emits_no_warnings_at_all():
    df = pd.DataFrame(
        {
            "Name ": [" Ann", "Bob", "NA", "Ann", None],
            "Age": ["34", "NA", "41", "34", "1000"],
            "Joined": ["05/01/2024", "13/02/2024", "-", "05/01/2024", "Mar 3 2024"],
            "Active": ["yes", "no", "?", "yes", "y"],
            "Score": [1.0, 2.0, np.nan, 1.0, 3.0],
            "empty": [None] * 5,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = clean(df)
        clean(df, missing="drop", outliers="drop")
        clean(df, missing="none", outliers="flag")
    assert result.df.shape[0] == 4
