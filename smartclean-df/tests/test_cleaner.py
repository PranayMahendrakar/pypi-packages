"""Cleaner.fit / transform / fit_transform: what is learned is what gets applied."""
import numpy as np
import pandas as pd
import pytest
from pandas.api import types as ptypes

from smartclean_df import FORWARD_FILL, Cleaner, CleanResult, clean


def training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": ["34", "NA", "41", "29", "38", "50"],
            "Joined": ["05/01/2024", "13/02/2024", "20/03/2024", "-", "02/04/2024", "15/05/2024"],
            "Active": ["yes", "no", "yes", "yes", "no", "yes"],
            "Score": [10, 12, 11, 13, 12, 200],
            "Note": [np.nan] * 6,
        }
    )


def test_fit_learns_types_values_bounds_and_empty_columns():
    cleaner = Cleaner().fit(training_frame())
    assert cleaner.fitted_
    types = cleaner.column_types_
    assert types["age"] == "numeric" and types["joined"] == "datetime"
    assert types["active"] == "boolean" and types["score"] == "numeric"
    assert cleaner.datetime_formats_["joined"]["dayfirst"] is True
    values = cleaner.impute_values_
    assert values["age"] == 38 and values["score"] == 12 and values["active"] is True
    assert values["joined"] is FORWARD_FILL
    assert cleaner.clip_bounds_ == {"age": (20, 56), "score": (7, 17)}
    assert cleaner.empty_columns_ == ["note"]
    assert "fitted=True" in repr(cleaner)


def test_transform_applies_learned_decisions_to_a_new_batch():
    cleaner = Cleaner().fit(training_frame())
    batch = pd.DataFrame(
        {
            "Age": ["NA", "x"],
            "Joined": ["07/06/2024", "junk"],
            "Active": ["y", "maybe"],
            "Score": [500, 5],
            "Note": [1, 2],
            "New": [" hi", "NA"],
        }
    )
    result = cleaner.transform(batch)
    assert isinstance(result, CleanResult)
    out = result.df
    assert list(out.columns) == ["age", "joined", "active", "score", "new"]
    assert out["age"].dtype == "int64" and out["age"].tolist() == [38, 38]
    assert out["joined"].tolist() == [pd.Timestamp("2024-06-07")] * 2
    assert out["active"].dtype == bool and out["active"].tolist() == [True, True]
    assert out["score"].tolist() == [17, 7]
    assert out["new"].tolist()[0] == "hi" and pd.isna(out["new"].iloc[1])
    kinds = [a.kind for a in result.actions]
    assert "drop_empty_column" in kinds and "clip_outliers" in kinds and "parse_datetime" in kinds
    # the same batch on its own would not pass the 90% rule for dates
    alone = clean(batch)
    assert alone.df["joined"].dtype == object


def test_fit_then_transform_matches_clean_on_the_same_data():
    df = training_frame()
    cleaner = Cleaner().fit(df)
    pd.testing.assert_frame_equal(cleaner.transform(df).df, clean(df).df)
    pd.testing.assert_frame_equal(Cleaner().fit_transform(df).df, clean(df).df)


def test_learned_values_fill_a_column_that_is_entirely_missing_in_the_batch():
    cleaner = Cleaner().fit(
        pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"], "k": [1, 2, 3, 4]})
    )
    result = cleaner.transform(pd.DataFrame({"a": [np.nan, np.nan], "b": [None, "y"], "k": [1, 2]}))
    assert result.df["a"].tolist() == [2.5, 2.5]
    assert result.df["b"].tolist() == ["x", "y"]
    unseen = cleaner.transform(pd.DataFrame({"a": [1.0], "c": [np.nan]}))
    assert list(unseen.df.columns) == ["a"]


def test_transform_before_fit_raises():
    cleaner = Cleaner()
    assert not cleaner.fitted_
    with pytest.raises(RuntimeError):
        cleaner.transform(pd.DataFrame({"a": [1]}))
    with pytest.raises(RuntimeError):
        cleaner.impute_values_


def test_fit_transform_dry_run_learns_but_returns_the_input_unchanged():
    df = training_frame()
    cleaner = Cleaner()
    result = cleaner.fit_transform(df, dry_run=True)
    pd.testing.assert_frame_equal(result.df, df)
    assert result.dry_run and cleaner.fitted_
    applied = cleaner.transform(df)
    assert applied.df["score"].tolist() == [10, 12, 11, 13, 12, 17]
    dry = cleaner.transform(df, dry_run=True)
    pd.testing.assert_frame_equal(dry.df, df)


def test_options_are_validated_and_exposed():
    with pytest.raises(ValueError):
        Cleaner(missing="later")
    with pytest.raises(ValueError):
        Cleaner(iqr_factor=float("nan"))
    cleaner = Cleaner(outliers="flag", iqr_factor=1.5, duplicates=False)
    assert cleaner.options["outliers"] == "flag"
    assert cleaner.options["iqr_factor"] == 1.5
    assert cleaner.options["duplicates"] is False
    assert cleaner.options["parse_dates"] is True


def test_fit_accepts_a_path(tmp_path):
    path = tmp_path / "train.csv"
    training_frame().to_csv(path, index=False)
    cleaner = Cleaner().fit(path)
    assert cleaner.column_types_["joined"] == "datetime"
    result = cleaner.transform(path)
    assert ptypes.is_datetime64_any_dtype(result.df["joined"])


def test_transform_keeps_a_learned_datetime_column_even_when_the_batch_is_one_row():
    cleaner = Cleaner().fit(pd.DataFrame({"d": ["01/02/2024", "13/02/2024", "20/03/2024"]}))
    result = cleaner.transform(pd.DataFrame({"d": ["05/06/2024"]}))
    assert result.df["d"].tolist() == [pd.Timestamp("2024-06-05")]
