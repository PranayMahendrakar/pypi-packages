import warnings

import numpy as np
import pandas as pd
import pytest

from synthetic_tabular import Synthesizer, generate


def test_single_column():
    df = pd.DataFrame({"x": np.arange(100, dtype="float64")})
    out = generate(df, n=30, random_state=0)
    assert list(out.columns) == ["x"] and len(out) == 30
    assert out["x"].between(0, 99).all()


def test_constant_numeric_column_is_copied():
    df = pd.DataFrame({"k": [7] * 20, "x": np.arange(20)})
    out = generate(df, n=40, random_state=0)
    assert (out["k"] == 7).all() and out["k"].dtype == "int64"
    assert Synthesizer().fit(df).column_kinds_["k"] == "constant"


def test_single_unique_category():
    df = pd.DataFrame({"c": ["only"] * 10, "x": range(10)})
    out = generate(df, n=25, random_state=0)
    assert (out["c"] == "only").all() and out["c"].dtype == object


def test_all_nan_float_column():
    df = pd.DataFrame({"empty": [np.nan] * 10, "x": range(10)})
    out = generate(df, n=15, random_state=0)
    assert out["empty"].isna().all() and out["empty"].dtype == "float64"


def test_all_none_object_column():
    df = pd.DataFrame({"empty": [None] * 10, "x": range(10)})
    out = generate(df, n=15, random_state=0)
    assert out["empty"].isna().all() and out["empty"].dtype == object


def test_empty_dataframe_round_trips_and_cannot_sample_rows():
    df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype=object)})
    out = generate(df)
    assert out.shape == (0, 2) and out.dtypes.equals(df.dtypes)
    with pytest.raises(ValueError, match="empty"):
        generate(df, n=5)


def test_zero_columns():
    out = generate(pd.DataFrame(index=range(4)), n=3)
    assert out.shape == (3, 0)


def test_single_row_is_repeated():
    df = pd.DataFrame({"a": [1], "b": ["x"], "c": [2.5], "d": [True]})
    out = generate(df, n=10, random_state=0)
    assert len(out) == 10
    assert out.dtypes.equals(df.dtypes)
    assert (out == df.iloc[0]).all().all()


def test_n_zero_returns_empty_frame_with_dtypes(rich_df):
    out = generate(rich_df, n=0)
    assert out.shape == (0, rich_df.shape[1]) and out.dtypes.equals(rich_df.dtypes)


def test_high_cardinality_text_is_resampled_and_flagged():
    names = [f"person-{i}" for i in range(100)]
    df = pd.DataFrame({"name": names, "age": np.arange(100)})
    with pytest.warns(UserWarning, match="'name'.*unique"):
        synth = Synthesizer(random_state=0).fit(df)
    assert synth.high_cardinality_columns_ == ["name"]
    assert synth.column_kinds_["name"] == "text"
    out = synth.sample(300)
    assert set(out["name"]) <= set(names)
    assert out["name"].dtype == object and out["age"].dtype == "int64"


def test_low_cardinality_strings_do_not_warn():
    df = pd.DataFrame({"city": ["a", "b", "c", "a", "b", "a"] * 10})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate(df)


def test_text_threshold_can_disable_resampling():
    df = pd.DataFrame({"name": [f"n{i}" for i in range(40)]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        synth = Synthesizer(random_state=0, text_threshold=1.0).fit(df)
    assert synth.column_kinds_["name"] == "categorical"


def test_unhashable_values_fall_back_to_resampling():
    df = pd.DataFrame({"tags": [["a"], ["b", "c"], ["a"], []] * 5, "x": range(20)})
    with pytest.warns(UserWarning, match="unhashable"):
        out = generate(df, n=10, random_state=0)
    assert all(isinstance(v, list) for v in out["tags"])


def test_unicode_values_survive():
    df = pd.DataFrame(
        {
            "city": ["München", "Zürich", "東京", "München", "東京", "Zürich"] * 10,
            "note": [f"ünïcödé-{i}-日本" for i in range(60)],
        }
    )
    with pytest.warns(UserWarning):
        out = generate(df, n=50, random_state=0)
    assert set(out["city"]) <= {"München", "Zürich", "東京"}
    assert set(out["note"]) <= set(df["note"])


def test_mixed_dtypes_all_round_trip():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "i": rng.integers(0, 100, n),
            "i8": rng.integers(0, 100, n).astype("int8"),
            "f32": rng.normal(size=n).astype("float32"),
            "f": rng.normal(size=n),
            "b": rng.random(n) < 0.5,
            "s": rng.choice(["x", "y", "z"], n),
            "cat": pd.Categorical(rng.choice(["p", "q"], n)),
            "dt": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"),
            "td": pd.to_timedelta(rng.integers(0, 60, n), unit="m"),
            "obj_mixed": [1, "a", 2.5, None] * (n // 4),
        }
    )
    out = generate(df, n=n, random_state=0)
    assert list(out.columns) == list(df.columns)
    assert out.dtypes.equals(df.dtypes)
    assert (out["dt"].dt.nanosecond == 0).all()


def test_infinite_values_do_not_crash():
    df = pd.DataFrame({"x": [1.0, 2.0, np.inf, 3.0, 4.0] * 10, "y": [np.inf] * 50})
    out = generate(df, n=20, random_state=0)
    assert np.isfinite(out["x"]).all()
    assert len(out) == 20


def test_two_row_input_with_perfect_correlation():
    df = pd.DataFrame({"a": [1, 2], "b": [10.0, 20.0], "c": ["x", "y"]})
    with pytest.warns(UserWarning, match="'c'"):  # two strings in two rows: 100% unique, so text
        out = generate(df, n=50, random_state=0)
    assert out["a"].between(1, 2).all() and out["b"].between(10, 20).all()
    assert set(out["c"]) <= {"x", "y"}


def test_non_string_column_labels():
    df = pd.DataFrame({0: np.arange(30), 1: np.arange(30) * 2.0})
    out = generate(df, n=10, random_state=0)
    assert list(out.columns) == [0, 1]
    assert "0" in Synthesizer().fit(df).summary()


def test_output_rows_need_not_exist_in_input(rich_df):
    """Synthetic rows are new rows, not copies: continuous columns are interpolated."""
    out = generate(rich_df, random_state=0)
    copied = out.merge(rich_df.drop_duplicates(), how="inner", on=list(out.columns))
    assert len(copied) < 0.01 * len(out)
    assert out["income"].nunique() > 100 and out["score"].nunique() > 100
