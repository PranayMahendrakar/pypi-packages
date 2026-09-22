import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy import stats

import synthetic_tabular as st
from synthetic_tabular import Synthesizer, generate


def test_public_api_exposed():
    assert st.__version__ == "0.1.0"
    for name in ("generate", "Synthesizer", "evaluate", "FidelityReport", "ColumnFidelity"):
        assert hasattr(st, name)


def test_generate_defaults_to_input_length_with_matching_columns_and_dtypes(rich_df):
    out = generate(rich_df)
    assert len(out) == len(rich_df)
    assert list(out.columns) == list(rich_df.columns)
    assert out.dtypes.equals(rich_df.dtypes)
    assert isinstance(out.index, pd.RangeIndex)


def test_n_larger_than_input(rich_df):
    out = generate(rich_df.head(10), n=500)
    assert len(out) == 500
    assert out.dtypes.equals(rich_df.dtypes)


def test_generate_is_deterministic_per_seed(rich_df):
    assert_frame_equal(generate(rich_df, random_state=3), generate(rich_df, random_state=3))
    assert not generate(rich_df, random_state=1).equals(generate(rich_df, random_state=2))


def test_synthesizer_sample_is_repeatable_and_reseedable(rich_df):
    synth = Synthesizer(random_state=0).fit(rich_df)
    assert_frame_equal(synth.sample(50), synth.sample(50))
    assert not synth.sample(50).equals(synth.sample(50, random_state=9))
    assert_frame_equal(synth.sample(50), generate(rich_df, 50, random_state=0))


def test_marginals_are_preserved(rich_df):
    out = generate(rich_df, random_state=0)
    for column in ("age", "income", "score"):
        assert stats.ks_2samp(rich_df[column], out[column]).statistic < 0.06
    real_freq = rich_df["city"].value_counts(normalize=True)
    synth_freq = out["city"].value_counts(normalize=True)
    assert (real_freq - synth_freq).abs().max() < 0.05
    assert abs(out["active"].mean() - rich_df["active"].mean()) < 0.05


def test_correlations_are_preserved(rich_df):
    out = generate(rich_df, random_state=0)
    cols = ["age", "income", "score"]
    real = rich_df[cols].corr(method="spearman")
    synth = out[cols].corr(method="spearman")
    assert (real - synth).abs().to_numpy().max() < 0.1
    assert synth.loc["age", "income"] > 0.8


def test_preserve_marginals_only_samples_independently(rich_df):
    out = generate(rich_df, random_state=0, preserve=("marginals",))
    assert abs(out[["age", "income"]].corr(method="spearman").iloc[0, 1]) < 0.1
    assert stats.ks_2samp(rich_df["income"], out["income"]).statistic < 0.06


def test_preserve_without_marginals_uses_smooth_normal_within_range():
    df = pd.DataFrame({"x": np.random.default_rng(0).exponential(size=1000)})
    out = generate(df, random_state=0, preserve=("correlations",))
    assert out["x"].min() >= df["x"].min() and out["x"].max() <= df["x"].max()
    assert abs(out["x"].skew()) < abs(df["x"].skew())


def test_preserve_accepts_a_string_and_rejects_unknown_values(rich_df):
    out = generate(rich_df.head(20), n=5, preserve="marginals")
    assert len(out) == 5
    with pytest.raises(ValueError, match="preserve"):
        generate(rich_df, preserve=("marginals", "magic"))


def test_integer_columns_stay_integer_within_observed_range(rich_df):
    out = generate(rich_df, random_state=1)
    assert out["age"].dtype == "int64"
    assert out["age"].between(rich_df["age"].min(), rich_df["age"].max()).all()


def test_float_columns_keep_observed_decimals_and_range(rich_df):
    out = generate(rich_df, random_state=1)
    assert np.allclose(out["income"], out["income"].round(2))
    assert out["income"].between(rich_df["income"].min(), rich_df["income"].max()).all()
    assert out["income"].nunique() > 100  # interpolated, not just copied


def test_bool_columns_stay_bool(rich_df):
    out = generate(rich_df, random_state=0)
    assert out["active"].dtype == bool
    assert set(out["active"].unique()) <= {True, False}


def test_datetime_columns_keep_dtype_range_and_day_resolution(rich_df):
    out = generate(rich_df, random_state=0)
    assert out["joined"].dtype == rich_df["joined"].dtype
    assert out["joined"].between(rich_df["joined"].min(), rich_df["joined"].max()).all()
    assert (out["joined"].dt.normalize() == out["joined"]).all()
    assert out["joined"].nunique() > 50


def test_tz_aware_datetime_round_trips():
    idx = pd.date_range("2021-03-01", periods=200, freq="h", tz="Asia/Kolkata")
    df = pd.DataFrame({"when": idx, "v": np.arange(200)})
    out = generate(df, n=300, random_state=0)
    assert str(out["when"].dtype) == str(df["when"].dtype)
    assert out["when"].min() >= df["when"].min() and out["when"].max() <= df["when"].max()
    assert (out["when"].dt.minute == 0).all()


def test_timedelta_columns_round_trip():
    df = pd.DataFrame({"wait": pd.to_timedelta(np.random.default_rng(0).integers(1, 500, 300), unit="s")})
    out = generate(df, random_state=0)
    assert out["wait"].dtype == df["wait"].dtype
    assert out["wait"].min() >= df["wait"].min() and out["wait"].max() <= df["wait"].max()
    assert (out["wait"].dt.nanoseconds == 0).all()


def test_category_dtype_is_preserved():
    df = pd.DataFrame({"size": pd.Categorical(["S", "M", "L", "M", "S", "M"] * 20, categories=["S", "M", "L"], ordered=True)})
    out = generate(df, random_state=0)
    assert out["size"].dtype == df["size"].dtype
    assert set(out["size"].unique()) <= {"S", "M", "L"}


def test_nullable_dtypes_are_preserved():
    df = pd.DataFrame(
        {
            "n": pd.array([1, 2, None, 4, 5, 6, None, 8] * 10, dtype="Int64"),
            "flag": pd.array([True, False, None, True] * 20, dtype="boolean"),
            "s": pd.array(["a", "b", None, "a"] * 20, dtype="string"),
        }
    )
    out = generate(df, random_state=0)
    assert out.dtypes.equals(df.dtypes)
    assert out["n"].isna().any() and out["flag"].isna().any() and out["s"].isna().any()


def test_missing_rate_is_preserved():
    rng = np.random.default_rng(0)
    values = rng.normal(size=2000)
    values[rng.random(2000) < 0.3] = np.nan
    df = pd.DataFrame({"x": values, "c": rng.choice(["a", "b"], 2000)})
    out = generate(df, random_state=0)
    assert out["x"].dtype == "float64"
    assert abs(out["x"].isna().mean() - 0.3) < 0.05
    assert out["c"].isna().sum() == 0


def test_csv_path_input(tmp_path, rich_df):
    path = tmp_path / "data.csv"
    rich_df.head(100).to_csv(path, index=False)
    out = generate(str(path), n=20)
    assert len(out) == 20 and list(out.columns) == list(rich_df.columns)
    out2 = Synthesizer().fit(path).sample(20)
    assert_frame_equal(out, out2)


def test_csv_iso_dates_are_parsed_as_datetimes(tmp_path):
    df = pd.DataFrame(
        {
            "day": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"] * 5,
            "stamp": ["2021-01-01 10:00:00", "2021-01-01 11:30:00", None, "2021-01-02 09:15:00"] * 5,
            "code": ["2021-01-0x", "2021-01-02", "2021-01-03", "2021-01-04"] * 5,
            "zip": ["02134", "10001", "60601", "94105"] * 5,
        }
    )
    path = tmp_path / "dates.csv"
    df.to_csv(path, index=False)
    synth = Synthesizer(random_state=0, text_threshold=1.0).fit(path)
    assert synth.column_kinds_ == {"day": "datetime", "stamp": "datetime", "code": "categorical", "zip": "numeric"}
    out = synth.sample(30)
    assert (out["day"].dt.normalize() == out["day"]).all()
    assert out["stamp"].isna().any() and (out["stamp"].dropna().dt.second == 0).all()


def test_parquet_path_input(tmp_path, rich_df):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    rich_df.head(100).to_parquet(path, index=False)
    out = generate(path, n=10)
    assert out.dtypes.equals(rich_df.dtypes)


def test_unsupported_inputs_raise(tmp_path):
    with pytest.raises(TypeError):
        generate([1, 2, 3])
    with pytest.raises(ValueError, match="Unsupported"):
        generate(str(tmp_path / "data.xlsx"))


def test_summary_and_attributes(rich_df):
    synth = Synthesizer(random_state=0).fit(rich_df)
    assert synth.n_rows_ == len(rich_df)
    assert synth.columns_ == list(rich_df.columns)
    assert synth.column_kinds_ == {
        "age": "numeric", "income": "numeric", "score": "numeric",
        "city": "categorical", "active": "bool", "joined": "datetime",
    }
    assert synth.high_cardinality_columns_ == []
    assert list(synth.correlation_.columns) == list(rich_df.columns)
    assert np.allclose(np.diag(synth.correlation_.to_numpy()), 1.0)
    text = synth.summary()
    assert "6 columns" in text and "day resolution" in text and "3 categories" in text
    assert "most common True (" in text and "np." not in text
    info = synth.to_dict()
    assert info["columns"]["age"]["kind"] == "numeric" and info["n_rows"] == len(rich_df)


def test_sample_before_fit_and_bad_n_raise(rich_df):
    with pytest.raises(RuntimeError, match="fit"):
        Synthesizer().sample(5)
    with pytest.raises(ValueError):
        Synthesizer().fit(rich_df).sample(-1)
    with pytest.raises(ValueError):
        Synthesizer(text_threshold=1.5)


def test_duplicate_column_names_raise():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        generate(df)


def test_custom_index_is_replaced_by_range_index(rich_df):
    df = rich_df.head(50).set_index(pd.Index([f"r{i}" for i in range(50)]))
    out = generate(df, random_state=0)
    assert isinstance(out.index, pd.RangeIndex) and len(out) == 50
