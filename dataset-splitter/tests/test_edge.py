"""Edge cases: empty, single row, all-NaN column, mixed dtypes, unicode, odd indexes."""
import json

import numpy as np
import pandas as pd
import pytest

from dataset_splitter import split


def test_empty_frame():
    df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")})
    s = split(df, target="b", group="a")
    assert len(s.train) == len(s.val) == len(s.test) == 0
    assert list(s.test.columns) == ["a", "b"]
    assert any("no rows" in w for w in s.warnings)
    r = s.report()
    assert r.ok and r.sizes == {"train": 0, "val": 0, "test": 0}
    json.dumps(r.to_dict())
    assert split(df, val_size=0).val is None


def test_single_row():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    s = split(df, target="b")
    assert len(s.train) == 1 and len(s.val) == 0 and len(s.test) == 0
    assert s.indices == {"train": [0], "val": [], "test": []}
    assert s.report().ok


def test_two_rows_with_time_and_group():
    df = pd.DataFrame({"ts": ["2024-01-01", "2024-01-02"], "g": [1, 2], "x": [0, 1]})
    s = split(df, time="ts", group="g", val_size=0, test_size=0.5)
    assert s.train["x"].tolist() == [0] and s.test["x"].tolist() == [1]
    assert s.report().ok


def test_all_nan_column_and_nan_group_target():
    df = pd.DataFrame({"empty": [np.nan] * 40, "y": ["a", "b"] * 20, "x": range(40)})
    s = split(df, target="y")
    r = s.report()
    assert r.ok and r.duplicate_leakage["duplicate_rows"] == 0
    s2 = split(df, target="empty", group="empty")
    assert s2.report().partition["exact"]


def test_mixed_dtypes_and_unicode():
    n = 60
    df = pd.DataFrame(
        {
            "int": np.arange(n),
            "float": np.linspace(0, 1, n),
            "text": ["café", "naïve", "東京", "München", None, "🙂"] * (n // 6),
            "flag": [True, False] * (n // 2),
            "when": pd.date_range("2020-01-01", periods=n, freq="D"),
            "cat": pd.Categorical(["α", "β", "γ"] * (n // 3)),
            "obj": [1, "1", None] * (n // 3),
            "città": range(n),
        }
    )
    s = split(df, target="cat", group="text")
    r = s.report()
    assert r.ok
    assert set(r.class_counts["all"]) == {"α", "β", "γ"}
    text = r.summary()
    assert "α" in text and "ok: True" in text
    json.dumps(r.to_dict(), ensure_ascii=False)
    s2 = split(df, time="when", target="int")
    assert s2.report().ok and s2.strategy["target_kind"] == "numeric"


def test_non_unique_index_is_preserved():
    df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25}, index=[0, 1, 2, 3, 4] * 10)
    s = split(df, target="y")
    all_idx = [v for name in s.indices for v in s.indices[name]]
    assert sorted(all_idx) == sorted(df.index.tolist())
    assert s.report().partition["exact"]
    assert sum(len(p) for p in s.frames().values()) == 50


def test_multiindex_rows():
    idx = pd.MultiIndex.from_product([["a", "b"], range(20)], names=["grp", "n"])
    df = pd.DataFrame({"x": range(40)}, index=idx)
    s = split(df)
    assert all(isinstance(v, tuple) for v in s.indices["test"])
    json.dumps(s.report().to_dict())
    pd.testing.assert_frame_equal(df.loc[s.indices["test"]], s.test)


def test_duplicate_column_names_are_rejected_by_name():
    df = pd.DataFrame([[1, 2, "a"], [3, 4, "b"], [1, 2, "a"], [5, 6, "c"]] * 5, columns=["v", "v", "y"])
    for kwargs in ({}, {"target": "y"}, {"group": "v"}, {"time": "v"}):
        with pytest.raises(ValueError, match=r"duplicate column name\(s\): 'v'"):
            split(df, **kwargs)
    renamed = df.set_axis(["v1", "v2", "y"], axis=1)
    assert split(renamed, target="y").report().ok


def test_many_duplicate_column_names_are_all_named():
    df = pd.DataFrame(np.arange(40).reshape(10, 4), columns=["a", "a", "b", "b"])
    with pytest.raises(ValueError) as exc:
        split(df)
    message = str(exc.value)
    assert "'a'" in message and "'b'" in message and "rename" in message


def test_large_frame_is_fast():
    import time as _time

    rng = np.random.default_rng(0)
    n = 200_000
    df = pd.DataFrame(
        {
            "customer_id": rng.integers(0, 20_000, size=n),
            "y": rng.choice(list("abc"), size=n),
            "x": rng.normal(size=n),
        }
    )
    t0 = _time.perf_counter()
    s = split(df, target="y", group="customer_id")
    r = s.report()
    assert _time.perf_counter() - t0 < 8
    assert r.ok and r.balance_max_deviation < 0.02
