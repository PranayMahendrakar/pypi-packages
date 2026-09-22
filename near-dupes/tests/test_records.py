import datetime as dt

import numpy as np
import pandas as pd
import pytest

import near_dupes
from near_dupes import DuplicateFinder, find_duplicates
from near_dupes import _records


def sample_df():
    return pd.DataFrame(
        {
            "name": ["Acme Corporation", "ACME Corporation", "Acme Corporaton", "Globex"],
            "city": ["New York", "new york", "New York", "Springfield"],
            "id": [1, 2, 3, 4],
        }
    )


def test_records_with_key_columns():
    df = sample_df()
    r = find_duplicates(df, key=["name", "city"], threshold=0.8)
    assert r.kind == "records"
    assert r.groups == [[0, 1, 2]]
    assert r.representatives == [0]  # first row wins
    scores = {(i, j): s for i, j, s in r.pairs}
    assert scores[(0, 1)] == 1.0 and 0.8 <= scores[(0, 2)] < 1.0
    clean = r.dedupe(df)
    assert isinstance(clean, pd.DataFrame)
    assert list(clean.index) == [0, 3]
    assert list(clean["name"]) == ["Acme Corporation", "Globex"]


def test_all_columns_by_default_and_exact_duplicates():
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 1]})
    r = find_duplicates(df)
    assert r.groups == [[0, 1]] and r.pairs == [(0, 1, 1.0)]
    assert find_duplicates(df, threshold=1.0).groups == [[0, 1]]


def test_key_as_single_string():
    df = sample_df()
    r = find_duplicates(df, key="city")
    assert r.groups == [[0, 1, 2]]


def test_missing_key_column_raises():
    with pytest.raises(KeyError):
        find_duplicates(sample_df(), key=["name", "nope"])
    with pytest.raises(ValueError):
        find_duplicates(sample_df(), key=[])


def test_nan_cells_count_as_empty():
    df = pd.DataFrame({"name": ["Ann Lee", "ann lee", "Bob"], "email": [np.nan, np.nan, "bob@x.com"]})
    r = find_duplicates(df, key=["name", "email"])
    assert r.groups == [[0, 1]]
    assert r.pairs == [(0, 1, 1.0)]


def test_all_nan_column_and_all_blank_rows():
    df = pd.DataFrame({"name": ["Zed", "zed", None, None], "note": [np.nan] * 4})
    r = find_duplicates(df)
    assert r.groups == [[0, 1]]  # rows 2 and 3 are entirely blank: never duplicates
    assert r.keep_indices == [0, 2, 3]
    only_blank = find_duplicates(df, key="note")
    assert only_blank.groups == [] and only_blank.keep_indices == [0, 1, 2, 3]


def test_empty_dataframe():
    df = pd.DataFrame({"a": pd.Series([], dtype=str), "b": pd.Series([], dtype=float)})
    r = find_duplicates(df)
    assert r.n_items == 0 and r.groups == [] and r.keep_indices == []
    out = r.dedupe(df)
    assert isinstance(out, pd.DataFrame) and len(out) == 0 and list(out.columns) == ["a", "b"]
    assert find_duplicates(pd.DataFrame()).n_items == 0


def test_single_row():
    df = pd.DataFrame({"a": ["one"], "b": [1]})
    r = find_duplicates(df)
    assert r.groups == [] and r.keep_indices == [0]
    assert r.dedupe(df).equals(df)


def test_mixed_dtypes():
    df = pd.DataFrame(
        {
            "int": [1, 1, 2, 1],
            "float": [2.0, 2.0, 2.5, np.nan],
            "flag": [True, True, False, True],
            "when": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-06-01", "2024-01-01"]),
            "text": ["Alpha", "alpha", "beta", "Alpha"],
            "obj": [None, None, {"k": 1}, None],
        }
    )
    r = find_duplicates(df)
    assert r.groups == [[0, 1, 3]]  # row 3 differs only by a NaN cell: a near-duplicate
    assert r.dedupe(df).shape == (2, 6)
    exact = find_duplicates(df, threshold=1.0)
    assert exact.groups == [[0, 1]]
    assert exact.dedupe(df).shape == (3, 6)


def test_integral_floats_match_ints():
    df = pd.DataFrame({"code": pd.Series([7, 7.0, 8], dtype=object), "x": ["a", "a", "a"]})
    assert find_duplicates(df).groups == [[0, 1]]


def test_unicode_records():
    df = pd.DataFrame({"name": ["Müller, Jürgen", "MÜLLER JÜRGEN", "Zoë"], "city": ["Köln", "köln", "Köln"]})
    r = find_duplicates(df)
    assert r.groups == [[0, 1]]


def test_punctuation_ignored_for_records():
    df = pd.DataFrame({"phone": ["+1 (555) 123-4567", "1 555 123 4567", "+1 (555) 999-0000"]})
    assert find_duplicates(df, threshold=1.0).groups == [[0, 1]]


def test_dedupe_keeps_original_index_labels():
    df = sample_df()
    df.index = ["r0", "r1", "r2", "r3"]
    r = find_duplicates(df, key="city")
    assert r.groups == [[0, 1, 2]]  # positions, not labels
    assert list(r.dedupe(df).index) == ["r0", "r3"]


def test_csv_path_input_and_dedupe(tmp_path):
    df = sample_df()
    path = tmp_path / "people.csv"
    df.to_csv(path, index=False)
    r = find_duplicates(str(path), key=["name", "city"], threshold=0.8)
    assert r.kind == "records" and r.groups == [[0, 1, 2]]
    clean = r.dedupe(path)
    assert isinstance(clean, pd.DataFrame) and len(clean) == 2
    assert len(near_dupes.dedupe(str(path), key="city")) == 2


def test_parquet_path_input(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "people.parquet"
    sample_df().to_parquet(path, index=False)
    assert find_duplicates(path, key="city").groups == [[0, 1, 2]]


def test_list_of_dicts_is_records():
    rows = [{"a": "X", "b": 1}, {"a": "x", "b": 1}, {"a": "y", "b": 2}]
    r = find_duplicates(rows)
    assert r.kind == "records" and r.groups == [[0, 1]]
    assert r.dedupe(rows) == [rows[0], rows[2]]
    assert near_dupes.dedupe(rows) == [rows[0], rows[2]]


def test_kind_mismatch_errors():
    with pytest.raises(ValueError):
        find_duplicates(sample_df(), kind="text")
    with pytest.raises(ValueError):
        find_duplicates(["a", "b"], kind="records")
    with pytest.raises(ValueError):
        find_duplicates(sample_df(), kind="images")


def test_row_keys_helper():
    df = pd.DataFrame({"a": ["Hi, there!", None], "b": [1.0, np.nan]})
    keys, blank = _records.row_keys(df)
    assert keys == ["hi there | 1", " | "] and blank == [False, True]
    raw, _ = _records.row_keys(df, normalize=False)
    assert raw[0] == "Hi, there! | 1"


def test_finder_with_records_and_threshold_one_only_exact():
    df = pd.DataFrame({"a": ["abc def", "abc deg", "abc def"]})
    r = DuplicateFinder(threshold=1.0).find(df)
    assert r.groups == [[0, 2]]
