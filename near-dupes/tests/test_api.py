import json

import numpy as np
import pandas as pd
import pytest

import near_dupes
from near_dupes import DuplicateFinder, DuplicateResult, dedupe, find_duplicates


ITEMS = ["alpha beta gamma", "Alpha Beta Gamma", "delta epsilon", "delta epsilon!", "zeta"]


def test_version_and_exports():
    assert near_dupes.__version__ == "0.1.0"
    for name in ("find_duplicates", "dedupe", "DuplicateFinder", "DuplicateResult", "text_similarity"):
        assert name in near_dupes.__all__


def test_dedupe_convenience_keeps_input_type():
    assert dedupe(ITEMS, threshold=0.8) == ["alpha beta gamma", "delta epsilon!", "zeta"]
    assert dedupe(tuple(ITEMS), threshold=0.8) == ("alpha beta gamma", "delta epsilon!", "zeta")
    arr = dedupe(np.array(ITEMS), threshold=0.8)
    assert isinstance(arr, np.ndarray) and arr.tolist() == ["alpha beta gamma", "delta epsilon!", "zeta"]
    series = dedupe(pd.Series(ITEMS, index=list("abcde")), threshold=0.8)
    assert isinstance(series, pd.Series) and list(series.index) == ["a", "d", "e"]
    assert dedupe(iter(ITEMS), threshold=0.8) == ["alpha beta gamma", "delta epsilon!", "zeta"]


def test_finder_dedupe_and_find():
    finder = DuplicateFinder(threshold=0.8)
    assert finder.dedupe(ITEMS) == finder.find(ITEMS).dedupe(ITEMS)
    assert finder.find(ITEMS).groups == [[0, 1], [2, 3]]


def test_result_to_dict_is_json_safe():
    r = find_duplicates(ITEMS, threshold=0.8)
    d = r.to_dict()
    text = json.dumps(d)
    back = json.loads(text)
    assert back["kind"] == "text" and back["threshold"] == 0.8
    assert back["n_items"] == 5 and back["n_groups"] == 2 and back["n_duplicates"] == 2
    assert back["groups"] == [[0, 1], [2, 3]]
    assert back["representatives"] == [0, 3]
    assert back["keep_indices"] == [0, 3, 4]
    assert back["pairs"][0] == [0, 1, 1.0]


def test_summary_text():
    r = find_duplicates(ITEMS, threshold=0.8)
    s = r.summary()
    assert "5 texts" in s and "2 duplicate groups" in s and "3 kept" in s
    assert "group 1: [0, 1] -> keep 0" in s
    assert "group 2: [2, 3] -> keep 3" in s
    many = find_duplicates([f"copy {k}" for k in range(15) for _ in range(2)])
    assert "more groups" in many.summary()


def test_dedupe_length_mismatch_raises():
    r = find_duplicates(ITEMS)
    with pytest.raises(ValueError):
        r.dedupe(ITEMS[:-1])
    with pytest.raises(ValueError):
        r.dedupe(pd.DataFrame({"a": [1]}))


def test_result_properties():
    r = find_duplicates(ITEMS, threshold=0.8)
    assert r.n_groups == 2 and r.n_duplicates == 2
    assert r.drop_indices == [1, 2]
    assert r.n_items == 5
    empty = DuplicateResult()
    assert empty.n_duplicates == 0 and empty.to_dict()["groups"] == []


def test_generator_and_series_input():
    r = find_duplicates(x for x in ITEMS)
    assert r.n_items == 5
    r2 = find_duplicates(pd.Series(ITEMS))
    assert r2.kind == "text" and r2.groups == [[0, 1], [2, 3]]
    assert find_duplicates(pd.Series(ITEMS), threshold=1.0).groups == [[0, 1]]


def test_pairs_are_sorted_and_unique():
    items = ["dup one here", "other text", "dup one here!", "dup one here", "other text"]
    r = find_duplicates(items, threshold=0.8)
    assert r.pairs == sorted(r.pairs, key=lambda t: (t[0], t[1]))
    assert len({(i, j) for i, j, _ in r.pairs}) == len(r.pairs)
    assert all(i < j for i, j, _ in r.pairs)
    assert r.groups == [[0, 2, 3], [1, 4]]


def test_duplicate_column_names_raise_clearly():
    df = pd.DataFrame([["a", "b"], ["a", "c"]], columns=["name", "name"])
    for call in (
        lambda: find_duplicates(df),
        lambda: find_duplicates(df, key="name"),
        lambda: dedupe(df),
        lambda: DuplicateFinder(kind="records").find(df),
    ):
        with pytest.raises(ValueError, match="duplicate column names"):
            call()
    # the message names the offending column
    with pytest.raises(ValueError, match="name"):
        find_duplicates(df)


def test_duplicate_keys_in_dict_records_are_reported():
    # dicts cannot repeat a key, but a DataFrame built from them can still be checked
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column names"):
        find_duplicates(df, kind="records")


def test_opaque_objects_are_rejected_not_compared_by_address():
    class Opaque:
        def __init__(self, v):
            self.v = v

    with pytest.raises(TypeError, match="memory address"):
        find_duplicates([Opaque(1), Opaque(1), Opaque(2)])
    with pytest.raises(TypeError):
        dedupe([Opaque(1), Opaque(2)])


def test_objects_with_a_repr_are_still_accepted():
    class Named:
        def __init__(self, v):
            self.v = v

        def __repr__(self):
            return f"Named({self.v})"

    r = find_duplicates([Named("alpha beta"), Named("alpha beta"), Named("zzz")])
    assert r.groups == [[0, 1]]
