"""Exact duplicate rows stay on one side unless dedupe=False, and the report notices."""
import numpy as np
import pandas as pd

from dataset_splitter import split
from dataset_splitter.groups import duplicate_ids


def duplicated_frame():
    base = pd.DataFrame({"a": np.arange(20), "b": list("abcdefghijklmnopqrst"), "c": [1.5, np.nan] * 10})
    return pd.concat([base, base, base], ignore_index=True).sample(frac=1, random_state=0)


def test_duplicates_are_kept_on_one_side():
    df = duplicated_frame()
    s = split(df)
    r = s.report()
    assert r.duplicate_leakage == {"checked": True, "duplicate_rows": 60, "leaked_rows": 0, "leaked_groups": 0}
    joined = pd.concat([part.assign(_side=name) for name, part in s.frames().items()])
    assert (joined.groupby("a")["_side"].nunique() == 1).all()
    assert s.strategy["duplicate_rows"] == 60
    assert r.ok


def test_dedupe_false_leaks_and_report_says_so():
    df = duplicated_frame()
    s = split(df, dedupe=False)
    r = s.report()
    assert r.duplicate_leakage["leaked_rows"] > 0
    assert r.ok is False
    assert any("duplicate rows appear in more than one split" in w for w in r.warnings)


def test_dedupe_with_missing_group_key_still_holds():
    df = pd.DataFrame({"customer_id": [np.nan] * 40 + list(range(20)), "x": [1, 2] * 30})
    s = split(df, group="customer_id")
    assert s.report().duplicate_leakage["leaked_rows"] == 0


def test_unhashable_cells_fall_back_to_text_comparison():
    df = pd.DataFrame({"tags": [[1, 2], [3], [1, 2], [3], [4]] * 6, "x": [0, 1, 0, 1, 2] * 6})
    ids = duplicate_ids(df)
    assert len(np.unique(ids)) == 3
    s = split(df)
    assert s.report().duplicate_leakage["leaked_rows"] == 0


def test_duplicate_ids_treats_nan_as_equal_and_none_like_nan():
    df = pd.DataFrame({"a": [np.nan, np.nan, None, 1.0], "b": ["x", "x", "x", "x"]})
    ids = duplicate_ids(df)
    assert ids[0] == ids[1] == ids[2] != ids[3]
