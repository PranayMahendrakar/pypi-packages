"""Group-aware splitting: explicit columns, composite keys, auto detection, linking."""
import numpy as np
import pandas as pd

from dataset_splitter import detect_id_columns, split
from dataset_splitter.groups import is_id_like


def sides_per_group(s, cols):
    joined = pd.concat([part.assign(_side=name) for name, part in s.frames().items()])
    return joined.groupby(cols)["_side"].nunique()


def test_single_group_column_never_straddles():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"customer_id": rng.integers(0, 40, size=300), "x": rng.normal(size=300)})
    s = split(df, group="customer_id")
    r = s.report()
    assert r.group_overlap["checked"] and r.group_overlap["overlapping_groups"] == 0
    assert (sides_per_group(s, "customer_id") == 1).all()
    assert abs(len(s.test) - 60) <= 12  # unit granularity, still close to 20%
    assert r.ok


def test_composite_key():
    df = pd.DataFrame(
        {"a": np.repeat([1, 2, 3, 4], 25), "b": np.tile(np.repeat([1, 2, 3, 4, 5], 5), 4), "x": range(100)}
    )
    s = split(df, group=["a", "b"])
    assert s.strategy["group_mode"] == "composite" and s.strategy["n_groups"] == 20
    assert (sides_per_group(s, ["a", "b"]) == 1).all()
    assert s.report().group_overlap["overlapping_groups"] == 0


def test_auto_detects_id_columns_and_skips_lookalikes():
    df = pd.DataFrame(
        {
            "customer_id": np.repeat(np.arange(20), 5),
            "width": np.arange(100),  # ends in "id" but is not an id
            "valid": [1, 0] * 50,
            "row_id": np.arange(100),  # unique per row: grouping would be a no-op
            "site_id": [7] * 100,  # constant: cannot be split on
            "y": [0, 1] * 50,
        }
    )
    assert detect_id_columns(df, exclude=["y"]) == ["customer_id"]
    s = split(df, group="auto", target="y")
    assert s.strategy["group"] == ["customer_id"] and s.strategy["group_detected"] is True
    assert s.report().group_overlap["overlapping_groups"] == 0


def test_auto_without_id_columns_warns_and_splits_rows():
    df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
    s = split(df, group="auto")
    assert s.strategy["group"] is None
    assert any("no id-like column" in w for w in s.warnings)
    assert s.report().ok


def test_auto_with_several_ids_links_rows():
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "device_id": [10, 20, 20, 30, 40, 40, 50, 50, 60, 60, 70, 70],
            "x": range(12),
        }
    )
    s = split(df, group="auto", val_size=0, test_size=0.4)
    assert s.strategy["group_mode"] == "linked"
    joined = pd.concat([part.assign(_side=name) for name, part in s.frames().items()])
    # customers 1 and 2 share device 20, so all their rows sit on one side
    assert joined.loc[joined.customer_id.isin([1, 2]), "_side"].nunique() == 1
    assert s.report().group_overlap["overlapping_groups"] == 0


def test_missing_group_keys_stand_alone():
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 10, size=200).astype(float)
    ids[::2] = np.nan
    df = pd.DataFrame({"customer_id": ids, "x": range(200)})
    s = split(df, group="customer_id")
    r = s.report()
    assert r.group_overlap["overlapping_groups"] == 0
    # the 100 NaN rows are spread over the parts instead of forming one giant group
    assert s.test["customer_id"].isna().sum() > 0 and s.train["customer_id"].isna().sum() > 0


def test_single_group_cannot_be_split_but_does_not_crash():
    df = pd.DataFrame({"g": ["same"] * 30, "x": range(30)})
    s = split(df, group="g")
    assert len(s.train) == 30 and len(s.test) == 0
    assert any("test split is empty" in w for w in s.warnings)
    assert s.report().ok


def test_categorical_group_column():
    df = pd.DataFrame({"g": pd.Categorical(np.repeat(list("abcdefgh"), 10)), "x": range(80)})
    s = split(df, group="g")
    assert s.report().group_overlap["overlapping_groups"] == 0


def test_is_id_like_names():
    for name in ("customer_id", "userId", "session_uuid", "patient_no", "customer", "ID", "account_number", "uuid"):
        assert is_id_like(name), name
    for name in ("width", "valid", "paid", "grid", "price", "session_length", "x", 3, "phone_number"):
        assert not is_id_like(name), name
