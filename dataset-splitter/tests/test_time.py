"""Chronological splits, alone and combined with groups, targets and duplicates."""
import numpy as np
import pandas as pd
import pytest

from dataset_splitter import split


def timed(n=100):
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
            "customer_id": np.arange(n) // 4,
            "y": np.arange(n) % 2,
            "x": np.arange(n),
        }
    )


def test_chronological_order_and_no_shuffle():
    df = timed().sample(frac=1, random_state=5)  # scrambled input order
    s = split(df, time="ts")
    r = s.report()
    assert s.strategy["method"] == "chronological" and s.strategy["shuffled"] is False
    assert (len(s.train), len(s.val), len(s.test)) == (70, 10, 20)
    assert s.train["ts"].max() <= s.val["ts"].min() <= s.val["ts"].max() <= s.test["ts"].min()
    assert sorted(s.train["x"]) == list(range(70)) and sorted(s.test["x"]) == list(range(80, 100))
    assert r.time_ordering["checked"] and r.time_ordering["ordered"] and r.time_ordering["level"] == "row"
    assert r.ok


def test_time_with_groups_orders_groups_and_keeps_them_whole():
    df = timed()
    df.loc[df.customer_id == 5, "ts"] = pd.Timestamp("2024-06-01")  # a group whose rows sit late
    s = split(df, time="ts", group="customer_id")
    r = s.report()
    assert r.group_overlap["overlapping_groups"] == 0
    assert r.time_ordering["level"] == "group" and r.time_ordering["ordered"]
    assert 5 in set(s.test["customer_id"])
    assert r.ok


def test_time_with_groups_reports_row_overlap_when_groups_straddle():
    df = timed(40)
    # customer 0 has one very early and one very late row: it must stay whole
    df.loc[0, "ts"] = pd.Timestamp("2023-01-01")
    df.loc[1, "ts"] = pd.Timestamp("2025-01-01")
    s = split(df, time="ts", group="customer_id", val_size=0)
    r = s.report()
    assert r.group_overlap["overlapping_groups"] == 0
    assert r.time_ordering["ordered"] is True
    assert r.time_ordering["overlap_rows"] >= 1
    assert any("groups were kept whole" in w for w in r.warnings)
    assert r.ok


def test_string_and_numeric_time_columns():
    df = pd.DataFrame({"when": [f"2024-03-{d:02d}" for d in range(1, 31)], "x": range(30)})
    s = split(df, time="when")
    assert s.test["x"].min() > s.train["x"].max()
    df2 = pd.DataFrame({"epoch": np.arange(30)[::-1], "x": range(30)})
    s2 = split(df2, time="epoch")
    assert s2.train["epoch"].max() < s2.test["epoch"].min()
    assert s2.report().ok


def test_time_with_missing_or_bad_values_raises():
    df = timed(20)
    df.loc[3, "ts"] = pd.NaT
    with pytest.raises(ValueError, match="missing"):
        split(df, time="ts")
    bad = pd.DataFrame({"ts": ["yesterday", "soon", "never"] * 5, "x": range(15)})
    with pytest.raises(ValueError, match="could not be parsed"):
        split(bad, time="ts")


def test_time_with_target_reports_balance_without_stratifying():
    s = split(timed(), time="ts", target="y")
    r = s.report()
    assert s.strategy["stratified"] is False and s.strategy["method"] == "chronological"
    assert r.class_balance is not None and set(r.class_balance["test"]) == {"0", "1"}


def test_time_keeps_duplicates_together():
    base = timed(30)
    df = pd.concat([base, base, base], ignore_index=True)
    s = split(df, time="ts")
    r = s.report()
    assert r.duplicate_leakage["duplicate_rows"] == 90 and r.duplicate_leakage["leaked_rows"] == 0
    assert r.time_ordering["ordered"] and r.ok


def test_tz_aware_time_and_json_ranges():
    df = timed(50)
    df["ts"] = df["ts"].dt.tz_localize("Europe/Rome")
    s = split(df, time="ts")
    d = s.report().to_dict()
    assert d["time_ordering"]["ordered"] is True
    assert isinstance(d["time_ordering"]["ranges"]["train"]["min"], str)
