"""Randomised invariants: every configuration partitions exactly and never leaks."""
import itertools

import numpy as np
import pandas as pd
import pytest

from dataset_splitter import Splitter


def make(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": rng.integers(0, max(1, n // 3), size=n),
            "ts": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 1000, size=n), unit="h"),
            "y": rng.choice(list("abc"), size=n, p=[0.6, 0.3, 0.1]),
            "score": rng.normal(size=n),
            "x": rng.integers(0, 5, size=n),
        }
    )
    if n >= 4:  # inject exact duplicates
        df = pd.concat([df, df.iloc[: max(1, n // 5)]], ignore_index=True)
    return df.sample(frac=1, random_state=seed)


CASES = list(
    itertools.product(
        [0, 1, 3, 10, 11, 37, 250],
        [None, "y", "score"],
        [None, "customer_id", "auto"],
        [None, "ts"],
        [True, False],
    )
)


@pytest.mark.parametrize("n,target,group,time,dedupe", CASES)
def test_invariants(n, target, group, time, dedupe):
    df = make(n, seed=n + 7)
    val_size = 0 if n % 2 else 0.1
    s = Splitter(target=target, group=group, time=time, dedupe=dedupe, val_size=val_size, random_state=n).split(df)
    r = s.report()
    frames = s.frames()
    assert r.partition["exact"], r.partition
    assert sum(len(f) for f in frames.values()) == len(df)
    assert sorted(v for name in s.indices for v in s.indices[name]) == sorted(df.index.tolist())
    for name, part in frames.items():
        assert s.indices[name] == part.index.tolist()
    if val_size == 0:
        assert s.val is None and "val" not in s.indices
    if group is not None and s.strategy["group"]:
        assert r.group_overlap["overlapping_groups"] == 0
    if dedupe:
        assert r.duplicate_leakage["leaked_rows"] == 0
        assert r.ok, r.warnings
    if time is not None:
        assert r.time_ordering["ordered"] is True
        assert s.strategy["shuffled"] is False
    if target is not None and n:
        assert r.class_balance is not None and "all" in r.class_balance
    if n >= 250 and group is None and time is None and target == "y":
        # row-level units balance to within one row of the 29-row val split; with dedupe the
        # duplicated rows form 2-row units, so the bound is one unit instead
        assert r.balance_max_deviation < (0.08 if dedupe else 0.04)
