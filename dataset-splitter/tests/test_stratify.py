"""Stratification on categorical and numeric targets, including the tiny cases."""
import numpy as np
import pandas as pd

from dataset_splitter import Splitter, split


def test_categorical_balance_is_kept():
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b", "c"], size=300, p=[0.6, 0.3, 0.1])
    df = pd.DataFrame({"y": y, "x": rng.normal(size=300)})
    s = split(df, target="y")
    r = s.report()
    assert s.strategy["method"] == "stratified" and s.strategy["target_kind"] == "categorical"
    assert r.balance_max_deviation < 0.03
    for name in ("train", "val", "test"):
        assert set(r.class_counts[name]) == {"a", "b", "c"}
        assert all(v > 0 for v in r.class_counts[name].values())
    assert r.ok


def test_numeric_target_uses_quantile_bins():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.exponential(size=300), "x": rng.normal(size=300)})
    s = split(df, target="y")
    r = s.report()
    assert s.strategy["target_kind"] == "numeric" and s.strategy["n_strata"] == 10
    assert r.balance_max_deviation < 0.05
    assert set(r.target_stats) == {"all", "train", "val", "test"}
    assert abs(r.target_stats["test"]["mean"] - r.target_stats["all"]["mean"]) < 0.4
    assert len(r.class_balance["all"]) == 10


def test_low_cardinality_numeric_target_is_categorical():
    df = pd.DataFrame({"y": np.arange(200) % 5, "x": range(200)})
    s = split(df, target="y")
    assert s.strategy["target_kind"] == "categorical"
    assert s.report().balance_max_deviation < 0.02


def test_single_sample_class_does_not_crash():
    df = pd.DataFrame({"y": ["a"] * 10 + ["b"] * 10 + ["c"], "x": range(21)})
    s = split(df, target="y")
    r = s.report()
    assert s.strategy["stratified"] is True
    assert any("pooled" in w and "'c'" in w for w in s.warnings)
    assert r.partition["exact"] and r.ok
    assert r.class_counts["all"]["c"] == 1


def test_tiny_dataset_falls_back_to_random_with_warning():
    df = pd.DataFrame({"y": list("abcdefghij"), "x": range(10)})
    s = split(df, target="y")
    assert s.strategy["method"] == "random"
    assert any("random" in w for w in s.warnings)
    assert (len(s.train), len(s.val), len(s.test)) == (7, 1, 2)
    assert s.report().ok


def test_tiny_dataset_with_two_classes_is_still_stratified():
    df = pd.DataFrame({"y": ["a", "b"] * 5, "x": range(10)})
    s = split(df, target="y")
    assert s.strategy["method"] == "stratified"
    assert (len(s.train), len(s.val), len(s.test)) == (7, 1, 2)
    assert sorted(s.test["y"]) == ["a", "b"]


def test_missing_target_values_form_their_own_stratum():
    df = pd.DataFrame({"y": ["a", "b"] * 30 + [None] * 10, "x": range(70)})
    s = split(df, target="y")
    r = s.report()
    assert "<missing>" in r.class_counts["all"]
    assert r.class_counts["all"]["<missing>"] == 10
    assert any("missing" in w for w in s.warnings)
    assert r.ok


def test_constant_target_warns_and_splits_randomly():
    df = pd.DataFrame({"y": ["only"] * 30, "x": range(30)})
    s = split(df, target="y")
    assert s.strategy["method"] == "random"
    assert any("single class" in w for w in s.warnings)


def test_stratify_false_still_reports_balance():
    df = pd.DataFrame({"y": ["a", "b"] * 50, "x": range(100)})
    s = Splitter(target="y", stratify=False).split(df)
    assert s.strategy["method"] == "random"
    assert s.report().class_balance is not None


def test_bool_and_categorical_targets():
    df = pd.DataFrame(
        {"flag": [True, False] * 40, "cat": pd.Categorical(list("xyzw") * 20), "x": range(80)}
    )
    assert split(df, target="flag").report().balance_max_deviation < 0.02
    s = split(df, target="cat")
    assert set(s.report().class_counts["all"]) == {"x", "y", "z", "w"}


def test_stratification_with_groups_keeps_groups_whole():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"g": np.repeat(np.arange(50), 4), "y": rng.choice(["p", "n"], size=200), "x": range(200)}
    )
    s = split(df, target="y", group="g")
    r = s.report()
    assert r.group_overlap["overlapping_groups"] == 0
    assert r.balance_max_deviation < 0.12
    assert r.ok
