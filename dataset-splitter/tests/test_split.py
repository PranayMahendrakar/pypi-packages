"""Core behaviour: sizes, partition, indices, determinism, parameters."""
import numpy as np
import pandas as pd
import pytest

from dataset_splitter import Split, Splitter, split


def frame(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(list("abc"), size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )


def assert_partition(df, s):
    frames = s.frames()
    assert sum(len(f) for f in frames.values()) == len(df)
    idx = [v for name in s.indices for v in s.indices[name]]
    assert sorted(idx) == sorted(df.index.tolist())
    pd.testing.assert_frame_equal(pd.concat(list(frames.values())).sort_index(), df.sort_index())
    assert s.report().partition["exact"] is True


def test_default_sizes_and_partition():
    df = frame()
    s = split(df)
    assert isinstance(s, Split)
    assert (len(s.train), len(s.val), len(s.test)) == (70, 10, 20)
    assert s.report().sizes == {"train": 70, "val": 10, "test": 20}
    assert_partition(df, s)


def test_indices_are_original_index_values():
    df = frame().set_index(pd.Index([f"r{i}" for i in range(100)], name="rid"))
    s = split(df)
    for name, part in s.frames().items():
        assert s.indices[name] == part.index.tolist()
        assert set(s.indices[name]) <= set(df.index)
        pd.testing.assert_frame_equal(df.loc[s.indices[name]], part)
    assert_partition(df, s)


def test_val_size_zero_gives_no_val():
    df = frame()
    s = split(df, val_size=0)
    assert s.val is None
    assert set(s.indices) == {"train", "test"}
    assert (len(s.train), len(s.test)) == (80, 20)
    assert set(s.report().sizes) == {"train", "test"}
    assert_partition(df, s)


def test_deterministic_and_seed_sensitive():
    df = frame()
    a, b, c = split(df, random_state=1), split(df, random_state=1), split(df, random_state=2)
    assert a.indices == b.indices
    assert a.indices != c.indices


def test_rows_are_shuffled_but_order_inside_parts_is_input_order():
    df = frame()
    s = split(df)
    assert s.train.index.tolist() != list(range(70))
    for part in s.frames().values():
        assert part.index.is_monotonic_increasing


def test_absolute_sizes():
    df = frame()
    s = split(df, test_size=25, val_size=5)
    assert (len(s.train), len(s.val), len(s.test)) == (70, 5, 25)
    assert_partition(df, s)


def test_test_size_zero_gives_empty_test_frame():
    df = frame()
    s = split(df, test_size=0.0, val_size=0.2)
    assert len(s.test) == 0 and list(s.test.columns) == list(df.columns)
    assert any("test split is empty" in w for w in s.warnings)
    assert_partition(df, s)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"test_size": 1.0}, ValueError),
        ({"test_size": -0.1}, ValueError),
        ({"test_size": 0.6, "val_size": 0.5}, ValueError),
        ({"test_size": True}, TypeError),
        ({"test_size": "0.2"}, TypeError),
        ({"group": 5}, TypeError),
        ({"group": []}, ValueError),
    ],
)
def test_bad_parameters(kwargs, exc):
    with pytest.raises(exc):
        split(frame(), **kwargs)


def test_bad_splitter_options():
    with pytest.raises(ValueError):
        Splitter(stratify="sometimes")
    with pytest.raises(ValueError):
        Splitter(n_bins=0)


def test_missing_columns_raise_helpful_errors():
    df = frame()
    with pytest.raises(ValueError, match="target column 'nope' not found"):
        split(df, target="nope")
    with pytest.raises(ValueError, match="group column"):
        split(df, group="nope")
    with pytest.raises(ValueError, match="time column"):
        split(df, time="nope")
    with pytest.raises(ValueError, match="exceeds"):
        split(df, test_size=500)
    with pytest.raises(ValueError, match="only 100"):
        split(df, test_size=60, val_size=50)
    with pytest.raises(TypeError):
        split([1, 2, 3])


def test_path_input(tmp_path):
    df = frame()
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    s = split(str(path), target="y")
    assert s.report().ok
    assert len(s.train) + len(s.val) + len(s.test) == 100
    tsv = tmp_path / "data.tsv"
    df.to_csv(tsv, index=False, sep="\t")
    assert split(tsv).report().ok
    with pytest.raises(ValueError, match="unsupported"):
        split(tmp_path / "data.xlsx")


def test_splitter_class_is_reusable():
    df = frame()
    splitter = Splitter(target="y", test_size=0.25, val_size=0.25, random_state=3)
    a, b = splitter.split(df), splitter.split(df)
    assert a.indices == b.indices
    assert a.strategy["method"] == "stratified"
    assert (len(a.train), len(a.val), len(a.test)) == (50, 25, 25)


def test_repr_and_strategy():
    s = split(frame(), target="y")
    text = repr(s)
    assert "train=70 rows" in text and "val=10 rows" in text and "method='stratified'" in text
    assert s.strategy["rows"] == 100 and s.strategy["random_state"] == 0
    grouped = split(frame(), target="y", group="cat")
    assert grouped.strategy["group"] == ["cat"] and grouped.strategy["n_groups"] == 3
    assert grouped.strategy["units"] == 3  # three indivisible units: sizes cannot be exact
    assert_partition(frame(), grouped)
