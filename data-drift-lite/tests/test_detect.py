import json

import numpy as np
import pandas as pd
import pytest

import data_drift_lite as ddl


def make_frames(n=400, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2]),
        }
    )
    rng = np.random.default_rng(seed + 1)
    current = pd.DataFrame(
        {
            "x": rng.normal(shift, 1.0, n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2]),
        }
    )
    return reference, current


def test_same_distribution_is_not_drift():
    reference, current = make_frames()
    report = ddl.detect(reference, current)
    assert not report.drifted
    assert report.drifted_columns == []
    assert report.drift_share == 0.0
    assert list(report.columns) == ["x", "cat"]
    x, cat = report.columns["x"], report.columns["cat"]
    assert x.kind == "numeric" and x.test == "ks" and x.p_value > 0.05 and x.psi < 0.2
    assert cat.kind == "categorical" and cat.test == "chi2" and cat.p_value > 0.05 and cat.psi < 0.2


def test_numeric_drift_is_detected():
    reference, current = make_frames(shift=1.5)
    report = ddl.detect(reference, current)
    assert report.drifted
    assert report.drifted_columns == ["x"]
    assert report.drift_share == pytest.approx(0.5)
    col = report.columns["x"]
    assert col.drifted and col.p_value < 0.05 and col.psi > 0.2
    assert col.reference_stats["mean"] == pytest.approx(0.0, abs=0.2)
    assert col.current_stats["mean"] == pytest.approx(1.5, abs=0.2)


def test_categorical_drift_is_detected():
    reference = pd.DataFrame({"plan": ["basic"] * 70 + ["pro"] * 30})
    current = pd.DataFrame({"plan": ["basic"] * 20 + ["pro"] * 80})
    col = ddl.detect(reference, current).columns["plan"]
    assert col.drifted and col.p_value < 0.05 and col.psi > 0.2
    assert col.reference_stats["top"] == {"basic": 0.7, "pro": 0.3}
    assert col.current_stats["n_categories"] == 2


def test_columns_restricts_to_subset():
    reference, current = make_frames(shift=1.5)
    reference["extra"] = 1
    current["extra"] = 2
    current["brand_new"] = 0
    report = ddl.detect(reference, current, columns=["x"])
    assert list(report.columns) == ["x"]
    assert report.new_columns == []
    assert report.missing_columns == []
    single = ddl.detect(reference, current, columns="cat")
    assert list(single.columns) == ["cat"]


def test_columns_validation():
    reference, current = make_frames()
    with pytest.raises(ValueError, match="not found in reference"):
        ddl.detect(reference, current, columns=["x", "typo"])
    with pytest.raises(ValueError, match="at least one column"):
        ddl.detect(reference, current, columns=[])


def test_thresholds_can_be_disabled():
    reference, current = make_frames(shift=1.5)
    assert ddl.detect(reference, current, threshold=None, psi_threshold=None).drifted is False
    assert ddl.detect(reference, current, threshold=None, psi_threshold=1e9).drifted is False
    assert ddl.detect(reference, current, psi_threshold=None).drifted is True
    assert ddl.detect(reference, current, threshold=None).drifted is True


def test_option_validation():
    reference, current = make_frames()
    with pytest.raises(ValueError):
        ddl.detect(reference, current, threshold=2.0)
    with pytest.raises(ValueError):
        ddl.detect(reference, current, psi_threshold=-1.0)
    with pytest.raises(ValueError):
        ddl.detect(reference, current, sample=0)
    with pytest.raises(ValueError):
        ddl.detect(reference, current, sample=True)


def test_sampling_is_capped_and_deterministic():
    reference, current = make_frames(n=500)
    first = ddl.detect(reference, current, sample=100, random_state=7)
    second = ddl.detect(reference, current, sample=100, random_state=7)
    assert first.reference_rows == 100 and first.current_rows == 100
    assert any("sampled" in note for note in first.notes)
    assert first.to_dict() == second.to_dict()
    unsampled = ddl.detect(reference, current, sample=None)
    assert unsampled.reference_rows == 500 and unsampled.notes == []


def test_accepts_csv_paths(tmp_path):
    reference, current = make_frames(shift=1.5)
    ref_path = tmp_path / "ref.csv"
    cur_path = tmp_path / "cur.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    from_paths = ddl.detect(ref_path, str(cur_path))
    from_frames = ddl.detect(reference, current)
    assert from_paths.drifted_columns == from_frames.drifted_columns == ["x"]
    assert from_paths.columns["x"].psi == pytest.approx(from_frames.columns["x"].psi)


def test_accepts_parquet_paths(tmp_path):
    pytest.importorskip("pyarrow")
    reference, current = make_frames(shift=1.5)
    ref_path = tmp_path / "ref.parquet"
    cur_path = tmp_path / "cur.parquet"
    reference.to_parquet(ref_path, index=False)
    current.to_parquet(cur_path, index=False)
    assert ddl.detect(ref_path, cur_path).drifted_columns == ["x"]


def test_file_errors(tmp_path):
    reference, current = make_frames()
    bad = tmp_path / "data.txt"
    bad.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        ddl.detect(bad, current)
    with pytest.raises(FileNotFoundError):
        ddl.detect(tmp_path / "missing.csv", current)
    with pytest.raises(TypeError):
        ddl.detect([1, 2, 3], current)


def test_series_input_becomes_one_column():
    reference = pd.Series(np.arange(100, dtype=float), name="score")
    current = pd.Series(np.arange(100, dtype=float) + 200, name="score")
    report = ddl.detect(reference, current)
    assert list(report.columns) == ["score"]
    assert report.drifted


def test_duplicate_column_names_are_rejected():
    reference = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    current = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="duplicate column names"):
        ddl.detect(reference, current)
    with pytest.raises(ValueError, match="duplicate column names"):
        ddl.detect(current, reference)


def test_inputs_are_not_mutated():
    reference, current = make_frames(n=50)
    reference.loc[3, "x"] = np.nan
    ref_copy, cur_copy = reference.copy(), current.copy()
    ddl.detect(reference, current)
    pd.testing.assert_frame_equal(reference, ref_copy)
    pd.testing.assert_frame_equal(current, cur_copy)


def test_report_round_trips_through_json():
    reference, current = make_frames(shift=1.5)
    payload = ddl.detect(reference, current).to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["drifted_columns"] == ["x"]
