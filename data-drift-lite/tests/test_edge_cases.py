import json
import logging
import math

import numpy as np
import pandas as pd
import pytest

import data_drift_lite as ddl


def _finite(value):
    return value is not None and math.isfinite(value)


def test_constant_column_has_finite_psi():
    reference = pd.DataFrame({"k": [5.0] * 100})
    same = ddl.detect(reference, pd.DataFrame({"k": [5.0] * 100})).columns["k"]
    assert same.psi == 0.0 and same.statistic == 0.0 and same.p_value == 1.0
    assert not same.drifted
    shifted = ddl.detect(reference, pd.DataFrame({"k": [7.0] * 100})).columns["k"]
    assert _finite(shifted.psi) and _finite(shifted.statistic) and _finite(shifted.p_value)
    assert shifted.drifted  # the KS test sees a shifted constant


def test_constant_categorical_column():
    reference = pd.DataFrame({"k": ["only"] * 50})
    same = ddl.detect(reference, reference.copy()).columns["k"]
    assert same.psi == 0.0 and same.p_value == 1.0 and not same.drifted
    other = ddl.detect(reference, pd.DataFrame({"k": ["else"] * 50})).columns["k"]
    assert _finite(other.psi) and other.drifted
    assert any("unseen in reference" in note for note in other.notes)


def test_categories_present_on_one_side_only():
    reference = pd.DataFrame({"c": ["a"] * 60 + ["b"] * 40})
    current = pd.DataFrame({"c": ["b"] * 40 + ["z"] * 60})
    col = ddl.detect(reference, current).columns["c"]
    assert _finite(col.psi) and _finite(col.statistic) and _finite(col.p_value)
    assert col.drifted
    assert any("'z'" in note for note in col.notes)
    assert col.current_stats["n_categories"] == 2
    payload = col.to_dict()
    json.dumps(payload, allow_nan=False)


def test_low_cardinality_numeric_flip_is_caught_both_ways():
    mostly_zero = pd.DataFrame({"flag": [0] * 95 + [1] * 5})
    mostly_one = pd.DataFrame({"flag": [0] * 5 + [1] * 95})
    assert ddl.detect(mostly_zero, mostly_one).columns["flag"].psi > 0.2
    assert ddl.detect(mostly_one, mostly_zero).columns["flag"].psi > 0.2
    assert ddl.detect(mostly_zero, mostly_zero.copy()).columns["flag"].psi == 0.0


def test_tiny_batch_produces_a_note_not_a_crash(caplog):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(size=200), "c": rng.choice(list("ab"), 200)})
    current = reference.head(5)
    with caplog.at_level(logging.WARNING, logger="data_drift_lite"):
        report = ddl.detect(reference, current)
    assert report.current_rows == 5
    assert any("fewer than 20" in note for note in report.notes)
    assert any("fewer than 20" in record.getMessage() for record in caplog.records)
    assert "notes:" in report.summary()


def test_tiny_reference_is_noted_too():
    reference = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    report = ddl.detect(reference, pd.DataFrame({"x": np.arange(50, dtype=float)}))
    assert any(note.startswith("reference has only 3 rows") for note in report.notes)


def test_single_row_batch():
    reference = pd.DataFrame({"x": np.arange(100, dtype=float), "c": ["a", "b"] * 50})
    report = ddl.detect(reference, reference.iloc[[0]])
    assert report.current_rows == 1
    assert any("only 1 row " in note for note in report.notes)
    for col in report.columns.values():
        assert _finite(col.psi)
    json.dumps(report.to_dict(), allow_nan=False)


def test_datetime_columns_are_numeric():
    reference = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=60, freq="D")})
    current = pd.DataFrame({"ts": pd.date_range("2024-06-01", periods=60, freq="D")})
    col = ddl.detect(reference, current).columns["ts"]
    assert col.kind == "numeric" and col.test == "ks"
    assert col.drifted
    assert col.reference_stats["min"].startswith("2024-01-01")
    assert col.current_stats["max"].startswith("2024-07-30")
    same = ddl.detect(reference, reference.copy()).columns["ts"]
    assert same.psi == 0.0 and not same.drifted


def test_timezone_aware_and_nat_datetimes():
    reference = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")})
    current = reference.copy()
    current.loc[:9, "ts"] = pd.NaT
    col = ddl.detect(reference, current).columns["ts"]
    assert col.kind == "numeric"
    assert _finite(col.psi) and col.current_stats["missing_share"] == 0.25
    assert col.reference_stats["min"].startswith("2024-01-01T00:00:00")


def test_timedelta_columns_are_numeric():
    reference = pd.DataFrame({"d": pd.to_timedelta(np.arange(30), unit="m")})
    current = pd.DataFrame({"d": pd.to_timedelta(np.arange(30) + 300, unit="m")})
    col = ddl.detect(reference, current).columns["d"]
    assert col.kind == "numeric" and col.drifted
    assert "days" in col.reference_stats["min"]


def test_bool_columns_are_categorical():
    reference = pd.DataFrame({"b": [True] * 50 + [False] * 50})
    current = pd.DataFrame({"b": [True] * 95 + [False] * 5})
    col = ddl.detect(reference, current).columns["b"]
    assert col.kind == "categorical" and col.test == "chi2"
    assert col.drifted
    assert set(col.reference_stats["top"]) == {"True", "False"}


def test_nullable_dtypes():
    reference = pd.DataFrame(
        {
            "i": pd.array([1, 2, 3, 4, None, 6] * 10, dtype="Int64"),
            "b": pd.array([True, False, None] * 20, dtype="boolean"),
            "s": pd.array(["x", "y", None] * 20, dtype="string"),
        }
    )
    report = ddl.detect(reference, reference.copy())
    assert report.columns["i"].kind == "numeric"
    assert report.columns["b"].kind == "categorical"
    assert report.columns["s"].kind == "categorical"
    assert not report.drifted
    for col in report.columns.values():
        assert col.psi == 0.0


def test_empty_frames():
    empty = pd.DataFrame()
    report = ddl.detect(empty, empty)
    assert report.columns == {} and not report.drifted and report.drift_share == 0.0
    assert "no columns compared" in report.summary()
    json.dumps(report.to_dict(), allow_nan=False)


def test_current_with_zero_rows():
    reference = pd.DataFrame({"x": np.arange(100, dtype=float), "c": ["a", "b"] * 50})
    report = ddl.detect(reference, reference.iloc[0:0])
    assert report.current_rows == 0 and not report.drifted
    for col in report.columns.values():
        assert col.p_value is None and col.psi is None and not col.drifted
        assert any("skipped" in note for note in col.notes)
    assert "n/a" in report.summary()
    json.dumps(report.to_dict(), allow_nan=False)


def test_reference_with_zero_rows():
    reference = pd.DataFrame({"x": pd.Series([], dtype=float), "c": pd.Series([], dtype=object)})
    current = pd.DataFrame({"x": np.arange(30, dtype=float), "c": ["a", "b", "c"] * 10})
    report = ddl.detect(reference, current)
    assert report.reference_rows == 0 and not report.drifted
    for col in report.columns.values():
        assert col.psi is None and col.p_value is None


def test_all_nan_column():
    reference = pd.DataFrame({"x": np.arange(100, dtype=float)})
    gone = ddl.detect(reference, pd.DataFrame({"x": [np.nan] * 100})).columns["x"]
    assert gone.p_value is None and gone.statistic is None
    assert _finite(gone.psi) and gone.psi > 0.2 and gone.drifted
    assert any("missing share moved" in note for note in gone.notes)
    both = pd.DataFrame({"x": [np.nan] * 50})
    same = ddl.detect(both, both.copy()).columns["x"]
    assert same.psi == 0.0 and not same.drifted
    assert same.reference_stats["mean"] is None
    json.dumps(same.to_dict(), allow_nan=False)


def test_missing_values_count_in_psi():
    reference = pd.DataFrame({"x": np.arange(200, dtype=float)})
    current = reference.copy()
    current.loc[::2, "x"] = np.nan  # same spread of values, but half of them vanished
    col = ddl.detect(reference, current).columns["x"]
    assert col.p_value > 0.05  # KS only sees the non-missing values
    assert col.psi > 0.2 and col.drifted  # PSI sees the missing bin


def test_inf_values_are_treated_as_missing():
    reference = pd.DataFrame({"x": np.arange(100, dtype=float)})
    current = reference.copy()
    current.loc[[1, 2], "x"] = [np.inf, -np.inf]
    col = ddl.detect(reference, current).columns["x"]
    assert _finite(col.psi) and _finite(col.statistic)
    assert col.current_stats["missing_share"] == 0.02
    assert any("non-finite" in note for note in col.notes)
    json.dumps(col.to_dict(), allow_nan=False)


def test_mixed_dtypes_and_unicode():
    n = 60
    reference = pd.DataFrame(
        {
            "i": np.arange(n),
            "f": np.linspace(0, 1, n),
            "s": ["café", "東京", "naïve"] * (n // 3),
            "b": [True, False] * (n // 2),
            "t": pd.date_range("2024-01-01", periods=n, freq="D"),
            "cat": pd.Categorical(["x", "y", "z"] * (n // 3)),
            "o": [1, "one", None] * (n // 3),
        }
    )
    report = ddl.detect(reference, reference.copy())
    kinds = {name: col.kind for name, col in report.columns.items()}
    assert kinds == {
        "i": "numeric",
        "f": "numeric",
        "s": "categorical",
        "b": "categorical",
        "t": "numeric",
        "cat": "categorical",
        "o": "categorical",
    }
    assert not report.drifted
    assert "東京" in report.columns["s"].reference_stats["top"]
    assert report.columns["o"].reference_stats["missing_share"] == pytest.approx(1 / 3)
    text = json.dumps(report.to_dict(), allow_nan=False, ensure_ascii=False)
    assert "café" in text
    assert isinstance(report.summary(), str)


def test_schema_missing_column_raises_the_flag():
    reference = pd.DataFrame({"x": np.arange(50, dtype=float), "y": np.arange(50, dtype=float)})
    report = ddl.detect(reference, reference[["x"]])
    assert report.missing_columns == ["y"]
    assert report.drifted and report.drifted_columns == [] and report.drift_share == 0.0
    assert report.schema.drifted
    assert "missing in current: y" in report.summary()
    assert report.to_dict()["schema"]["missing_columns"] == ["y"]


def test_schema_new_column_is_reported_but_not_drift():
    reference = pd.DataFrame({"x": np.arange(50, dtype=float)})
    current = reference.assign(extra=1)
    report = ddl.detect(reference, current)
    assert report.new_columns == ["extra"]
    assert not report.drifted and not report.schema.drifted and report.schema.changed
    assert "new in current: extra" in report.summary()


def test_schema_dtype_change_is_reported_and_excluded():
    reference = pd.DataFrame({"x": np.arange(50, dtype="int64"), "y": np.arange(50, dtype="int64")})
    current = pd.DataFrame({"x": [str(v) for v in range(50)], "y": np.arange(50, dtype="int64")})
    report = ddl.detect(reference, current)
    assert report.dtype_changed == {"x": ("int64", "object")}
    assert "x" not in report.columns and "y" in report.columns
    assert report.drifted and report.drifted_columns == []
    assert "dtype changed: x (int64 -> object)" in report.summary()
    assert report.to_dict()["schema"]["dtype_changed"] == {"x": {"reference": "int64", "current": "object"}}


def test_int_to_float_is_not_a_dtype_change():
    reference = pd.DataFrame({"x": np.arange(50)})
    current = pd.DataFrame({"x": np.arange(50, dtype=float)})
    report = ddl.detect(reference, current)
    assert report.dtype_changed == {} and not report.drifted
    assert report.columns["x"].current_stats["dtype"] == "float64"


def test_high_cardinality_note():
    reference = pd.DataFrame({"id": [f"u{i}" for i in range(300)]})
    col = ddl.detect(reference, reference.copy()).columns["id"]
    assert any("high cardinality" in note for note in col.notes)
    assert col.psi == 0.0
