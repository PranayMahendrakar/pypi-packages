"""The happy paths: the quickstart, every input shape, and every method."""
import json

import numpy as np
import pandas as pd
import pytest

import timeseries_anomaly
from timeseries_anomaly import METHODS, AnomalyResult, Detector, detect

QUICKSTART = [10, 11, 10, 12, 11, 10, 11, 60, 10, 11, 12, 10, 11, 10, 12]


def spiky(n=120, spike_at=47, spike=99.0):
    """A calm series with one obvious spike in the middle."""
    values = [10.0 + (i % 5) * 0.2 for i in range(n)]
    values[spike_at] = spike
    return values


def test_quickstart_from_the_readme():
    result = detect(QUICKSTART)
    assert result.anomalies == [7]
    assert result.n_anomalies == 1
    assert result.method_used == "zscore"
    assert result.rate == pytest.approx(1 / 15)
    assert "60" in result.summary()


def test_version_and_exports():
    assert timeseries_anomaly.__version__ == "0.1.0"
    assert set(METHODS) == {"auto", "all", "zscore", "iqr", "rolling", "seasonal", "ewma"}
    for name in ("detect", "Detector", "AnomalyResult", "METHODS"):
        assert hasattr(timeseries_anomaly, name)


@pytest.mark.parametrize("method", ["zscore", "iqr", "rolling", "ewma", "auto", "all"])
def test_every_method_finds_the_one_spike(method):
    result = detect(spiky(), method=method)
    assert result.anomalies == [47]
    assert result.method_used in METHODS
    assert result.mask.dtype == bool
    assert result.scores.shape == result.values.shape == result.expected.shape


def test_seasonal_method_with_an_explicit_period():
    result = detect(spiky(), method="seasonal", seasonality=5)
    assert result.anomalies == [47]
    assert result.seasonality == 5
    assert result.method_used == "seasonal"


def test_seasonal_period_inferred_from_a_datetime_index():
    index = pd.date_range("2026-01-01", periods=96, freq="h")
    values = 20 + 5 * np.sin(2 * np.pi * np.arange(96) / 24.0)
    values[40] = 60.0
    result = detect(pd.Series(values, index=index, name="temp_C"), method="seasonal")
    assert result.anomalies == [40]
    assert result.seasonality == 24
    assert any("inferred a season" in note for note in result.warnings)


def test_seasonal_period_inferred_from_the_shape_without_timestamps():
    values = list(np.tile([1.0, 4.0, 9.0, 4.0, 1.0, 0.0], 12))
    values[37] = 40.0
    result = detect(values, method="seasonal")
    assert result.seasonality == 6
    assert result.anomalies == [37]


@pytest.mark.parametrize(
    "wrap",
    [
        lambda v: v,
        lambda v: tuple(v),
        lambda v: np.asarray(v, dtype=float),
        lambda v: pd.Series(v),
        lambda v: pd.DataFrame({"value": v}),
    ],
)
def test_every_in_memory_input_shape_gives_the_same_answer(wrap):
    values = spiky()
    assert detect(wrap(values)).anomalies == [47]


def test_dataframe_with_named_value_and_time_columns():
    frame = pd.DataFrame(
        {
            "recorded_at": pd.date_range("2026-03-01", periods=120, freq="min"),
            "temperature": spiky(),
        }
    )
    result = detect(frame, value="temperature", time="recorded_at")
    assert result.anomalies == [47]
    assert result.label == "temperature"
    assert result.time_label == "recorded_at"


def test_csv_round_trip(tmp_path):
    path = tmp_path / "readings.csv"
    pd.DataFrame(
        {"recorded_at": pd.date_range("2026-03-01", periods=120, freq="min"), "temp": spiky()}
    ).to_csv(path, index=False, encoding="utf-8")
    result = detect(str(path), value="temp", time="recorded_at")
    assert result.anomalies == [47]


def test_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "readings.parquet"
    pd.DataFrame({"temp": spiky()}).to_parquet(path, index=False)
    assert detect(str(path)).anomalies == [47]


def test_sensitivity_trades_recall_for_quiet():
    values = spiky()
    counts = [detect(values, sensitivity=s).n_anomalies for s in (2.0, 3.0, 50.0, 500.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_to_frame_columns_and_alignment():
    result = detect(spiky())
    frame = result.to_frame()
    assert list(frame.columns) == ["time", "value", "expected", "score", "is_anomaly"]
    assert len(frame) == result.n_points
    assert bool(frame.loc[frame.index[47], "is_anomaly"])


def test_to_frame_preserves_the_callers_index():
    labels = [f"row{i}" for i in range(120)]
    series = pd.Series(spiky(), index=labels, name="temp")
    frame = detect(series).to_frame()
    assert list(frame.index) == labels
    assert frame.loc["row47", "is_anomaly"]


def test_to_dict_is_json_safe_and_complete():
    result = detect(spiky())
    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text)["n_anomalies"] == 1
    for key in ("method", "method_used", "n_points", "rate", "anomalies", "warnings", "summary"):
        assert key in payload
    assert payload["anomalies"][0]["index"] == 47


def test_summary_top_plot_data_and_dunders():
    result = detect(spiky())
    assert isinstance(result.summary(), str)
    assert len(result) == 120
    assert bool(result) is True
    assert result.top(3)[0]["index"] == 47
    plot = result.plot_data()
    assert plot["anomaly_index"] == [47]
    assert len(plot["value"]) == 120
    assert result.time_at(0) == 0
    assert isinstance(result, AnomalyResult)


def test_quiet_series_is_falsy():
    assert not detect([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02])


@pytest.mark.parametrize("method", list(METHODS))
def test_results_are_deterministic(method):
    values = spiky()
    first = detect(values, method=method, seasonality=5)
    second = detect(values, method=method, seasonality=5)
    assert np.array_equal(first.mask, second.mask)
    assert np.allclose(first.scores, second.scores, equal_nan=True)
    assert first.to_dict() == second.to_dict()


def test_all_needs_a_strict_majority():
    result = detect(spiky(), method="all")
    assert result.method_used == "all"
    assert len(result.methods_used) > 1
    assert result.anomalies == [47]


def test_detector_is_reachable_from_the_package_root():
    assert Detector().detect(QUICKSTART).anomalies == [7]


def test_text_dates_in_a_csv_are_found_without_being_named(tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame(
        {
            "recorded_at": pd.date_range("2026-03-01", periods=120, freq="min"),
            "temperature": spiky(),
        }
    ).to_csv(path, index=False, encoding="utf-8")
    result = detect(str(path))
    assert result.time_label == "recorded_at"
    assert result.label == "temperature"
    assert result.anomalies == [47]
    assert "2026-03-01" in result.summary()
    assert any("time axis" in note for note in result.warnings)


def test_plain_text_is_not_mistaken_for_a_clock():
    frame = pd.DataFrame(
        {
            "part-code": ["a-1", "b-2", "c-3", "d-4", "e-5", "f-6", "g-7", "h-8"],
            "reading": [1.0, 1.1, 0.9, 1.0, 9.0, 1.0, 1.1, 0.9],
        }
    )
    result = detect(frame)
    assert result.time_label is None
    assert result.label == "reading"
    assert result.anomalies == [4]


def test_text_dates_are_sorted_like_real_timestamps(tmp_path):
    path = tmp_path / "shuffled.csv"
    stamps = [f"2026-04-{d:02d} 00:00:00" for d in (5, 1, 2, 3, 4, 6, 7, 8, 9, 10)]
    pd.DataFrame({"when": stamps, "v": [1.0, 1.0, 1.0, 40.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}).to_csv(
        path, index=False, encoding="utf-8"
    )
    result = detect(str(path), method="zscore")
    assert result.time_label == "when"
    assert any("not in order" in note for note in result.warnings)
    assert result.anomalies == [3]
