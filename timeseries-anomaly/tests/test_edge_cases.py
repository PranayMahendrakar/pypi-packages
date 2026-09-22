"""The awkward data: nothing here may raise, and every fix must be recorded."""
import numpy as np
import pandas as pd
import pytest

from timeseries_anomaly import detect


def test_empty_list_returns_an_empty_result():
    result = detect([])
    assert result.n_points == 0
    assert result.anomalies == []
    assert result.mask.shape == (0,)
    assert result.scores.shape == (0,)
    assert result.rate == 0.0
    assert result.warnings
    assert isinstance(result.summary(), str)


def test_empty_dataframe_returns_an_empty_result():
    assert detect(pd.DataFrame({"value": []})).n_anomalies == 0
    assert detect(pd.DataFrame()).n_points == 0


def test_single_point_returns_an_empty_result():
    result = detect([42.0])
    assert result.n_points == 1
    assert result.n_anomalies == 0
    assert result.mask.tolist() == [False]
    assert any("single point" in note for note in result.warnings)


def test_single_row_dataframe():
    result = detect(pd.DataFrame({"value": [3.0]}))
    assert result.n_points == 1
    assert result.n_anomalies == 0


def test_constant_series_flags_nothing_and_never_divides_by_zero():
    result = detect([5.0] * 60)
    assert result.n_anomalies == 0
    assert result.scale is None
    assert result.scale_kind == "none"
    assert np.all(np.isfinite(result.scores))
    assert np.all(result.scores == 0.0)
    assert any("no point can be an anomaly" in note for note in result.warnings)


def test_a_perfect_ramp_is_not_an_anomaly():
    assert detect([float(i) for i in range(80)]).n_anomalies == 0


def test_mad_of_zero_falls_back_to_the_standard_deviation():
    values = [5.0] * 30 + [9.0] + [5.0] * 4
    result = detect(values, method="zscore")
    assert result.scale_kind == "std"
    assert result.anomalies == [30]
    assert any("standard deviation" in note for note in result.warnings)


def test_no_usable_spread_at_all_reports_nothing():
    result = detect([2.5] * 40, method="zscore")
    assert result.scale is None
    assert result.n_anomalies == 0


def test_series_shorter_than_the_rolling_window_falls_back_to_zscore():
    result = detect([1.0, 2.0, 3.0], method="rolling")
    assert result.method_used == "zscore"
    assert any("fell back to zscore" in note for note in result.warnings)
    assert any("rolling window" in note for note in result.warnings)


def test_all_nan_series_is_handled():
    result = detect([np.nan] * 12)
    assert result.n_points == 12
    assert result.n_valid == 0
    assert result.n_missing == 12
    assert result.n_anomalies == 0


def test_nan_values_are_ignored_and_never_flagged():
    values = [1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 40.0, 1.0, 1.0, 1.0]
    result = detect(values, method="zscore")
    assert result.anomalies == [6]
    assert not bool(result.mask[3])
    assert result.n_missing == 1
    assert result.n_valid == 9


def test_nan_does_not_distort_the_baseline():
    clean = [1.0, 1.2, 0.8, 1.0, 9.0, 1.1, 0.9, 1.0, 1.0, 1.2]
    holed = clean + [np.nan] * 5
    assert detect(holed, method="zscore").anomalies == detect(clean, method="zscore").anomalies


def test_unsorted_timestamps_are_sorted_and_recorded():
    stamps = pd.to_datetime(
        [
            "2026-01-05",
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
            "2026-01-10",
        ]
    )
    values = [1.0, 1.0, 1.0, 40.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = detect(pd.Series(values, index=stamps), method="zscore")
    assert any("not in order" in note for note in result.warnings)
    assert result.anomalies == [3]
    assert result.values[3] == 40.0
    assert bool(result.mask[3])


def test_duplicate_timestamps_are_recorded_and_every_point_is_kept():
    stamps = pd.to_datetime(["2026-01-01"] * 2 + [f"2026-01-{d:02d}" for d in range(2, 10)])
    values = [1.0, 1.0, 1.0, 1.0, 40.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = detect(pd.Series(values, index=stamps), method="zscore")
    assert result.n_points == 10
    assert any("duplicate timestamp" in note for note in result.warnings)
    assert result.anomalies == [4]


def test_mask_and_scores_align_with_the_input_the_caller_passed():
    stamps = pd.to_datetime([f"2026-02-{d:02d}" for d in range(1, 21)])[::-1]
    values = [1.0] * 20
    values[7] = 50.0
    result = detect(pd.Series(values, index=stamps), method="zscore")
    assert result.mask.shape == (20,)
    assert bool(result.mask[7])
    assert result.anomalies == [7]
    assert result.values.tolist() == values
    frame = result.to_frame()
    assert frame["value"].tolist() == values
    assert list(frame.index) == list(stamps)


def test_mixed_dtypes_pick_the_numeric_column():
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2026-01-01", periods=10, freq="D"),
            "note": list("abcdefghij"),
            "reading": [1.0, 1.1, 0.9, 1.0, 9.0, 1.0, 1.1, 0.9, 1.0, 1.2],
        }
    )
    result = detect(frame)
    assert result.label == "reading"
    assert result.time_label == "when"
    assert result.anomalies == [4]


def test_unicode_labels_survive_the_whole_pipeline():
    frame = pd.DataFrame(
        {
            "時刻": pd.date_range("2026-01-01", periods=10, freq="D"),
            "température_°C": [1.0, 1.1, 0.9, 1.0, 9.0, 1.0, 1.1, 0.9, 1.0, 1.2],
        }
    )
    result = detect(frame, value="température_°C", time="時刻")
    assert result.anomalies == [4]
    assert "température_°C" in result.summary()
    assert result.to_dict()["label"] == "température_°C"


def test_duplicate_column_names_raise_a_clear_error():
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["v", "v"])
    with pytest.raises(ValueError, match="duplicate column names"):
        detect(frame)


def test_helpful_errors_for_bad_options_and_missing_columns():
    with pytest.raises(ValueError, match="unknown method"):
        detect([1.0, 2.0, 3.0], method="magic")
    with pytest.raises(ValueError, match="sensitivity"):
        detect([1.0, 2.0, 3.0], sensitivity=0.0)
    with pytest.raises(ValueError, match="seasonality"):
        detect([1.0, 2.0, 3.0], seasonality=1)
    with pytest.raises(ValueError, match="not in the table"):
        detect(pd.DataFrame({"a": [1.0, 2.0]}), value="b")
    with pytest.raises(ValueError, match="not in the table"):
        detect(pd.DataFrame({"a": [1.0, 2.0]}), time="b")
    with pytest.raises(FileNotFoundError):
        detect("no-such-file.csv")
    with pytest.raises(ValueError, match="unsupported file type"):
        detect(__file__)
    with pytest.raises(TypeError, match="unsupported data type"):
        detect(object())
    with pytest.raises(ValueError, match="not numbers"):
        detect(["a", "b", "c"])
    with pytest.raises(ValueError, match="no numeric column"):
        detect(pd.DataFrame({"a": list("xyz")}))


def test_infinities_are_treated_as_missing_not_as_anomalies():
    values = [1.0, 1.1, np.inf, 1.0, 9.0, 1.0, 1.1, 0.9, 1.0, 1.2]
    result = detect(values, method="zscore")
    assert not bool(result.mask[2])
    assert result.anomalies == [4]
