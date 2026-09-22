"""Awkward input: empty, one row, all NaN, mixed dtypes, unicode, duplicate columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine_health import HealthMonitor, score


def test_empty_frame_raises_a_clear_error():
    with pytest.raises(ValueError) as excinfo:
        score(pd.DataFrame({"temp": []}))
    assert "no rows" in str(excinfo.value)


def test_frame_with_no_columns_raises():
    with pytest.raises(ValueError) as excinfo:
        score(pd.DataFrame())
    assert "no columns" in str(excinfo.value)


def test_single_row_is_scored_not_rejected():
    result = score(pd.DataFrame({"temp": [42.0]}))
    assert 0 <= result.value <= 100
    assert result.n_rows == 1
    assert any("compares the data with itself" in note for note in result.notes)


def test_two_rows_still_produce_a_score():
    result = score(pd.DataFrame({"temp": [42.0, 43.0]}))
    assert 0 <= result.value <= 100


def test_all_nan_channel_scores_zero_on_availability_alone():
    result = score(pd.DataFrame({"temp": [np.nan] * 50}))
    assert result.value == 0.0
    assert result.grade == "F"
    assert result.components["availability"] == 0.0
    assert result.weights["availability"] == pytest.approx(1.0)
    assert result.weights["stability"] == 0.0
    assert any("stability was not scored" in note for note in result.notes)


def test_one_good_channel_survives_an_all_nan_neighbour():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"good": rng.normal(10, 1, 200), "dead": [np.nan] * 200})
    result = score(df)
    assert 0 < result.value < 100
    assert result.contributors["dead"] > result.contributors["good"]


def test_no_numeric_channel_lists_the_columns():
    df = pd.DataFrame({"site": ["a", "b"], "shift": ["day", "night"]})
    with pytest.raises(ValueError) as excinfo:
        score(df)
    assert "no numeric channel" in str(excinfo.value)
    assert "site" in str(excinfo.value)


def test_mixed_dtypes_are_sorted_out(unicode_df):
    result = score(unicode_df)
    assert result.channels == ["温度", "vibración"]
    assert any("non-numeric" in note for note in result.notes)
    assert "opérateur" in " ".join(result.notes)


def test_unicode_survives_summary_and_json(unicode_df):
    import json

    result = score(unicode_df, rules={"温度": {"max": 65}})
    text = result.summary()
    assert "温度" in text
    json.dumps(result.to_dict(), ensure_ascii=False)
    assert result.violations[0].channel == "温度"


def test_boolean_channel_is_usable():
    values = [True] * 40 + [False] * 10
    result = score(pd.DataFrame({"pump_on": values}))
    assert 0 <= result.value <= 100


def test_integer_channel_is_usable():
    result = score(pd.DataFrame({"count": list(range(100))}))
    assert 0 <= result.value <= 100


def test_duplicate_columns_raise_value_error():
    df = pd.DataFrame(np.arange(20.0).reshape(10, 2), columns=["temp", "temp"])
    with pytest.raises(ValueError) as excinfo:
        score(df)
    assert "duplicate column names" in str(excinfo.value)
    assert "temp" in str(excinfo.value)


def test_infinite_readings_count_as_missing():
    values = np.r_[np.full(50, 1.0), np.full(10, np.inf), np.full(40, 1.0)]
    result = score(pd.DataFrame({"x": values}))
    assert result.components["availability"] < 100


def test_unknown_channel_argument_lists_what_exists():
    with pytest.raises(ValueError) as excinfo:
        score(pd.DataFrame({"temp": [1.0, 2.0]}), channels=["pressure"])
    assert "pressure" in str(excinfo.value)
    assert "temp" in str(excinfo.value)


def test_non_numeric_channel_argument_is_explained():
    with pytest.raises(ValueError) as excinfo:
        score(pd.DataFrame({"temp": [1.0, 2.0], "site": ["a", "b"]}), channels=["site"])
    assert "not numeric" in str(excinfo.value)


def test_unknown_time_column_lists_what_exists():
    with pytest.raises(ValueError) as excinfo:
        score(pd.DataFrame({"temp": [1.0, 2.0]}), time="when")
    assert "when" in str(excinfo.value)


def test_unusable_timestamps_are_a_note_not_a_crash():
    df = pd.DataFrame(
        {
            "ts": ["2026-01-01", "not a date", "2026-01-03", "2026-01-04"],
            "temp": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = score(df, time="ts")
    assert any("unusable" in note for note in result.notes)
    assert 0 <= result.value <= 100


def test_bad_baseline_arguments_are_rejected():
    df = pd.DataFrame({"temp": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ValueError):
        score(df, baseline=1.5)
    with pytest.raises(ValueError):
        score(df, baseline=0)
    with pytest.raises(ValueError):
        score(df, baseline=[False] * 5)
    with pytest.raises(ValueError):
        score(df, baseline=[True, False])
    with pytest.raises(TypeError):
        score(df, baseline=True)


def test_baseline_mask_covering_everything_still_scores():
    df = pd.DataFrame({"temp": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = score(df, baseline=[True] * 5)
    assert 0 <= result.value <= 100
    assert any("covers every row" in note for note in result.notes)


def test_empty_baseline_frame_is_rejected():
    df = pd.DataFrame({"temp": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        score(df, baseline=pd.DataFrame({"temp": []}))


def test_baseline_frame_missing_a_channel_is_noted():
    df = pd.DataFrame({"a": [1.0] * 20, "b": [2.0] * 20})
    result = score(df, baseline=pd.DataFrame({"a": [1.0] * 10}))
    assert any("no column for channel" in note for note in result.notes)
    assert 0 <= result.value <= 100


def test_monitor_rejects_an_empty_baseline():
    with pytest.raises(ValueError):
        HealthMonitor(pd.DataFrame({"temp": []}))


def test_wrong_input_type_is_a_type_error():
    with pytest.raises(TypeError):
        score(42)


def test_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        score(tmp_path / "nope.csv")


def test_unsupported_file_type(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("not really a spreadsheet", encoding="utf-8")
    with pytest.raises(ValueError):
        score(path)


def test_load_table_reads_csv_tsv_and_frames(tmp_path):
    from machine_health import load_table

    df = pd.DataFrame({"temp": [1.0, 2.0], "site": ["a", "b"]})
    assert load_table(df) is df
    assert load_table(pd.Series([1.0, 2.0], name="temp")).shape == (2, 1)

    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    assert load_table(csv).equals(df)

    tsv = tmp_path / "t.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    assert load_table(tsv).equals(df)


def test_parquet_round_trip(tmp_path, drifting_df):
    pytest.importorskip("pyarrow")
    path = tmp_path / "telemetry.parquet"
    drifting_df.to_parquet(path)
    assert score(path).value == pytest.approx(score(drifting_df).value)
