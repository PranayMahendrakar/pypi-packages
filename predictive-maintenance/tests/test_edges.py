"""Awkward frames: empty, tiny, missing, unicode, unsorted, from disk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import predictive_maintenance as pm
from conftest import make_frame


def test_empty_frame_is_rejected_clearly():
    with pytest.raises(ValueError, match="no numeric sensor channels"):
        pm.health_score(pd.DataFrame())


def test_frame_with_columns_but_no_rows_is_rejected():
    with pytest.raises(ValueError, match="at least 2 rows"):
        pm.health_score(pd.DataFrame({"vibration": pd.Series(dtype=float)}))


def test_single_row_is_rejected_clearly():
    with pytest.raises(ValueError, match="at least 2 rows"):
        pm.health_score(pd.DataFrame({"vibration": [1.0]}))


def test_two_rows_still_answers():
    health = pm.health_score(pd.DataFrame({"vibration": [1.0, 1.4]}))
    assert 0.0 <= health.score <= 100.0
    assert any("directional rather than precise" in note for note in health.notes)


def test_all_nan_channel_is_carried_as_constant():
    df = make_frame(n_rows=120, ramp=60)
    df["dead_sensor"] = np.nan
    health = pm.health_score(df)
    assert health.contributors["dead_sensor"] == 0.0
    assert any("entirely missing" in note for note in health.notes)
    assert health.series.notna().all()


def test_partly_missing_channel_is_filled_and_noted():
    df = make_frame(n_rows=120, ramp=60)
    df.loc[10:20, "vibration"] = np.nan
    health = pm.health_score(df)
    assert health.series.notna().all()
    assert any("carried forward" in note for note in health.notes)


def test_infinities_do_not_leak_into_the_score():
    df = make_frame(n_rows=120, ramp=60)
    df.loc[30, "temp_c"] = np.inf
    df.loc[31, "temp_c"] = -np.inf
    health = pm.health_score(df)
    assert np.isfinite(health.series.to_numpy()).all()
    assert 0.0 <= health.score <= 100.0


def test_mixed_dtypes_pick_only_the_numeric_channels():
    df = make_frame(n_rows=120, ramp=60)
    df["operator"] = "anna"
    df["shift"] = pd.Categorical(["day", "night"] * 60)
    df["running"] = True
    health = pm.health_score(df)
    assert set(health.channels) == {"vibration", "temp_c", "rpm"}


def test_non_numeric_channel_choice_is_rejected():
    df = make_frame(n_rows=60, ramp=30)
    df["operator"] = "anna"
    with pytest.raises(ValueError, match="must hold numbers"):
        pm.health_score(df, channels=["operator"])


def test_duplicate_column_names_raise_a_named_value_error():
    df = pd.DataFrame(np.ones((40, 3)), columns=["vibration", "vibration", "temp_c"])
    with pytest.raises(ValueError, match="duplicate column names: vibration"):
        pm.health_score(df)


def test_unicode_channel_names_survive_everywhere():
    df = make_frame(n_rows=160, ramp=80).rename(
        columns={"vibration": "vibración_mm/s", "temp_c": "温度_℃"}
    )
    health = pm.health_score(df)
    assert "vibración_mm/s" in health.contributors
    assert "温度_℃" in health.to_dict()["contributors"]
    text = health.summary()
    assert "vibraci" in text
    text.encode("utf-8")


def test_rows_out_of_time_order_are_sorted():
    df = make_frame(n_rows=160, ramp=80)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    ordered = pm.health_score(df)
    resorted = pm.health_score(shuffled)
    assert resorted.score == pytest.approx(ordered.score)
    assert any("sorted by the time axis" in note for note in resorted.notes)


def test_identical_timestamps_fall_back_to_row_order():
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01")] * 60,
            "vibration": np.linspace(1.0, 3.0, 60),
        }
    )
    health = pm.health_score(df)
    assert any("every timestamp is identical" in note for note in health.notes)
    assert 0.0 <= health.score <= 100.0


def test_numeric_cycle_column_is_used_as_the_time_axis():
    df = pd.DataFrame(
        {"cycle": np.arange(120.0), "vibration": np.linspace(1.0, 2.0, 120)}
    )
    health = pm.health_score(df)
    assert health.channels == ["vibration"]
    assert len(health.series) == 120


def test_unparseable_time_column_falls_back_to_row_order():
    df = pd.DataFrame(
        {"time": ["early"] * 30 + ["late"] * 30, "vibration": np.linspace(1.0, 2.0, 60)}
    )
    health = pm.health_score(df, time="time")
    assert any("using row order" in note for note in health.notes)


def test_reads_a_csv_from_disk(tmp_path):
    df = make_frame(n_rows=160, ramp=80)
    path = tmp_path / "sensors.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    from_disk = pm.health_score(str(path))
    assert from_disk.score == pytest.approx(pm.health_score(df).score)


def test_reads_a_tsv_from_disk(tmp_path):
    df = make_frame(n_rows=120, ramp=60)
    path = tmp_path / "sensors.tsv"
    df.to_csv(path, index=False, sep="\t", encoding="utf-8")
    assert 0.0 <= pm.health_score(str(path)).score <= 100.0


def test_missing_and_unsupported_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        pm.health_score(str(tmp_path / "nope.csv"))
    odd = tmp_path / "sensors.xlsx"
    odd.write_bytes(b"not really a spreadsheet")
    with pytest.raises(ValueError, match="unsupported file type"):
        pm.health_score(str(odd))


def test_a_series_is_accepted_as_one_channel():
    series = pd.Series(np.linspace(1.0, 2.5, 80), name="vibration")
    health = pm.health_score(series)
    assert health.channels == ["vibration"]


def test_wrong_input_type_is_rejected():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        pm.health_score(42)


def test_empty_channel_list_is_rejected():
    with pytest.raises(ValueError, match="at least one channel"):
        pm.health_score(make_frame(n_rows=60, ramp=30), channels=[])


def test_package_exports_and_version():
    assert pm.__version__ == "0.1.0"
    for name in ("health_score", "estimate_rul", "MaintenanceModel",
                 "HealthResult", "RULResult", "FailureRisk"):
        assert name in pm.__all__
        assert hasattr(pm, name)


def test_library_never_prints(capsys, degrading, labelled):
    df, labels = labelled
    pm.health_score(degrading)
    pm.estimate_rul(degrading)
    pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=10).predict(df)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_a_positional_baseline_follows_the_rows_when_they_are_sorted():
    """baseline=mask names rows, not positions in the order they arrived."""
    n_rows = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "v": np.concatenate([rng.normal(0.0, 0.1, 100), np.linspace(0.0, 5.0, 100)]),
        }
    )
    mask = np.zeros(n_rows, dtype=bool)
    mask[:60] = True

    ordered = pm.health_score(df, baseline=mask)
    reversed_frame = df.iloc[::-1].reset_index(drop=True)
    reordered = pm.health_score(reversed_frame, baseline=mask[::-1].copy())
    positions = pm.health_score(reversed_frame, baseline=list(range(140, 200)))

    assert reordered.score == pytest.approx(ordered.score)
    assert positions.score == pytest.approx(ordered.score)
    assert reordered.baseline_rows == ordered.baseline_rows
    assert any("carried through the sort" in note for note in reordered.notes)
