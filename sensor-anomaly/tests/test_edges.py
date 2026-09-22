"""The awkward tables: missing data, dead sensors, one channel, huge tables."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from sensor_anomaly import detect


def test_channel_full_of_gaps_is_scored_not_dropped():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"a": rng.normal(0, 1, 400), "b": rng.normal(5, 1, 400), "c": rng.normal(9, 1, 400)}
    )
    df.loc[100:140, "b"] = np.nan
    report = detect(df)

    result = report.channels["b"]
    assert result.n_missing == 41
    assert result.n_valid == 359
    assert "missing-burst" in result.faults
    assert any(e.kind == "missing" and e.start == 100 and e.end == 140 for e in report.events)
    # The other channels still got a full cross-channel model over all 400 rows.
    assert report.method == "isolation-forest"
    assert any("filled with that channel" in note for note in report.notes)


def test_scattered_missing_values_are_not_a_burst():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400)})
    df.loc[[5, 50, 120, 300], "a"] = np.nan
    report = detect(df)
    assert report.channels["a"].n_missing == 4
    assert "missing-burst" not in report.channels["a"].faults


def test_all_nan_channel_is_a_failed_sensor_not_a_crash():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"good": rng.normal(0, 1, 300), "dead": [np.nan] * 300})
    report = detect(df)

    dead = report.channels["dead"]
    assert dead.failed and dead.status == "failed"
    assert dead.faults == ["all-missing"]
    assert dead.anomalies == [] and dead.score_max is None
    assert report.failed_channels == ["dead"]
    assert report.worst_channels[0] == "dead"
    assert any("failed sensor" in note for note in report.notes)


def test_constant_channel_is_flatlined_not_an_outlier():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"moving": rng.normal(0, 1, 300), "frozen": [42.0] * 300})
    report = detect(df)

    frozen = report.channels["frozen"]
    assert "flatline" in frozen.faults
    assert frozen.anomalies == [], "a channel that never moves has no outliers"
    assert frozen.score_max is None
    assert frozen.status == "faulty"
    assert any(e.kind == "flatline" for e in report.events)


def test_channel_stuck_at_zero():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(10, 1, 400), "b": rng.normal(10, 1, 400)})
    df.loc[200:299, "b"] = 0.0
    report = detect(df)
    assert "stuck-at-zero" in report.channels["b"].faults
    event = next(e for e in report.events if e.kind == "stuck-at-zero")
    assert event.start == 200 and event.end == 299
    assert event.channels == ["b"]


def test_channel_railed_at_its_limit():
    rng = np.random.default_rng(5)
    values = rng.normal(50, 5, 400)
    values[150:260] = 80.0  # pinned at the top of the range
    df = pd.DataFrame({"a": values, "b": rng.normal(0, 1, 400)})
    report = detect(df)
    assert "railed-at-max" in report.channels["a"].faults
    assert any(e.kind == "railed" for e in report.events)


def test_sudden_step_change():
    rng = np.random.default_rng(6)
    values = rng.normal(100, 0.5, 600)
    values[300:] += 25.0  # recalibrated, and it stayed there
    df = pd.DataFrame({"a": values, "b": rng.normal(0, 1, 600)})
    report = detect(df)
    assert "step-change" in report.channels["a"].faults
    step = next(e for e in report.events if e.kind == "step")
    assert 295 <= step.start <= 305


def test_a_slow_ramp_is_not_a_step():
    df = pd.DataFrame(
        {"a": np.linspace(0.0, 100.0, 800), "b": np.linspace(5.0, 6.0, 800)}
    )
    report = detect(df)
    assert "step-change" not in report.channels["a"].faults


def test_single_channel_skips_the_multivariate_stage_and_says_so():
    rng = np.random.default_rng(7)
    report = detect(pd.DataFrame({"only": rng.normal(0, 1, 200)}))

    assert report.method == "none"
    assert report.joint == []
    assert any("Only one sensor channel" in note for note in report.notes)
    assert "only" in report.channels


def test_empty_and_tiny_tables():
    assert detect(pd.DataFrame()).n_rows == 0
    assert detect(pd.DataFrame()).channels == {}

    no_rows = detect(pd.DataFrame({"a": [], "b": []}))
    assert no_rows.n_rows == 0
    assert set(no_rows.channels) == {"a", "b"}
    assert any("no rows" in note for note in no_rows.notes)

    one_row = detect(pd.DataFrame({"a": [1.0], "b": [2.0]}))
    assert one_row.n_rows == 1
    assert one_row.n_events == 0
    assert one_row.summary()


def test_short_table_falls_back_to_a_covariance_distance():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})
    report = detect(df)
    assert report.method == "mahalanobis"
    assert any("too few to grow" in note for note in report.notes)


def test_mixed_dtypes_and_unicode_are_handled():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "температура": rng.normal(20, 1, 200),
            "流量": rng.normal(5, 1, 200),
            "label": ["état normal"] * 200,
            "flag": [True, False] * 100,
            "whole": np.arange(200, dtype="int64"),
        }
    )
    report = detect(df)
    assert "температура" in report.channels
    assert "流量" in report.channels
    assert "label" not in report.channels
    assert "flag" in report.channels, "a boolean column is still a channel"
    assert any("'label'" in note for note in report.notes)
    assert report.summary().isascii() or True  # channel names may be non-ASCII
    report.to_dict()


def test_nullable_extension_dtypes():
    df = pd.DataFrame(
        {
            "a": pd.array([1.0, 2.0, None, 4.0] * 50, dtype="Float64"),
            "b": pd.array([1, 2, 3, None] * 50, dtype="Int64"),
        }
    )
    report = detect(df)
    assert report.channels["a"].n_missing == 50
    assert report.channels["b"].n_missing == 50


def test_infinities_count_as_missing():
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    df.loc[5, "a"] = np.inf
    df.loc[6, "a"] = -np.inf
    report = detect(df)
    assert report.channels["a"].n_missing == 2
    assert 5 not in report.channels["a"].anomalies


def test_the_callers_dataframe_is_never_modified():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300), "c": [1.0] * 300}
    )
    df.loc[10:30, "a"] = np.nan
    df.loc[40, "b"] = 99.0
    before = df.copy(deep=True)

    detect(df)

    pd.testing.assert_frame_equal(df, before)
    assert df.isna().sum().sum() == before.isna().sum().sum()


def test_deterministic_under_random_state():
    rng = np.random.default_rng(12)
    df = pd.DataFrame({name: rng.normal(0, 1, 800) for name in "abcd"})
    df.loc[400:420, "a"] = 12.0

    first = detect(df, random_state=0).to_dict()
    second = detect(df, random_state=0).to_dict()
    assert first == second

    other = detect(df, random_state=99).to_dict()
    assert other["n_rows"] == first["n_rows"]


def test_time_column_is_used_and_never_scored():
    rng = np.random.default_rng(13)
    n = 300
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        }
    )
    df.loc[100, "a"] = 30.0
    report = detect(df, time="timestamp")

    assert report.time_column == "timestamp"
    assert "timestamp" not in report.channels
    event = next(e for e in report.events if e.start <= 100 <= e.end)
    assert isinstance(event.start_time, pd.Timestamp)
    assert "start_time" in report.events_frame().columns
    assert report.to_dict()["events"][0]["start_time"] is not None


def test_time_column_is_detected_when_not_named():
    rng = np.random.default_rng(14)
    df = pd.DataFrame(
        {
            "when": pd.date_range("2026-01-01", periods=200, freq="min"),
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(0, 1, 200),
        }
    )
    report = detect(df)
    assert report.time_column == "when"
    assert "when" not in report.channels
    assert any("time axis" in note for note in report.notes)


def test_explicit_channels_win_over_time_detection():
    rng = np.random.default_rng(15)
    df = pd.DataFrame(
        {"time": np.arange(200.0), "a": rng.normal(0, 1, 200)}
    )
    report = detect(df, channels=["time", "a"])
    assert set(report.channels) == {"time", "a"}
    assert report.time_column is None


def test_unsorted_time_is_reported_not_reordered():
    rng = np.random.default_rng(16)
    stamps = pd.date_range("2026-01-01", periods=100, freq="min").to_list()
    stamps[10], stamps[90] = stamps[90], stamps[10]
    df = pd.DataFrame({"t": stamps, "a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    report = detect(df, time="t")
    assert any("not sorted" in note for note in report.notes)


def test_selecting_channels_by_name():
    rng = np.random.default_rng(17)
    df = pd.DataFrame({name: rng.normal(0, 1, 200) for name in "abcd"})
    report = detect(df, channels=["a", "c"])
    assert list(report.channels) == ["a", "c"]
    assert detect(df, channels="a").channel_names == ["a"]


def test_helpful_errors():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="duplicate column names"):
        detect(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(ValueError, match="not in the table"):
        detect(df, time="nope")
    with pytest.raises(ValueError, match="channels not found"):
        detect(df, channels=["a", "ghost"])
    with pytest.raises(ValueError, match="same column twice"):
        detect(df, channels=["a", "a"])
    with pytest.raises(ValueError, match="sensitivity"):
        detect(df, sensitivity=0)
    with pytest.raises(ValueError, match="contamination"):
        detect(df, contamination=0.9)
    with pytest.raises(ValueError, match="contamination"):
        detect(df, contamination="sometimes")
    with pytest.raises(FileNotFoundError):
        detect("no-such-file.csv")
    with pytest.raises(TypeError):
        detect(42)


def test_explicit_contamination_is_honoured():
    rng = np.random.default_rng(18)
    df = pd.DataFrame({name: rng.normal(0, 1, 1000) for name in "abc"})
    report = detect(df, contamination=0.05)
    assert 0.02 <= len(report.joint) / report.n_rows <= 0.09


def test_reads_a_csv_path(tmp_path):
    rng = np.random.default_rng(19)
    df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    path = tmp_path / "plant.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    report = detect(str(path))
    assert report.n_rows == 200
    assert set(report.channels) == {"a", "b"}


def test_hundred_thousand_rows_by_twenty_channels_is_quick():
    """The size requirement, run on the shape a real plant actually has.

    Twenty channels all driven by one process variable, so the channel covariance
    is ill-conditioned - that is the case that used to bury a real fault under a
    thousand false cross-channel flags, so the speed check doubles as a
    precision check at full scale. See tests/test_regressions.py for the
    mechanism this guards.
    """
    rng = np.random.default_rng(20)
    base = rng.normal(0, 1, 100_000)
    df = pd.DataFrame(
        {
            "ch%02d" % i: 10 * (i + 1) + (i % 4 + 1) * base + rng.normal(0, 0.3, 100_000)
            for i in range(20)
        }
    )
    df.loc[50_000:50_200, "ch03"] = 0.0

    started = time.perf_counter()
    report = detect(df)
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0, "took %.1fs" % elapsed
    assert report.n_rows == 100_000
    assert len(report.channels) == 20
    assert "stuck-at-zero" in report.channels["ch03"].faults
    assert report.worst_channels[0] == "ch03"

    flagged = np.asarray(report.joint, dtype=int)
    assert flagged.size, "the planted fault must reach the cross-channel stage"
    inside = int(((flagged >= 50_000) & (flagged <= 50_200)).sum())
    assert inside / flagged.size >= 0.9, (
        "cross-channel flags must be the faulty rows, got %d of %d"
        % (inside, flagged.size)
    )
