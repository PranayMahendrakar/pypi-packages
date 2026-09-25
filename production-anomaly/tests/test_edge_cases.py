"""Awkward input: schedules, counters, irregular and timezone-aware time, tiny and messy tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import production_anomaly as pa
from conftest import line


def kinds(report):
    return [e.kind for e in report.events]


# ---------------------------------------------------------------- unscheduled nights
def test_line_not_scheduled_overnight_is_not_downtime_when_shift_hours_given():
    df = line(days=3, every=5, rate=300, hours=(6, 22))  # idle 22:00-06:00 every night
    report = pa.analyze(df, shift_hours=(6, 22))
    assert report.stoppages == []
    assert "downtime" not in kinds(report)
    assert report.availability == 1.0
    assert report.downtime_minutes == 0
    assert report.unscheduled_minutes == 3 * 8 * 60
    assert report.scheduled_minutes == 3 * 16 * 60
    assert set(report.intervals.loc[report.intervals["time"].dt.hour < 6, "state"]) == {"unscheduled"}
    assert report.findings[0].startswith("No downtime")


def test_without_shift_hours_the_nights_are_downtime_and_the_report_says_what_to_pass():
    df = line(days=3, every=5, rate=300, hours=(6, 22))
    report = pa.analyze(df)
    assert report.downtime_minutes == pytest.approx(3 * 8 * 60)
    assert report.availability == pytest.approx(16 / 24)
    hint = [f for f in report.findings if f.startswith("Schedule:")]
    assert hint and "shift_hours=(6, 22)" in hint[0] and "22:00-06:00" in hint[0]


def test_shifts_with_a_lunch_break_are_suggested_as_two_windows():
    df = line(days=2, every=5, rate=300, hours=(6, 22))
    df.loc[df["time"].dt.hour == 12, "units"] = 0.0
    hint = [f for f in pa.analyze(df).findings if f.startswith("Schedule:")][0]
    assert "shift_hours=[(6, 12), (13, 22)]" in hint


def test_missing_overnight_rows_with_shift_hours_are_not_gaps():
    df = line(days=3, every=5, rate=300, hours=(6, 22))
    df = df[(df["time"].dt.hour >= 6) & (df["time"].dt.hour < 22)]
    report = pa.analyze(df, shift_hours=(6, 22))
    assert report.events == [] and report.availability == 1.0


def test_night_shift_that_wraps_midnight_belongs_to_the_day_it_started():
    df = line(days=2, every=15, rate=100)
    report = pa.analyze(df, shift_hours={"A": (6, 14), "B": (14, 22), "C": (22, 6)})
    by_shift = report.by_shift
    night = by_shift[by_shift["shift"] == "C"]
    assert str(night["date"].iloc[1]) == "2026-03-02"
    assert night["start"].iloc[1] == pd.Timestamp("2026-03-02 22:00")
    assert night["end"].iloc[1] == pd.Timestamp("2026-03-03 06:00")
    assert night["scheduled_min"].iloc[1] == 480


# ---------------------------------------------------------------- counter resets
def _counter_with_reset(at=200, n=480, tz=None):
    t = pd.date_range("2026-03-02 06:00", periods=n, freq="min", tz=tz)
    per_minute = np.full(n, 60.0)
    total = np.cumsum(per_minute) + 5_000
    total[at:] = np.cumsum(per_minute[at:]) - per_minute[at]  # restarts from 0 at `at`
    return pd.DataFrame({"time": t, "total": total})


def test_counter_reset_mid_shift_is_reported_as_a_reset():
    df = _counter_with_reset()
    report = pa.analyze(df, shift_hours=(6, 14))
    assert report.mode == "counter"
    resets = [e for e in report.events if e.kind == "counter_reset"]
    assert len(resets) == 1
    assert resets[0].start == df["time"][199] and resets[0].value == 0
    assert "not as a stoppage or a negative rate" in resets[0].detail
    assert report.stoppages == []
    assert "spike" not in kinds(report)
    assert np.nanmin(report.intervals["units"].to_numpy(dtype=float)) >= 0
    assert report.availability == 1.0 and report.performance == pytest.approx(1.0)
    assert report.unknown_minutes == 1
    assert any(f.startswith("Counter reset:") for f in report.findings)


def test_negative_count_from_a_differenced_counter_is_a_reset(steady_day):
    df = steady_day.copy()
    df.loc[300, "units"] = -18_000.0
    report = pa.analyze(df)
    assert [e.kind for e in report.events] == ["counter_reset"]
    assert report.stoppages == []
    assert report.availability == 1.0
    assert (report.intervals["units"].dropna() >= 0).all()


def test_counter_reset_while_stopped_keeps_one_stoppage():
    t = pd.date_range("2026-03-02 06:00", periods=120, freq="min")
    total = np.r_[np.arange(1, 41) * 60.0, np.full(40, 2400.0), np.arange(0, 40) * 60.0]
    total[60] = 0.0  # the counter is zeroed half-way through the stop
    total[61:80] = 0.0
    report = pa.analyze(pd.DataFrame({"time": t, "total": total}))
    downtime = [s for s in report.stoppages if s.kind == "downtime"]
    assert len(downtime) == 1
    assert downtime[0].minutes == pytest.approx(40, abs=1)


def test_counter_can_be_forced_either_way(steady_day):
    df = steady_day.copy()
    df["units"] = np.arange(len(df), dtype=float)  # a strict ramp looks like a counter
    assert pa.analyze(df).mode == "counter"
    assert pa.ProductionAnalyzer(counter=False).analyze(df).mode == "counts"


# ---------------------------------------------------------------- irregular timestamps
def test_irregular_timestamps_are_resampled_with_a_note():
    rng = np.random.default_rng(8)
    gaps = rng.uniform(40, 80, 600)
    times = pd.Timestamp("2026-03-02 06:00:07") + pd.to_timedelta(np.cumsum(gaps), unit="s")
    units = gaps / 60 * 60.0  # 60 units a minute, whatever the spacing
    units[300:330] = 0.0
    df = pd.DataFrame({"time": times, "units": units})
    report = pa.analyze(df)
    assert any("irregular" in n and "resampled onto a regular 1 min grid" in n for n in report.notes)
    assert report.interval_minutes == 1
    assert (report.intervals["time"].dt.second == 0).all()
    assert report.intervals["units"].sum() == pytest.approx(units[:299].sum() + units[330:].sum(), rel=0.02)
    assert "data_gap" not in kinds(report)
    downtime = [s for s in report.stoppages if s.kind == "downtime"]
    assert len(downtime) == 1
    assert downtime[0].minutes == pytest.approx(gaps[300:330].sum() / 60, abs=2)


def test_regular_data_with_missing_rows_is_not_resampled(steady_day):
    df = steady_day.drop(index=range(500, 520))
    report = pa.analyze(df)
    assert not any("irregular" in n for n in report.notes)
    gaps = [e for e in report.events if e.kind == "data_gap"]
    assert len(gaps) == 1 and gaps[0].minutes == 20
    assert report.stoppages == []
    assert report.unknown_minutes == 20
    assert report.availability == 1.0


# ---------------------------------------------------------------- timezones
def test_timezone_aware_timestamps_keep_their_timezone():
    df = line(days=2, every=5, rate=300, hours=(6, 22), tz="Asia/Kolkata")
    df.loc[150:161, "units"] = 0.0
    report = pa.analyze(df, shift_hours=(6, 22))
    assert report.timezone == "Asia/Kolkata"
    assert len(report.stoppages) == 1  # the local nights are unscheduled
    stop = report.stoppages[0]
    assert str(stop.start.tz) == "Asia/Kolkata" and stop.start == df["time"][150]
    assert str(report.by_shift["start"].iloc[0].tz) == "Asia/Kolkata"
    assert str(report.intervals["time"].dt.tz) == "Asia/Kolkata"
    data = report.to_dict()
    assert data["stoppages"][0]["start"].endswith("+05:30")
    assert data["timezone"] == "Asia/Kolkata"


def test_offset_strings_keep_their_offset(tmp_path):
    times = pd.date_range("2026-03-02 06:00", periods=240, freq="min")
    text = [t.strftime("%Y-%m-%dT%H:%M:%S") + "-05:00" for t in times]
    units = [60] * 100 + [0] * 30 + [60] * 110
    path = tmp_path / "line.csv"
    pd.DataFrame({"ts": text, "count": units}).to_csv(path, index=False)
    report = pa.analyze(path, shift_hours=(6, 22))
    assert report.stoppages[0].start.utcoffset() == pd.Timedelta(hours=-5)
    assert report.stoppages[0].start.hour == 7 and report.stoppages[0].start.minute == 40


def test_mixed_offsets_are_converted_to_utc_with_a_note():
    text = ["2026-03-02T06:00:00+01:00", "2026-03-02T06:01:00+01:00", "2026-03-02T05:02:00+00:00"]
    report = pa.analyze(pd.DataFrame({"time": text, "units": [60, 60, 60]}))
    assert report.timezone == "UTC"
    assert any("mixes UTC offsets" in n for n in report.notes)


# ---------------------------------------------------------------- tiny tables
def test_single_row_returns_a_report():
    df = pd.DataFrame({"time": [pd.Timestamp("2026-03-02 06:00")], "units": [50]})
    report = pa.analyze(df)
    assert isinstance(report, pa.ProductionReport)
    assert report.availability == 1.0 and report.units == 50
    assert any("single timestamp" in n for n in report.notes)
    json.dumps(report.to_dict(), allow_nan=False)
    assert "production report" in report.summary()


def test_single_row_of_zero_is_a_stop():
    df = pd.DataFrame({"time": ["2026-03-02 06:00"], "units": [0]})
    report = pa.analyze(df)
    assert report.availability == 0.0 and report.oee_partial == 0.0
    assert len(report.stoppages) == 1


def test_single_counter_reading_returns_a_report():
    report = pa.ProductionAnalyzer(counter=True).analyze(
        pd.DataFrame({"time": ["2026-03-02 06:00"], "total": [12345]})
    )
    assert report.availability is None
    assert report.findings[0].startswith("Nothing to measure")


def test_empty_dataframe_returns_a_report():
    df = pd.DataFrame({"time": pd.to_datetime([]), "units": pd.Series([], dtype=float)})
    report = pa.analyze(df)
    assert report.availability is None and report.performance is None and report.oee_partial is None
    assert report.findings == ["Nothing to measure: the table has no rows."]
    assert report.by_shift.empty
    json.dumps(report.to_dict(), allow_nan=False)
    assert "nothing measured" in report.summary()


def test_all_nan_output_column():
    df = pd.DataFrame({"time": pd.date_range("2026-03-02", periods=30, freq="min"), "units": np.nan})
    report = pa.analyze(df)
    assert report.availability is None
    assert report.findings == ["Nothing to measure: the output column has no usable numbers."]
    assert report.unknown_minutes == 30


def test_all_zero_output_is_one_long_stop():
    df = pd.DataFrame({"time": pd.date_range("2026-03-02", periods=60, freq="min"), "units": 0})
    report = pa.analyze(df)
    assert report.availability == 0.0 and report.performance is None and report.oee_partial == 0.0
    assert len(report.stoppages) == 1 and report.stoppages[0].minutes == 60
    assert report.lost_units is None


# ---------------------------------------------------------------- messy tables
def test_mixed_dtypes_in_the_output_column():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-03-02", periods=6, freq="min"),
            "units": ["60", 60, "n/a", None, 59.5, "61"],
            "line": ["L1"] * 6,
        }
    )
    report = pa.analyze(df, output="units")
    assert any("not numbers" in n for n in report.notes)
    assert report.unknown_minutes == 2
    assert report.units == pytest.approx(60 + 60 + 59.5 + 61)


def test_unicode_column_names_and_text():
    t = pd.date_range("2026-03-02", periods=120, freq="min")
    df = pd.DataFrame({"Zeit": t, "Stückzahl-产量": [60.0] * 90 + [0.0] * 30, "Linie": ["Presse ü"] * 120})
    report = pa.analyze(df)
    assert report.output_column == "Stückzahl-产量"
    assert report.time_column == "Zeit"
    assert "Stückzahl-产量" in report.summary()
    assert json.loads(json.dumps(report.to_dict(), ensure_ascii=False))["output_column"] == "Stückzahl-产量"


def test_summary_is_plain_ascii_for_ascii_data():
    df = line(days=3, every=5, rate=300, hours=(6, 22))
    df.loc[130:141, "units"] = 0.0
    df.loc[200, "units"] = 600.0
    for report in (pa.analyze(df), pa.analyze(df, shift_hours=(6, 22), target_rate=3000)):
        report.summary().encode("ascii")
        for finding in report.findings:
            finding.encode("ascii")


def test_duplicate_column_names_raise_a_clear_value_error():
    df = pd.DataFrame([[pd.Timestamp("2026-03-02"), 1, 2]], columns=["time", "units", "units"])
    with pytest.raises(ValueError, match="duplicate column names: 'units'"):
        pa.analyze(df)


def test_unknown_columns_raise_value_errors(steady_day):
    with pytest.raises(ValueError, match="time column 'ts' not found"):
        pa.analyze(steady_day, time="ts")
    with pytest.raises(ValueError, match="output column 'pcs' not found"):
        pa.analyze(steady_day, output="pcs")
    with pytest.raises(ValueError, match="no time column found"):
        pa.analyze(pd.DataFrame({"units": [1, 2, 3]}))
    with pytest.raises(ValueError, match="no numeric column"):
        pa.analyze(pd.DataFrame({"time": ["2026-03-02"], "name": ["x"]}))


def test_output_is_picked_by_name_and_the_choice_is_noted(steady_day):
    df = steady_day.assign(temperature=21.5, units_good=steady_day["units"])
    df = df.drop(columns="units")
    report = pa.analyze(df)
    assert report.output_column == "units_good"
    assert any("output not given; used 'units_good'" in n for n in report.notes)


def test_no_obvious_output_column_falls_back_with_a_warning(steady_day):
    df = pd.DataFrame({"time": steady_day["time"], "a": 60.0, "b": 3.0})
    report = pa.analyze(df)
    assert report.output_column == "a"
    assert any(n.startswith("WARNING: output not given") for n in report.notes)


def test_unsorted_duplicated_and_undated_rows(steady_day):
    df = steady_day.sample(frac=1.0, random_state=0)
    df = pd.concat([df, df.iloc[:5]])  # five double-logged rows
    df = pd.concat([df, pd.DataFrame({"time": [pd.NaT], "units": [60.0]})])
    report = pa.analyze(df)
    assert any("sorted by time" in n for n in report.notes)
    assert any("5 of them repeat the same value" in n and "drop_duplicates" in n for n in report.notes)
    assert any("no readable timestamp" in n for n in report.notes)
    # Rows sharing a timestamp are always added up, so a reading logged twice doubles its
    # interval, which is then reported as a double count and kept out of the totals.
    spikes = [e for e in report.events if e.kind == "spike"]
    assert len(spikes) == 5 and all("double count" in e.detail for e in spikes)
    assert report.units == pytest.approx(1440 * 60)


def test_same_timestamp_different_values_are_added(steady_day):
    extra = steady_day.iloc[[10]].assign(units=5.0)
    report = pa.analyze(pd.concat([steady_day, extra]))
    assert any("shared a timestamp" in n for n in report.notes)
    assert report.intervals["units"].iloc[10] == 65


def test_datetime_index_and_csv_path(tmp_path, steady_day):
    report = pa.analyze(steady_day.set_index("time"))
    assert report.time_column == "time" and report.availability == 1.0
    path = tmp_path / "line.csv"
    steady_day.to_csv(path, index=False)
    assert pa.analyze(path).units == pytest.approx(1440 * 60)
    with pytest.raises(FileNotFoundError):
        pa.analyze(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="unsupported file type"):
        (tmp_path / "line.txt").write_text("x", encoding="utf-8")
        pa.analyze(tmp_path / "line.txt")


# ---------------------------------------------------------------- the caller's data
def test_caller_dataframe_is_never_modified():
    frames = [
        line(days=2, every=5, rate=300, hours=(6, 22), tz="Europe/Berlin"),
        _counter_with_reset(),
        line(days=1).sample(frac=1.0, random_state=1),
    ]
    frames[2].loc[frames[2].index[:10], "units"] = np.nan
    for df in frames:
        before = df.copy(deep=True)
        pa.analyze(df, shift_hours=(6, 22), target_rate=3600)
        pd.testing.assert_frame_equal(df, before)
        assert list(df.columns) == list(before.columns)
        assert df.index.equals(before.index)


def test_read_only_input_arrays_work():
    """pandas 3 hands out read-only arrays; the library must never write into them."""
    t = pd.date_range("2026-03-02", periods=600, freq="min").to_numpy()
    units = np.full(600, 60.0)
    units[100:130] = 0.0
    t.setflags(write=False)
    units.setflags(write=False)
    report = pa.analyze(pd.DataFrame({"time": t, "units": units}, copy=False))
    assert len(report.stoppages) == 1


def test_copy_on_write_mode():
    """pandas 3 always copies on write; on pandas 2 switch it on to get the same arrays."""
    import contextlib

    df = _counter_with_reset()
    df.loc[300:320, "total"] = df.loc[300, "total"]
    major = int(pd.__version__.split(".")[0])
    if major >= 3:
        mode = contextlib.nullcontext()
    else:
        try:
            pd.get_option("mode.copy_on_write")
        except (KeyError, pd.errors.OptionError):
            pytest.skip("this pandas has no copy_on_write option")
        mode = pd.option_context("mode.copy_on_write", True)
    with mode:
        report = pa.analyze(df, shift_hours=(6, 14))
    assert report.mode == "counter" and len(report.stoppages) == 1
