"""Each detector finds what it should, and stays quiet when there is nothing to find."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import production_anomaly as pa
from conftest import line


def kinds(report):
    return [e.kind for e in report.events]


# ---------------------------------------------------------------- downtime
def test_zero_run_becomes_one_stoppage_with_start_end_and_duration(steady_day):
    df = steady_day.copy()
    df.loc[100:159, "units"] = 0.0
    report = pa.analyze(df)
    assert len(report.stoppages) == 1
    stop = report.stoppages[0]
    assert stop.kind == "downtime"
    assert stop.start == df["time"][100]
    assert stop.end == df["time"][160]
    assert stop.minutes == 60
    assert report.availability == pytest.approx((1440 - 60) / 1440)
    assert report.performance == pytest.approx(1.0)
    assert report.lost_by_cause["downtime"] == pytest.approx(60 * 60)
    assert report.findings[0].startswith("Downtime: stopped for 60 min")


def test_near_zero_trickle_is_part_of_the_stop(steady_day):
    df = steady_day.copy()
    df.loc[300:339, "units"] = [0, 1, 2, 0, 1] * 8
    report = pa.analyze(df)
    downtime = [s for s in report.stoppages if s.kind == "downtime"]
    assert len(downtime) == 1 and downtime[0].minutes == 40


def test_two_stops_stay_separate(steady_day):
    df = steady_day.copy()
    df.loc[100:119, "units"] = 0.0
    df.loc[400:429, "units"] = 0.0
    report = pa.analyze(df)
    assert [s.minutes for s in report.stoppages] == [20, 30]
    assert report.downtime_minutes == 50


def test_short_stop_is_a_micro_stop_not_downtime(steady_day):
    df = steady_day.copy()
    df.loc[500:501, "units"] = 0.0
    report = pa.analyze(df)
    assert [s.kind for s in report.stoppages] == ["micro_stop"]
    assert report.availability == 1.0
    assert report.performance == pytest.approx((1440 - 2) / 1440)
    assert "micro_stop" in kinds(report)


def test_stop_at_the_start_of_the_data_says_so(steady_day):
    df = steady_day.copy()
    df.loc[:29, "units"] = 0.0
    report = pa.analyze(df)
    event = [e for e in report.events if e.kind == "downtime"][0]
    assert "already stopped when the data begins" in event.detail


# ---------------------------------------------------------------- slow running
def test_slow_hour_below_typical_rate(steady_day):
    df = steady_day.copy()
    df.loc[600:659, "units"] = 42.0
    report = pa.analyze(df)
    slow = [e for e in report.events if e.kind == "slow_running"]
    assert len(slow) == 1
    assert slow[0].minutes == 60
    assert slow[0].units_lost == pytest.approx(60 * 18)
    assert report.lost_by_cause["slow_running"] == pytest.approx(60 * 18)
    assert not report.stoppages
    assert any(f.startswith("Slow running: 1 period") for f in report.findings)


def test_slow_spell_survives_noise():
    df = line(days=1, noise=2.0, seed=4)
    df.loc[700:760, "units"] = df.loc[700:760, "units"] * 0.7
    report = pa.analyze(df)
    slow = [e for e in report.events if e.kind == "slow_running"]
    assert len(slow) == 1
    assert 55 <= slow[0].minutes <= 65


def test_a_brief_dip_is_not_slow_running(steady_day):
    df = steady_day.copy()
    df.loc[800:804, "units"] = 40.0  # five minutes, shorter than min_slow_minutes
    report = pa.analyze(df)
    assert "slow_running" not in kinds(report)
    assert "micro_stop" in kinds(report)


def test_running_below_target_rate(steady_day):
    df = steady_day.copy()
    df["units"] = 50.0
    report = pa.analyze(df, target_rate=3600)
    assert report.reference == "target"
    assert report.reference_rate == pytest.approx(60.0)
    assert report.performance == pytest.approx(50 / 60)
    assert report.lost_by_cause["speed_loss"] == pytest.approx(1440 * 10)
    assert any(f.startswith("Running speed:") and "below target_rate" in f for f in report.findings)


def test_slow_against_target_when_the_line_beats_it(steady_day):
    df = steady_day.copy()
    df["units"] = 70.0
    df.loc[200:259, "units"] = 40.0  # below 80% of the 60/min target
    report = pa.analyze(df, target_rate=3600)
    slow = [e for e in report.events if e.kind == "slow_running"]
    assert len(slow) == 1 and "target_rate" in slow[0].detail
    assert report.performance == 1.0  # capped: the line otherwise beats target
    assert any("capped at 100%" in n for n in report.notes)


def test_target_rate_in_the_wrong_unit_is_flagged(steady_day):
    report = pa.analyze(steady_day, target_rate=60)  # 60 per hour, but it makes 60 a minute
    assert any("units per HOUR" in n for n in report.notes)


# ---------------------------------------------------------------- drift
def _six_shifts(levels, bad_hour=None):
    t = pd.date_range("2026-03-02", "2026-03-05", freq="5min", inclusive="left")
    scheduled = (t.hour >= 6) & (t.hour < 22)
    days = np.floor(np.asarray((t - t[0]).total_seconds(), dtype=float) / 86400).astype(int)
    shift_number = np.clip(days * 2 + np.asarray(t.hour >= 14, dtype=int), 0, 5)
    rng = np.random.default_rng(11)
    units = np.asarray(levels, dtype=float)[shift_number] + rng.normal(0, 3, len(t))
    units = np.where(scheduled, units, 0.0)
    if bad_hour is not None:
        units[bad_hour:bad_hour + 12] *= 0.6
    return pd.DataFrame({"time": t, "units": units})


def test_sustained_decline_across_shifts_is_drift():
    df = _six_shifts([300, 290, 280, 268, 255, 245])
    report = pa.analyze(df, shift_hours=[(6, 14), (14, 22)])
    drift = [e for e in report.events if e.kind == "rate_drift"]
    assert len(drift) == 1
    assert drift[0].value == pytest.approx(-0.18, abs=0.04)
    assert drift[0].units_lost > 0
    assert any(f.startswith("Rate drift:") for f in report.findings)
    assert len(report.by_shift) == 6


def test_one_bad_hour_is_not_drift():
    df = _six_shifts([300] * 6, bad_hour=12 * 24 * 2 + 12 * 18)  # the last day, 18:00
    report = pa.analyze(df, shift_hours=[(6, 14), (14, 22)])
    assert "rate_drift" not in kinds(report)
    assert "slow_running" in kinds(report)


def test_steady_shifts_are_not_drift():
    report = pa.analyze(_six_shifts([300] * 6), shift_hours=[(6, 14), (14, 22)])
    assert "rate_drift" not in kinds(report)


def test_drift_needs_three_shifts(steady_day):
    report = pa.analyze(steady_day.iloc[:600])
    assert any("rate drift is judged across shifts" in n for n in report.notes)


# ---------------------------------------------------------------- micro-stops
def test_frequent_short_drops_add_up():
    df = line(days=1, hours=(6, 22))
    rng = np.random.default_rng(2)
    scheduled = np.flatnonzero(df["units"].to_numpy() > 0)
    picks = rng.choice(scheduled[5:-5], 40, replace=False)
    units = df["units"].to_numpy(copy=True)
    units[picks] = 30.0
    df["units"] = units
    report = pa.analyze(df, shift_hours=(6, 22))
    micro = [e for e in report.events if e.kind == "micro_stop"]
    assert 30 <= len(micro) <= 40
    assert report.availability == 1.0
    assert report.lost_by_cause["micro_stops"] == pytest.approx(40 * 30)
    finding = [f for f in report.findings if f.startswith("Micro-stops:")][0]
    assert "together they cost about 1,200 units" in finding
    assert "20 min of full-speed running" in finding


def test_hourly_data_cannot_show_micro_stops():
    df = line(days=2, every=60)
    report = pa.analyze(df)
    assert any("cannot be seen one by one" in n for n in report.notes)


# ---------------------------------------------------------------- spikes
def test_double_count_is_a_spike_and_does_not_inflate_units(steady_day):
    df = steady_day.copy()
    df.loc[720, "units"] = 120.0
    report = pa.analyze(df)
    spikes = [e for e in report.events if e.kind == "spike"]
    assert len(spikes) == 1 and spikes[0].value == 120
    assert "double count" in spikes[0].detail
    assert report.units == pytest.approx(1440 * 60)
    assert report.performance == pytest.approx(1.0)


def test_huge_spike_reads_as_a_counter_artifact(steady_day):
    df = steady_day.copy()
    df.loc[10, "units"] = 60 * 50
    report = pa.analyze(df)
    spike = [e for e in report.events if e.kind == "spike"][0]
    assert "counter reset or rollover" in spike.detail
    assert any(f.startswith("Spikes: 1 interval") for f in report.findings)


# ---------------------------------------------------------------- quiet when healthy
def test_poisson_noise_alone_raises_nothing():
    t = pd.date_range("2026-03-02", periods=2 * 1440, freq="min")
    units = np.random.default_rng(5).poisson(60, len(t))
    report = pa.analyze(pd.DataFrame({"time": t, "units": units}))
    assert report.events == []
    assert report.stoppages == []
    assert report.performance == pytest.approx(1.0)
    assert report.findings[0].startswith("No downtime, slow running")


def test_lost_units_is_the_sum_of_its_causes():
    df = _six_shifts([300, 290, 280, 268, 255, 245], bad_hour=100)
    df.loc[300:320, "units"] = 0.0
    report = pa.analyze(df, shift_hours=(6, 22), target_rate=3600)
    assert report.lost_units == pytest.approx(sum(report.lost_by_cause.values()))
    assert set(report.lost_by_cause) == {"downtime", "micro_stops", "slow_running", "speed_loss"}
    assert all(v >= 0 for v in report.lost_by_cause.values())


def test_oee_partial_is_availability_times_performance():
    df = _six_shifts([300] * 6, bad_hour=150)
    df.loc[400:430, "units"] = 0.0
    report = pa.analyze(df, shift_hours=(6, 22))
    assert report.oee_partial == pytest.approx(report.availability * report.performance)
    assert 0 < report.oee_partial < 1
