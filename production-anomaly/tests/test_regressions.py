"""Regressions from review: low whole-part counts, irregular spacing, days off, noise, duplicates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import production_anomaly as pa
from conftest import line

START = pd.Timestamp("2026-04-06 06:00")  # a Monday


def kinds(report):
    return [e.kind for e in report.events]


def _steady_machine(cycle_s, stops=(), bin_="1min", hours=16):
    """A machine finishing one part every ``cycle_s`` seconds, counted per ``bin_``."""
    stamps = START + pd.to_timedelta(np.arange(1, int(hours * 3600 / cycle_s)) * cycle_s, "s")
    keep = np.ones(stamps.size, dtype=bool)
    for offset, minutes in stops:
        begin = START + pd.Timedelta(offset)
        keep &= ~((stamps >= begin) & (stamps < begin + pd.Timedelta(minutes=minutes)))
    stamps = stamps[keep]
    counts = pd.Series(1.0, index=stamps).resample(bin_).sum()
    grid = pd.date_range(START, START + pd.Timedelta(hours=hours), freq=bin_, inclusive="left")
    counts = counts.reindex(grid, fill_value=0.0)
    return pd.DataFrame({"time": counts.index, "units": counts.to_numpy()}), int(keep.sum())


# ---------------------------------------------------------------- low whole-part counts
@pytest.mark.parametrize("cycle_s", [45, 90, 120, 600])
def test_steady_line_with_few_parts_per_interval_raises_nothing(cycle_s):
    # 45 s reads 1, 1, 2, 1, 1, 2 ... and 90 s reads 0, 1, 1, 0, 1, 1 ... per minute.
    df, made = _steady_machine(cycle_s)
    report = pa.analyze(df, shift_hours=(6, 22))
    assert report.events == [] and report.stoppages == []
    assert report.units == made
    assert report.performance == pytest.approx(1.0) and report.availability == 1.0
    assert report.interval_minutes > 1
    assert any(n.startswith("counts are low") for n in report.notes)
    assert report.findings[0].startswith("No downtime")


def test_real_stops_on_a_line_with_few_parts_per_interval_are_still_found():
    df, made = _steady_machine(45, stops=[("2h", 30), ("6h", 60)])
    report = pa.analyze(df, shift_hours=(6, 22))
    downtime = [s for s in report.stoppages if s.kind == "downtime"]
    assert [s.minutes for s in downtime] == [30, 60]
    assert downtime[0].start == START + pd.Timedelta("2h")
    assert report.units == made
    assert "spike" not in kinds(report)


def test_counter_glitch_on_a_line_with_few_parts_is_still_a_spike():
    df, made = _steady_machine(45)
    df.loc[300, "units"] = 5_000.0
    report = pa.analyze(df, shift_hours=(6, 22))
    assert kinds(report) == ["spike"]
    assert report.units == pytest.approx(made, abs=15)  # counted at the typical rate


def test_one_second_counts_with_a_cycle_longer_than_the_interval():
    n = 4 * 3600
    t = pd.date_range(START, periods=n, freq="1s")
    parts = np.diff(np.floor(np.arange(n + 1) / 1.25))  # 0, 1, 1, 1, 1, 0, 1 ...
    report = pa.analyze(pd.DataFrame({"time": t, "units": parts}))
    assert report.events == []
    assert report.units == parts.sum()


# ---------------------------------------------------------------- irregular spacing
@pytest.mark.parametrize("low, high, interval", [(20, 100, 1.0), (10, 120, 1.0), (5, 60, 0.5)])
def test_irregular_spacing_is_spread_in_full_and_raises_nothing(low, high, interval):
    rng = np.random.default_rng(3)
    secs = np.cumsum(rng.uniform(low, high, 3000))
    stamps = pd.Timestamp("2026-04-06 06:00:03") + pd.to_timedelta(secs, "s")
    following = np.append(np.diff(secs), np.median(np.diff(secs)))
    units = following / 60 * 60  # a steady 60 units a minute
    report = pa.analyze(pd.DataFrame({"time": stamps, "units": units}))
    assert report.events == []
    assert report.units == pytest.approx(units.sum(), rel=1e-3)
    assert report.interval_minutes == interval  # a round interval, not 65 s or 32 s
    note = [n for n in report.notes if "irregular" in n][0]
    assert "round interval nearest the typical gap" in note


def test_one_row_per_part_with_random_arrivals():
    rng = np.random.default_rng(1)
    stamps = START + pd.to_timedelta(np.cumsum(rng.exponential(30, 1900)), "s")
    report = pa.analyze(pd.DataFrame({"time": stamps, "units": 1}), shift_hours=(6, 22))
    assert report.events == []
    assert report.units == pytest.approx(1900, rel=0.02)
    assert any("irregular" in n and "resampled" in n for n in report.notes)


def test_a_logger_outage_in_irregular_data_is_still_a_data_gap():
    rng = np.random.default_rng(4)
    secs = np.cumsum(rng.uniform(40, 80, 900))
    secs[400:] += 45 * 60  # the logger was off for 45 minutes
    stamps = START + pd.to_timedelta(secs, "s")
    units = np.append(np.diff(secs), 60.0) / 60 * 60
    units[399] = 60.0  # the reading before the outage covers one normal gap
    report = pa.analyze(pd.DataFrame({"time": stamps, "units": units}))
    gaps = [e for e in report.events if e.kind == "data_gap"]
    assert len(gaps) == 1 and 40 <= gaps[0].minutes <= 47
    assert "downtime" not in kinds(report) and "spike" not in kinds(report)


# ---------------------------------------------------------------- days off
def _weekdays_only(days=14, drop_weekends=False):
    t = pd.date_range("2026-04-06", periods=days * 288, freq="5min")
    rng = np.random.default_rng(2)
    on = (t.hour >= 6) & (t.hour < 22) & (t.dayofweek < 5)
    df = pd.DataFrame({"time": t, "units": np.where(on, rng.poisson(300, t.size), 0).astype(float)})
    return df[t.dayofweek < 5] if drop_weekends else df


def test_idle_weekends_are_pointed_out_as_probably_unscheduled():
    report = pa.analyze(_weekdays_only(), shift_hours=(6, 22))
    hint = [f for f in report.findings if f.startswith("Schedule:")]
    assert len(hint) == 1
    assert "4 whole scheduled days" in hint[0] and "Sat 2026-04-11" in hint[0]
    assert "every Saturday and Sunday in the data" in hint[0]
    assert "leave their rows out" in hint[0]


def test_leaving_the_days_off_out_is_not_downtime_and_not_a_data_gap_finding():
    report = pa.analyze(_weekdays_only(drop_weekends=True), shift_hours=(6, 22))
    assert report.downtime_minutes == 0 and report.availability == 1.0
    assert report.stoppages == []
    assert report.findings[0].startswith("No downtime")
    days = [f for f in report.findings if f.startswith("Days without data:")]
    assert days and "Sat 2026-04-11, Sun 2026-04-12" in days[0]
    assert not any(f.startswith("Data gaps:") for f in report.findings)


def test_a_breakdown_day_among_working_days_is_not_called_a_weekday_pattern():
    df = _weekdays_only(days=5)
    df.loc[df["time"].dt.day == 8, "units"] = 0.0  # the whole of Wednesday
    report = pa.analyze(df, shift_hours=(6, 22))
    hint = [f for f in report.findings if f.startswith("Schedule:")][0]
    assert "Wed 2026-04-08" in hint and "every" not in hint
    assert report.downtime_minutes == 16 * 60


# ---------------------------------------------------------------- slow running and noise
@pytest.mark.parametrize("seed", range(5))
def test_noise_at_15_minute_intervals_is_not_slow_running(seed):
    t = pd.date_range("2026-04-06", periods=5 * 96, freq="15min")
    rng = np.random.default_rng(seed)
    units = np.where((t.hour >= 6) & (t.hour < 22), rng.normal(900, 90, t.size), 0.0)
    report = pa.analyze(pd.DataFrame({"time": t, "units": units}), shift_hours=(6, 22))
    assert "slow_running" not in kinds(report)


def test_a_real_slow_spell_at_15_minute_intervals_is_found():
    t = pd.date_range("2026-04-06", periods=5 * 96, freq="15min")
    rng = np.random.default_rng(9)
    units = np.where((t.hour >= 6) & (t.hour < 22), rng.normal(900, 45, t.size), 0.0)
    units[140:148] *= 0.7  # two hours at 70%
    report = pa.analyze(pd.DataFrame({"time": t, "units": units}), shift_hours=(6, 22))
    slow = [e for e in report.events if e.kind == "slow_running"]
    assert len(slow) == 1 and 90 <= slow[0].minutes <= 120


# ---------------------------------------------------------------- rows sharing a timestamp
def test_rows_sharing_a_timestamp_follow_one_rule():
    t = pd.date_range(START, periods=120, freq="1min")
    same = pd.DataFrame({"time": np.repeat(t, 2), "machine": ["A", "B"] * 120, "units": 60.0})
    differ = same.assign(units=np.tile([60.0, 61.0], 120))
    a = pa.analyze(same, output="units")
    b = pa.analyze(differ, output="units")
    assert a.units == 120 * 120 and b.units == 120 * 121
    assert any("repeat the same value" in n for n in a.notes)
    assert not any("repeat the same value" in n for n in b.notes)


def test_steady_high_count_line_is_not_merged():
    report = pa.analyze(line(days=1))
    assert report.interval_minutes == 1
    assert not any("counts are low" in n for n in report.notes)


def test_a_healthy_line_with_rounded_counts_is_not_full_of_micro_stops():
    """Regression: MAD-based sigma is badly biased on rounded/integer data - rounding
    piles many values onto the exact median, collapsing the 'typical absolute
    deviation' below the true spread. On mean 20/min, true sd 2, 7 days of 1-min
    counts, that bias alone produced ~29 false micro-stops a week on data with
    nothing wrong on it (0.2/week for the identical values left as floats)."""
    import numpy as np
    import pandas as pd

    import production_anomaly as pa

    for seed in range(10):
        rng = np.random.default_rng(seed)
        t = pd.date_range("2026-02-02", periods=7 * 1440, freq="1min")
        u = np.round(rng.normal(20, 2, len(t)))
        report = pa.analyze(pd.DataFrame({"time": t, "parts": u}))
        micro = [s for s in report.stoppages if "micro" in str(s.kind).lower()]
        assert len(micro) <= 2, f"seed {seed}: {len(micro)} false micro-stops on healthy data"


def test_noisy_but_healthy_coarse_interval_lines_are_not_reported_as_slow():
    """The same MAD bias fed the slow-running threshold: a healthy line sampled at
    15-minute or hourly intervals, with realistic (10-15%) coefficient of variation,
    was often reported as running slow purely from statistical noise."""
    import numpy as np
    import pandas as pd

    import production_anomaly as pa

    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        t = pd.date_range("2026-02-02", periods=1440, freq="1h")
        u = np.round(rng.normal(3600, 360, len(t)))
        report = pa.analyze(pd.DataFrame({"t": t, "u": u}))
        slow = [s for s in report.stoppages if "slow" in str(s.kind).lower()]
        assert not slow, f"seed {seed}: healthy hourly data reported as slow"

    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        n = 96 * 28
        t = pd.date_range("2026-02-02", periods=n, freq="15min")
        u = np.round(rng.normal(900, 135, n))
        report = pa.analyze(pd.DataFrame({"t": t, "u": u}))
        slow = [s for s in report.stoppages if "slow" in str(s.kind).lower()]
        assert not slow, f"seed {seed}: healthy 15-min data reported as slow"


def test_a_genuine_stoppage_is_still_caught_after_the_sigma_fix():
    """The fix must not blunt real detection: a clear, sustained downtime run must
    still be found."""
    import numpy as np
    import pandas as pd

    import production_anomaly as pa

    rng = np.random.default_rng(0)
    t = pd.date_range("2026-02-02", periods=7 * 1440, freq="1min")
    u = np.round(rng.normal(20, 2, len(t)))
    u[1000:1040] = 0  # a genuine 40-minute stoppage
    report = pa.analyze(pd.DataFrame({"time": t, "parts": u}))
    downtime = [s for s in report.stoppages if s.kind == "downtime"]
    assert downtime, "a genuine 40-minute stoppage must still be reported"
