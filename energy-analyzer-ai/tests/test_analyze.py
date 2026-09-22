"""The main path: what analyze() finds and how it explains itself."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import energy_analyzer_ai as ea
from tests.conftest import meter


def test_quickstart_from_the_readme():
    """The exact block in README.md must keep working."""
    hours = pd.date_range("2026-03-01", periods=24 * 21, freq="h")
    kwh = 0.4 + 1.8 * ((hours.hour >= 9) & (hours.hour < 18))
    readings = pd.Series(
        kwh + np.random.default_rng(0).normal(0, 0.05, hours.size), index=hours, name="kwh"
    )
    readings.iloc[300] = 9.0

    report = ea.analyze(readings, tariff=0.28)
    text = report.summary()

    assert isinstance(text, str) and text.startswith("energy-analyzer-ai:")
    assert report.n_anomalies == 1
    assert report.anomalies[0].when == hours[300]
    assert report.anomalies[0].observed == pytest.approx(9.0)
    assert report.total_cost == pytest.approx(report.total * 0.28)
    text.encode("ascii")  # the summary must survive any console


def test_the_real_spike_is_found_and_priced(readings):
    report = ea.analyze(readings, tariff=0.28)
    worst = report.top(1)[0]

    assert worst.kind == "spike"
    assert worst.excess > 5.0
    assert worst.cost == pytest.approx(worst.excess * 0.28)
    assert report.excess_units == pytest.approx(sum(a.excess for a in report.spikes))
    assert report.excess_cost == pytest.approx(report.excess_units * 0.28)


def test_evening_peak_is_not_an_anomaly():
    """A same-phase baseline must not flag the daily shape itself."""
    report = ea.analyze(meter(days=21, seed=3))
    assert report.n_anomalies == 0
    assert "Nothing stood out" in " ".join(report.findings)


def test_monday_9am_is_compared_with_other_monday_9ams():
    """A weekend-only pattern must not make weekday mornings look unusual."""
    index = pd.date_range("2026-03-02", periods=24 * 28, freq="h")
    weekday = 0.4 + 1.8 * ((index.hour >= 9) & (index.hour < 18)) * (index.dayofweek < 5)
    weekend = 2.5 * (index.dayofweek >= 5)
    series = pd.Series(
        weekday + weekend + np.random.default_rng(1).normal(0, 0.04, index.size),
        index=index,
        name="kwh",
    )
    report = ea.analyze(series)

    assert report.baseline_method.startswith("day of week")
    assert report.n_anomalies == 0


def test_by_period_lines_up_with_the_report(readings):
    report = ea.analyze(readings, tariff=0.28)
    frame = report.by_period

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == report.n_periods
    for column in ("observed", "expected", "excess", "score", "is_anomaly", "is_gap"):
        assert column in frame.columns
    assert int(frame["is_anomaly"].sum()) == report.n_anomalies
    assert frame["cost"].sum() == pytest.approx(report.total_cost)
    assert report.to_frame() is frame


def test_standby_load_is_the_overnight_floor(readings):
    report = ea.analyze(readings)
    assert report.baseline_load == pytest.approx(ea.baseline_load(readings))
    assert 0.3 < report.baseline_load < 0.5
    assert 0.0 < report.baseline_share < 1.0
    assert "overnight minimum" in report.baseline_how


def test_a_rising_baseline_is_reported():
    base = meter(days=28, seed=4)
    ramp = 1.0 + 0.04 * np.arange(base.size) / (24 * 7)
    report = ea.analyze(pd.Series(base.to_numpy() * ramp, index=base.index, name="kwh"))

    assert report.trend.direction == "rising"
    assert report.trend.confident is True
    assert report.trend.pct_per_week == pytest.approx(4.0, abs=1.5)
    assert bool(report.trend) is True
    assert "rising" in " ".join(report.findings)


def test_a_flat_series_has_no_trend(readings):
    report = ea.analyze(readings)
    assert report.trend.direction == "flat"
    assert bool(report.trend) is False
    assert str(report.trend) == "flat"


def test_one_step_change_is_reported_once():
    """A level that jumps and stays must not be sliced into several steps."""
    series = meter(days=28, seed=9).copy()
    series.iloc[24 * 14 :] *= 1.45
    report = ea.analyze(series)

    assert len(report.steps) == 1
    step = report.steps[0]
    assert step.when.date() == pd.Timestamp("2026-03-15").date()
    assert step.change > 0
    assert step.pct > 20
    assert "stepped up" in str(step)


def test_a_smooth_ramp_is_not_a_step():
    base = meter(days=28, seed=10)
    ramp = 1.0 + 0.05 * np.arange(base.size) / (24 * 7)
    report = ea.analyze(pd.Series(base.to_numpy() * ramp, index=base.index, name="kwh"))
    assert report.steps == []


def test_findings_are_plain_sentences(readings):
    findings = ea.analyze(readings, tariff=0.28).findings
    assert findings and all(isinstance(line, str) for line in findings)
    assert all(line.endswith(".") for line in findings)
    assert all(line[0].isupper() or line[0].isdigit() for line in findings)
    assert all(len(line.split()) >= 4 for line in findings)


def test_to_dict_is_json_safe(readings):
    payload = ea.analyze(readings, tariff=0.28).to_dict()
    text = json.dumps(payload)  # must not raise

    assert json.loads(text)["n_anomalies"] == payload["n_anomalies"]
    assert payload["unit"] == "kWh"
    assert payload["granularity"] == "1h"
    assert payload["anomalies"][0]["when"].startswith("2026-03-")
    assert payload["tariff"]["kind"] == "flat"


def test_report_dunders(readings):
    report = ea.analyze(readings)
    assert len(report) == report.n_periods
    assert bool(report) is (report.n_anomalies > 0)
    assert report.span == (report.by_period.index[0], report.by_period.index[-1])
    assert len(report.spikes) + len(report.drops) == report.n_anomalies


def test_dataframe_input_with_named_columns(frame):
    report = ea.analyze(frame, value="kwh", time="recorded_at")
    assert report.label == "kwh"
    assert report.time_label == "recorded_at"
    assert report.n_anomalies == 1


def test_columns_are_guessed_when_not_named(frame):
    report = ea.analyze(frame)
    assert report.label == "kwh"
    assert report.n_anomalies == 1


def test_legacy_pandas_offsets_do_not_leak_a_warning():
    """'H' and 'T' were canonical before pandas 2.2; reading them is our business."""
    series = meter(days=20, seed=4)
    for alias in ("H", "T", "S", "min", "1h"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                ea.analyze(series, granularity=alias)
            except ValueError:
                pass  # some of these are refused, which is fine; silence is not
        assert caught == [], f"granularity={alias!r} leaked {[str(w.message) for w in caught]}"


def test_an_hourly_alias_still_works_and_is_read_as_hourly():
    report = ea.analyze(meter(days=20, seed=4), granularity="H")
    assert report.granularity == "1h"
