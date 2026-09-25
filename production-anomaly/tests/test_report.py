"""The report explains itself: summary text, JSON-safe dicts and self-describing objects."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import production_anomaly as pa
from conftest import line
from production_anomaly import QUALITY_NOTE, Event, ProductionReport, Stoppage


@pytest.fixture
def busy_report():
    df = line(days=3, every=5, rate=300, hours=(6, 22), noise=4, seed=3)
    df.loc[130:141, "units"] = 0.0
    df.loc[400:439, "units"] *= 0.7
    df.loc[700, "units"] = 3000.0
    df.loc[750, "units"] = -9000.0
    df.loc[800:803, "units"] = np.nan
    return pa.analyze(df, shift_hours=[(6, 14), (14, 22)], target_rate=3600)


def test_to_dict_is_json_safe(busy_report):
    data = busy_report.to_dict()
    text = json.dumps(data, allow_nan=False)
    assert json.loads(text)["quality"] is None
    assert data["quality_note"] == QUALITY_NOTE
    assert set(data) >= {
        "availability", "performance", "oee_partial", "stoppages", "events", "by_shift",
        "lost_units", "lost_by_cause", "findings", "notes", "schedule", "timezone",
    }
    assert len(data["by_shift"]) == len(busy_report.by_shift) == 6
    kinds = {e["kind"] for e in data["events"]}
    assert kinds >= {"downtime", "slow_running", "spike", "counter_reset", "data_gap"}


def test_quality_is_never_claimed(busy_report):
    assert busy_report.quality is None
    assert busy_report.notes[0] == QUALITY_NOTE
    assert "quality not measured" in busy_report.summary()
    assert busy_report.oee_partial == pytest.approx(
        busy_report.availability * busy_report.performance
    )


def test_summary_sections(busy_report):
    text = busy_report.summary()
    for piece in (
        "production report for 'units'", "availability", "performance", "oee_partial",
        "findings:", "stoppages (", "events:", "notes:", "vs target_rate",
    ):
        assert piece in text
    assert str(busy_report) == text
    assert repr(busy_report).startswith("ProductionReport(availability=")


def test_summary_lists_the_longest_stoppages_when_there_are_many(steady_day):
    df = steady_day.copy()
    for start in range(30, 1400, 60):
        df.loc[start:start + 1, "units"] = 0.0
    df.loc[700:760, "units"] = 0.0
    report = pa.analyze(df)
    text = report.summary(max_items=5)
    assert "61 min  downtime" in text
    assert "shorter ones (see report.stoppages)" in text


def test_stoppage_and_event_objects():
    start, end = pd.Timestamp("2026-03-02 10:50"), pd.Timestamp("2026-03-02 11:50")
    stop = Stoppage(start, end, 60.0, "downtime")
    assert str(stop) == "2026-03-02 10:50 to 2026-03-02 11:50  60 min  downtime"
    assert stop.to_dict() == {
        "start": "2026-03-02T10:50:00",
        "end": "2026-03-02T11:50:00",
        "minutes": 60.0,
        "kind": "downtime",
    }
    event = Event("spike", start, end, 5.0, "warning", "600 units", value=np.float64(600))
    assert event.to_dict()["value"] == 600.0 and str(event) == "[spike] 600 units"


def test_by_shift_columns_and_values(busy_report):
    frame = busy_report.by_shift
    assert list(frame.columns) == [
        "date", "shift", "start", "end", "scheduled_min", "measured_min", "downtime_min",
        "run_min", "units", "rate_per_hour", "availability", "performance", "oee_partial",
        "stoppages", "micro_stops",
    ]
    assert frame["stoppages"].sum() == 1
    assert frame["downtime_min"].sum() == busy_report.downtime_minutes
    assert frame["availability"].between(0, 1).all()


def test_intervals_table_labels_every_interval(busy_report):
    states = set(busy_report.intervals["state"])
    assert states >= {"running", "unscheduled", "downtime", "slow", "spike", "reset", "unknown"}
    assert len(busy_report.intervals) == 3 * 288


def test_events_are_in_time_order_and_explain_themselves(busy_report):
    starts = [e.start for e in busy_report.events]
    assert starts == sorted(starts)
    for event in busy_report.events:
        assert event.detail and event.severity in {"info", "warning", "critical"}
        assert event.minutes > 0


def test_findings_put_the_biggest_loss_first(busy_report):
    assert busy_report.findings[0].startswith(("Downtime", "Slow running", "Running speed"))
    assert all(isinstance(f, str) and f.endswith(".") for f in busy_report.findings)


def test_report_dataclass_defaults():
    report = ProductionReport(None, None, None, [], [], pd.DataFrame(), None, [])
    assert report.to_dict()["availability"] is None
    assert "production report" in report.summary()


def test_analyzer_class_with_custom_thresholds(steady_day):
    df = steady_day.copy()
    df.loc[100:103, "units"] = 0.0  # a 4-minute stop
    default = pa.analyze(df)
    strict = pa.ProductionAnalyzer(min_stop_minutes=3, interval="1min").analyze(df)
    assert [s.kind for s in default.stoppages] == ["micro_stop"]
    assert [s.kind for s in strict.stoppages] == ["downtime"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target_rate": -5}, "target_rate"),
        ({"target_rate": "fast"}, "target_rate"),
        ({"counter": "maybe"}, "counter"),
        ({"near_zero": 2}, "near_zero"),
        ({"slow_fraction": 1.5}, "slow_fraction"),
        ({"spike_factor": 1.0}, "spike_factor"),
        ({"drift_threshold": 0}, "drift_threshold"),
        ({"interval": "soon"}, "interval"),
        ({"min_stop_minutes": 0}, "min_stop_minutes"),
    ],
)
def test_bad_parameters_raise_value_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pa.ProductionAnalyzer(**kwargs)


def test_interval_accepts_retired_pandas_aliases(steady_day):
    for alias in ("1T", "1min", 1, "60S", "60s"):
        report = pa.ProductionAnalyzer(interval=alias).analyze(steady_day)
        assert report.interval_minutes == pytest.approx(1.0)
    hourly = pa.ProductionAnalyzer(interval="1H").analyze(line(days=1, every=60))
    assert hourly.interval_minutes == 60
