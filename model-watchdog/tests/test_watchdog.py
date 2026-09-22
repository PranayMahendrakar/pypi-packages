"""The Watchdog itself: logging, reading back, reports, alerts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import model_watchdog
from conftest import WINDOW_START, log_healthy


def test_storage_directory_is_created_on_demand(tmp_path):
    storage = tmp_path / "deep" / "nested" / "logs"
    watchdog = model_watchdog.Watchdog("m", storage=storage)
    assert not storage.exists()
    watchdog.log(prediction=1)
    assert (storage / "events.jsonl").is_file()
    assert len(watchdog) == 1


def test_default_storage_is_under_the_default_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    watchdog = model_watchdog.Watchdog("Checkout Model v2")
    watchdog.log(prediction=1)
    expected = tmp_path / model_watchdog.DEFAULT_ROOT / "Checkout_Model_v2"
    assert (expected / "events.jsonl").is_file()
    assert repr(watchdog).startswith("Watchdog('Checkout Model v2'")


def test_timestamps_are_timezone_aware_utc(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    before = model_watchdog.utc_now()
    watchdog.log(prediction=1)
    frame = watchdog.metrics()
    assert str(frame["ts"].dt.tz) == "UTC"
    stamp = frame["ts"].iloc[0].to_pydatetime()
    assert stamp.tzinfo is not None
    assert stamp >= before - timedelta(seconds=5)
    report = watchdog.check()
    assert report.first_ts.tzinfo is not None
    assert report.to_dict()["first_ts"].endswith("+00:00")


def test_metrics_columns_cover_features_and_meta(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(
        features={"age": 31, "plan": "pro"},
        prediction=0.7,
        actual=1,
        latency_ms=9.5,
        request_id="abc",
        model_version="2.1",
    )
    frame = watchdog.metrics()
    assert list(frame.columns) == [
        "ts",
        "prediction",
        "actual",
        "latency_ms",
        "age",
        "plan",
        "request_id",
        "model_version",
    ]
    row = frame.iloc[0]
    assert row["prediction"] == 0.7
    assert row["plan"] == "pro"
    assert row["request_id"] == "abc"
    assert frame.attrs["feature_columns"] == ["age", "plan"]


def test_feature_named_like_a_core_column_is_kept(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(features={"prediction": 3}, prediction=1, actual=0)
    frame = watchdog.metrics()
    assert frame["prediction"].iloc[0] == 1
    assert frame["prediction_feature"].iloc[0] == 3


def test_check_window_limits_the_records(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    for index in range(50):
        watchdog.log(prediction=index)
    assert watchdog.check().records == 50
    assert watchdog.check(window=10).records == 10


def test_report_since_filters_by_timestamp(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    for index in range(10):
        watchdog.log(prediction=index, ts=WINDOW_START + timedelta(hours=index))
    everything = watchdog.report()
    assert everything.records == 10
    assert everything.window is None
    recent = watchdog.report(since=WINDOW_START + timedelta(hours=6))
    assert recent.records == 4
    assert "since 2026-09-22T06:00:00" in recent.summary()
    as_text = watchdog.report(since="2026-09-22T06:00:00+00:00")
    assert as_text.records == 4
    naive = watchdog.report(since=datetime(2026, 9, 22, 6, 0, 0))
    assert naive.records == 4


def test_alert_callable_fires_once_per_failing_check(tmp_path, reference):
    seen = []
    watchdog = model_watchdog.Watchdog(
        "m", reference=reference, storage=tmp_path / "m", alert=seen.append
    )
    for index in range(60):
        watchdog.log(prediction=1, latency_ms=11, ts=WINDOW_START + timedelta(seconds=36 * index))
    report = watchdog.check()
    assert not report.ok
    assert len(seen) == 1
    alert = seen[0]
    assert isinstance(alert, model_watchdog.Alert)
    assert alert.name == "m"
    assert alert.report is report
    assert [check.name for check in alert.failed] == [c.name for c in report.failed]
    assert "ALERT" in str(alert)
    assert alert.to_dict()["report"]["ok"] is False


def test_alert_is_not_fired_when_everything_is_fine(tmp_path, reference):
    seen = []
    watchdog = model_watchdog.Watchdog(
        "m", reference=reference, storage=tmp_path / "m", alert=seen.append
    )
    log_healthy(watchdog)
    report = watchdog.check()
    assert report.ok, report.summary()
    assert seen == []
    assert report.failed == []
    assert report.active, "at least one monitor should have run"


def test_a_broken_alert_callable_does_not_break_check(tmp_path, reference):
    def explode(alert):
        raise RuntimeError("pagerduty is down")

    watchdog = model_watchdog.Watchdog(
        "m", reference=reference, storage=tmp_path / "m", alert=explode
    )
    for index in range(30):
        watchdog.log(prediction=1, ts=WINDOW_START + timedelta(seconds=36 * index))
    report = watchdog.check()
    assert not report.ok


def test_report_alone_does_not_alert(tmp_path, reference):
    seen = []
    watchdog = model_watchdog.Watchdog(
        "m", reference=reference, storage=tmp_path / "m", alert=seen.append
    )
    for index in range(30):
        watchdog.log(prediction=1)
    assert not watchdog.report().ok
    assert seen == []


def test_thresholds_can_be_tuned(tmp_path, reference):
    watchdog = model_watchdog.Watchdog("m", reference=reference, storage=tmp_path / "m")
    for index in range(30):
        watchdog.log(prediction=1, ts=WINDOW_START + timedelta(seconds=36 * index))
    assert not watchdog.check().get("constant_output").ok
    relaxed = model_watchdog.Thresholds(constant_run=1000, min_records=1000)
    assert watchdog.check(thresholds=relaxed).get("constant_output").active is False
    watchdog.thresholds = relaxed
    assert watchdog.check().get("constant_output").active is False
    assert relaxed.to_dict()["constant_run"] == 1000


def test_reference_can_be_set_later_from_the_log(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    for index in range(40):
        watchdog.log(features={"age": 20 + index % 30}, prediction=index % 5, latency_ms=10.0)
    assert watchdog.check().get("prediction_drift").active is False
    watchdog.reference = watchdog.metrics()
    assert isinstance(watchdog.reference, model_watchdog.ReferenceProfile)
    report = watchdog.check()
    assert report.get("prediction_drift").active is True
    assert report.get("prediction_drift").ok
    assert "predictions" in watchdog.reference.describe()
    assert watchdog.reference.to_dict()["records"] == 40


def test_monitor_names_are_the_report_order(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    report = watchdog.check()
    assert [check.name for check in report.checks] == model_watchdog.MONITOR_NAMES
    assert watchdog.monitors == model_watchdog.MONITOR_NAMES
    with pytest.raises(KeyError):
        report["nope"]
    assert report.get("nope") is None


def test_storage_object_reads_what_was_written(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(prediction=1)
    store = model_watchdog.JsonlStorage(tmp_path / "m")
    assert store.exists()
    assert store.count() == 1
    assert store.read()[0]["prediction"] == 1
    assert store.read(limit=2)[0]["prediction"] == 1
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        store.read(limit=0)
    with pytest.raises(TypeError, match="limit must be a positive integer"):
        store.read(limit="two")


def test_a_one_row_dataframe_is_accepted_as_features(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(features=pd.DataFrame([{"age": 4, "plan": "pro"}]), prediction=1)
    watchdog.log(features=pd.Series({"age": 5, "plan": "basic"}), prediction=0)
    watchdog.log(features=[7, 8], prediction=1)
    frame = watchdog.metrics()
    assert frame["age"].tolist()[:2] == [4, 5]
    assert frame["f0"].iloc[2] == 7
