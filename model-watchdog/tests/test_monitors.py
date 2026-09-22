"""Every monitor, passing and failing, on deterministic data."""

from __future__ import annotations

from datetime import timedelta

import pytest

import model_watchdog
from conftest import PLANS, WINDOW_START, log_healthy


@pytest.fixture
def watchdog(tmp_path, reference):
    return model_watchdog.Watchdog("m", reference=reference, storage=tmp_path / "m")


def _log(watchdog, count=40, step=36, **overrides):
    for index in range(count):
        record = {
            "features": {"age": 20 + (index % 40), "plan": PLANS[index % 3]},
            "prediction": index % 2,
            "actual": index % 2,
            "latency_ms": 10 + (index % 20),
            "ts": WINDOW_START + timedelta(seconds=step * index),
        }
        for key, value in overrides.items():
            record[key] = value(index) if callable(value) else value
        watchdog.log(**record)


def test_healthy_traffic_passes_every_active_monitor(watchdog):
    log_healthy(watchdog)
    report = watchdog.check()
    assert report.ok, report.summary()
    assert report.failed == []
    names = {check.name for check in report.active}
    assert names == set(model_watchdog.MONITOR_NAMES), report.summary()
    assert "OK: 7 of 7 checks passed" in report.summary()


def test_prediction_drift_fires(watchdog):
    _log(watchdog, prediction=1)
    check = watchdog.check().get("prediction_drift")
    assert check.failed
    assert check.value > check.threshold
    assert "major shift" in check.message
    assert check.details["reference_n"] == 200


def test_feature_drift_names_the_worst_feature(watchdog):
    # 80 records, not 40: "age" is continuous, so its PSI is binned by
    # reference quantiles and stays inactive below min_drift_records.
    _log(watchdog, count=80, features=lambda index: {"age": 900 + index, "plan": PLANS[index % 3]})
    check = watchdog.check().get("feature_drift")
    assert check.failed
    assert check.details["drifted"] == ["age"]
    assert "worst: age" in check.message
    assert check.details["per_feature"]["plan"] < check.threshold


def test_accuracy_drop_fires_when_labels_go_wrong(watchdog):
    _log(watchdog, actual=lambda index: 1 - (index % 2))
    check = watchdog.check().get("accuracy_drop")
    assert check.failed
    assert check.value == 0.0
    assert check.details["metric"] == "accuracy"
    assert "vs reference 0.9000" in check.message


def test_accuracy_drop_passes_when_labels_still_match(watchdog):
    _log(watchdog)
    check = watchdog.check().get("accuracy_drop")
    assert check.ok and check.active
    assert check.value == 1.0


def test_regression_reference_uses_absolute_error(tmp_path):
    reference = {"prediction": [float(i) for i in range(100)], "actual": [float(i) + 0.5 for i in range(100)]}
    watchdog = model_watchdog.Watchdog("r", reference=reference, storage=tmp_path / "r")
    assert watchdog.reference.task == "regression"
    for index in range(30):
        watchdog.log(prediction=float(index), actual=float(index) + 9.0)
    check = watchdog.check().get("accuracy_drop")
    assert check.failed
    assert check.details["metric"] == "mae"
    assert "mean absolute error" in check.message


def test_latency_fires_on_p95(watchdog):
    _log(watchdog, latency_ms=lambda index: 400 if index % 10 == 0 else 12)
    check = watchdog.check().get("latency")
    assert check.failed
    assert "p95 too slow" in check.message
    assert check.details["reference_p95"] > 0


def test_latency_passes_when_it_is_the_same_speed(watchdog):
    _log(watchdog)
    check = watchdog.check().get("latency")
    assert check.ok and check.active
    assert check.details["p50"] is not None


def test_null_rate_fires(watchdog):
    _log(watchdog, features=lambda index: {"age": None, "plan": PLANS[index % 3]})
    check = watchdog.check().get("null_rate")
    assert check.failed
    assert check.value == pytest.approx(0.5)
    assert check.details["per_feature"]["age"] == pytest.approx(1.0)


def test_constant_output_catches_the_silent_failure(watchdog):
    _log(watchdog, count=200, prediction=0.5)
    check = watchdog.check().get("constant_output")
    assert check.failed
    assert check.value == 200
    assert check.threshold == 50
    assert "model looks stuck" in check.message
    assert check.details["repeated_value"] == 0.5


def test_constant_output_ignores_a_short_run(watchdog):
    _log(watchdog, count=200)
    check = watchdog.check().get("constant_output")
    assert check.ok and check.active
    assert check.value == 1


def test_volume_spike_and_drop(watchdog, tmp_path, reference):
    _log(watchdog, count=60, step=2)
    check = watchdog.check().get("volume")
    assert check.failed
    assert "traffic spike" in check.message
    assert check.value > check.threshold

    quiet = model_watchdog.Watchdog("q", reference=reference, storage=tmp_path / "q")
    _log(quiet, count=20, step=600)
    check = quiet.check().get("volume")
    assert check.failed
    assert "traffic dropped" in check.message
    assert check.value < check.threshold


def test_every_monitor_can_fail_at_once(watchdog):
    _log(
        watchdog,
        count=120,
        step=2,
        prediction=1,
        actual=0,
        latency_ms=500,
        features=lambda index: {"age": 900, "plan": None},
    )
    report = watchdog.check()
    assert not report.ok
    assert {check.name for check in report.failed} == set(model_watchdog.MONITOR_NAMES)
    assert "ALERT: 7 of 7 active checks failed" in report.summary()
    assert report.to_dict()["failed"] == model_watchdog.MONITOR_NAMES


def test_one_broken_monitor_does_not_hide_the_others(watchdog, monkeypatch):
    from model_watchdog import _monitors

    def explode(window, reference, thresholds):
        raise RuntimeError("boom")

    explode.__name__ = "latency"
    monkeypatch.setattr(_monitors, "MONITORS", [_monitors.prediction_drift, explode])
    _log(watchdog, prediction=1)
    report = watchdog.check()
    assert report.get("prediction_drift").failed
    broken = report.get("latency")
    assert broken.active is False
    assert "monitor raised RuntimeError: boom" in broken.message


def test_missing_prediction_in_reference_does_not_slide_it_out_of_alignment():
    """Regression: predictions and actuals were filtered for missing values
    independently, so one gap slid every later prediction onto the wrong actual.
    A perfect baseline read as 0.03 accuracy, and a collapsed model then looked
    unchanged against it."""
    import tempfile

    import numpy as np
    import pandas as pd

    import model_watchdog as mw

    reference = pd.DataFrame({"prediction": [1, 0] * 50, "actual": [1, 0] * 50})
    reference.loc[3, "prediction"] = np.nan  # a single gap

    watchdog = mw.Watchdog("align", reference=reference, storage=tempfile.mkdtemp())
    profile = getattr(watchdog, "reference", None) or getattr(watchdog, "_reference")
    assert profile.accuracy == 1.0, "a perfect baseline must survive one missing prediction"

    for _ in range(200):  # the model now gets every single case wrong
        watchdog.log(prediction=0, actual=1)
    report = watchdog.check()
    accuracy_checks = [c for c in report.checks if "accur" in c.name.lower()]
    assert accuracy_checks, "an accuracy monitor should be active once actuals are logged"
    assert any(not c.ok for c in accuracy_checks), "a total collapse must not pass"
