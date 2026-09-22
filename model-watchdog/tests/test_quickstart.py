"""The README quickstart, run exactly as written."""

from __future__ import annotations

import json

import model_watchdog


def test_readme_quickstart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # --- README quickstart, verbatim ------------------------------------
    wd = model_watchdog.Watchdog(
        "checkout-model", reference={"prediction": [0.1, 0.2, 0.3, 0.4, 0.5] * 20}
    )
    for score in [0.92, 0.93, 0.94, 0.95, 0.96] * 20:
        wd.log(features={"cart_items": 3}, prediction=score, latency_ms=12.0)
    summary = wd.check().summary()
    # --------------------------------------------------------------------

    assert "model-watchdog: checkout-model" in summary
    assert "prediction_drift" in summary
    report = wd.check()
    assert report.records == 100
    assert not report.ok
    assert [check.name for check in report.failed] == ["prediction_drift"]
    assert report["prediction_drift"].value > 0.2
    assert (tmp_path / ".model_watchdog" / "checkout-model" / "events.jsonl").is_file()


def test_quickstart_report_is_json_safe(tmp_path):
    wd = model_watchdog.Watchdog(
        "q", reference={"prediction": [0.1, 0.2, 0.3]}, storage=tmp_path / "q"
    )
    for score in (0.9, 0.8, 0.7):
        wd.log(prediction=score)
    payload = wd.check().to_dict()
    assert json.loads(json.dumps(payload))["name"] == "q"
    assert len(payload["checks"]) == len(model_watchdog.MONITOR_NAMES)


def test_three_lines_is_enough(tmp_path):
    """The one-line convenience path finds an existing log."""
    storage = tmp_path / "shared"
    model_watchdog.Watchdog("m", storage=storage).log(prediction=1)
    report = model_watchdog.check("m", storage=storage)
    assert report.records == 1
    assert report.ok
    assert str(report) == report.summary()
