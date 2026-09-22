"""Regressions for everything an independent review found in 0.1.0.

One test (or a small cluster) per reported defect, each named after the thing
that used to go wrong. They are here to fail loudly if any of it comes back.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import model_watchdog
from model_watchdog import _stats

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# -- blocker: constant_output cried wolf on imbalanced classifiers --------
#
# A fraud/churn/click model that answers 0 for 95% of requests repeats itself
# constantly by design: the longest run of zeros in a healthy 1000-request
# window is about 87. The monitor compared that to a flat limit of 50 and
# called every such model stuck, while prediction_drift said, in the same
# report, that the distribution had not moved at all.


def _imbalanced(rate, count, seed):
    rng = np.random.default_rng(seed)
    return [int(value) for value in (rng.random(count) < rate)]


@pytest.mark.parametrize("rate, seed", [(0.05, 0), (0.02, 1), (0.10, 2)])
def test_a_healthy_imbalanced_classifier_is_not_called_stuck(tmp_path, rate, seed):
    """The window is drawn from the reference's own distribution."""
    reference = pd.DataFrame({"prediction": _imbalanced(rate, 5000, seed)})
    watchdog = model_watchdog.Watchdog(
        "fraud", reference=reference, storage=tmp_path / "fraud"
    )
    for value in _imbalanced(rate, 1000, seed + 100):
        watchdog.log(prediction=value)

    report = watchdog.check()
    stuck = report.get("constant_output")
    assert stuck.ok, stuck.message
    assert stuck.active, "there is plenty of data; it should have run"
    assert report.get("prediction_drift").ok, "the distribution did not move"
    assert report.ok, report.summary()
    assert report.failed == []
    # The limit came from the reference, not from thresholds.constant_run.
    assert stuck.threshold > stuck.details["flat_limit"]
    assert stuck.details["reference_share"] == pytest.approx(1.0 - rate, abs=0.05)


def test_a_genuinely_stuck_imbalanced_model_still_fails(tmp_path):
    """The same 95/5 reference, but now the model only ever says 0."""
    reference = pd.DataFrame({"prediction": _imbalanced(0.05, 5000, 0)})
    watchdog = model_watchdog.Watchdog(
        "fraud", reference=reference, storage=tmp_path / "fraud"
    )
    for _ in range(1000):
        watchdog.log(prediction=0)
    stuck = watchdog.check().get("constant_output")
    assert stuck.failed
    assert "model looks stuck" in stuck.message
    assert stuck.value == 1000


def test_a_balanced_reference_keeps_the_flat_limit(tmp_path):
    """Nothing about the fix loosens the classic even-odds case."""
    reference = {"prediction": [index % 2 for index in range(400)]}
    watchdog = model_watchdog.Watchdog("b", reference=reference, storage=tmp_path / "b")
    for _ in range(200):
        watchdog.log(prediction=0.5)
    stuck = watchdog.check().get("constant_output")
    assert stuck.failed
    assert stuck.threshold == 50, "a 50/50 reference earns no extra slack"


def test_a_reference_too_lopsided_to_judge_says_so(tmp_path):
    """At a 98% majority class a short window cannot tell stuck from normal."""
    reference = pd.DataFrame({"prediction": _imbalanced(0.02, 5000, 3)})
    watchdog = model_watchdog.Watchdog("q", reference=reference, storage=tmp_path / "q")
    for _ in range(120):
        watchdog.log(prediction=0)
    stuck = watchdog.check().get("constant_output")
    assert stuck.active is False, stuck.message
    assert stuck.ok, "inactive is not a failure"
    assert "not enough to tell a stuck model" in stuck.message
    assert stuck.details["needed"] > 120


def test_a_constant_reference_cannot_judge_a_constant_window(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "c", reference={"prediction": [1] * 300}, storage=tmp_path / "c"
    )
    for _ in range(100):
        watchdog.log(prediction=1)
    stuck = watchdog.check().get("constant_output")
    assert stuck.active is False
    assert "one constant value" in stuck.message


def test_without_a_reference_the_flat_limit_still_catches_a_stuck_model(tmp_path):
    watchdog = model_watchdog.Watchdog("n", storage=tmp_path / "n")
    for _ in range(200):
        watchdog.log(prediction=0.5)
    stuck = watchdog.check().get("constant_output")
    assert stuck.failed
    assert stuck.threshold == 50


# -- blocker: decile PSI ran on far too little data -----------------------
#
# min_records was 10, so a ten-bin PSI was computed from ten points: each
# decile held one sample and the number came out around 0.43 against a 0.2
# alarm line. Two samples of the *same* distribution failed 98% of the time.


def _log_normal(watchdog, count, seed):
    rng = np.random.default_rng(seed)
    for value in rng.normal(0.0, 1.0, count):
        watchdog.log(features={"x": float(value)}, prediction=float(value))


def _same_distribution_watchdog(tmp_path, seed):
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame(
        {
            "prediction": rng.normal(0.0, 1.0, 1000),
            "x": rng.normal(0.0, 1.0, 1000),
        }
    )
    return model_watchdog.Watchdog(
        "m", reference=reference, storage=tmp_path / ("m%d" % seed)
    )


@pytest.mark.parametrize("count", [10, 12, 20, 30, 50])
def test_the_psi_monitors_stay_inactive_on_too_few_records(tmp_path, count):
    watchdog = _same_distribution_watchdog(tmp_path, count)
    _log_normal(watchdog, count, seed=count + 7)
    report = watchdog.check()
    for name in ("prediction_drift", "feature_drift"):
        check = report.get(name)
        assert check.active is False, check.message
        assert check.ok, "not enough data is not a failure"
        assert "only %d records; need" % count in check.message
    assert report.ok, report.summary()
    assert "ALERT" not in report.summary()


@pytest.mark.parametrize("seed", range(12))
def test_identical_traffic_does_not_raise_a_drift_alarm(tmp_path, seed):
    """The window is drawn from the reference's distribution, at every size."""
    for count in (80, 150, 400):
        watchdog = _same_distribution_watchdog(tmp_path / str(count), seed)
        _log_normal(watchdog, count, seed=1000 + seed)
        report = watchdog.check()
        drift = report.get("prediction_drift")
        assert drift.active, drift.message
        assert drift.ok, "%d records: %s" % (count, drift.message)
        assert report.get("feature_drift").ok, report.get("feature_drift").message


def test_real_drift_is_still_caught_once_the_monitors_are_active(tmp_path):
    watchdog = _same_distribution_watchdog(tmp_path, 4)
    rng = np.random.default_rng(55)
    for value in rng.normal(3.0, 1.0, 120):  # shifted three standard deviations
        watchdog.log(features={"x": float(value)}, prediction=float(value))
    report = watchdog.check()
    assert report.get("prediction_drift").failed, report.summary()
    assert report.get("feature_drift").failed


def test_a_label_comparison_needs_only_min_records(tmp_path):
    """A handful of classes is binned per value, which is sound on small n."""
    watchdog = model_watchdog.Watchdog(
        "labels", reference={"prediction": [0, 1] * 100}, storage=tmp_path / "labels"
    )
    for index in range(20):
        watchdog.log(prediction=index % 2)
    drift = watchdog.check().get("prediction_drift")
    assert drift.active, drift.message
    assert drift.ok


def test_the_bin_count_never_outruns_the_records():
    reference = list(np.random.default_rng(0).normal(0.0, 1.0, 1000))
    for count, expected in ((60, 3), (100, 5), (200, 10), (1000, 10)):
        current = list(np.random.default_rng(1).normal(0.0, 1.0, count))
        kind, bins = _stats.psi_plan(reference, current)
        assert kind == _stats.PSI_BY_QUANTILE
        assert bins == expected, count


# -- major: duplicates born from expanding the 'features' column ----------
#
# The duplicate guard ran before the features dict was exploded into columns,
# so the concat put "age" in twice. frame["age"] was then a DataFrame, and
# as_series handed back its *column names* as the feature's values.


def _log_shaped_frame():
    """What pd.read_json('events.jsonl', lines=True) gives on our own format."""
    return pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3, 0.4] * 25,
            "age": [30, 40, 50, 60] * 25,
            "features": [{"age": 1.0, "plan": "pro"}] * 100,
        }
    )


def test_duplicates_made_by_expanding_features_raise():
    with pytest.raises(ValueError, match="duplicate column names: age"):
        model_watchdog.ReferenceProfile.from_frame(_log_shaped_frame())


def test_that_duplicate_becomes_a_visible_note_not_a_fabricated_alarm(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "dup", reference=_log_shaped_frame(), storage=tmp_path / "dup"
    )
    assert any("duplicate column names" in note for note in watchdog.reference.notes)
    for _ in range(60):
        watchdog.log(features={"age": 35.0, "plan": "pro"}, prediction=0.2)
    report = watchdog.check()
    drift = report.get("feature_drift")
    assert drift.failed is False, drift.message
    assert "age" not in watchdog.reference.features
    assert any("duplicate column names" in note for note in report.notes)


def test_as_series_refuses_a_dataframe_instead_of_returning_column_names():
    frame = pd.DataFrame({"age": [1, 2, 3], "plan": ["a", "b", "c"]})
    with pytest.raises(TypeError, match="got a DataFrame"):
        _stats.as_series(frame)


def test_duplicate_feature_columns_in_a_mapping_reference_raise():
    features = pd.DataFrame([[1, 2]], columns=["age", "age"])
    with pytest.raises(ValueError, match="duplicate column names: age"):
        model_watchdog.ReferenceProfile.from_mapping(
            {"prediction": [0, 1], "features": features}
        )


# -- major: a wrong thresholds= disabled all seven monitors, silently -----


def test_a_wrong_thresholds_type_raises_instead_of_blinding_the_watchdog(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "t", reference={"prediction": [0.0, 1.0] * 250}, storage=tmp_path / "t"
    )
    for _ in range(300):
        watchdog.log(prediction=999.0)
    assert watchdog.check().ok is False, "the model really is broken"

    for wrong in ("bad", {"min_records": 10}, 7, object()):
        with pytest.raises(TypeError, match="thresholds must be a Thresholds"):
            watchdog.check(thresholds=wrong)
        with pytest.raises(TypeError, match="thresholds must be a Thresholds"):
            watchdog.report(thresholds=wrong)
    assert watchdog.check(thresholds=model_watchdog.Thresholds()).ok is False


def test_a_report_where_every_monitor_crashed_is_not_ok(tmp_path, monkeypatch):
    """Belt and braces: blind must never read healthy, whatever broke."""
    from model_watchdog import _monitors

    def make(name):
        def explode(window, reference, thresholds):
            raise RuntimeError("everything is down")

        explode.__name__ = name
        return explode

    monkeypatch.setattr(
        _monitors, "MONITORS", [make(name) for name in model_watchdog.MONITOR_NAMES]
    )
    watchdog = model_watchdog.Watchdog("blind", storage=tmp_path / "blind")
    for _ in range(20):
        watchdog.log(prediction=1.0)
    report = watchdog.check()
    assert report.blind is True
    assert report.ok is False
    assert bool(report) is False
    assert "every monitor failed to run" in report.headline()
    assert report.to_dict()["ok"] is False


def test_one_crashed_monitor_still_leaves_the_report_ok(tmp_path, monkeypatch):
    """Only *all* of them going down is blindness; one is just inactive."""
    from model_watchdog import _monitors

    def explode(window, reference, thresholds):
        raise RuntimeError("boom")

    explode.__name__ = "latency"
    monkeypatch.setattr(
        _monitors, "MONITORS", [_monitors.null_rate, explode, _monitors.volume]
    )
    watchdog = model_watchdog.Watchdog("one", storage=tmp_path / "one")
    for _ in range(20):
        watchdog.log(prediction=1.0, features={"a": 1})
    report = watchdog.check()
    assert report.blind is False
    assert report.ok is True


# -- major: report(since=<unparseable>) widened the window to everything ---


@pytest.fixture
def daily_log(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "t", reference={"prediction": [0.0] * 200}, storage=tmp_path / "t"
    )
    for index in range(300):
        watchdog.log(prediction=float(index), ts=BASE + timedelta(days=index))
    return watchdog


def test_report_since_still_filters_on_everything_readable(daily_log):
    assert daily_log.report().records == 300
    assert daily_log.report(since="2026-10-01").records == 27
    assert daily_log.report(since=BASE + timedelta(days=290)).records == 10
    assert daily_log.report(since=datetime(2026, 10, 1)).records == 27
    assert daily_log.report(since=pd.Timestamp("2026-10-01")).records == 27


@pytest.mark.parametrize(
    "bad", ["2026-13-45", "not-a-date", "", 42, 1758537600, 3.5, object(), [2026, 1, 1]]
)
def test_an_unreadable_since_raises_instead_of_reading_the_whole_log(daily_log, bad):
    with pytest.raises(ValueError, match="could not read since="):
        daily_log.report(since=bad)


def test_the_since_error_names_the_accepted_formats(daily_log):
    with pytest.raises(ValueError) as failure:
        daily_log.report(since="2026-13-45")
    message = str(failure.value)
    assert "2026-13-45" in message
    assert "2026-09-22T10:00:00+00:00" in message


# -- minor: a pandas UserWarning escaped to_utc(), and could lose a record -


def test_no_pandas_warning_escapes_into_the_callers_filters(tmp_path):
    watchdog = model_watchdog.Watchdog("t", storage=tmp_path / "t")
    nanoseconds = "2026-09-22T10:00:00.123456789+00:00"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        watchdog.log(prediction=1.0, ts=nanoseconds)
        watchdog.report(since=nanoseconds)
        watchdog.check()
        watchdog.metrics()
    leaked = [str(item.message) for item in caught]
    assert leaked == [], leaked


def test_a_nanosecond_timestamp_is_not_dropped_under_werror(tmp_path):
    """`python -W error::UserWarning` is ordinary in CI; it must not lose a record."""
    script = tmp_path / "werror.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import model_watchdog

            watchdog = model_watchdog.Watchdog("t", storage=sys.argv[1])
            watchdog.log(prediction=1.0, ts="2026-09-22T10:00:00.123456789+00:00")
            watchdog.log(prediction=2.0)
            print(len(watchdog))
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", str(script), str(tmp_path / "w")],
        capture_output=True,
        env=environment,
        timeout=120,
    )
    stderr = completed.stderr.decode("utf-8", "replace")
    assert completed.returncode == 0, stderr
    assert completed.stdout.decode("utf-8").strip() == "2", stderr


def test_a_nanosecond_timestamp_still_round_trips_to_the_microsecond(tmp_path):
    watchdog = model_watchdog.Watchdog("t", storage=tmp_path / "t")
    watchdog.log(prediction=1.0, ts="2026-09-22T10:00:00.123456789+00:00")
    stamp = watchdog.metrics()["ts"].iloc[0]
    assert stamp.microsecond == 123456
    assert stamp.utcoffset().total_seconds() == 0


# -- minor: check(window=0) reported over nothing and called it healthy ----


@pytest.fixture
def forty(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "k", reference={"prediction": [1.0, 2.0] * 60}, storage=tmp_path / "k"
    )
    for index in range(40):
        watchdog.log(prediction=float(index % 3))
    return watchdog


@pytest.mark.parametrize("window", [0, -1, -5])
def test_a_non_positive_window_raises_instead_of_reporting_on_nothing(forty, window):
    with pytest.raises(ValueError, match="window must be a positive integer"):
        forty.check(window=window)


@pytest.mark.parametrize("window", ["ten", 1.5, None, True])
def test_a_non_integer_window_names_the_argument(forty, window):
    with pytest.raises(TypeError, match="window must be a positive integer"):
        forty.check(window=window)


def test_a_good_window_is_untouched(forty):
    assert forty.check(window=10).records == 10
    assert forty.check().records == 40


def test_the_cli_turns_a_bad_window_into_a_usage_error(forty, capsys):
    """argparse's exit 2 and one clear line, not a traceback."""
    from model_watchdog import cli

    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(forty.storage), "--window", "0"])
    assert exit_info.value.code == 2
    assert "window must be a positive integer" in capsys.readouterr().err


# -- minor: a flawless reference failed on any error at all ---------------


def _perfect_reference():
    return {
        "prediction": [float(index) for index in range(200)],
        "actual": [float(index) for index in range(200)],
    }


def test_a_perfect_regression_reference_tolerates_a_rounding_error(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "z", reference=_perfect_reference(), storage=tmp_path / "z"
    )
    assert watchdog.reference.error == 0.0
    for index in range(60):
        watchdog.log(prediction=float(index), actual=float(index) + 0.001)
    check = watchdog.check().get("accuracy_drop")
    assert check.ok, check.message
    assert check.active, "there are labels; it should have run"
    assert check.threshold > 0.0, "a zero limit fails on any error at all"
    assert "accuracy_drop" not in [item.name for item in watchdog.check().failed]


def test_a_perfect_reference_still_catches_a_real_regression(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "z", reference=_perfect_reference(), storage=tmp_path / "z"
    )
    for index in range(60):
        watchdog.log(prediction=float(index), actual=float(index) + 50.0)
    check = watchdog.check().get("accuracy_drop")
    assert check.failed, check.message


def test_a_nonzero_reference_error_keeps_its_proportional_limit(tmp_path):
    """The floor must not loosen a reference that has real error in it."""
    watchdog = model_watchdog.Watchdog(
        "p",
        reference={
            "prediction": [float(index) for index in range(200)],
            "actual": [float(index) + 1.0 for index in range(200)],
        },
        storage=tmp_path / "p",
    )
    assert watchdog.reference.error == pytest.approx(1.0)
    for index in range(60):
        watchdog.log(prediction=float(index), actual=float(index) + 2.0)
    check = watchdog.check().get("accuracy_drop")
    assert check.failed, check.message
    assert check.threshold == pytest.approx(1.25)


# -- minor: names were coerced, and long ones collided --------------------


@pytest.mark.parametrize("name", [None, 42, 3.5, b"bytes", ["a"], object()])
def test_a_non_string_name_is_rejected(name):
    with pytest.raises(TypeError, match="name must be a str"):
        model_watchdog.Watchdog(name)


def test_names_that_differ_late_do_not_share_one_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = model_watchdog.Watchdog("x" * 100 + "-alpha")
    second = model_watchdog.Watchdog("x" * 100 + "-beta")
    assert first.storage != second.storage
    first.log(prediction=1.0)
    second.log(prediction=2.0)
    assert len(first) == 1 and len(second) == 1
    assert first.metrics()["prediction"].tolist() == [1.0]
    assert second.metrics()["prediction"].tolist() == [2.0]


def test_names_made_only_of_separators_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = model_watchdog.Watchdog("***").storage
    second = model_watchdog.Watchdog("!!!").storage
    assert first != second


def test_a_readable_name_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    watchdog = model_watchdog.Watchdog("checkout-model")
    assert watchdog.storage.name == "checkout-model"
    assert watchdog.name == "checkout-model"


# -- the logging path is still unbreakable --------------------------------


def test_none_of_the_new_validation_leaks_into_log(tmp_path):
    """log() still swallows everything; only check/report are allowed to raise."""
    watchdog = model_watchdog.Watchdog("safe", storage=tmp_path / "safe")
    watchdog.log(prediction=1.0, ts=object())
    watchdog.log(prediction=2.0, ts="2026-13-45")
    watchdog.log(features=object(), prediction=3.0)
    watchdog.log(prediction=4.0)
    assert len(watchdog) >= 1
    assert watchdog.check().ok


def test_the_report_stays_json_safe_with_the_new_details(tmp_path):
    reference = pd.DataFrame({"prediction": _imbalanced(0.05, 2000, 9)})
    watchdog = model_watchdog.Watchdog("j", reference=reference, storage=tmp_path / "j")
    for value in _imbalanced(0.05, 600, 11):
        watchdog.log(prediction=value, features={"a": 1.0}, latency_ms=2.0)
    payload = json.loads(json.dumps(watchdog.check().to_dict()))
    stuck = [item for item in payload["checks"] if item["name"] == "constant_output"][0]
    assert stuck["details"]["reference_share"] is not None
    assert payload["ok"] is True
