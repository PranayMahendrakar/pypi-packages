"""The awkward cases: no reference, no actuals, bad disks, bad lines, odd data.

Monitoring must never take the host application down, so most of this file is
about things going wrong and the caller not noticing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import model_watchdog
from conftest import WINDOW_START


# -- no reference ---------------------------------------------------------


def test_logging_works_with_no_reference_at_all(tmp_path):
    """Logging before a reference exists is normal, not an error."""
    watchdog = model_watchdog.Watchdog("fresh", storage=tmp_path / "fresh")
    for index in range(30):
        watchdog.log(features={"age": index}, prediction=index % 3, latency_ms=5.0)
    report = watchdog.check()
    assert report.records == 30
    assert report.ok, report.summary()
    assert report.failed == []


def test_check_names_the_inactive_monitors_instead_of_failing(tmp_path):
    watchdog = model_watchdog.Watchdog("fresh", storage=tmp_path / "fresh")
    for index in range(30):
        watchdog.log(prediction=index % 3)
    report = watchdog.check()
    inactive = {check.name for check in report.inactive}
    assert inactive == {
        "prediction_drift",
        "feature_drift",
        "accuracy_drop",
        "latency",
        "null_rate",
        "volume",
    }
    for check in report.inactive:
        assert check.ok, "an inactive monitor is not a failure"
        assert check.status == "inactive"
        assert check.message, "an inactive monitor must say why"
        assert check.value is None and check.threshold is None
    assert "inactive monitors (not failures" in report.summary()
    assert set(report.to_dict()["inactive"]) == inactive


def test_no_records_at_all_reports_cleanly(tmp_path):
    watchdog = model_watchdog.Watchdog("empty", storage=tmp_path / "empty")
    report = watchdog.check()
    assert report.records == 0
    assert report.ok
    assert "no records logged yet" in report.summary()
    assert watchdog.metrics().empty
    assert len(watchdog) == 0
    assert json.loads(json.dumps(report.to_dict()))["records"] == 0


# -- no actuals -----------------------------------------------------------


def test_without_actuals_the_accuracy_monitor_is_inactive_not_failed(tmp_path, reference):
    watchdog = model_watchdog.Watchdog("m", reference=reference, storage=tmp_path / "m")
    for index in range(40):
        watchdog.log(prediction=index % 2, ts=WINDOW_START + timedelta(seconds=36 * index))
    check = watchdog.check().get("accuracy_drop")
    assert check.active is False
    assert check.ok is True
    assert check.failed is False
    assert "no actuals logged" in check.message


def test_actuals_logged_later_activate_the_monitor(tmp_path, reference):
    watchdog = model_watchdog.Watchdog("m", reference=reference, storage=tmp_path / "m")
    for index in range(40):
        watchdog.log(prediction=index % 2)
    assert watchdog.check().get("accuracy_drop").active is False
    for index in range(40):
        watchdog.log(prediction=index % 2, actual=index % 2)
    assert watchdog.check().get("accuracy_drop").active is True


# -- storage --------------------------------------------------------------


def test_storage_directory_is_created_on_demand_many_levels_deep(tmp_path):
    storage = tmp_path / "a" / "b" / "c" / "d"
    watchdog = model_watchdog.Watchdog("m", storage=storage)
    assert not storage.exists()
    watchdog.log(prediction=1)
    assert (storage / "events.jsonl").is_file()


def test_concurrent_appends_from_several_processes_do_not_corrupt(tmp_path):
    """Four separate processes append at once; every line must survive whole."""
    storage = tmp_path / "shared"
    per_process = 60
    workers = 4
    script = tmp_path / "worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import model_watchdog

            storage, tag, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
            watchdog = model_watchdog.Watchdog("shared", storage=storage)
            for index in range(count):
                watchdog.log(
                    features={"worker": tag, "index": index},
                    prediction=index % 7,
                    latency_ms=1.0 + index,
                )
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    running = [
        subprocess.Popen(
            [sys.executable, str(script), str(storage), str(tag), str(per_process)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for tag in range(workers)
    ]
    for process in running:
        _, errors = process.communicate(timeout=120)
        assert process.returncode == 0, errors.decode("utf-8", "replace")

    path = storage / "events.jsonl"
    raw_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(raw_lines) == workers * per_process, "a line was lost or overwritten"
    for line in raw_lines:
        json.loads(line)  # every line is whole and parseable

    frame = model_watchdog.Watchdog("shared", storage=storage).metrics()
    assert len(frame) == workers * per_process
    counts = frame["worker"].value_counts()
    assert sorted(counts.tolist()) == [per_process] * workers


def test_a_partial_trailing_line_is_skipped_with_a_warning(tmp_path, caplog):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(prediction=1)
    watchdog.log(prediction=2)
    path = tmp_path / "m" / "events.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026-09-22T00:00:00+00:00", "predic')

    with caplog.at_level(logging.WARNING, logger="model_watchdog._storage"):
        report = watchdog.check()
    assert report.records == 2, "the two good records still read back"
    assert "skipped 1 unreadable line" in caplog.text
    assert watchdog.metrics()["prediction"].tolist() == [1, 2]


def test_a_corrupt_line_in_the_middle_is_skipped(tmp_path):
    path = tmp_path / "m" / "events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"ts": "2026-09-22T00:00:00+00:00", "prediction": 1}\n'
        "this is not json at all\n"
        "[1, 2, 3]\n"
        '{"ts": "2026-09-22T00:00:01+00:00", "prediction": 2}\n',
        encoding="utf-8",
    )
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    assert watchdog.check().records == 2
    assert watchdog.metrics()["prediction"].tolist() == [1, 2]


def test_log_never_raises_when_the_storage_path_is_unusable(tmp_path, caplog):
    """The storage 'directory' is actually a file: log() must not raise."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("in the way", encoding="utf-8")
    watchdog = model_watchdog.Watchdog("m", storage=blocked)
    with caplog.at_level(logging.WARNING, logger="model_watchdog._watchdog"):
        watchdog.log(prediction=1, features={"age": 3})
    assert "could not log a record" in caplog.text
    assert watchdog.metrics().empty
    assert len(watchdog) == 0


def test_log_never_raises_when_the_disk_write_fails(tmp_path, monkeypatch, caplog):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")

    def broken_open(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "open", broken_open)
    with caplog.at_level(logging.WARNING, logger="model_watchdog._watchdog"):
        watchdog.log(prediction=1)  # must not raise
    assert "No space left on device" in caplog.text


def test_log_never_raises_on_unloggable_features(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(features=42, prediction=1)  # not a mapping or sequence
    watchdog.log(features=pd.DataFrame([{"a": 1}, {"a": 2}]), prediction=1)  # two rows
    watchdog.log(prediction=1)  # this one is fine
    assert len(watchdog) == 1


def test_a_bad_reference_becomes_a_note_not_a_crash(tmp_path):
    watchdog = model_watchdog.Watchdog("m", reference=object(), storage=tmp_path / "m")
    watchdog.log(prediction=1)
    report = watchdog.check()
    assert report.ok
    assert any("reference could not be read" in note for note in report.notes)


def test_a_missing_reference_file_is_a_note_not_a_crash(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "m", reference=tmp_path / "nope.csv", storage=tmp_path / "m"
    )
    watchdog.log(prediction=1)
    assert watchdog.check().ok


# -- timestamps -----------------------------------------------------------


def test_every_timestamp_is_timezone_aware_utc(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(prediction=1)
    watchdog.log(prediction=2, ts="2026-09-22T10:00:00")           # naive string
    watchdog.log(prediction=3, ts=WINDOW_START)                    # aware datetime
    watchdog.log(prediction=4, ts="2026-09-22T10:00:00+05:30")     # other offset
    frame = watchdog.metrics()
    assert str(frame["ts"].dt.tz) == "UTC"
    assert frame["ts"].notna().all()
    for stamp in frame["ts"]:
        assert stamp.tzinfo is not None
        assert stamp.utcoffset().total_seconds() == 0
    raw = (tmp_path / "m" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    for line in raw:
        assert json.loads(line)["ts"].endswith("+00:00")
    report = watchdog.check()
    assert report.first_ts.utcoffset().total_seconds() == 0
    assert report.created_at.tzinfo is not None
    assert model_watchdog.utc_now().tzinfo is not None


def test_whole_second_and_microsecond_timestamps_mix_without_loss(tmp_path):
    """Backfilled records land on whole seconds, live ones carry microseconds.

    Both shapes must survive in one log: parsing the column with a single
    inferred format would turn one of the two into NaT and silently throw
    away half the traffic.
    """
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    for index in range(5):  # whole seconds, as a backfill writes them
        watchdog.log(prediction=index, ts=WINDOW_START + timedelta(hours=index))
    for index in range(5):  # live records, with microseconds
        watchdog.log(prediction=index)
    frame = watchdog.metrics()
    assert len(frame) == 10
    assert frame["ts"].notna().all(), "a timestamp shape was parsed away to NaT"

    report = watchdog.check()
    assert report.records == 10
    assert report.first_ts is not None and report.last_ts is not None
    # A cutoff before every record keeps all ten: a NaT would have dropped out.
    early = WINDOW_START - timedelta(days=3650)
    assert watchdog.report(since=early).records == 10


def test_a_volume_reference_of_mixed_timestamp_shapes_still_measures_a_rate(tmp_path):
    reference = pd.DataFrame(
        {
            "ts": ["2026-09-01T00:00:00+00:00", "2026-09-01T00:36:00.500000+00:00"]
            + ["2026-09-01T%02d:00:00+00:00" % hour for hour in range(1, 12)],
            "prediction": [0, 1] * 6 + [0],
        }
    )
    watchdog = model_watchdog.Watchdog("m", reference=reference, storage=tmp_path / "m")
    assert watchdog.reference.rate_per_hour is not None


# -- odd data -------------------------------------------------------------


def test_empty_dataframe_reference(tmp_path):
    watchdog = model_watchdog.Watchdog(
        "m", reference=pd.DataFrame(), storage=tmp_path / "m"
    )
    watchdog.log(prediction=1)
    report = watchdog.check()
    assert report.ok
    assert watchdog.reference.empty
    assert watchdog.reference.describe() == "no reference"


def test_single_row_reference_and_a_single_record(tmp_path):
    reference = pd.DataFrame([{"prediction": 1, "age": 30, "latency_ms": 10.0}])
    watchdog = model_watchdog.Watchdog("m", reference=reference, storage=tmp_path / "m")
    watchdog.log(features={"age": 30}, prediction=1, latency_ms=10.0)
    report = watchdog.check()
    assert report.records == 1
    assert report.ok, report.summary()


def test_an_all_nan_feature_column(tmp_path):
    reference = pd.DataFrame({"prediction": [0, 1] * 50, "blank": [None] * 100})
    watchdog = model_watchdog.Watchdog("m", reference=reference, storage=tmp_path / "m")
    for index in range(30):
        watchdog.log(features={"blank": None}, prediction=index % 2)
    report = watchdog.check()
    nulls = report.get("null_rate")
    assert nulls.failed
    assert nulls.value == pytest.approx(1.0)
    assert report.get("feature_drift").active is False


def test_mixed_dtypes_in_one_column(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(prediction="yes", features={"x": 1, "y": "a"})
    watchdog.log(prediction=3, features={"x": "b", "y": 2.5})
    watchdog.log(prediction=None, features={"x": None, "y": True})
    frame = watchdog.metrics()
    assert frame["prediction"].tolist()[:2] == ["yes", 3]
    assert len(frame) == 3
    report = watchdog.check()
    # Nothing crashed on the mixed column; the one missing feature value is
    # the only thing the monitors object to.
    assert [check.name for check in report.failed] == ["null_rate"]
    assert report.get("null_rate").value == pytest.approx(1.0 / 6.0)


def test_numpy_scalars_survive_the_round_trip(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(
        features={"a": np.int64(3), "b": np.float64(1.5), "c": np.bool_(True)},
        prediction=np.float32(0.25),
        actual=np.int32(1),
        latency_ms=np.float64(7.5),
    )
    row = watchdog.metrics().iloc[0]
    assert row["a"] == 3 and row["b"] == 1.5 and bool(row["c"]) is True
    assert row["prediction"] == pytest.approx(0.25)
    assert row["latency_ms"] == 7.5


def test_not_a_number_latency_and_infinities_do_not_crash(tmp_path):
    watchdog = model_watchdog.Watchdog("m", storage=tmp_path / "m")
    watchdog.log(prediction=float("inf"), latency_ms=float("nan"))
    watchdog.log(prediction=float("-inf"), latency_ms=float("inf"))
    watchdog.log(prediction=0.5, latency_ms=10.0)
    assert watchdog.check().ok
    assert len(watchdog.metrics()) == 3


def test_unicode_survives_names_features_and_summaries(tmp_path):
    watchdog = model_watchdog.Watchdog("modele-cafe-模型", storage=tmp_path / "unicode")
    for _ in range(3):
        watchdog.log(
            features={"città": "München", "名前": "テスト"},
            prediction="класс-A",
        )
    frame = watchdog.metrics()
    assert "città" in frame.columns and "名前" in frame.columns
    assert frame["名前"].iloc[0] == "テスト"
    report = watchdog.check()
    assert "modele-cafe-模型" in report.summary()
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "città" in payload
    raw = (tmp_path / "unicode" / "events.jsonl").read_text(encoding="utf-8")
    assert "München" in raw, "written as real UTF-8, not escapes"


def test_a_messy_watchdog_name_makes_a_safe_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    watchdog = model_watchdog.Watchdog("checkout/model:v2 *test*")
    watchdog.log(prediction=1)
    assert watchdog.storage.is_dir()
    assert len(watchdog) == 1


def test_duplicate_reference_columns_raise_a_clear_error(tmp_path):
    frame = pd.DataFrame([[1, 2, 3]], columns=["prediction", "age", "age"])
    with pytest.raises(ValueError, match="duplicate column names: age"):
        model_watchdog.ReferenceProfile.from_frame(frame)
    # The Watchdog turns that into a note rather than a crash.
    watchdog = model_watchdog.Watchdog("m", reference=frame, storage=tmp_path / "m")
    watchdog.log(prediction=1)
    assert any("duplicate column names" in note for note in watchdog.check().notes)


def test_a_csv_reference_is_read_from_disk(tmp_path):
    path = tmp_path / "reference.csv"
    pd.DataFrame(
        {"prediction": [0, 1] * 50, "actual": [0, 1] * 50, "age": list(range(100))}
    ).to_csv(path, index=False)
    watchdog = model_watchdog.Watchdog("m", reference=path, storage=tmp_path / "m")
    assert watchdog.reference.n == 100
    for index in range(30):
        watchdog.log(features={"age": index}, prediction=index % 2, actual=index % 2)
    assert watchdog.check().get("prediction_drift").active is True


def test_an_unsupported_reference_file_type_is_a_note(tmp_path):
    path = tmp_path / "reference.xlsx"
    path.write_bytes(b"not really a spreadsheet")
    watchdog = model_watchdog.Watchdog("m", reference=path, storage=tmp_path / "m")
    assert watchdog.reference.empty
    assert watchdog.reference.notes
