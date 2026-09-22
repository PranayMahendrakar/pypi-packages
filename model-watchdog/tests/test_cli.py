"""The command line interface, including the Windows encoding trap.

Earlier packages in this family shipped a CLI that died with
UnicodeEncodeError the moment its output was piped instead of shown in a
terminal. The subprocess tests at the bottom of this file exist to make sure
this one does not.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from datetime import timedelta

import pytest

import model_watchdog
from conftest import WINDOW_START, reference_frame
from model_watchdog import cli


@pytest.fixture
def logged(tmp_path):
    """A storage directory with healthy traffic and a reference CSV beside it."""
    storage = tmp_path / "logs"
    watchdog = model_watchdog.Watchdog("checkout", storage=storage)
    for index in range(40):
        watchdog.log(
            features={"città": "München", "plan": "pro"},
            prediction=index % 2,
            actual=index % 2,
            latency_ms=10.0 + (index % 5),
            ts=WINDOW_START + timedelta(seconds=36 * index),
        )
    reference_path = tmp_path / "reference.csv"
    reference_frame().to_csv(reference_path, index=False, encoding="utf-8")
    return storage, reference_path


def run(argv, capsys):
    """Run the CLI in-process and return ``(status, stdout)``."""
    status = cli.main(argv)
    return status, capsys.readouterr().out


# -- the basics -----------------------------------------------------------


def test_help_works(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "model-watchdog" in out
    assert "--json" in out and "--window" in out


def test_version_works(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert model_watchdog.__version__ in capsys.readouterr().out


def test_summary_is_printed_for_a_storage_directory(logged, capsys):
    storage, _ = logged
    status, out = run([str(storage)], capsys)
    assert status == 0
    assert "model-watchdog:" in out
    assert "records: 40" in out
    for name in model_watchdog.MONITOR_NAMES:
        assert name in out


def test_json_output_is_valid_json(logged, capsys):
    storage, _ = logged
    status, out = run([str(storage), "--json"], capsys)
    assert status == 0
    payload = json.loads(out)
    assert payload["records"] == 40
    assert len(payload["checks"]) == len(model_watchdog.MONITOR_NAMES)
    assert payload["ok"] is True


def test_reference_makes_the_monitors_active(logged, capsys):
    storage, reference_path = logged
    status, out = run([str(storage), "--reference", str(reference_path), "--json"], capsys)
    assert status == 0
    payload = json.loads(out)
    active = {check["name"] for check in payload["checks"] if check["active"]}
    assert "prediction_drift" in active
    assert "accuracy_drop" in active


def test_metrics_prints_csv(logged, capsys):
    storage, _ = logged
    status, out = run([str(storage), "--metrics"], capsys)
    assert status == 0
    header = out.splitlines()[0]
    assert header.startswith("ts,prediction,actual,latency_ms")
    assert len(out.strip().splitlines()) == 41  # header plus 40 records


def test_metrics_csv_has_no_doubled_line_endings(unicode_storage, tmp_path):
    """Redirected to a file, --metrics must be ordinary CSV.

    pandas writes "\\r\\n" itself; if the text layer translates the "\\n" a
    second time the file ends up with "\\r\\r\\n" and a blank row between
    every record in anything stricter than csv.reader.
    """
    destination = tmp_path / "metrics.csv"
    with open(destination, "wb") as handle:
        completed = subprocess.run(
            [sys.executable, "-m", "model_watchdog", str(unicode_storage), "--metrics"],
            stdout=handle,
            stderr=subprocess.PIPE,
            env=_child_environment(),
            timeout=120,
        )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    raw = destination.read_bytes()
    assert b"\r\r\n" not in raw, "line endings were translated twice"

    with open(destination, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    assert len(text.splitlines()) == 31, "header plus 30 records, no blank rows"
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) == 31
    assert len({len(row) for row in rows}) == 1, "ragged CSV"


def test_window_limits_the_records(logged, capsys):
    storage, _ = logged
    status, out = run([str(storage), "--window", "10", "--json"], capsys)
    assert status == 0
    assert json.loads(out)["records"] == 10


def test_since_filters_the_records(logged, capsys):
    storage, _ = logged
    cutoff = (WINDOW_START + timedelta(seconds=36 * 20)).isoformat()
    status, out = run([str(storage), "--since", cutoff, "--json"], capsys)
    assert status == 0
    payload = json.loads(out)
    assert payload["records"] == 20
    assert payload["window"] is None


def test_output_writes_the_report_to_a_file(logged, tmp_path, capsys):
    storage, _ = logged
    destination = tmp_path / "report.json"
    status, _ = run([str(storage), "--output", str(destination)], capsys)
    assert status == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["records"] == 40


def test_fail_on_alert_sets_the_exit_status(tmp_path, capsys):
    storage = tmp_path / "stuck"
    watchdog = model_watchdog.Watchdog("stuck", storage=storage)
    for _ in range(80):
        watchdog.log(prediction=0.5)  # the model is stuck on one value

    status, out = run([str(storage)], capsys)
    assert status == 0, "without the flag a failing check is still exit 0"
    assert "constant_output" in out

    status, _ = run([str(storage), "--fail-on-alert"], capsys)
    assert status == 1


def test_a_name_resolves_under_the_default_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model_watchdog.Watchdog("checkout").log(prediction=1)
    status, out = run(["checkout", "--json"], capsys)
    assert status == 0
    assert json.loads(out)["records"] == 1


def test_an_unreadable_reference_becomes_a_visible_note(logged, tmp_path, capsys):
    """A bad reference must not take the run down, but must not hide either."""
    storage, _ = logged
    bad = tmp_path / "reference.xlsx"
    bad.write_bytes(b"not a spreadsheet")
    status, out = run([str(storage), "--reference", str(bad)], capsys)
    assert status == 0
    assert "reference could not be read" in out
    assert "use .csv or .parquet" in out


def test_an_unreadable_since_is_rejected_not_ignored(logged, capsys):
    """A typo must not silently widen the window to the whole log."""
    storage, _ = logged
    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(storage), "--since", "not-a-date"])
    assert exit_info.value.code == 2  # argparse's usage-error status
    assert "could not read --since" in capsys.readouterr().err


def test_a_date_only_since_is_accepted(logged, capsys):
    storage, _ = logged
    status, out = run([str(storage), "--since", "2026-09-22", "--json"], capsys)
    assert status == 0
    assert json.loads(out)["records"] == 40


def test_an_empty_storage_directory_still_reports(tmp_path, capsys):
    status, out = run([str(tmp_path / "nothing-here")], capsys)
    assert status == 0
    assert "no records logged yet" in out


# -- encoding safety ------------------------------------------------------
#
# These run the real console script in a child process with stdout attached to
# a pipe (not a terminal) and the legacy Windows code page as the ambient
# encoding. That is exactly the shape that used to raise UnicodeEncodeError.


def _child_environment():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    # The trap: claim a legacy single-byte code page, so any stream the CLI
    # does not reconfigure will refuse non-Latin text.
    environment["PYTHONIOENCODING"] = "cp1252"
    environment.pop("PYTHONUTF8", None)
    environment["PYTHONLEGACYWINDOWSSTDIO"] = "1"
    return environment


def _run_cli(arguments, cwd=None):
    """Run ``python -m model_watchdog`` with stdout and stderr piped."""
    completed = subprocess.run(
        [sys.executable, "-m", "model_watchdog"] + list(arguments),
        capture_output=True,  # a pipe, never a console
        env=_child_environment(),
        cwd=str(cwd) if cwd else None,
        timeout=120,
    )
    return completed


@pytest.fixture
def unicode_storage(tmp_path):
    """Traffic whose names, features and predictions are all non-ASCII."""
    storage = tmp_path / "unicode-logs"
    watchdog = model_watchdog.Watchdog("модель", storage=storage)
    for index in range(30):
        watchdog.log(
            features={"città": "München", "名前": "テスト", "prix": "12,50 €"},
            prediction="класс-%d" % (index % 3),
            actual="класс-%d" % (index % 3),
            latency_ms=10.0,
            note="naïve café — résumé",
            ts=WINDOW_START + timedelta(seconds=36 * index),
        )
    return storage


def test_help_survives_a_pipe(tmp_path):
    completed = _run_cli(["--help"], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert b"model-watchdog" in completed.stdout
    assert b"UnicodeEncodeError" not in completed.stderr


def test_summary_with_non_ascii_data_survives_a_pipe(unicode_storage):
    completed = _run_cli([str(unicode_storage)])
    stderr = completed.stderr.decode("utf-8", "replace")
    assert "UnicodeEncodeError" not in stderr, stderr
    assert completed.returncode == 0, stderr
    out = completed.stdout.decode("utf-8", "replace")
    assert "città" in out
    assert "null_rate" in out


def test_json_with_non_ascii_data_survives_a_pipe(unicode_storage):
    completed = _run_cli([str(unicode_storage), "--json"])
    stderr = completed.stderr.decode("utf-8", "replace")
    assert "UnicodeEncodeError" not in stderr, stderr
    assert completed.returncode == 0, stderr
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert payload["records"] == 30
    assert "città" in json.dumps(payload, ensure_ascii=False)


def test_metrics_with_non_ascii_data_survives_a_pipe(unicode_storage):
    completed = _run_cli([str(unicode_storage), "--metrics"])
    stderr = completed.stderr.decode("utf-8", "replace")
    assert "UnicodeEncodeError" not in stderr, stderr
    assert completed.returncode == 0, stderr
    out = completed.stdout.decode("utf-8", "replace")
    assert "名前" in out.splitlines()[0]
    assert "München" in out


def test_the_summary_is_plain_ascii_punctuation(unicode_storage):
    """Only the data may be exotic; the report's own furniture stays ASCII."""
    report = model_watchdog.Watchdog("модель", storage=unicode_storage).check()
    scaffolding = report.summary()
    for value in ("модель", "città", "München", "名前", "テスト", "класс", "€", "—", "ï", "é"):
        scaffolding = scaffolding.replace(value, "")
    assert scaffolding.isascii(), "summary furniture must be plain ASCII"
    for forbidden in ("->", "→", "•", "┌", "└"):
        assert forbidden not in report.summary()
