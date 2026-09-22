"""The predictive-maintenance command line."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import predictive_maintenance as pm
from predictive_maintenance.cli import main
from conftest import make_frame


@pytest.fixture
def csv_path(tmp_path):
    df = make_frame(n_rows=200, ramp=90)
    path = tmp_path / "sensors.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def unicode_csv_path(tmp_path):
    df = make_frame(n_rows=160, ramp=80).rename(
        columns={"vibration": "vibración_mm/s", "temp_c": "温度_℃"}
    )
    path = tmp_path / "capteurs.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    assert "predictive-maintenance" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert pm.__version__ in capsys.readouterr().out


def test_no_arguments_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_bare_file_runs_health(csv_path, capsys):
    assert main([str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert "health score" in out
    assert "share of degradation" in out


def test_health_subcommand_json(csv_path, capsys):
    assert main(["health", str(csv_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"score", "trend", "contributors", "channels"}


def test_rul_subcommand(csv_path, capsys):
    assert main(["rul", str(csv_path)]) == 0
    assert "remaining useful life" in capsys.readouterr().out


def test_rul_json_and_threshold(csv_path, capsys):
    assert main(["rul", str(csv_path), "--json", "--threshold", "45"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["threshold"] == 45.0
    assert "confidence" in payload


def test_channels_baseline_and_window(csv_path, capsys):
    code = main(
        [
            "health",
            str(csv_path),
            "--channels",
            "vibration,temp_c",
            "--baseline",
            "0.25",
            "--window",
            "15",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["channels"] == ["vibration", "temp_c"]
    assert payload["window"] == 15


def test_time_flag(csv_path, capsys):
    assert main(["health", str(csv_path), "--time", "time", "--json"]) == 0
    json.loads(capsys.readouterr().out)


def test_output_writes_utf8_json(unicode_csv_path, tmp_path, capsys):
    target = tmp_path / "nested" / "result.json"
    assert main(["health", str(unicode_csv_path), "--output", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "温度_℃" in payload["contributors"]
    assert "result written to" in capsys.readouterr().err


def test_unicode_summary_reaches_stdout(unicode_csv_path, capsys):
    assert main(["health", str(unicode_csv_path)]) == 0
    out = capsys.readouterr().out
    assert "vibraci" in out


def test_bad_baseline_is_a_friendly_error(csv_path, capsys):
    assert main(["health", str(csv_path), "--baseline", "soon"]) == 1
    assert "--baseline takes a row count" in capsys.readouterr().err


def test_empty_channels_flag_is_a_friendly_error(csv_path, capsys):
    assert main(["health", str(csv_path), "--channels", " , "]) == 1
    assert "named no columns" in capsys.readouterr().err


def test_missing_file_is_a_friendly_error(tmp_path, capsys):
    assert main(["health", str(tmp_path / "gone.csv")]) == 1
    assert "no such file" in capsys.readouterr().err


def test_unknown_channel_is_a_friendly_error(csv_path, capsys):
    assert main(["health", str(csv_path), "--channels", "torque"]) == 1
    assert "channels not found" in capsys.readouterr().err


def _run_piped(args, cwd):
    """Run the CLI as a subprocess with a non-UTF-8 stdout pipe."""
    env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        "PYTHONIOENCODING": "cp1252",
    }
    return subprocess.run(
        [sys.executable, "-m", "predictive_maintenance.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        env=env,
    )


def test_piped_output_with_non_ascii_data_does_not_crash(unicode_csv_path, tmp_path):
    """A legacy code page on the pipe must not raise UnicodeEncodeError."""
    for args in ([str(unicode_csv_path)], ["rul", str(unicode_csv_path)],
                 ["health", str(unicode_csv_path), "--json"]):
        done = _run_piped(args, tmp_path)
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        assert b"UnicodeEncodeError" not in done.stderr
        assert done.stdout
