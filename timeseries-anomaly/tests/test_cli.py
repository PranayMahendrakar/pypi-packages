"""The command line, including the encoding traps a piped Windows console sets."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from timeseries_anomaly.cli import build_parser, main, write_output
from timeseries_anomaly import detect

UNICODE_VALUE_COLUMN = "température_°C"
UNICODE_TIME_COLUMN = "時刻"


def spiky(n=120, spike_at=47, spike=99.0):
    values = [10.0 + (i % 5) * 0.2 for i in range(n)]
    values[spike_at] = spike
    return values


@pytest.fixture()
def readings_csv(tmp_path):
    path = tmp_path / "readings.csv"
    pd.DataFrame(
        {
            "recorded_at": pd.date_range("2026-03-01", periods=120, freq="min"),
            "temperature": spiky(),
        }
    ).to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture()
def unicode_csv(tmp_path):
    path = tmp_path / "capteur.csv"
    pd.DataFrame(
        {
            UNICODE_TIME_COLUMN: pd.date_range("2026-03-01", periods=40, freq="min"),
            UNICODE_VALUE_COLUMN: spiky(40, spike_at=17, spike=88.0),
        }
    ).to_csv(path, index=False, encoding="utf-8")
    return path


def test_parser_builds_and_help_does_not_crash(capsys):
    parser = build_parser()
    assert parser.prog == "timeseries-anomaly"
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "timeseries-anomaly" in capsys.readouterr().out


def test_summary_run(readings_csv, capsys):
    assert main([str(readings_csv)]) == 0
    out = capsys.readouterr().out
    assert "timeseries-anomaly:" in out
    assert "1 anomaly" in out


def test_named_columns_and_options(readings_csv, capsys):
    code = main(
        [
            str(readings_csv),
            "--value",
            "temperature",
            "--time",
            "recorded_at",
            "--method",
            "zscore",
            "--sensitivity",
            "4",
        ]
    )
    assert code == 0
    assert "zscore" in capsys.readouterr().out


def test_json_output_is_parseable(readings_csv, capsys):
    assert main([str(readings_csv), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_anomalies"] == 1
    assert payload["anomalies"][0]["index"] == 47
    assert "method_used" in payload


def test_output_file_is_written(readings_csv, tmp_path, capsys):
    target = tmp_path / "scored.csv"
    assert main([str(readings_csv), "--output", str(target)]) == 0
    assert "wrote" in capsys.readouterr().out
    written = pd.read_csv(target)
    assert list(written.columns) == ["time", "value", "expected", "score", "is_anomaly"]
    assert len(written) == 120


def test_write_output_helper_handles_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    target = tmp_path / "scored.parquet"
    rows = write_output(str(target), detect(spiky()))
    assert rows == 120
    assert len(pd.read_parquet(target)) == 120


def test_bad_input_exits_one_with_a_message(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 1
    assert "error" in capsys.readouterr().err


def test_seasonality_option(readings_csv, capsys):
    assert main([str(readings_csv), "--method", "seasonal", "--seasonality", "5"]) == 0
    assert "season of 5 points" in capsys.readouterr().out


def _run(args, env_extra=None):
    """Run the CLI in a child process with its output piped, never a console."""
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "timeseries_anomaly", *args],
        capture_output=True,
        env=env,
        timeout=120,
    )


def test_help_through_a_pipe():
    done = _run(["--help"])
    assert done.returncode == 0
    assert b"timeseries-anomaly" in done.stdout


def test_version_through_a_pipe():
    done = _run(["--version"])
    assert done.returncode == 0
    assert b"0.1.0" in done.stdout


@pytest.mark.parametrize("extra", [[], ["--json"]])
def test_non_ascii_output_survives_a_narrow_piped_console(unicode_csv, extra):
    """A cp1252 (or ascii) stdout must not turn unicode data into a crash."""
    done = _run([str(unicode_csv), *extra], env_extra={"PYTHONIOENCODING": "cp1252"})
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    text = done.stdout.decode("utf-8", "replace")
    assert UNICODE_VALUE_COLUMN in text


def test_non_ascii_output_survives_an_ascii_piped_console(unicode_csv):
    done = _run([str(unicode_csv)], env_extra={"PYTHONIOENCODING": "ascii"})
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr


def test_json_written_through_a_pipe_is_still_valid_json(unicode_csv):
    done = _run([str(unicode_csv), "--json"], env_extra={"PYTHONIOENCODING": "cp1252"})
    assert done.returncode == 0
    payload = json.loads(done.stdout.decode("utf-8"))
    assert payload["label"] == UNICODE_VALUE_COLUMN
    assert payload["n_anomalies"] >= 1
