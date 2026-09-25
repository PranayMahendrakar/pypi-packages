"""The command line: help, summary, JSON, files, errors, and non-ASCII text through a pipe."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from conftest import line
from production_anomaly.cli import build_parser, main


@pytest.fixture
def csv_path(tmp_path):
    df = line(days=2, every=5, rate=300, hours=(6, 22))
    df.loc[130:141, "units"] = 0.0
    path = tmp_path / "line.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def unicode_csv(tmp_path):
    df = line(days=1, every=5, rate=300, hours=(6, 22))
    df.loc[100:111, "units"] = 0.0
    df = df.rename(columns={"time": "Zeit", "units": "Stückzahl-产量"})
    df["Linie"] = "Presse ü 冲压"
    path = tmp_path / "linie.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--shift-hours" in capsys.readouterr().out
    assert build_parser().prog == "production-anomaly"


def test_prints_the_summary(csv_path, capsys):
    assert main([str(csv_path), "--shift-hours", "6-22"]) == 0
    out = capsys.readouterr().out
    assert "availability" in out and "60 min  downtime" in out
    assert "outside the shifts not counted" in out


def test_json_output(csv_path, capsys):
    assert main([str(csv_path), "--json", "--target-rate", "3600", "--column", "units"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["reference"] == "target" and data["target_rate"] == 3600
    assert data["quality"] is None


def test_writes_json_and_by_shift_files(csv_path, tmp_path, capsys):
    out_json = tmp_path / "out" / "report.json"
    out_csv = tmp_path / "out" / "shifts.csv"
    code = main([str(csv_path), "--shift-hours", "06:00-14:00,14:00-22:00",
                 "--output", str(out_json), "--by-shift", str(out_csv)])
    assert code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data["by_shift"]) == 4
    assert len(pd.read_csv(out_csv)) == 4


def test_errors_exit_with_status_2(tmp_path, csv_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 2
    assert "error: " in capsys.readouterr().err
    assert main([str(csv_path), "--shift-hours", "6-14,12-20"]) == 2
    assert "overlap" in capsys.readouterr().err
    assert main([str(csv_path), "--time", "nope"]) == 2
    with pytest.raises(SystemExit):
        main([str(csv_path), "--target-rate", "-1"])


def test_non_ascii_output_file(unicode_csv, tmp_path, capsys):
    target = tmp_path / "bericht.json"
    assert main([str(unicode_csv), "--output", str(target)]) == 0
    text = target.read_text(encoding="utf-8")
    assert "Stückzahl-产量" in text  # written as UTF-8, not \u escapes


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
def test_non_ascii_through_a_pipe_does_not_crash(unicode_csv, encoding):
    """A narrow console encoding must not raise UnicodeEncodeError when output is piped."""
    env = dict(os.environ, PYTHONIOENCODING=encoding)
    env.pop("PYTHONUTF8", None)
    for extra in ([], ["--json"]):
        done = subprocess.run(
            [sys.executable, "-m", "production_anomaly.cli", str(unicode_csv), *extra],
            capture_output=True,
            env=env,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        out = done.stdout.decode("utf-8")
        assert "Stückzahl-产量" in out
