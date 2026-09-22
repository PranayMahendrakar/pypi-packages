"""The command line, including the encoding traps that a piped console sets."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sensor_anomaly.cli import build_parser, main


@pytest.fixture()
def plant_csv(tmp_path):
    """A small plant table whose channel names and labels are not ASCII."""
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "température": rng.normal(70, 1, n),
            "débit": rng.normal(12, 0.5, n),
            "振動": rng.normal(2.0, 0.1, n),
            "note": ["état normal"] * n,
        }
    )
    frame.loc[200:260, "débit"] = 0.0
    path = tmp_path / "usine.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def test_help_works():
    parser = build_parser()
    assert parser.prog == "sensor-anomaly"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_summary_run(plant_csv, capsys):
    assert main([str(plant_csv)]) == 0
    out = capsys.readouterr().out
    assert "sensor-anomaly" in out
    assert "débit" in out
    assert "stuck at zero" in out


def test_json_run(plant_csv, capsys):
    assert main([str(plant_csv), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_rows"] == 400
    assert "stuck-at-zero" in payload["channels"]["débit"]["faults"]
    assert payload["time_column"] == "timestamp"


def test_options_are_wired_through(plant_csv, capsys):
    assert (
        main(
            [
                str(plant_csv),
                "--time",
                "timestamp",
                "--channels",
                "température,débit",
                "--sensitivity",
                "4",
                "--random-state",
                "5",
                "--limit",
                "3",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["channels"]) == {"température", "débit"}
    assert payload["params"]["sensitivity"] == 4.0
    assert payload["params"]["random_state"] == 5


def test_output_files(plant_csv, tmp_path, capsys):
    csv_out = tmp_path / "channels.csv"
    assert main([str(plant_csv), "--output", str(csv_out)]) == 0
    assert "wrote" in capsys.readouterr().out
    written = pd.read_csv(csv_out)
    assert "débit" in set(written["channel"])

    json_out = tmp_path / "report.json"
    assert main([str(plant_csv), "--output", str(json_out)]) == 0
    capsys.readouterr()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 400
    # ensure_ascii=False, so the file keeps the real characters.
    assert "débit" in json_out.read_text(encoding="utf-8")


def test_bad_input_is_a_message_not_a_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 1
    assert "sensor-anomaly: error:" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def _run_piped(args, cwd):
    """Run the CLI in a child process with pipes and a hostile console encoding."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONLEGACYWINDOWSSTDIO"] = "1"  # force the old, narrow Windows console path
    return subprocess.run(
        [sys.executable, "-m", "sensor_anomaly", *args],
        capture_output=True,
        cwd=str(cwd),
        env=env,
    )


def test_piped_output_with_non_ascii_data_does_not_crash(plant_csv, tmp_path):
    """Earlier packages in this family died here with UnicodeEncodeError."""
    done = _run_piped([str(plant_csv)], tmp_path)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    assert "débit" in done.stdout.decode("utf-8", "replace")


def test_piped_json_with_non_ascii_data_does_not_crash(plant_csv, tmp_path):
    done = _run_piped([str(plant_csv), "--json"], tmp_path)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    payload = json.loads(done.stdout.decode("utf-8"))
    assert "débit" in payload["channels"]


def test_help_through_a_pipe(tmp_path):
    done = _run_piped(["--help"], tmp_path)
    assert done.returncode == 0
    assert b"sensor-anomaly" in done.stdout
