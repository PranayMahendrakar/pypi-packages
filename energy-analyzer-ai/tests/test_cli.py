"""The command line, including the encoding traps a Windows console sets."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from energy_analyzer_ai import __version__
from energy_analyzer_ai.cli import build_parser, main, parse_baseline, parse_tariff
from tests.conftest import meter


@pytest.fixture
def meter_csv(tmp_path):
    """A csv with a non-ASCII column name and a non-ASCII text column."""
    series = meter(days=21, seed=0).copy()
    series.iloc[300] = 9.0
    path = tmp_path / "zähler.csv"
    pd.DataFrame(
        {
            "Zeitstempel": series.index,
            "Verbrauch_kWh": series.to_numpy(),
            "standort": ["Gebäude Süd"] * series.size,
        }
    ).to_csv(path, index=False, encoding="utf-8")
    return path


def test_help_works():
    parser = build_parser()
    text = parser.format_help()
    assert "energy-analyzer-ai" in text
    assert "--tariff" in text and "--granularity" in text and "--json" in text


def test_version_is_the_package_version():
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ == "0.1.0"


def test_summary_is_printed(meter_csv, capsys):
    assert main([str(meter_csv), "--value", "Verbrauch_kWh", "--tariff", "0.28"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("energy-analyzer-ai:")
    assert "always-on" in out and "findings" in out


def test_json_output_is_valid(meter_csv, capsys):
    assert main([str(meter_csv), "--value", "Verbrauch_kWh", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["meter"] == "Verbrauch_kWh"
    assert payload["total_cost"] is None
    assert payload["n_anomalies"] == 1


def test_tariff_as_a_json_map(meter_csv, capsys):
    argv = [str(meter_csv), "--value", "Verbrauch_kWh", "--tariff", '{"0": 0.12, "7": 0.31}', "--json"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["tariff"]["kind"] == "time-of-use"
    assert payload["total_cost"] > 0


def test_tariff_from_a_json_file(meter_csv, tmp_path, capsys):
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({str(h): 0.2 for h in range(24)}), encoding="utf-8")
    assert main([str(meter_csv), "--value", "Verbrauch_kWh", "--tariff", str(rates), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tariff"]["kind"] == "time-of-use"


def test_output_file_is_written(meter_csv, tmp_path, capsys):
    out_path = tmp_path / "by_period.csv"
    argv = [str(meter_csv), "--value", "Verbrauch_kWh", "--output", str(out_path)]
    assert main(argv) == 0

    assert "wrote" in capsys.readouterr().out
    written = pd.read_csv(out_path)
    assert len(written) == 24 * 21
    assert "is_anomaly" in written.columns


def test_forecast_flag(meter_csv, capsys):
    argv = [str(meter_csv), "--value", "Verbrauch_kWh", "--forecast", "12"]
    assert main(argv) == 0
    assert "forecast  : next 12" in capsys.readouterr().out


def test_forecast_appears_in_json(meter_csv, capsys):
    argv = [str(meter_csv), "--value", "Verbrauch_kWh", "--forecast", "6", "--json"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["forecast"]) == 6
    assert payload["forecast"][0]["value"] > 0


def test_granularity_and_sensitivity_flags(meter_csv, capsys):
    argv = [str(meter_csv), "--value", "Verbrauch_kWh", "--granularity", "daily", "--sensitivity", "2.5"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "1D periods" in out
    assert "threshold 2.5 sigmas" in out


def test_cumulative_flag(tmp_path, capsys):
    used = meter(days=14, seed=2)
    register = pd.Series(np.cumsum(used.to_numpy()) + 9000.0, index=used.index, name="kwh")
    path = tmp_path / "register.csv"
    register.reset_index().rename(columns={"index": "time"}).to_csv(path, index=False)

    assert main([str(path), "--cumulative", "no", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cumulative_meter"] is False

    assert main([str(path), "--cumulative", "auto", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cumulative_meter"] is True


def test_a_missing_file_exits_with_one(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_bad_tariff_exits_with_one(meter_csv, capsys):
    assert main([str(meter_csv), "--value", "Verbrauch_kWh", "--tariff", "nonsense"]) == 1
    assert "could not read --tariff" in capsys.readouterr().err


def test_parse_tariff_reads_every_form(tmp_path):
    assert parse_tariff(None) is None
    assert parse_tariff("  ") is None
    assert parse_tariff("0.28") == 0.28
    assert parse_tariff('{"0": 0.1}') == {"0": 0.1}
    with pytest.raises(ValueError, match="does not exist"):
        parse_tariff(str(tmp_path / "gone.json"))


def test_parse_baseline_reads_every_form():
    assert parse_baseline(None) is None
    assert parse_baseline("") is None
    assert parse_baseline("1.5") == 1.5
    assert parse_baseline("normal.csv") == "normal.csv"


# ------------------------------------------------------------ encoding safety
def _run(args, csv_path):
    """Run the CLI in a child process with its output captured as bytes."""
    return subprocess.run(
        [sys.executable, "-m", "energy_analyzer_ai", str(csv_path)] + args,
        capture_output=True,
        timeout=120,
    )


def test_piped_output_with_non_ascii_data_does_not_crash(meter_csv):
    """The failure this guards against is UnicodeEncodeError on a piped stdout."""
    done = _run(["--value", "Verbrauch_kWh", "--tariff", "0.28"], meter_csv)

    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    assert done.stdout.decode("utf-8").startswith("energy-analyzer-ai:")


def test_piped_json_with_non_ascii_data_does_not_crash(meter_csv):
    done = _run(["--value", "Verbrauch_kWh", "--json"], meter_csv)

    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    payload = json.loads(done.stdout.decode("utf-8"))
    assert payload["meter"] == "Verbrauch_kWh"


def test_piped_output_with_a_non_ascii_meter_name(tmp_path):
    """The column name itself is echoed into the summary, so it must survive."""
    series = meter(days=10, seed=1)
    path = tmp_path / "meter.csv"
    pd.DataFrame({"время": series.index, "замер кВтч": series.to_numpy()}).to_csv(
        path, index=False, encoding="utf-8"
    )
    done = _run([], path)

    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert "замер кВтч".encode("utf-8") in done.stdout


def test_help_survives_a_pipe(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "energy_analyzer_ai", "--help"],
        capture_output=True,
        timeout=120,
    )
    assert done.returncode == 0
    assert b"--tariff" in done.stdout
