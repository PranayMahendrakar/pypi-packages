"""The command line: arguments, JSON output, exit status and encoding safety."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from machine_health.cli import build_parser, main


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=500, freq="min"),
            "temp": np.r_[rng.normal(60, 1, 400), rng.normal(68, 3, 100)],
            "vibration": rng.normal(0.2, 0.02, 500),
        }
    )
    path = tmp_path / "telemetry.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def unicode_csv(tmp_path):
    path = tmp_path / "unicode.csv"
    path.write_text(
        "opérateur,ciudad,温度,flag\n"
        "Zoë,Kraków,60,yes\n"
        "李雷,北京,61,no\n"
        "José,Ñuñoa,,yes\n"
        "Zoë,Kraków,59,yes\n"
        "Ann,Paris,95,no\n",
        encoding="utf-8",
    )
    return path


def test_help_works(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_version_works(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "0.1.0" in capsys.readouterr().out


def test_plain_run_prints_the_summary(csv_path, capsys):
    assert main([str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert "machine health:" in out
    assert "components (score, weight, points lost):" in out


def test_json_output_is_valid_json(csv_path, capsys):
    assert main([str(csv_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0 <= payload["value"] <= 100
    assert payload["grade"] in {"A", "B", "C", "D", "F"}
    assert set(payload["components"]) == {"stability", "compliance", "anomaly", "availability"}


def test_rules_time_and_channels(csv_path, capsys):
    code = main(
        [
            str(csv_path),
            "--time",
            "ts",
            "--channels",
            "temp",
            "--rule",
            "temp:max=65,warn_max=62",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["channels"] == ["temp"]
    assert any(v["limit"] == "max" for v in payload["violations"])


def test_weight_and_baseline_options(csv_path, capsys):
    assert main([str(csv_path), "--baseline", "0.4", "--weight", "compliance=0.5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_baseline_rows"] == 200


def test_baseline_can_be_a_row_count(csv_path, capsys):
    assert main([str(csv_path), "--baseline", "150", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_baseline_rows"] == 150


def test_baseline_can_be_a_file(tmp_path, csv_path, capsys):
    healthy = pd.read_csv(csv_path).iloc[:300]
    path = tmp_path / "healthy.csv"
    healthy.to_csv(path, index=False)
    assert main([str(csv_path), "--baseline", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_baseline_rows"] == 300


def test_output_file_is_written(csv_path, tmp_path, capsys):
    out = tmp_path / "nested" / "health.json"
    assert main([str(csv_path), "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "value" in payload


def test_rules_from_a_json_file(csv_path, tmp_path, capsys):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"temp": {"max": 65}}), encoding="utf-8")
    assert main([str(csv_path), "--rules", str(rules), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["violations"]


def test_fail_under_sets_the_exit_status(csv_path, capsys):
    assert main([str(csv_path), "--fail-under", "0"]) == 0
    assert main([str(csv_path), "--fail-under", "100"]) == 1


def test_bad_file_exits_two(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 2
    assert "error:" in capsys.readouterr().err


def test_bad_rule_channel_exits_two(csv_path, capsys):
    assert main([str(csv_path), "--rule", "pressure:max=1"]) == 2
    assert "available channels" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["nocolon", "temp:", "temp:max", "temp:maxx=1", "temp:max=hot", ":max=1"])
def test_malformed_rule_text_is_rejected(bad, csv_path):
    with pytest.raises(SystemExit) as excinfo:
        main([str(csv_path), "--rule", bad])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("bad", ["stability", "nonsense=1", "stability=lots"])
def test_malformed_weight_text_is_rejected(bad, csv_path):
    with pytest.raises(SystemExit):
        main([str(csv_path), "--weight", bad])


def test_unicode_data_prints_without_crashing(unicode_csv, capsys):
    assert main([str(unicode_csv), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["channels"] == ["温度"]


def test_unicode_survives_a_pipe(unicode_csv):
    """The failure this family hit before: non-ASCII output through a pipe."""
    result = subprocess.run(
        [sys.executable, "-m", "machine_health.cli", str(unicode_csv)],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in result.stderr
    assert "温度".encode("utf-8") in result.stdout


def test_unicode_json_survives_a_pipe(unicode_csv):
    result = subprocess.run(
        [sys.executable, "-m", "machine_health.cli", str(unicode_csv), "--json"],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    payload = json.loads(result.stdout.decode("utf-8"))
    assert payload["channels"] == ["温度"]


def test_parser_is_documented():
    parser = build_parser()
    text = parser.format_help()
    for flag in ("--time", "--rule", "--rules", "--weight", "--baseline", "--json", "--output", "--fail-under"):
        assert flag in text
