"""The command line: help, summaries, JSON, files written, and piped unicode."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from quality_predictor.cli import build_parser, main


@pytest.fixture()
def runs_csv(runs, tmp_path):
    path = tmp_path / "runs.csv"
    runs.to_csv(path, index=False, encoding="utf-8")
    return path


def test_help_works():
    parser = build_parser()
    text = parser.format_help()
    assert "quality-predictor" in text
    assert "--target" in text
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--help"])
    assert exit_info.value.code == 0


def test_no_arguments_prints_help(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_summary_run(runs_csv, capsys):
    assert main([str(runs_csv), "--target", "quality"]) == 0
    printed = capsys.readouterr().out
    assert "WHAT DRIVES QUALITY" in printed
    assert "held-out scores" in printed


def test_target_defaults_to_the_last_column(runs_csv, capsys):
    assert main([str(runs_csv)]) == 0
    printed = capsys.readouterr().out
    assert "no --target was given" in printed
    assert "quality" in printed


def test_json_output_and_output_file(runs_csv, tmp_path, capsys):
    report = tmp_path / "out" / "report.json"
    assert main([str(runs_csv), "-t", "quality", "--json", "-o", str(report)]) == 0
    printed = capsys.readouterr().out
    payload = json.loads(printed)
    assert payload["target"] == "quality"
    assert payload["feature_importance"]
    assert json.loads(report.read_text(encoding="utf-8"))["task"] == "classification"


def test_predict_file_explain_and_written_predictions(runs, runs_csv, tmp_path, capsys):
    fresh = tmp_path / "today.csv"
    runs.drop(columns=["quality"]).head(3).to_csv(fresh, index=False, encoding="utf-8")
    out = tmp_path / "scored.csv"
    code = main(
        [
            str(runs_csv),
            "--target", "quality",
            "--predict", str(fresh),
            "--explain", "0",
            "--predictions", str(out),
            "--features", "temperature,pressure,machine",
            "--top", "2",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "WHAT DROVE IT" in printed
    scored = pd.read_csv(out)
    assert "predicted_quality" in scored.columns
    assert len(scored) == 3


def test_missing_target_column_exits_one(runs_csv, capsys):
    assert main([str(runs_csv), "--target", "ghost"]) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_file_exits_one(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv"), "--target", "quality"]) == 1
    assert "error:" in capsys.readouterr().err


def test_explain_out_of_range(runs_csv, capsys):
    assert main([str(runs_csv), "-t", "quality", "--explain", "99999"]) == 1
    assert "out of range" in capsys.readouterr().err


def _clean_env():
    """A child environment that does NOT pre-arrange a UTF-8 console for us."""
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONLEGACYWINDOWSSTDIO", None)
    return env


def test_piped_unicode_output_does_not_crash(unicode_runs, tmp_path):
    path = tmp_path / "unicode.csv"
    unicode_runs.to_csv(path, index=False, encoding="utf-8")
    for extra in ([], ["--json"]):
        finished = subprocess.run(
            [sys.executable, "-m", "quality_predictor.cli", str(path), "-t", "qualité"] + extra,
            capture_output=True,
            env=_clean_env(),
        )
        assert finished.returncode == 0, finished.stderr.decode("utf-8", "replace")
        out = finished.stdout.decode("utf-8", "replace")
        assert "température" in out
        assert "UnicodeEncodeError" not in finished.stderr.decode("utf-8", "replace")


def test_console_script_help_is_installed():
    finished = subprocess.run(
        [sys.executable, "-m", "quality_predictor.cli", "--help"],
        capture_output=True,
        env=_clean_env(),
    )
    assert finished.returncode == 0
    assert b"--target" in finished.stdout


def test_version_flag():
    from quality_predictor import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ == "0.1.0"
