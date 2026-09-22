"""The argparse command line interface."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ml_feature_check import __version__
from ml_feature_check.cli import build_parser, main


@pytest.fixture
def csv_path(tmp_path, quickstart_frame):
    path = tmp_path / "train.csv"
    quickstart_frame.to_csv(path, index=False, encoding="utf-8")
    return str(path)


def test_help_mentions_the_options():
    text = build_parser().format_help()
    for flag in ("--target", "--json", "--markdown", "--output", "--apply", "--sample"):
        assert flag in text
    assert "ml-feature-check" in text


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_summary_run(csv_path, capsys):
    assert main([csv_path, "--target", "churned"]) == 0
    out = capsys.readouterr().out
    assert "ml-feature-check:" in out
    assert "Drop recommended" in out
    assert "row_id" in out


def test_json_run(csv_path, capsys):
    assert main([csv_path, "--target", "churned", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "churned"
    assert "row_id" in payload["drop_recommended"]


def test_markdown_run(csv_path, capsys):
    assert main([csv_path, "--markdown"]) == 0
    assert "| Column | Verdict | Findings |" in capsys.readouterr().out


def test_output_and_apply(tmp_path, csv_path, capsys):
    report_path = tmp_path / "report.json"
    clean_path = tmp_path / "clean.csv"
    code = main(
        [csv_path, "--target", "churned", "--output", str(report_path), "--apply", str(clean_path)]
    )
    capsys.readouterr()
    assert code == 0

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["n_features"] == 6

    cleaned = pd.read_csv(clean_path)
    assert list(cleaned.columns) == ["age", "signed_up", "churned"]


@pytest.mark.parametrize("suffix, needle", [(".md", "# ml-feature-check report"), (".txt", "ml-feature-check:")])
def test_output_format_follows_the_suffix(tmp_path, csv_path, capsys, suffix, needle):
    path = tmp_path / f"report{suffix}"
    assert main([csv_path, "--output", str(path)]) == 0
    capsys.readouterr()
    assert needle in path.read_text(encoding="utf-8")


def test_threshold_flags(csv_path, capsys):
    assert main([csv_path, "--corr-threshold", "0.5", "--missing-threshold", "0.2", "--sample", "50"]) == 0
    out = capsys.readouterr().out
    assert "checked" in out


def test_missing_file_exits_one(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 1
    assert "ml-feature-check: error:" in capsys.readouterr().err


def test_bad_target_exits_one(csv_path, capsys):
    assert main([csv_path, "--target", "not_a_column"]) == 1
    assert "is not a column" in capsys.readouterr().err


def test_bad_threshold_exits_one(csv_path, capsys):
    assert main([csv_path, "--corr-threshold", "3"]) == 1
    assert "corr_threshold" in capsys.readouterr().err


def test_unicode_survives_a_captured_pipe(tmp_path):
    df = pd.DataFrame({"説明": ["日本語", "テキスト"] * 15, "v": np.linspace(0, 1, 30)})
    path = tmp_path / "unicode.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "ml_feature_check", str(path)],
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "説明" in proc.stdout.decode("utf-8", "replace")


def test_module_entry_point(csv_path):
    proc = subprocess.run(
        [sys.executable, "-m", "ml_feature_check", csv_path, "--target", "churned", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["target"] == "churned"


def test_version_string():
    assert __version__ == "0.1.0"


# --------------------------------------------------------------------- QA round 2


def test_apply_to_parquet_without_pyarrow_is_one_line(tmp_path, csv_path, capsys):
    """The write path must translate pandas' engine dump like the read path does.

    README.md promises "the command exits 1 with a one-line message on a bad
    file"; --apply out.parquet used to paste four-plus lines of raw pandas text
    after the error prefix.
    """
    try:
        import pyarrow  # noqa: F401

        pytest.skip("pyarrow is installed, so the engine error cannot be raised")
    except ImportError:
        pass

    assert main([csv_path, "--apply", str(tmp_path / "out.parquet")]) == 1
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0] == (
        "ml-feature-check: error: writing parquet needs pyarrow: "
        "pip install 'ml-feature-check[parquet]'"
    )


def test_apply_to_an_unsupported_extension_is_also_one_line(tmp_path, csv_path, capsys):
    assert main([csv_path, "--apply", str(tmp_path / "out.xlsx")]) == 1
    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "unsupported output type '.xlsx'" in lines[0]
