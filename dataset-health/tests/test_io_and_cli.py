"""Reading files from disk, and the ``dataset-health`` command."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_health import diagnose
from dataset_health._io import load_table
from dataset_health.cli import build_parser, main


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": range(1, 21),
            "age": [23, 34, 45, None, 31, 29, 38, 52, 41, 27] * 2,
            "ciudad": ["München", "北京"] * 10,
            "churn": [0, 1] * 10,
        }
    )


@pytest.fixture
def csv_path(tmp_path, frame):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


# ----- loading -------------------------------------------------------------------------


def test_load_table_passes_a_dataframe_through(frame):
    assert load_table(frame) is frame


def test_load_table_wraps_a_series():
    out = load_table(pd.Series([1, 2, 3], name="s"))
    assert list(out.columns) == ["s"]


def test_diagnose_reads_a_csv(csv_path, frame):
    from_path = diagnose(csv_path, target="churn")
    from_frame = diagnose(frame, target="churn")
    assert from_path.n_rows == from_frame.n_rows == 20
    assert from_path.score == from_frame.score


def test_diagnose_reads_a_tsv(tmp_path, frame):
    path = tmp_path / "data.tsv"
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8")
    assert diagnose(path).n_columns == 4


def test_diagnose_reads_a_parquet_file(tmp_path, frame):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, index=False)
    assert diagnose(str(path), target="churn").n_rows == 20


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        diagnose(tmp_path / "absent.csv")


def test_an_unsupported_extension_says_so(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("not really a spreadsheet", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        diagnose(path)


# ----- CLI -----------------------------------------------------------------------------


def test_parser_help_mentions_the_options():
    text = build_parser().format_help()
    for flag in ("--target", "--json", "--markdown", "--output", "--sample", "--fail-below"):
        assert flag in text


def test_cli_prints_the_summary(csv_path, capsys):
    assert main([str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("dataset-health: score")
    assert "rows: 20 | columns: 4" in out


def test_cli_with_a_target(csv_path, capsys):
    assert main([str(csv_path), "--target", "churn"]) == 0
    assert "target: churn (classification)" in capsys.readouterr().out


def test_cli_json_output(csv_path, capsys):
    assert main([str(csv_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_rows"] == 20
    assert "issues" in payload


def test_cli_markdown_output(csv_path, capsys):
    assert main([str(csv_path), "--markdown"]) == 0
    assert capsys.readouterr().out.startswith("# dataset-health report")


def test_cli_writes_markdown_and_json_files(csv_path, tmp_path, capsys):
    md = tmp_path / "report.md"
    js = tmp_path / "report.json"
    assert main([str(csv_path), "--output", str(md)]) == 0
    assert main([str(csv_path), "--output", str(js)]) == 0
    capsys.readouterr()
    assert md.read_text(encoding="utf-8").startswith("# dataset-health report")
    assert json.loads(js.read_text(encoding="utf-8"))["n_rows"] == 20


def test_cli_sample_zero_reads_every_row(tmp_path, capsys):
    path = tmp_path / "big.csv"
    pd.DataFrame({"a": range(400)}).to_csv(path, index=False)
    assert main([str(path), "--sample", "0"]) == 0
    assert "random sample" not in capsys.readouterr().out


def test_cli_sampling_is_noted(tmp_path, capsys):
    path = tmp_path / "big.csv"
    pd.DataFrame({"a": range(400), "b": np.arange(400) % 5}).to_csv(path, index=False)
    assert main([str(path), "--sample", "100", "--random-state", "1"]) == 0
    assert "analyzed: 100 rows (random sample)" in capsys.readouterr().out


def test_cli_fail_below(csv_path, capsys):
    assert main([str(csv_path), "--fail-below", "100"]) == 1
    assert main([str(csv_path), "--fail-below", "1"]) == 0
    capsys.readouterr()


def test_cli_fail_on_critical(tmp_path, capsys):
    path = tmp_path / "leak.csv"
    y = np.linspace(0, 1, 100)
    pd.DataFrame({"copy": y * 2, "y": y}).to_csv(path, index=False)
    assert main([str(path), "--target", "y", "--fail-on-critical"]) == 1
    assert main([str(path), "--target", "y"]) == 0
    capsys.readouterr()


def test_cli_rejects_an_unknown_target(csv_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([str(csv_path), "--target", "nope"])
    assert excinfo.value.code == 2
    assert "nope" in capsys.readouterr().err


def test_cli_rejects_a_missing_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "absent.csv")])
    assert excinfo.value.code == 2
    assert "no such file" in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_cli_handles_unicode_without_raising(tmp_path, capsys):
    path = tmp_path / "unicode.csv"
    pd.DataFrame({"ciudad": ["München", "北京"] * 10, "n": range(20)}).to_csv(
        path, index=False, encoding="utf-8"
    )
    assert main([str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "ciudad" in payload["columns"]
