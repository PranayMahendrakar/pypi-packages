import json
import subprocess
import sys

import pandas as pd
import pytest

from synthetic_tabular import __version__
from synthetic_tabular.cli import build_parser, main


@pytest.fixture
def csv_path(tmp_path, rich_df):
    path = tmp_path / "data.csv"
    rich_df.head(300).to_csv(path, index=False)
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "synthetic-tabular" in out and "--output" in out and "--evaluate" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_module_entry_point_help():
    result = subprocess.run(
        [sys.executable, "-m", "synthetic_tabular.cli", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0 and "usage: synthetic-tabular" in result.stdout


def test_default_run_prints_fidelity_report(csv_path, capsys):
    assert main([str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert "Fidelity score" in out and "city" in out


def test_json_output_is_parseable(csv_path, capsys):
    assert main([str(csv_path), "--json", "--n", "40"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_synthetic"] == 40 and payload["n_real"] == 300
    assert "age" in payload["columns"]


def test_output_file_is_written(csv_path, tmp_path, capsys):
    target = tmp_path / "synth.csv"
    assert main([str(csv_path), "--n", "77", "--output", str(target), "--random-state", "5"]) == 0
    written = pd.read_csv(target)
    assert written.shape == (77, 6)
    assert list(written.columns) == list(pd.read_csv(csv_path).columns)
    assert "Wrote 77 synthetic rows" in capsys.readouterr().out


def test_quiet_only_reports_the_write(csv_path, tmp_path, capsys):
    target = tmp_path / "synth.tsv"
    assert main([str(csv_path), "-o", str(target), "-q"]) == 0
    out = capsys.readouterr().out
    assert "Fidelity score" not in out and "Wrote 300" in out
    assert pd.read_csv(target, sep="\t").shape == (300, 6)


def test_evaluate_existing_file(csv_path, tmp_path, capsys):
    target = tmp_path / "synth.csv"
    main([str(csv_path), "-o", str(target), "-q"])
    capsys.readouterr()
    assert main([str(csv_path), "--evaluate", str(target)]) == 0
    assert "Fidelity score" in capsys.readouterr().out


def test_no_correlations_flag(csv_path, capsys):
    assert main([str(csv_path), "--json"]) == 0
    with_correlations = json.loads(capsys.readouterr().out)
    assert main([str(csv_path), "--no-correlations", "--json"]) == 0
    independent = json.loads(capsys.readouterr().out)
    assert independent["correlation_mad"] > 0.05
    assert independent["correlation_mad"] > 2 * with_correlations["correlation_mad"]


def test_csv_dates_are_modelled_not_resampled(csv_path, capsys):
    assert main([str(csv_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert "note:" not in captured.err
    assert json.loads(captured.out)["columns"]["joined"]["metric"] == "ks"


def test_missing_input_file_returns_1(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 1
    assert "error:" in capsys.readouterr().err


def test_high_cardinality_note_on_stderr(tmp_path, capsys):
    path = tmp_path / "names.csv"
    pd.DataFrame({"name": [f"n{i}" for i in range(50)], "x": range(50)}).to_csv(path, index=False)
    assert main([str(path), "-q"]) == 0
    assert "note:" in capsys.readouterr().err


def test_parser_prog_name():
    assert build_parser().prog == "synthetic-tabular"


def test_unicode_survives_a_piped_subprocess(tmp_path):
    """Non-ASCII output must not raise UnicodeEncodeError when stdout is a pipe."""
    path = tmp_path / "unicode.csv"
    pd.DataFrame(
        {
            "city": ["München", "東京", "Zürich"] * 20,
            "x": range(60),
        }
    ).to_csv(path, index=False, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "synthetic_tabular.cli", str(path), "--json"],
        capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    payload = json.loads(result.stdout.decode("utf-8"))
    assert payload["columns"]["city"]["metric"] == "tvd"


def test_unicode_output_file_round_trips(tmp_path, capsys):
    source = tmp_path / "in.csv"
    target = tmp_path / "out.csv"
    pd.DataFrame({"city": ["München", "東京", "Zürich"] * 20, "x": range(60)}).to_csv(
        source, index=False, encoding="utf-8"
    )
    assert main([str(source), "-o", str(target), "-q"]) == 0
    written = pd.read_csv(target, encoding="utf-8")
    assert set(written["city"]) <= {"München", "東京", "Zürich"}
