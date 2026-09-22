"""The command line: summary, JSON, output files, flags and errors."""
import json

import pandas as pd
import pytest

from smartclean_df import __version__
from smartclean_df.cli import build_parser, main


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "A B": [1, 2, 2, 3, 1000],
            "C": ["x", "y", "y", "z", "w"],
            "When": ["2024-01-01", "2024-01-02", "2024-01-02", "NA", "2024-01-04"],
        }
    ).to_csv(path, index=False)
    return path


def test_summary_output(csv_path, capsys):
    assert main([str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("smartclean-df: 5 rows x 3 columns -> 4 rows x 3 columns")
    assert "drop_duplicates" in out and "rename_column" in out


def test_json_and_output_file(csv_path, tmp_path, capsys):
    target = tmp_path / "clean.csv"
    assert main([str(csv_path), "--json", "--output", str(target)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_shape"] == [4, 3]
    assert payload["output"] == str(target)
    written = pd.read_csv(target)
    assert list(written.columns) == ["a_b", "c", "when"]
    assert len(written) == 4


def test_tsv_output(csv_path, tmp_path):
    target = tmp_path / "clean.tsv"
    assert main([str(csv_path), "--output", str(target)]) == 0
    assert list(pd.read_csv(target, sep="\t").columns) == ["a_b", "c", "when"]


def test_flags_are_passed_through(csv_path, capsys):
    code = main(
        [
            str(csv_path),
            "--missing",
            "drop",
            "--outliers",
            "flag",
            "--iqr-factor",
            "1.5",
            "--keep-duplicates",
            "--keep-column-names",
            "--no-parse-dates",
            "--no-parse-numbers",
            "--no-parse-booleans",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    kinds = [a["kind"] for a in payload["actions"]]
    assert "drop_duplicates" not in kinds and "rename_column" not in kinds
    assert "parse_datetime" not in kinds and "clip_outliers" not in kinds
    assert "drop_missing_rows" in kinds and "flag_outliers" in kinds
    assert payload["dtypes"]["When"] == "object"


def test_dry_run_prints_and_never_writes(csv_path, tmp_path, capsys):
    assert main([str(csv_path), "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main([str(csv_path), "--dry-run", "--output", str(tmp_path / "x.csv")])
    assert exc.value.code == 2
    assert not (tmp_path / "x.csv").exists()


def test_help_and_version():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert build_parser().prog == "smartclean-df"
    assert __version__ == "0.1.0"


def test_errors_exit_with_status_two(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 2
    assert "error:" in capsys.readouterr().err
    bad = tmp_path / "data.xlsx"
    bad.write_text("nope", encoding="utf-8")
    assert main([str(bad)]) == 2
