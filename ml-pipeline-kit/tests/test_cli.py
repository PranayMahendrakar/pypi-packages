"""The command line, including the console encoding it has to survive."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from ml_pipeline_kit import Pipeline
from ml_pipeline_kit.cli import build_parser, data_line, main, parse_range, parse_schema


UNICODE_ROWS = {
    "ville": ["Zürich", "東京", "São Paulo"],
    "température": [21.5, 33.0, 99.0],
    "id": [1, 2, 3],
}


@pytest.fixture()
def csv_path(tmp_path):
    path = tmp_path / "relevés.csv"
    pd.DataFrame(UNICODE_ROWS).to_csv(path, index=False, encoding="utf-8")
    return path


def test_help_works():
    parser = build_parser()
    text = parser.format_help()
    assert "ml-pipeline-kit" in text
    assert "--expect-schema" in text
    assert "--describe" in text


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert "ml-pipeline-kit 0.1.0" in capsys.readouterr().out


def test_passing_checks_print_the_summary_and_exit_zero(csv_path, capsys):
    code = main([str(csv_path), "--expect-schema", "id:int,ville:str", "--expect-range", "id:1:3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "data: " in out and "3 rows, 3 columns" in out
    assert "ville" in out and "température" in out
    assert "pipeline 'checks' ok" in out


def test_failing_checks_name_the_column_and_exit_one(csv_path, capsys):
    code = main([str(csv_path), "--expect-range", "température:0:40"])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAILED" in out
    assert "température" in out
    assert "1 of 3 values above 40" in out


def test_not_null_check(tmp_path, capsys):
    path = tmp_path / "gaps.csv"
    pd.DataFrame({"id": [1, None, 3]}).to_csv(path, index=False, encoding="utf-8")
    assert main([str(path), "--not-null", "id"]) == 1
    assert "1 of 3 values missing" in capsys.readouterr().out
    assert main([str(path), "--not-null", "ghost"]) == 1


def test_warn_turns_failures_into_warnings(csv_path, capsys):
    code = main([str(csv_path), "--expect-range", "température:0:40", "--warn"])
    out = capsys.readouterr().out
    assert code == 0
    assert "warnings" in out and "warned" in out


def test_json_output_is_utf8_and_parses(csv_path, capsys):
    code = main([str(csv_path), "--expect-schema", "température:float", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["steps"][0]["name"] == "schema"
    assert payload["data"].endswith("relevés.csv")


def test_output_file_is_written(csv_path, tmp_path, capsys):
    target = tmp_path / "report.json"
    code = main([str(csv_path), "--expect-range", "température:0:40", "--output", str(target)])
    assert code == 1
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert "température" in payload["failures"][0]
    assert "wrote     :" in capsys.readouterr().out


def test_no_checks_says_so(csv_path, capsys):
    code = main([str(csv_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "give --expect-schema" in out


def test_describe_prints_a_saved_pipeline(tmp_path, capsys):
    path = tmp_path / "saved.json"
    Pipeline("scoring").expect_schema({"age": "int"}).add(lambda df: df, name="clean").save(path)
    assert main(["--describe", str(path)]) == 0
    out = capsys.readouterr().out
    assert "pipeline 'scoring' with 2 step(s)" in out
    assert "age:int" in out
    assert "needs bind()" in out


def test_missing_file_and_bad_extension_report_cleanly(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err

    other = tmp_path / "data.xlsx"
    other.write_text("not really", encoding="utf-8")
    assert main([str(other)]) == 2
    assert "expected .csv" in capsys.readouterr().err


def test_duplicate_columns_are_reported_by_name(tmp_path, capsys):
    path = tmp_path / "dupes.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    assert main([str(path), "--expect-schema", "a:int"]) == 2
    assert "duplicate column names: a" in capsys.readouterr().err


def test_bad_spec_strings_are_explained(tmp_path, capsys):
    path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False, encoding="utf-8")
    assert main([str(path), "--expect-range", "a:0"]) == 2
    assert "column:low:high" in capsys.readouterr().err
    assert main([str(path), "--expect-range", "a:low:high"]) == 2
    assert "is not a number" in capsys.readouterr().err
    assert main([str(path), "--expect-schema", ":int"]) == 2
    assert "column name is missing" in capsys.readouterr().err


def test_no_arguments_at_all_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2
    assert "--describe" in capsys.readouterr().err


def test_spec_parsers():
    assert parse_schema("age:int, city ") == {"age": "int", "city": None}
    assert parse_range("age:18:100") == ("age", 18.0, 100.0)
    assert parse_range("age:18:") == ("age", 18.0, None)
    assert parse_range("age::100") == ("age", None, 100.0)
    with pytest.raises(ValueError, match="at least one bound"):
        parse_range("age::")


def test_data_line_truncates_a_wide_table():
    wide = pd.DataFrame({"c{0}".format(i): [1] for i in range(12)})
    line = data_line("wide.csv", wide)
    assert "12 columns" in line and "and 4 more" in line


def test_cli_output_survives_being_piped_on_a_non_utf8_console(csv_path, tmp_path):
    """The bug this family shipped before: non-ASCII output through a pipe."""
    environment = dict(os.environ)
    for variable in ("PYTHONIOENCODING", "PYTHONUTF8", "PYTHONLEGACYWINDOWSSTDIO"):
        environment.pop(variable, None)
    environment["PYTHONIOENCODING"] = "cp1252:strict"
    finished = subprocess.run(
        [sys.executable, "-m", "ml_pipeline_kit", str(csv_path), "--expect-range", "température:0:40"],
        capture_output=True,
        env=environment,
        cwd=str(tmp_path),
    )
    assert finished.returncode == 1, finished.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in finished.stderr
    text = finished.stdout.decode("utf-8", "replace")
    assert "température" in text
    assert "FAILED" in text


# ------------------------------------------- reviewer regressions: the CLI
def test_output_creates_a_missing_folder_the_way_save_does(csv_path, tmp_path, capsys):
    """Pipeline.save() makes its parents, so --output does too."""
    target = tmp_path / "nope" / "dir" / "report.json"
    code = main([str(csv_path), "--expect-range", "température:0:40", "--output", str(target)])
    assert code == 1
    capsys.readouterr()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is False


def test_output_that_cannot_be_written_explains_itself(csv_path, tmp_path, capsys):
    """A write that still fails says what is wrong, never a bare errno."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a folder", encoding="utf-8")
    code = main([str(csv_path), "--expect-schema", "id:int", "--output", str(blocker / "r.json")])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("ml-pipeline-kit: error: --output ")
    assert "Traceback" not in err


def test_range_warning_about_unreadable_values_reaches_the_cli(tmp_path, capsys):
    path = tmp_path / "junk.csv"
    path.write_text("id,price\n1,10.5\n2,free\n3,20.0\n", encoding="utf-8")
    code = main([str(path), "--expect-range", "price:0:1000"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 of 3 values that are not numbers" in out
