import json
import subprocess
import sys

import pandas as pd
import pytest

from schema_guard import Schema
from schema_guard.cli import main


@pytest.fixture
def files(tmp_path, train, messy):
    train_csv, new_csv = tmp_path / "train.csv", tmp_path / "new.csv"
    train.to_csv(train_csv, index=False)
    messy.to_csv(new_csv, index=False)
    schema_json = Schema.infer(train_csv).save(tmp_path / "schema.json")
    return {"train": str(train_csv), "new": str(new_csv), "schema": str(schema_json)}


def test_no_arguments_prints_help(capsys):
    assert main([]) == 2
    assert "usage: schema-guard" in capsys.readouterr().out


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "infer" in out and "validate" in out and "enforce" in out
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0 and capsys.readouterr().out.strip() == "schema-guard 0.1.0"


def test_bare_data_file_runs_infer(files, capsys):
    assert main([files["train"]]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Schema with 6 columns") and "city" in out and "string" in out


def test_infer_json_and_output(files, capsys, tmp_path):
    target = tmp_path / "out" / "schema.json"
    argv = ["infer", files["train"], "--json", "--output", str(target), "--max-categories", "2", "--no-ranges", "--nullable", "always"]
    assert main(argv) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "schema written to" in captured.err
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    columns = {c["name"]: c for c in payload["columns"]}
    assert columns["id"]["min"] is None and columns["city"]["categories"] is None and columns["id"]["nullable"] is True
    assert columns["tier"]["categories"] == ["a", "b"]


def test_validate_exit_codes_and_outputs(files, capsys, tmp_path):
    assert main(["validate", files["schema"], files["train"]]) == 0
    assert capsys.readouterr().out.startswith("Schema validation: OK")
    assert main(["validate", files["schema"], files["new"]]) == 1
    out = capsys.readouterr().out
    assert out.startswith("Schema validation: FAILED") and "unknown_category 'city'" in out
    target = tmp_path / "result.json"
    assert main(["validate", files["schema"], files["new"], "--json", "--output", str(target)]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False and json.loads(target.read_text(encoding="utf-8")) == payload
    assert "result written to" in captured.err


def test_validate_strict_ranges(files, capsys, tmp_path):
    high = tmp_path / "high.csv"
    pd.read_csv(files["train"]).assign(score=[9.0, 9.0, 9.0, 9.0]).to_csv(high, index=False)
    assert main(["validate", files["schema"], str(high)]) == 0
    assert "[warning] out_of_range 'score'" in capsys.readouterr().out
    assert main(["validate", files["schema"], str(high), "--strict-ranges"]) == 1
    assert "[error] out_of_range 'score'" in capsys.readouterr().out


def test_enforce_writes_the_fixed_file(files, capsys, tmp_path):
    target = tmp_path / "fixed.csv"
    assert main(["enforce", files["schema"], files["new"], "--output", str(target)]) == 0
    assert capsys.readouterr().out.strip() == f"wrote 2 rows x 6 columns to {target}"
    fixed = pd.read_csv(target)
    assert list(fixed.columns) == ["id", "city", "score", "active", "signup", "tier"]
    assert fixed["id"].tolist()[0] == 5 and pd.isna(fixed["id"].iloc[1])
    kept = tmp_path / "kept.tsv"
    assert main(["enforce", files["schema"], files["new"], "--output", str(kept), "--extra", "keep"]) == 0
    assert list(pd.read_csv(kept, sep="\t").columns)[-1] == "extra"


def test_enforce_failures_exit_1_with_the_problems_on_stderr(files, capsys, tmp_path):
    target = tmp_path / "never.csv"
    assert main(["enforce", files["schema"], files["new"], "--output", str(target), "--strict"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("DataFrame does not match the schema") and "wrong_order" in err
    assert main(["enforce", files["schema"], files["new"], "--output", str(target), "--extra", "raise"]) == 1
    assert "extra_column 'extra'" in capsys.readouterr().err
    partial = tmp_path / "partial.csv"
    pd.DataFrame({"id": [1]}).to_csv(partial, index=False)
    assert main(["enforce", files["schema"], str(partial), "--output", str(target), "--missing", "raise"]) == 1
    assert "missing_column" in capsys.readouterr().err
    assert not target.exists()


def test_bad_inputs_are_reported_not_raised(files, capsys, tmp_path):
    assert main(["validate", files["schema"], str(tmp_path / "data.xlsx")]) == 1
    assert "schema-guard: error: unsupported file type" in capsys.readouterr().err
    assert main(["validate", files["schema"], str(tmp_path / "missing.csv")]) == 1
    assert "schema-guard: error:" in capsys.readouterr().err
    assert main(["enforce", files["schema"], files["new"], "--output", str(tmp_path / "out.xlsx")]) == 1
    assert "unsupported output type" in capsys.readouterr().err


def test_module_entry_point_runs_in_a_subprocess():
    completed = subprocess.run([sys.executable, "-m", "schema_guard", "--help"], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0 and "usage: schema-guard" in completed.stdout
