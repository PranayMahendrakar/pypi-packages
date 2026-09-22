import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from data_drift_lite.cli import main


@pytest.fixture
def csv_pair(tmp_path):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 200), "c": rng.choice(["a", "b"], 200)})
    current = pd.DataFrame({"x": rng.normal(3, 1, 200), "c": rng.choice(["a", "b"], 200)})
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    return str(ref_path), str(cur_path)


def test_summary_output(csv_pair, capsys):
    assert main(list(csv_pair)) == 0
    out = capsys.readouterr().out
    assert out.startswith("data-drift-lite: DRIFT DETECTED: 1 of 2 columns drifted")
    assert "DRIFTED" in out


def test_json_output(csv_pair, capsys):
    assert main([*csv_pair, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["drifted"] is True and payload["drifted_columns"] == ["x"]


def test_output_file(csv_pair, tmp_path, capsys):
    target = tmp_path / "report.json"
    assert main([*csv_pair, "--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["drifted_columns"] == ["x"]
    assert "data-drift-lite:" in capsys.readouterr().out  # summary still printed


def test_fail_on_drift_exit_code(csv_pair, capsys):
    assert main([*csv_pair, "--fail-on-drift"]) == 1
    assert main([*csv_pair, "--fail-on-drift", "--columns", "c"]) == 0


def test_options_are_passed_through(csv_pair, capsys):
    assert main([*csv_pair, "--columns", "c", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["columns"]) == ["c"]
    assert main([*csv_pair, "--threshold", "0", "--psi-threshold", "100", "--sample", "50", "--random-state", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["drifted"] is False and payload["reference_rows"] == 50
    assert main([*csv_pair, "--sample", "0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reference_rows"] == 200


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: data-drift-lite" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_bad_inputs_exit_with_usage_error(csv_pair, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "missing.csv"), csv_pair[1]])
    assert exc.value.code == 2
    assert "no such file" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main([*csv_pair, "--columns", "nope"])
    assert exc.value.code == 2


def test_python_m_entry_point():
    result = subprocess.run([sys.executable, "-m", "data_drift_lite", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage: data-drift-lite" in result.stdout


def test_unicode_survives_json_output_and_file(tmp_path, capsys):
    reference = pd.DataFrame({"città": ["café"] * 40 + ["東京"] * 40})
    current = pd.DataFrame({"città": ["café"] * 10 + ["東京"] * 70})
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    reference.to_csv(ref_path, index=False, encoding="utf-8")
    current.to_csv(cur_path, index=False, encoding="utf-8")
    target = tmp_path / "out.json"
    assert main([str(ref_path), str(cur_path), "--json", "--output", str(target)]) == 0
    printed = capsys.readouterr().out
    # ensure_ascii=False: the characters themselves, never escaped to an ASCII form
    assert "東京" in printed and "café" in printed
    assert chr(92) + "u6771" not in printed
    on_disk = target.read_text(encoding="utf-8")
    assert "東京" in on_disk and "città" in on_disk
    assert json.loads(on_disk)["drifted_columns"] == ["città"]
