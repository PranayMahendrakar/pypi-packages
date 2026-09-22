"""The command line: offline-ml [MODELS] [options]."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from offline_ml import cli

UNICODE_NAME = "模型-Café-7b"


def test_help_exits_clean(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "offline-ml" in out and "--prefer" in out


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "offline-ml 0.1.0" in capsys.readouterr().out


def test_no_arguments_reports_the_machine(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("offline-ml:")
    assert "ram" in out and "device" in out


def test_json_of_the_machine(capsys):
    assert cli.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu_count"] >= 1
    assert payload["device"] in ("cuda", "mps", "cpu")


def test_inline_models(capsys):
    code = cli.main(["--model", "tiny:0.5", "--model", "mistral-7b-q4:4.1", "--prefer", "smallest"])
    assert code == 0
    out = capsys.readouterr().out
    assert "run 'tiny'" in out or "nothing fits" in out


def test_inline_models_json(capsys):
    assert cli.main(["--model", "tiny:0.5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requirements"]["tiny"]["disk_gb"] == 0.5
    assert payload["prefer"] == "balanced"


def test_models_file(tmp_path, capsys):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            [
                {"name": "tiny", "size_gb": 0.4, "quality": 2},
                {"name": UNICODE_NAME, "size_gb": 0.6, "quality": 9},
            ]
        ),
        encoding="utf-8",
    )
    assert cli.main([str(path), "--prefer", "quality"]) == 0
    assert UNICODE_NAME in capsys.readouterr().out


def test_models_file_with_a_models_key(tmp_path, capsys):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"models": [{"name": "tiny", "size_gb": 0.4}]}), encoding="utf-8")
    assert cli.main([str(path)]) == 0
    assert "tiny" in capsys.readouterr().out


def test_output_file_is_utf8(tmp_path, capsys):
    out_path = tmp_path / "pick.json"
    assert cli.main(["--model", f"{UNICODE_NAME}:0.6", "--output", str(out_path)]) == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["model"]["name"] == UNICODE_NAME
    assert "wrote" in capsys.readouterr().out


def test_bad_inline_model_is_a_clear_error(capsys):
    assert cli.main(["--model", "mistral-7b"]) == 1
    assert "NAME:SIZE_GB" in capsys.readouterr().err


def test_bad_inline_size_is_a_clear_error(capsys):
    assert cli.main(["--model", "mistral:huge"]) == 1
    assert "not a size in GiB" in capsys.readouterr().err


def test_missing_models_file_is_a_clear_error(capsys):
    assert cli.main(["no-such-file.json"]) == 1
    assert "offline-ml: error:" in capsys.readouterr().err


def test_malformed_models_file_is_a_clear_error(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert cli.main([str(path)]) == 1
    assert "offline-ml: error:" in capsys.readouterr().err


def test_models_file_of_the_wrong_shape_is_a_clear_error(tmp_path, capsys):
    path = tmp_path / "wrong.json"
    path.write_text("42", encoding="utf-8")
    assert cli.main([str(path)]) == 1
    assert "expected a JSON list" in capsys.readouterr().err


def test_empty_models_file_is_an_error_not_a_machine_report(tmp_path, capsys):
    """Asking to choose between models and supplying none is a mistake, not a fallback."""
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    assert cli.main([str(path)]) == 1
    captured = capsys.readouterr()
    assert "no models to choose from" in captured.err
    assert "cpu       :" not in captured.out


def test_empty_models_key_is_an_error(tmp_path, capsys):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"models": []}), encoding="utf-8")
    assert cli.main([str(path)]) == 1
    assert "no models to choose from" in capsys.readouterr().err


def test_no_arguments_is_still_a_machine_report(capsys):
    """The empty-source error must not swallow the plain `offline-ml` case."""
    assert cli.main([]) == 0
    assert "cpu       :" in capsys.readouterr().out


def test_parse_inline_model():
    assert cli.parse_inline_model("mistral-7b-q4:4.1") == {
        "name": "mistral-7b-q4",
        "size_gb": 4.1,
    }
    with pytest.raises(ValueError):
        cli.parse_inline_model(":4.1")


def test_piped_output_with_non_ascii_does_not_crash(tmp_path):
    """The regression this family has hit before: UnicodeEncodeError under a pipe."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "ascii"  # the worst case a CI log can hand us
    env["OFFLINE_ML_NO_TORCH"] = "1"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.path.join(root, "src") + os.pathsep + env.get("PYTHONPATH", "")
    for extra in ([], ["--json"]):
        proc = subprocess.run(
            [sys.executable, "-m", "offline_ml", "--model", f"{UNICODE_NAME}:0.6"] + extra,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=90,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        assert b"UnicodeEncodeError" not in proc.stderr
        assert proc.stdout


def test_negative_limit_is_rejected_with_a_clear_message(capsys):
    """`--limit -1` used to print one alternative and then 'and 3 more'."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--model", "a:1", "--model", "b:2", "--model", "c:3", "--limit", "-1"])
    assert exit_info.value.code == 2
    assert "must be 0 or more" in capsys.readouterr().err


def test_limit_zero_lists_nothing_and_counts_what_it_withheld(capsys):
    argv = ["--model", "a:1", "--model", "b:2", "--model", "c:3", "--limit", "0"]
    assert cli.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    if "  alternatives:" in lines:
        assert "    and 2 more" in lines
        assert "    a (1.0 GiB)" not in lines


def test_non_numeric_limit_is_rejected(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--model", "a:1", "--limit", "lots"])
    assert "whole number" in capsys.readouterr().err
