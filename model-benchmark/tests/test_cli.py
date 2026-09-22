"""The command line interface, including console encoding safety."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from model_benchmark import cli

_MODULE = '''\
# -*- coding: utf-8 -*-
SAMPLE = ["café", "naïve", "日本語", "Здравствуй"]


def rapide(rows):
    return [len(str(value)) for value in rows]


def lent(rows):
    return [len(str(value).upper()) for value in rows]


def casse(rows):
    raise RuntimeError("ce modèle est cassé")


class Estimateur:
    def predict(self, rows):
        return [1 for _ in rows]
'''


@pytest.fixture()
def project(tmp_path, monkeypatch):
    (tmp_path / "mesmodeles.py").write_text(_MODULE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_help_works():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0


def test_version_works():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0


def test_demo_prints_a_summary(capsys):
    assert cli.main(["--demo", "--repeats", "2"]) == 0
    out = capsys.readouterr().out
    assert "model-benchmark:" in out
    assert "always_zero" in out
    assert "broken" in out  # the deliberately broken demo model is reported, not fatal


def test_demo_json_is_valid(capsys):
    assert cli.main(["--demo", "--repeats", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "accuracy"
    assert payload["n_failed"] == 1
    assert payload["best"]["score"] == "two_features"


def test_best_by_prints_only_the_winner(capsys):
    assert cli.main(["--demo", "--repeats", "2", "--best-by", "score"]) == 0
    assert capsys.readouterr().out.strip() == "two_features"


def test_output_file_is_written(tmp_path, capsys):
    target = tmp_path / "table.csv"
    assert cli.main(["--demo", "--repeats", "2", "--output", str(target)]) == 0
    assert "wrote" in capsys.readouterr().out
    frame = pd.read_csv(target)
    assert list(frame["model"]) == ["always_zero", "threshold", "two_features", "broken"]


def test_models_are_imported_from_the_current_directory(project, capsys):
    code = cli.main(
        ["rapide=mesmodeles:rapide", "lent=mesmodeles:lent", "--data", "mesmodeles:SAMPLE",
         "--repeats", "2"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "rapide" in out and "lent" in out


def test_unicode_model_names_survive_the_summary(project, capsys):
    code = cli.main(
        ["café=mesmodeles:rapide", "モデル=mesmodeles:lent", "--data", "mesmodeles:SAMPLE",
         "--repeats", "2"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "café" in out and "モデル" in out


def test_a_broken_model_named_on_the_command_line_is_reported(project, capsys):
    code = cli.main(
        ["ok=mesmodeles:rapide", "casse=mesmodeles:casse", "--data", "mesmodeles:SAMPLE",
         "--repeats", "2"]
    )
    out = capsys.readouterr().out
    assert code == 0  # one broken model is a finding, not a CLI failure
    assert "cassé" in out


def test_an_object_with_predict_works_from_the_command_line(project, capsys):
    code = cli.main(
        ["est=mesmodeles:Estimateur().predict", "--data", "mesmodeles:SAMPLE", "--repeats", "2"]
    )
    assert code == 2  # calling an expression is not supported, and it says so
    assert "attribute" in capsys.readouterr().err


def test_csv_with_a_target_column(project, capsys):
    frame = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8], "label": [0, 1, 0, 1]})
    frame.to_csv(project / "rows.csv", index=False)
    code = cli.main(
        [
            "threshold=mesmodeles:rapide",
            "--data",
            "rows.csv",
            "--target",
            "label",
            "--metric",
            "accuracy",
            "--repeats",
            "2",
        ]
    )
    assert code == 0
    assert "accuracy" in capsys.readouterr().out


def test_metric_without_labels_is_refused(project, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            ["m=mesmodeles:rapide", "--data", "mesmodeles:SAMPLE", "--metric", "accuracy"]
        )
    assert exit_info.value.code == 2
    assert "--target" in capsys.readouterr().err


def test_no_models_and_no_demo_is_refused(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "--demo" in capsys.readouterr().err


def test_unknown_module_is_a_clear_error(project, capsys):
    assert cli.main(["m=nosuchmodule:thing"]) == 2
    assert "cannot import module" in capsys.readouterr().err


def test_unknown_metric_is_a_clear_error(project, capsys):
    assert cli.main(["m=mesmodeles:rapide", "--data", "mesmodeles:SAMPLE", "--metric", "bleu"]) == 2
    assert "unknown metric" in capsys.readouterr().err


def test_missing_data_file_is_a_clear_error(project, capsys):
    assert cli.main(["m=mesmodeles:rapide", "--data", "missing.csv"]) == 2
    assert "no such data file" in capsys.readouterr().err


def test_missing_target_column_is_a_clear_error(project, capsys):
    pd.DataFrame({"x": [1, 2]}).to_csv(project / "rows.csv", index=False)
    code = cli.main(
        ["m=mesmodeles:rapide", "--data", "rows.csv", "--target", "ghost", "--metric", "accuracy"]
    )
    assert code == 2
    assert "no column 'ghost'" in capsys.readouterr().err


def test_parse_batch_sizes():
    assert cli.parse_batch_sizes("1,16,all") == (1, 16, "all")
    assert cli.parse_batch_sizes(None) == (1,)
    with pytest.raises(ValueError, match="not a whole number"):
        cli.parse_batch_sizes("many")


def test_parse_model_specs_names_models_after_their_attribute(project):
    models = cli.parse_model_specs(["mesmodeles:rapide"])
    assert list(models) == ["rapide"]
    with pytest.raises(ValueError, match="duplicate model name"):
        cli.parse_model_specs(["mesmodeles:rapide", "mesmodeles:rapide"])
    with pytest.raises(ValueError, match="module:attribute"):
        cli.parse_model_specs(["nocolon"])


def test_load_data_variants(project):
    data, has_target = cli.load_data(None, None)
    assert data is None and has_target is False

    data, has_target = cli.load_data("mesmodeles:SAMPLE", None)
    assert data[0] == "café" and has_target is False

    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(project / "rows.csv", index=False)
    data, has_target = cli.load_data("rows.csv", "y")
    assert has_target is True
    assert list(data[0].columns) == ["x"]
    assert list(data[1]) == [0, 1]


def test_build_parser_is_inspectable():
    parser = cli.build_parser()
    assert parser.prog == "model-benchmark"
    assert "--demo" in parser.format_help()


def test_piped_output_with_non_ascii_never_raises_unicodeencodeerror(project):
    """The family regression: a piped CLI must not die on a non-ASCII model name."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"  # a console that cannot encode Japanese
    env["PYTHONPATH"] = str(project)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_benchmark",
            "café=mesmodeles:rapide",
            "モデル=mesmodeles:lent",
            "--data",
            "mesmodeles:SAMPLE",
            "--repeats",
            "2",
        ],
        cwd=str(project),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in completed.stderr
    text = completed.stdout.decode("utf-8", "replace")
    assert "café" in text and "モデル" in text
