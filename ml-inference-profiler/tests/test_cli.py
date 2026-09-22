"""The command line: the demo, reading saved reports, and encoding safety."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ml_inference_profiler import Profiler
from ml_inference_profiler.cli import build_parser, demo_report, main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ml-inference-profiler" in capsys.readouterr().out


def test_no_arguments_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_demo_prints_a_report(capsys):
    assert main(["--demo", "--repeats", "1"]) == 0
    out = capsys.readouterr().out

    assert "Bottleneck:" in out
    assert "Suggestions:" in out
    assert "modèle" in out  # non-ASCII labels survive the round trip


def test_demo_json_is_valid_json(capsys):
    assert main(["--demo", "--repeats", "1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)

    assert data["stages"]
    assert data["bottleneck"]
    assert any("modèle" in s["label"] for s in data["stages"])


def test_demo_tree_only(capsys):
    assert main(["--demo", "--repeats", "1", "--tree"]) == 0
    out = capsys.readouterr().out

    assert "Suggestions:" not in out
    assert "prétraitement" in out


def test_demo_report_has_the_expected_shape():
    report = demo_report(repeats=2)

    assert report.find("prétraitement/normalisation par élément").calls > 100
    assert report.find("modèle").self_ms > 0
    assert report.find("modèle/chargement des poids (cache froid)").calls == 1
    assert report.repeats == 2


def test_saved_report_can_be_printed_again(tmp_path, capsys):
    profiler = Profiler("saved")
    with profiler.stage("étape"):
        pass
    path = profiler.report().save(tmp_path / "r.json")

    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "saved" in out and "étape" in out


def test_output_writes_json(tmp_path, capsys):
    target = tmp_path / "out.json"
    assert main(["--demo", "--repeats", "1", "--output", str(target)]) == 0

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["name"].startswith("démo")
    assert "report written to" in capsys.readouterr().err


def test_a_missing_file_is_an_error_not_a_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_report_file_and_demo_together_is_a_usage_error(tmp_path, capsys):
    assert main([str(tmp_path / "x.json"), "--demo"]) == 2
    assert "not both" in capsys.readouterr().err


def test_demo_rejects_zero_repeats(capsys):
    assert main(["--demo", "--repeats", "0"]) == 1
    assert "repeats must be >= 1" in capsys.readouterr().err


def test_version_flag_prints_the_version(capsys):
    from ml_inference_profiler import __version__

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"ml-inference-profiler {__version__}" in capsys.readouterr().out


def test_every_documented_cli_flag_is_in_the_readme():
    """Regression: --repeats and --version worked but the README's CLI section hid them."""
    readme = (
        pathlib.Path(__file__).resolve().parents[1] / "README.md"
    ).read_text(encoding="utf-8")
    cli_section = readme.split("## CLI", 1)[1].split("## License", 1)[0]

    for flag in ("--demo", "--tree", "--json", "--output", "--repeats", "--version"):
        assert flag in cli_section, f"{flag} is accepted by the parser but undocumented"
    # and the parser really does accept the two that were missing
    args = build_parser().parse_args(["--demo", "--repeats", "7"])
    assert args.repeats == 7


def test_repeats_is_documented_as_demo_only():
    # argparse rewraps help text, so compare on collapsed whitespace.
    help_text = " ".join(build_parser().format_help().split())

    assert "ignored when a REPORT file is given" in help_text


def test_repeats_is_accepted_and_ignored_with_a_report_file(tmp_path, capsys):
    profiler = Profiler("saved")
    with profiler.stage("étape"):
        pass
    path = profiler.report().save(tmp_path / "r.json")

    assert main([str(path), "--repeats", "9"]) == 0
    assert "étape" in capsys.readouterr().out  # the saved numbers, not a re-run


def test_parser_is_buildable_on_its_own():
    parser = build_parser()
    args = parser.parse_args(["report.json", "--json"])

    assert args.report == "report.json" and args.json is True


def test_piping_non_ascii_output_does_not_crash():
    """The regression this package must never have: a pipe with an ASCII encoding."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "ascii"  # the harshest console a user can hand us
    completed = subprocess.run(
        [sys.executable, "-m", "ml_inference_profiler", "--demo", "--repeats", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in completed.stderr
    assert "modèle".encode("utf-8") in completed.stdout
