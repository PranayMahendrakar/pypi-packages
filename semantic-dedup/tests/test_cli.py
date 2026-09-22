"""The command line interface, including its behaviour under a pipe."""
import json
import os
import subprocess
import sys

import pytest

from semantic_dedup import __version__
from semantic_dedup.cli import build_parser, main

LINES = [
    "The meeting was postponed.",
    "the meeting was postponed",
    "We moved the meeting to a later date.",
    "Lunch is at noon.",
]
UNICODE_LINES = ["会議は延期されました", "会議は延期されました", "Café fermé — naïve", "Lunch is at noon."]


@pytest.fixture()
def notes(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture()
def unicode_notes(tmp_path):
    path = tmp_path / "uni.txt"
    path.write_text("\n".join(UNICODE_LINES) + "\n", encoding="utf-8")
    return str(path)


def test_help_and_version_exit_zero():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    assert "semantic-dedup" in parser.format_help()
    assert "--threshold" in parser.format_help()


def test_default_run_prints_a_summary(notes, capsys):
    assert main([notes]) == 0
    out = capsys.readouterr().out
    assert "semantic-dedup: 4 texts" in out
    assert "group 1" in out
    assert "removed" in out


def test_json_output_is_parseable(notes, capsys):
    assert main([notes, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_texts"] == 4
    assert payload["groups"] == [[0, 1, 2]]
    assert payload["method"] == "tfidf"


def test_report_mode_removes_nothing(notes, capsys):
    assert main([notes, "--report", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_removed"] == 0
    assert payload["dropped"] is False
    assert payload["groups"] == [[0, 1, 2]]


def test_output_file_holds_the_survivors(notes, tmp_path, capsys):
    target = tmp_path / "clean.txt"
    assert main([notes, "--output", str(target)]) == 0
    assert "wrote 2 texts" in capsys.readouterr().out
    kept = target.read_text(encoding="utf-8").splitlines()
    assert kept == ["We moved the meeting to a later date.", "Lunch is at noon."]


def test_flags_are_forwarded(notes, capsys):
    assert main([notes, "--threshold", "1.0", "--keep", "first", "--method", "tfidf", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["threshold"] == 1.0
    assert payload["keep"] == "first"
    assert payload["method"] == "tfidf"
    # No two of these lines are identical, so threshold 1.0 finds nothing.
    assert payload["groups"] == []

    assert main([notes, "--keep", "first", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["groups"] == [[0, 1, 2]]
    assert payload["kept"] == [0, 3]


def test_max_groups_limits_the_listing(tmp_path, capsys):
    path = tmp_path / "many.txt"
    lines = []
    for i in range(6):
        lines += [f"topic {i} was discussed", f"topic {i} was discussed"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main([str(path), "--max-groups", "2"]) == 0
    out = capsys.readouterr().out
    assert "and 4 more groups" in out


def test_column_selection_from_a_csv(tmp_path, capsys):
    path = tmp_path / "faq.csv"
    path.write_text("id,question\n1,reset my password\n2,reset my password\n", encoding="utf-8")
    assert main([str(path), "--column", "question", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["groups"] == [[0, 1]]


def test_errors_exit_one_with_a_message(tmp_path, capsys):
    missing = str(tmp_path / "nope.txt")
    assert main([missing]) == 1
    assert "error:" in capsys.readouterr().err

    path = tmp_path / "ok.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    assert main([str(path), "--threshold", "3"]) == 1
    assert "threshold" in capsys.readouterr().err


def test_unicode_summary_prints(unicode_notes, capsys):
    assert main([unicode_notes]) == 0
    out = capsys.readouterr().out
    assert "会議は延期されました" in out


# -- the real thing: a child process whose stdout is a pipe --------------------
def _run(args, extra_env=None):
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONUTF8"] = "0"          # do not let UTF-8 mode paper over the bug
    env["PYTHONLEGACYWINDOWSSTDIO"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "semantic_dedup", *args],
        capture_output=True, env=env,
    )


def test_cli_help_runs_as_a_subprocess():
    done = _run(["--help"])
    assert done.returncode == 0
    assert b"semantic-dedup" in done.stdout


@pytest.mark.parametrize("flags", [[], ["--json"], ["--report"]])
def test_piped_non_ascii_output_never_raises_unicodeencodeerror(unicode_notes, flags):
    # stdout is a pipe here, so Python picks the locale encoding, which on a
    # Windows box is usually cp1252 and cannot encode Japanese at all.
    done = _run([unicode_notes, *flags])
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    assert done.stdout
    text = done.stdout.decode("utf-8", "replace")
    if "--json" in flags:
        assert json.loads(text)["n_texts"] == 4
    else:
        assert "semantic-dedup:" in text


def test_piped_output_under_a_hostile_encoding(unicode_notes):
    done = _run([unicode_notes], extra_env={"PYTHONIOENCODING": "cp1252"})
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr


def test_module_entry_point_reports_the_version():
    done = _run(["--version"])
    assert done.returncode == 0
    assert __version__.encode() in done.stdout
