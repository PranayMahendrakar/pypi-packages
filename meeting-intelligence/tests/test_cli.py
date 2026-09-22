"""The command line interface, including its behaviour under a pipe."""
from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

import meeting_intelligence as mi
from meeting_intelligence import cli

from conftest import MEETING, UNICODE, VTT


@pytest.fixture
def transcript_file(tmp_path):
    path = tmp_path / "standup.txt"
    path.write_text(MEETING, encoding="utf-8")
    return path


def test_help_builds_and_names_every_option(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--json", "--markdown", "--actions", "--decisions", "--output", "--speakers"):
        assert flag in text
    assert "never opens or decodes audio" in text


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert mi.__version__ in capsys.readouterr().out


def test_default_run_prints_the_summary(transcript_file, capsys):
    assert cli.main([str(transcript_file)]) == 0
    out = capsys.readouterr().out
    assert "Meeting report" in out
    assert "Action items" in out
    assert "Postgres" in out


def test_json_output_is_parseable(transcript_file, capsys):
    assert cli.main([str(transcript_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_turns"] == 10
    assert payload["decisions"]


def test_markdown_output(transcript_file, capsys):
    assert cli.main([str(transcript_file), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Meeting report")
    assert "| Owner | Task | Due | Confidence |" in out


def test_actions_and_decisions_shortcuts(transcript_file, capsys):
    assert cli.main([str(transcript_file), "--actions"]) == 0
    actions = capsys.readouterr().out
    assert "Bob:" in actions
    assert cli.main([str(transcript_file), "--decisions"]) == 0
    assert "Postgres" in capsys.readouterr().out


def test_shortcuts_say_so_when_there_is_nothing(tmp_path, capsys):
    path = tmp_path / "quiet.txt"
    path.write_text("Alice: Morning. The weather is fine outside today.\n", encoding="utf-8")
    cli.main([str(path), "--actions"])
    assert "no action items found" in capsys.readouterr().out
    cli.main([str(path), "--decisions"])
    assert "no decisions found" in capsys.readouterr().out


def test_speakers_roster_resolves_a_first_name(tmp_path, capsys):
    path = tmp_path / "roster.txt"
    path.write_text("Alice: Sam will handle the archive job by Friday.\n", encoding="utf-8")
    assert cli.main([str(path), "--speakers", "Alice,Sam Okafor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action_items"][0]["owner"] == "Sam Okafor"
    assert payload["speakers"] == ["Alice", "Sam Okafor"]


def test_parse_speakers_cleans_the_list():
    assert cli.parse_speakers("Alice, Bob , Alice, ") == ["Alice", "Bob"]
    assert cli.parse_speakers(None) == []
    assert cli.parse_speakers("") == []


@pytest.mark.parametrize("name, marker", [("out.json", '"summary_text"'), ("out.md", "# Meeting report"), ("out.txt", "Meeting report")])
def test_output_file_format_follows_the_suffix(transcript_file, tmp_path, capsys, name, marker):
    target = tmp_path / name
    assert cli.main([str(transcript_file), "--output", str(target)]) == 0
    written = target.read_text(encoding="utf-8")
    assert marker in written
    assert "wrote the report to" in capsys.readouterr().out


def test_a_missing_file_is_a_clean_error_not_a_traceback(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.txt")]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("meeting-intelligence: error:")
    assert "nope.txt" in captured.err


def test_stdin_is_read_when_the_target_is_a_dash(monkeypatch, capsys):
    stream = io.TextIOWrapper(io.BytesIO(VTT.encode("utf-8")), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stream)
    assert cli.main(["-", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_format"] == "vtt"


def test_an_empty_stdin_transcript_is_an_empty_report(monkeypatch, capsys):
    stream = io.TextIOWrapper(io.BytesIO(b""), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stream)
    assert cli.main(["-"]) == 0
    assert "Empty transcript" in capsys.readouterr().out


# ------------------------------------------------- the real process, under a pipe


def _run(args, stdin_bytes=None, env_extra=None):
    """Run the module as a subprocess with stdout captured (a pipe, not a tty)."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "meeting_intelligence.cli"] + args,
        input=stdin_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def test_the_installed_module_runs_help():
    done = _run(["--help"])
    assert done.returncode == 0
    assert b"meeting-intelligence" in done.stdout


def test_non_ascii_data_through_a_pipe_does_not_crash(tmp_path):
    path = tmp_path / "unicode.txt"
    path.write_text(UNICODE, encoding="utf-8")
    for extra in ([], ["--json"], ["--markdown"], ["--actions"]):
        done = _run([str(path)] + extra)
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        assert b"UnicodeEncodeError" not in done.stderr
        assert b"Trace" not in done.stderr
        text = done.stdout.decode("utf-8", errors="replace")
        assert "Zo" in text


def test_non_ascii_transcript_on_stdin_through_a_pipe():
    done = _run(["-", "--json"], stdin_bytes=UNICODE.encode("utf-8"))
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    payload = json.loads(done.stdout.decode("utf-8"))
    assert "Zoë Müller" in payload["speakers"]
    assert "李雷" in payload["speakers"]
