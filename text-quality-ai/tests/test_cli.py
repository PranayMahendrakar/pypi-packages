"""The command line interface, including encoding safety under a pipe."""
import json
import os
import subprocess
import sys

import pytest

from text_quality_ai.cli import build_parser, main

UNICODE_TEXT = (
    "Le café était très chaud et le résumé naïf. "
    "यह एक परीक्षण है। "
    "这是一个测试。"
)
SAMPLE = ("The report was written by the team. The findings were reviewed by the board. "
          "The decision was taken by the chair. However, the minutes were never circulated.")


def test_help_works():
    parser = build_parser()
    text = parser.format_help()
    assert "text-quality-ai" in text
    assert "--json" in text and "--target" in text and "--fail-under" in text


def test_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_text_argument_prints_a_summary(capsys):
    assert main(["--text", SAMPLE]) == 0
    out = capsys.readouterr().out
    assert "Text quality:" in out
    assert "What to fix first:" in out


def test_json_output_is_parseable(capsys):
    assert main(["--text", SAMPLE, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["grade"] in {"A", "B", "C", "D", "F"}
    assert payload["inputs"] == ["--text"]
    assert set(payload["components"]) == {"readability", "repetition", "structure", "clarity", "vocabulary"}


def test_file_input_and_target(tmp_path, capsys):
    path = tmp_path / "draft.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    assert main([str(path), "--target", "marketing"]) == 0
    assert "marketing copy" in capsys.readouterr().out


def test_several_files_are_listed_one_per_line(tmp_path, capsys):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text(SAMPLE, encoding="utf-8")
    second.write_text("We wrote the report. The board liked it. Short and clear.", encoding="utf-8")
    assert main([str(first), str(second)]) == 0
    out = capsys.readouterr().out
    assert "Per document:" in out
    assert "a.txt" in out and "b.txt" in out


def test_output_file_is_written_as_utf8(tmp_path, capsys):
    destination = tmp_path / "report.json"
    repeated = ("Le café était chaud. Le café était froid. "
                "Le café était bon. Le café reste un café.")
    assert main(["--text", repeated, "--json", "--output", str(destination)]) == 0
    assert "wrote" in capsys.readouterr().out
    written = destination.read_text(encoding="utf-8")
    payload = json.loads(written)
    assert payload["stats"]["words"] > 0
    assert "café" in written                       # written straight through, not escaped


def test_compare_flag(tmp_path, capsys):
    other = tmp_path / "final.txt"
    other.write_text("We wrote the report. The board reviewed it. The chair decided.", encoding="utf-8")
    assert main(["--text", SAMPLE, "--compare", str(other)]) == 0
    out = capsys.readouterr().out
    assert "Compared with" in out
    assert "clarity" in out


def test_compare_flag_json(tmp_path, capsys):
    other = tmp_path / "final.txt"
    other.write_text("We wrote the report. The board reviewed it.", encoding="utf-8")
    assert main(["--text", SAMPLE, "--compare", str(other), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "compare" in payload and "report" in payload
    assert payload["compare"]["better"] in {"a", "b", "tie"}


def test_fail_under_sets_exit_code(capsys):
    assert main(["--text", "word " * 200, "--fail-under", "95"]) == 2
    assert "below" in capsys.readouterr().err


def test_missing_file_is_a_clear_error(capsys):
    assert main(["does-not-exist.txt"]) == 1
    assert "error" in capsys.readouterr().err


def test_text_and_files_together_is_an_error(tmp_path, capsys):
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    assert main([str(path), "--text", "hello"]) == 1
    assert "not both" in capsys.readouterr().err


def test_stdin_is_read_when_no_file_is_given(monkeypatch, capsys):
    import io

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(SAMPLE.encode("utf-8"))))
    assert main([]) == 0
    assert "Text quality:" in capsys.readouterr().out


def _run(args, extra_env=None):
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "text_quality_ai", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=120,
    )


def test_piped_output_with_non_ascii_text_does_not_crash():
    """A pipe on Windows defaults to the legacy codepage; the CLI must survive it."""
    done = _run(["--text", UNICODE_TEXT])
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    out = done.stdout.decode("utf-8")
    assert "Text quality:" in out


def test_piped_json_with_non_ascii_text_stays_utf8():
    done = _run(["--text", UNICODE_TEXT, "--json"])
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    payload = json.loads(done.stdout.decode("utf-8"))
    assert payload["stats"]["words"] > 0


def test_module_entry_point_help():
    done = _run(["--help"])
    assert done.returncode == 0
    assert b"text-quality-ai" in done.stdout
