"""The command line interface."""
import io
import json
import os
import subprocess
import sys

import pytest

from hallucination_check.cli import build_parser, load_sources, main

ANSWER = "The Eiffel Tower is in Paris. It was completed in 1889. It cost 42 million francs."
SOURCE = "The Eiffel Tower stands in Paris, France, and was completed in 1889."


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_literal_text_arguments(capsys):
    assert main([ANSWER, "-s", SOURCE]) == 0
    out = capsys.readouterr().out
    assert "66.7 / 100 grounded" in out
    assert "42 million" in out


def test_files_on_disk(tmp_path, capsys):
    answer = tmp_path / "answer.txt"
    source = tmp_path / "source.txt"
    answer.write_text(ANSWER, encoding="utf-8")
    source.write_text(SOURCE, encoding="utf-8")
    assert main([str(answer), "-s", str(source)]) == 0
    out = capsys.readouterr().out
    assert "66.7 / 100 grounded" in out
    assert "source.txt" in out or "grounded" in out


def test_json_output_is_valid(capsys):
    assert main([ANSWER, "-s", SOURCE, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["score"] == 66.7
    assert payload["unsupported"] == [2]


def test_output_file_is_utf8_json(tmp_path, capsys):
    out_path = tmp_path / "report.json"
    assert main(["Zoë a confirmé 12 %.", "-s", "Zoë a confirmé 12 % hier.", "--output", str(out_path)]) == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "Zoë" in payload["claims"][0]["text"]
    assert "wrote the JSON report" in capsys.readouterr().out


def test_stdin_answer(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(ANSWER))
    assert main(["-", "-s", SOURCE]) == 0
    assert "grounded" in capsys.readouterr().out


def test_jsonl_sources(tmp_path, capsys):
    path = tmp_path / "retrieved.jsonl"
    path.write_text(
        json.dumps({"id": "wiki:eiffel#3", "text": SOURCE}, ensure_ascii=False) + "\n"
        + json.dumps({"id": "wiki:paris#1", "text": "Paris is the capital of France."}) + "\n",
        encoding="utf-8",
    )
    assert main([ANSWER, "-s", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_ids"] == ["wiki:eiffel#3", "wiki:paris#1"]
    assert payload["citations"]["0"] == "wiki:eiffel#3"


def test_a_broken_jsonl_file_is_an_error(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    assert main([ANSWER, "-s", str(path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_granularity_and_threshold_flags(capsys):
    assert main([ANSWER, "-s", SOURCE, "--granularity", "paragraph", "--threshold", "0.2"]) == 0
    assert "granularity: paragraph" in capsys.readouterr().out


def test_no_fact_flags(capsys):
    assert main(["The trial enrolled 4,200 patients.", "-s", "The trial enrolled 1,200 patients.", "--no-fact-flags"]) == 0
    assert "100.0 / 100 grounded" in capsys.readouterr().out


def test_fail_under_returns_2(capsys):
    assert main([ANSWER, "-s", SOURCE, "--fail-under", "80"]) == 2
    assert "below --fail-under" in capsys.readouterr().err
    assert main([ANSWER, "-s", SOURCE, "--fail-under", "50"]) == 0


def test_a_bad_threshold_is_a_clean_error(capsys):
    assert main([ANSWER, "-s", SOURCE, "--threshold", "5"]) == 1
    assert "error:" in capsys.readouterr().err


def test_no_sources_still_reports(capsys):
    assert main([ANSWER]) == 0
    assert "0.0 / 100 grounded" in capsys.readouterr().out


def test_load_sources_reads_plain_files_with_an_id(tmp_path):
    path = tmp_path / "passage.txt"
    path.write_text(SOURCE, encoding="utf-8")
    loaded = load_sources([str(path), "literal text"])
    assert loaded[0] == {"text": SOURCE, "id": "passage.txt"}
    assert loaded[1] == "literal text"


def test_non_ascii_output_survives_a_pipe_in_a_subprocess(tmp_path):
    """The exact failure QA found in sibling packages: piped non-ASCII output.

    The child is forced onto a legacy codepage, which is what a Windows console or a
    captured pipe does; the CLI must still write the report instead of raising
    UnicodeEncodeError.
    """
    answer = tmp_path / "answer.txt"
    source = tmp_path / "source.txt"
    answer.write_text("Zoë Kraków a confirmé 12 %. 李雷 a signé le 12 mars 2024.", encoding="utf-8")
    source.write_text("Zoë Kraków a confirmé 12 % hier.", encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "cp1252"
    for extra in ([], ["--json"]):
        proc = subprocess.run(
            [sys.executable, "-m", "hallucination_check.cli", str(answer), "-s", str(source)] + extra,
            capture_output=True,
            env=env,
        )
        stderr = proc.stderr.decode("utf-8", "replace")
        assert proc.returncode == 0, stderr
        assert "UnicodeEncodeError" not in stderr
        assert proc.stdout


def test_non_ascii_stdin_survives_a_legacy_codepage(tmp_path):
    """Piping UTF-8 into the CLI must not fail on the console's own codepage."""
    source = tmp_path / "source.txt"
    source.write_text("Zoë Kraków a confirmé 12 % hier.", encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "cp1252"
    proc = subprocess.run(
        [sys.executable, "-m", "hallucination_check.cli", "-", "-s", str(source)],
        input="Zoë Kraków a confirmé 12 %. 李雷 a signé le 12 mars 2024.".encode("utf-8"),
        capture_output=True,
        env=env,
    )
    stderr = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0, stderr
    assert "codec can't decode" not in stderr
    assert b"grounded" in proc.stdout
