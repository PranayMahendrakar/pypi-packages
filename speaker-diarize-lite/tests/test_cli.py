"""The command line tool, including output through pipes that cannot encode the text."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from _voices import SR
from speaker_diarize_lite import __version__, write_wav
from speaker_diarize_lite.cli import build_parser, main

UNICODE_NAME = "réunion_会議"


@pytest.fixture(scope="module")
def recording(tmp_path_factory, two_voices):
    signal, _ = two_voices
    path = tmp_path_factory.mktemp("cli") / (UNICODE_NAME + ".wav")
    write_wav(path, signal, SR, bits=16)
    return path


def run_cli(*args, encoding="cp1252"):
    """Run the CLI in a child process whose stdout encoding cannot represent CJK text."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = encoding
    env.pop("PYTHONUTF8", None)
    code = "import sys; from speaker_diarize_lite.cli import main; sys.exit(main())"
    return subprocess.run(
        [sys.executable, "-c", code, *[str(a) for a in args]],
        capture_output=True,
        env=env,
        timeout=120,
    )


def test_help_works(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage: speaker-diarize-lite" in out and "--rttm" in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_summary_by_default(recording, capsys):
    assert main([str(recording)]) == 0
    out = capsys.readouterr().out
    assert "2 speakers (estimated from the recording)" in out
    assert UNICODE_NAME in out


def test_json_output(recording, capsys):
    assert main([str(recording), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_speakers"] == 2 and data["file_id"] == UNICODE_NAME


def test_rttm_output_and_file_id(recording, capsys):
    assert main([str(recording), "--rttm", "--file-id", "call 7"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines and all(line.startswith("SPEAKER call_7 1 ") for line in lines)


def test_speakers_option(recording, capsys):
    assert main([str(recording), "--speakers", "3"]) == 0
    assert "3 speakers (as requested)" in capsys.readouterr().out


def test_output_files_are_utf8(recording, tmp_path, capsys):
    as_json = tmp_path / "out.json"
    as_rttm = tmp_path / "out.rttm"
    assert main([str(recording), "--output", str(as_json)]) == 0
    assert main([str(recording), "--output", str(as_rttm)]) == 0
    capsys.readouterr()
    data = json.loads(as_json.read_text(encoding="utf-8"))
    assert data["file_id"] == UNICODE_NAME
    assert UNICODE_NAME in as_json.read_text(encoding="utf-8")  # ensure_ascii=False
    rttm = as_rttm.read_text(encoding="utf-8")
    assert rttm.startswith("SPEAKER " + UNICODE_NAME + " 1 ")


def test_errors_exit_2(tmp_path, capsys):
    assert main([str(tmp_path / "missing.wav")]) == 2
    assert "error" in capsys.readouterr().err
    unwritable = tmp_path / "no_such_dir" / "out.json"
    silent = tmp_path / "silent.wav"
    write_wav(silent, [0.0] * 16000, 16000)
    assert main([str(silent), "--output", str(unwritable)]) == 2
    with pytest.raises(SystemExit) as exit_info:
        main([str(silent), "--speakers", "0"])
    assert exit_info.value.code == 2


def test_parser_is_exposed():
    parser = build_parser()
    assert parser.prog == "speaker-diarize-lite"


@pytest.mark.parametrize("flag", [None, "--json", "--rttm"])
def test_non_ascii_output_through_a_narrow_pipe_does_not_crash(recording, flag):
    args = [recording] + ([flag] if flag else [])
    proc = run_cli(*args)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    text = proc.stdout.decode("utf-8")
    assert UNICODE_NAME in text


def test_error_message_with_non_ascii_path_does_not_crash(tmp_path):
    proc = run_cli(tmp_path / (UNICODE_NAME + "_missing.wav"))
    assert proc.returncode == 2
    assert UNICODE_NAME in proc.stderr.decode("utf-8")
