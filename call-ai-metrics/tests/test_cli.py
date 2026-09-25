"""The command line tool, including output that must survive a non-UTF-8 console."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from call_ai_metrics.cli import main
from conftest import write_wav

UNICODE_CSV = (
    "start,end,speaker,text\n"
    "0.0,6.0,Дмитрий,\"Добрый день, чем могу помочь?\"\n"
    "6.5,9.0,山田,予約を変更したいです\n"
    "9.4,15.0,Дмитрий,Конечно — одну минуту\n"
)


def _csv(tmp_path, text=UNICODE_CSV, name="звонок.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_help(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "call-ai-metrics" in out and "--speakers" in out


def test_summary_json_and_output(tmp_path, capsys):
    source = _csv(tmp_path)
    assert main([source]) == 0
    out = capsys.readouterr().out
    assert "Дмитрий" in out and "山田" in out

    target = tmp_path / "отчёт.json"
    assert main([source, "--json", "--output", str(target)]) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(target.read_text(encoding="utf-8"))
    assert printed == written
    assert "Дмитрий" in target.read_text(encoding="utf-8")  # ensure_ascii=False
    assert written["speakers"]["山田"]["words"] == 10


def test_explain_and_settings(tmp_path, capsys):
    source = _csv(tmp_path, "0,10,A\n9.2,14,B\n")
    assert main([source, "--explain", "--grace", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "How this report was measured" in out
    assert "0 interruptions" in out  # 0.8 s of overlap is inside a 1.0 s grace


def test_wav_with_speaker_names(tmp_path, capsys, call_audio):
    path = tmp_path / "call.wav"
    write_wav(str(path), call_audio)
    assert main([str(path), "--speakers", "agent", "customer"]) == 0
    out = capsys.readouterr().out
    assert "agent" in out and "customer" in out and "energy threshold" in out


def test_exit_codes(tmp_path, capsys):
    assert main([]) == 2
    assert main([str(tmp_path / "missing.wav")]) == 2
    assert "no such file" in capsys.readouterr().err
    empty = _csv(tmp_path, "", "empty.csv")
    assert main([empty, "--quiet"]) == 1
    assert capsys.readouterr().out == ""
    bad = _csv(tmp_path, "2,1,A\n", "bad.csv")
    assert main([bad]) == 2
    assert "ends" in capsys.readouterr().err


def test_piped_output_survives_an_ascii_console(tmp_path):
    """Non-ASCII names through a pipe whose encoding cannot hold them must not crash."""
    source = _csv(tmp_path)
    env = dict(os.environ, PYTHONIOENCODING="ascii")
    env.pop("PYTHONUTF8", None)
    for extra in ([], ["--json"]):
        done = subprocess.run(
            [sys.executable, "-m", "call_ai_metrics.cli", source] + extra,
            capture_output=True,
            env=env,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        text = done.stdout.decode("utf-8")
        assert "Дмитрий" in text and "山田" in text
