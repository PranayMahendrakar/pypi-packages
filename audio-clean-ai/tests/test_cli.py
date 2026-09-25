"""The command line tool, the README quickstart, and console encoding safety."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import wave
from contextlib import redirect_stdout

import numpy as np
import pytest

from audio_clean_ai.cli import main
from conftest import SR, voice_like, white, write_pcm

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def _noisy_wav(folder: str, name: str = "noisy.wav", seconds: float = 2.0) -> str:
    n = int(seconds * SR)
    audio = voice_like(n, lead_s=0.4) + white(n, 0.05)
    path = os.path.join(folder, name)
    write_pcm(path, np.round(audio * 30000).astype(np.int16), SR, 2)
    return path


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "audio-clean-ai" in out and "--output" in out and "heuristic" in out


def test_dry_run_prints_the_summary_and_writes_nothing(tmp_path, capsys):
    path = _noisy_wav(str(tmp_path))
    assert main([path]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("audio clean: noise down")
    assert "dry run" in captured.err
    assert sorted(os.listdir(str(tmp_path))) == ["noisy.wav"]


def test_output_json_and_report(tmp_path, capsys):
    path = _noisy_wav(str(tmp_path))
    out = os.path.join(str(tmp_path), "clean.wav")
    report = os.path.join(str(tmp_path), "report.json")
    code = main([path, "--output", out, "--json", "--report", report, "--strength", "0.6"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["strength"] == 0.6 and printed["output"] == out
    with open(report, encoding="utf-8") as handle:
        assert json.load(handle)["n_samples"] == 2 * SR
    with wave.open(out, "rb") as handle:
        assert handle.getnframes() == 2 * SR and handle.getsampwidth() == 2


def test_noise_clip_bits_quiet_and_no_preserve(tmp_path, capsys):
    path = _noisy_wav(str(tmp_path))
    clip = os.path.join(str(tmp_path), "room.wav")
    write_pcm(clip, np.round(white(SR, 0.05, seed=8) * 30000).astype(np.int16), SR, 2)
    out = os.path.join(str(tmp_path), "clean24.wav")
    code = main([path, "-o", out, "--noise-clip", clip, "--bits", "24", "--no-preserve-speech", "--quiet"])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    with wave.open(out, "rb") as handle:
        assert handle.getsampwidth() == 3


def test_bad_input_exits_two(tmp_path, capsys):
    assert main([os.path.join(str(tmp_path), "missing.wav")]) == 2
    assert "no such audio file" in capsys.readouterr().err
    assert main([]) == 2
    assert main([_noisy_wav(str(tmp_path)), "--strength", "3"]) == 2
    assert "strength" in capsys.readouterr().err


def test_non_ascii_output_survives_an_ascii_pipe(tmp_path):
    """A cp1252 or ASCII stdout must not crash on a non-Latin file name."""
    name = "\u4f1a\u8b70_r\u00e9union_\u0437\u0430\u043f\u0438\u0441\u044c.wav"
    path = _noisy_wav(str(tmp_path), name=name)
    env = dict(os.environ, PYTHONIOENCODING="ascii")
    for extra in ([], ["--json"]):
        done = subprocess.run(
            [sys.executable, "-m", "audio_clean_ai.cli", path] + extra,
            capture_output=True,
            env=env,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        text = done.stdout.decode("utf-8")
        assert name in text


def _quickstart_block() -> str:
    with open(README, encoding="utf-8") as handle:
        readme = handle.read()
    section = readme.split("## Quickstart", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_readme_quickstart_runs_and_matches_its_printed_output():
    code = _quickstart_block()
    assert 3 <= len([line for line in code.splitlines() if line.strip()]) <= 6
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(code, "README-quickstart", "exec"), {"__name__": "__main__"})
    printed = buffer.getvalue().strip()
    with open(README, encoding="utf-8") as handle:
        section = handle.read().split("## Quickstart", 1)[1]
    shown = re.search(r"```python\n.*?```\s*```\n(.*?)```", section, re.S).group(1).strip()
    assert printed == shown
