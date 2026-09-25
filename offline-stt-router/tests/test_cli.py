"""The command line, including non-ASCII output through a pipe."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from offline_stt_router.cli import main, with_default_command

from conftest import tone, write_wav
from test_transcribe import FAKE_WHISPER_CLI

DEVANAGARI = "नमस्ते"


@pytest.fixture(autouse=True)
def cli_sees_only_the_fake_world(request, monkeypatch):
    """In-process CLI runs look for engines in the fake site folder only."""
    if "world" not in request.fixturenames:
        return
    world = request.getfixturevalue("world")
    import offline_stt_router.cli as cli
    from offline_stt_router import Router

    monkeypatch.setattr(cli, "Router", lambda **kw: Router(search_paths=[str(world.site)], **kw))


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as done:
        main(["--help"])
    assert done.value.code == 0
    assert "usage: offline-stt-router" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "0.1.0" in capsys.readouterr().out


def test_default_command_routing():
    assert with_default_command([]) == ["choose"]
    assert with_default_command(["--language", "hi"]) == ["choose", "--language", "hi"]
    assert with_default_command(["talk.wav"]) == ["transcribe", "talk.wav"]
    assert with_default_command(["engines"]) == ["engines"]
    assert with_default_command(["-h"]) == ["-h"]


def test_bare_command_says_what_to_install(world, capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "No speech-to-text engine is ready" in out and "pip install faster-whisper" in out


def test_engines_and_machine(world, capsys):
    world.faster_whisper()
    world.ct2_model("Systran/faster-whisper-small")
    assert main(["engines"]) == 0
    out = capsys.readouterr().out
    assert "engines ready" in out and "faster-whisper" in out
    assert main(["engines", "--json"]) == 0
    names = [e["name"] for e in json.loads(capsys.readouterr().out)]
    assert names == ["faster-whisper", "openai-whisper", "whisper.cpp", "vosk"]
    assert main(["machine", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cpu_physical"] >= 1
    assert main(["machine"]) == 0
    assert "RAM" in capsys.readouterr().out


def test_choose_json_to_a_utf8_file(world, tmp_path, capsys):
    target = tmp_path / "choice.json"
    assert main(["choose", "--language", "हिन्दी", "--prefer", "accurate", "--max-ram-gb", "2",
                 "--json", "--output", str(target)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["language"] == "hi" and saved["prefer"] == "accurate" and saved["ok"] is False


def test_bad_input_is_refused_cleanly(world, tmp_path, capsys):
    csv = tmp_path / "u.csv"
    csv.write_text("name,city\nZoë,Kraków\n李雷,北京\n", encoding="utf-8")
    assert main([str(csv)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("offline-stt-router: error:") and "not look like an audio file" in err
    assert main([str(csv), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and "audio file" in payload["error"]
    assert main(["choose", "--language", "not a language at all"]) == 1
    assert "unknown language" in capsys.readouterr().err


def test_transcribe_failure_as_json_carries_the_choice(world, tmp_path, capsys):
    audio = write_wav(tmp_path / "a.wav", tone(440, 0.2))
    assert main(["transcribe", str(audio), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["choice"]["ok"] is False


@pytest.fixture
def cpp_in_unicode_folder(world, tmp_path):
    world.script("whisper-cli", FAKE_WHISPER_CLI)
    folder = tmp_path / "модели 模型"
    world.ggml_model(folder, "base")
    audio = write_wav(tmp_path / "Zoë 李雷.wav", tone(440, 0.3, 44100), rate=44100)
    return folder, audio


def test_transcribe_prints_a_non_ascii_transcript(cpp_in_unicode_folder, capsys):
    folder, audio = cpp_in_unicode_folder
    assert main([str(audio), "--models-dir", str(folder), "--language", "hi"]) == 0
    assert capsys.readouterr().out.startswith(DEVANAGARI + " 16000 Hz")
    assert main(["transcribe", str(audio), "--models-dir", str(folder), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "whisper.cpp" and payload["text"].startswith(DEVANAGARI)


def test_piped_output_survives_an_ascii_console(cpp_in_unicode_folder):
    folder, audio = cpp_in_unicode_folder
    env = dict(os.environ, PYTHONIOENCODING="ascii")
    env.pop("PYTHONUTF8", None)
    for extra in ([], ["--json"]):
        done = subprocess.run(
            [sys.executable, "-m", "offline_stt_router.cli", str(audio), "--models-dir", str(folder)] + extra,
            capture_output=True, env=env, timeout=120,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        text = done.stdout.decode("utf-8")
        assert DEVANAGARI in text
        assert b"Traceback" not in done.stderr
    choose = subprocess.run(
        [sys.executable, "-m", "offline_stt_router.cli", "choose", "--language", "हिन्दी",
         "--models-dir", str(folder)],
        capture_output=True, env=env, timeout=120,
    )
    assert choose.returncode == 0 and "модели" in choose.stdout.decode("utf-8")
