"""The README quickstart, run exactly as written, plus edge-case inputs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def quickstart_code() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(r"##\s*Quickstart\s*\n+```(?:python)?\n(.*?)```", text, re.S)
    assert match, "README has no Quickstart python block"
    return match.group(1)


def test_readme_quickstart_runs_verbatim(world, capsys):
    code = quickstart_code()
    assert 3 <= len([line for line in code.splitlines() if line.strip()]) <= 6
    exec(compile(code, "README-quickstart", "exec"), {"__name__": "__main__"})
    out = capsys.readouterr().out
    assert "faster-whisper" in out and "Checked:" in out and "Machine:" in out


def test_readme_sections_are_in_family_order():
    headings = [line for line in README.read_text(encoding="utf-8").splitlines() if line.startswith("## ")]
    assert headings == ["## Install", "## Quickstart", "## What it does", "## API", "## CLI", "## License"]
    first_paragraph = README.read_text(encoding="utf-8").split("\n\n")[1]
    assert "heuristic" in first_paragraph


def test_single_sample_and_all_nan_audio(world, tmp_path):
    from test_transcribe import FAKE_WHISPER_CLI

    router = world.router()
    router.register("counter", lambda: {"quality": 0.9}, lambda audio: "{0} samples".format(len(audio)))
    assert router.transcribe([0.5]) == "1 samples"
    world.script("whisper-cli", FAKE_WHISPER_CLI)
    world.ggml_model(world.home / ".cache" / "whisper.cpp", "base")
    only_cpp = world.router()
    text = only_cpp.transcribe([float("nan")] * 1600, sample_rate=16000)
    assert "16000 Hz 1 ch 16 bit" in text  # NaN became silence, nothing crashed


def test_every_public_name_is_exported():
    import offline_stt_router as stt

    for name in stt.__all__:
        assert hasattr(stt, name), name
    assert stt.__version__ == "0.1.0"
    assert callable(stt.available) and callable(stt.choose) and callable(stt.register)
    assert callable(stt.transcribe) and callable(stt.unregister) and callable(stt.machine)
    with pytest.raises(TypeError):
        stt.Router(machine="big")
