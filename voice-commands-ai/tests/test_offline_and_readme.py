"""No network and no model anywhere; the README quickstart runs as documented."""

import contextlib
import io
import pathlib
import re
import socket

import pytest

import voice_commands_ai
from voice_commands_ai import Commands, parse_number

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "voice_commands_ai"


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("voice-commands-ai tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_everything_works_with_the_network_blocked(no_network):
    cmds = Commands().add("set temperature to {value:number} degrees", lambda value: value)
    assert cmds.run("um set the temperature to twenty one degrees") == 21
    assert cmds.listen(lambda audio: "set temperature to 5 degrees", b"").slots == {"value": 5}
    assert parse_number("a hundred and five") == 105


def test_source_imports_no_network_speech_or_model_code():
    forbidden = re.compile(
        r"^\s*(import|from)\s+(socket|ssl|http|urllib|requests|httpx|aiohttp|subprocess|"
        r"torch|tensorflow|transformers|whisper|vosk|speech_recognition|numpy|scipy)\b",
        re.M,
    )
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not forbidden.search(text), path.name


def test_no_runtime_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"^dependencies = \[\]$", pyproject, re.M)


def test_version():
    assert voice_commands_ai.__version__ == "0.1.0"


def _readme_blocks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick = readme.split("## Quickstart", 1)[1]
    code = re.search(r"```python\n(.*?)```", quick, re.S).group(1)
    expected = re.search(r"```text\n(.*?)```", quick, re.S).group(1)
    return readme, code, expected


def test_readme_first_line_says_it_maps_text_not_speech():
    readme, _, _ = _readme_blocks()
    first = readme.splitlines()[2]
    assert "maps TEXT to actions" in first and "does not recognise speech" in first
    assert "heuristic" in first


def test_readme_quickstart_runs_and_prints_what_it_says():
    _, code, expected = _readme_blocks()
    buffer = io.StringIO()
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "README-quickstart", "exec"), namespace)
    assert buffer.getvalue().strip() == expected.strip()
    # the sentence under the quickstart
    assert namespace["cmds"].match("what time is it") is None


def test_readme_transcriber_example_runs():
    readme, quickstart, _ = _readme_blocks()
    section = readme.split("### Plugging in a transcriber", 1)[1]
    example = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(quickstart + "\n" + example, "README-listen", "exec"), {"__name__": "__main__"})
    assert "switch {'state': 'on', 'device': 'porch light'}" in buffer.getvalue()
