"""The README examples run exactly as written and print exactly what it shows."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from conftest import CALL_DURATION, CALL_SCRIPT, make_call, write_wav

README = Path(__file__).resolve().parent.parent / "README.md"


def _blocks():
    if not README.exists():
        pytest.skip("README.md is not beside the tests")
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"```(\w*)\n(.*?)```", section, flags=re.S)


def test_quickstart_prints_what_the_readme_shows():
    blocks = _blocks()
    code = blocks[0][1]
    shown = blocks[1][1]
    assert blocks[0][0] == "python" and blocks[1][0] == ""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README quickstart", "exec"), {})
    assert out.getvalue().strip() == shown.strip()


def test_stereo_example_runs(tmp_path, monkeypatch):
    code = _blocks()[2][1]
    write_wav(str(tmp_path / "call.wav"), make_call(CALL_SCRIPT, CALL_DURATION, seed=1))
    monkeypatch.chdir(tmp_path)
    scope = {}
    exec(compile("from call_ai_metrics import analyze\n" + code, "README stereo", "exec"), scope)
    report = scope["report"]
    assert list(report.speakers) == ["agent", "customer"]
    assert report["customer"].interruptions == 1
