"""The README quickstart must run exactly as printed, and print what it claims."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def quickstart_block() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(r"##\s*Quickstart\s*\n+```(?:python)?\n(.*?)```", text, re.S | re.I)
    assert match, "README.md has no ## Quickstart python block"
    return match.group(1)


@pytest.mark.skipif(not README.exists(), reason="README.md not next to the tests")
def test_quickstart_runs_verbatim(capsys):
    exec(compile(quickstart_block(), "README-quickstart", "exec"), {})
    printed = capsys.readouterr().out
    assert "machine health:" in printed
    assert "components (score, weight, points lost):" in printed
    for component in ("stability", "compliance", "anomaly", "availability"):
        assert component in printed


@pytest.mark.skipif(not README.exists(), reason="README.md not next to the tests")
def test_quickstart_is_short_and_ascii():
    lines = [line for line in quickstart_block().splitlines() if line.strip()]
    assert len(lines) <= 10
    quickstart_block().encode("ascii")  # the copy-pasteable block stays plain ASCII


def test_three_lines_are_enough():
    """The advertised shape: import, data, score."""
    import pandas as pd

    from machine_health import score

    result = score(pd.DataFrame({"temp": [60, 61, 60, 59, 60, 61, 88, 90, 91, 92]}))
    assert 0 <= result.value <= 100
    assert result.grade in {"A", "B", "C", "D", "F"}
    assert isinstance(result.summary(), str)
