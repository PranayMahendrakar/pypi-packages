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
    assert "availability   97.9%" in printed
    assert "oee_partial" in printed and "quality not measured" in printed
    assert "Downtime: stopped for 60 min" in printed
    assert "Slow running: 1 period" in printed
    assert "double count" in printed
    assert "2026-03-02 10:50 to 2026-03-02 11:50  60 min  downtime" in printed


@pytest.mark.skipif(not README.exists(), reason="README.md not next to the tests")
def test_readme_shows_the_real_output(capsys):
    """The output block under the quickstart is what the quickstart prints today."""
    exec(compile(quickstart_block(), "README-quickstart", "exec"), {})
    printed = capsys.readouterr().out.strip()
    text = README.read_text(encoding="utf-8")
    shown = re.search(r"##\s*Quickstart.*?```python\n.*?```\s*\n```\n(.*?)```", text, re.S).group(1)
    assert printed == shown.strip()


@pytest.mark.skipif(not README.exists(), reason="README.md not next to the tests")
def test_quickstart_is_short_and_ascii():
    lines = [line for line in quickstart_block().splitlines() if line.strip()]
    assert 3 <= len(lines) <= 6
    quickstart_block().encode("ascii")


@pytest.mark.skipif(not README.exists(), reason="README.md not next to the tests")
def test_readme_is_honest_about_oee():
    text = README.read_text(encoding="utf-8")
    assert "does not report OEE" in text
    assert "quality needs scrap" in text


def test_three_lines_are_enough():
    import pandas as pd

    import production_anomaly as pa

    t = pd.date_range("2026-03-02 06:00", periods=120, freq="min")
    report = pa.analyze(pd.DataFrame({"time": t, "units": [60] * 50 + [0] * 20 + [60] * 50}))
    assert report.stoppages and report.stoppages[0].minutes == 20
    assert isinstance(report.summary(), str)
