"""The README quickstart runs as printed and prints what the README shows."""

from __future__ import annotations

import contextlib
import io
import os

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def _read():
    with open(README, encoding="utf-8") as handle:
        return handle.read()


def _quickstart_blocks():
    section = _read().split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    blocks = []
    current = None
    for line in section.splitlines():
        if current is None and line.startswith("```"):
            current = []
        elif current is not None and line == "```":
            blocks.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    return blocks[0], blocks[1]


def test_quickstart_is_short_and_runs_verbatim():
    code, shown = _quickstart_blocks()
    lines = [line for line in code.splitlines() if line.strip()]
    assert 3 <= len(lines) <= 6
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README-quickstart", "exec"), {})
    printed = out.getvalue()
    assert "2 speakers (estimated from the recording)" in printed
    assert printed.strip() == shown.strip()


def test_readme_sections_in_family_order():
    heads = [line for line in _read().splitlines() if line.startswith("## ")]
    assert heads == ["## Install", "## Quickstart", "## What it does", "## API", "## CLI", "## License"]


def test_readme_says_heuristic_up_front():
    first_paragraph = _read().split("\n\n")[1]
    assert "heuristic" in first_paragraph and "embed" in first_paragraph
