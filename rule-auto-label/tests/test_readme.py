"""The README quickstart must run exactly as written."""

import pathlib
import re

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def quickstart_block() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("## ", 1)[0]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    assert match, "README has no python block under ## Quickstart"
    return match.group(1)


def test_quickstart_runs_and_labels_everything(capsys):
    code = quickstart_block()
    namespace = {}
    exec(compile(code, str(README), "exec"), namespace)
    result = namespace["result"]
    assert result.coverage == 1.0
    assert result.labels[:6] == ["spam", "work", "spam", "work", "spam", "work"]
    assert result.source[:6] == ["rule"] * 6
    assert result.source[6:] == ["model", "model"]
    assert result.labels[6:] == ["spam", "work"]
    out = capsys.readouterr().out
    assert "auto-label: 8/8 items labeled (100.0%)" in out
    assert "confidence" in out


def test_readme_section_order():
    text = README.read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, re.M)
    assert headings == ["Install", "Quickstart", "What it does", "API", "CLI", "License"]
    # The distribution is rule-auto-label; the import name stays auto_label.
    # PyPI blocks the plain name as too close to 'autolabel' and 'auto-labeler'.
    assert text.startswith("# rule-auto-label\n")
