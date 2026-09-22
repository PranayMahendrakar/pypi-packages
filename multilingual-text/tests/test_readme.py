"""The README quickstart must run exactly as printed."""
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

REQUIRED_SECTIONS = (
    "## Install",
    "## Quickstart",
    "## What it does",
    "## API",
    "## CLI",
    "## License",
)


def quickstart_block():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    assert match, "README needs a python quickstart block"
    return match.group(1)


@pytest.mark.skipif(not README.exists(), reason="README not shipped with the wheel")
def test_quickstart_runs_verbatim(capsys):
    namespace = {}
    exec(compile(quickstart_block(), "README-quickstart", "exec"), namespace)
    out = capsys.readouterr().out
    assert "multilingual-text: es (Spanish)" in out
    assert '"Smart" quotes - and hard spaces' in out
    assert "namaste duniya | Privet, mir!" in out
    assert "Arabic | rtl: True | empty: und" in out


@pytest.mark.skipif(not README.exists(), reason="README not shipped with the wheel")
def test_quickstart_output_is_pure_ascii(capsys):
    exec(compile(quickstart_block(), "README-quickstart", "exec"), {})
    assert capsys.readouterr().out.isascii()


@pytest.mark.skipif(not README.exists(), reason="README not shipped with the wheel")
def test_readme_has_the_required_sections_in_order():
    text = README.read_text(encoding="utf-8")
    positions = []
    for heading in REQUIRED_SECTIONS:
        assert heading in text, heading
        positions.append(text.index(heading))
    assert positions == sorted(positions)


@pytest.mark.skipif(not README.exists(), reason="README not shipped with the wheel")
def test_readme_states_which_scripts_transliterate_covers():
    text = README.read_text(encoding="utf-8")
    assert "Devanagari" in text and "Cyrillic" in text and "Greek" in text
    assert "returned completely unchanged" in text
