"""The README quickstart must run exactly as printed."""
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


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
    assert "rag-chunker:" in out
    assert "chunks;" in out


@pytest.mark.skipif(not README.exists(), reason="README not shipped with the wheel")
def test_readme_has_the_required_sections():
    text = README.read_text(encoding="utf-8")
    for heading in ("## Install", "## Quickstart", "## What it does", "## API",
                    "## CLI", "## License"):
        assert heading in text
