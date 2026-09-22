"""The README quickstart block, run exactly as written."""
import re
from pathlib import Path

import pandas as pd

import near_dupes

README = Path(__file__).resolve().parents[1] / "README.md"


def quickstart_block() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    assert match, "README quickstart block not found"
    return match.group(1)


def test_readme_quickstart_runs(capsys):
    code = quickstart_block()
    assert 3 <= len([line for line in code.splitlines() if line.strip()]) <= 6
    namespace = {}
    exec(compile(code, "README-quickstart", "exec"), namespace)
    out = capsys.readouterr().out
    assert "3 texts" in out and "1 duplicate group" in out and "2 kept" in out
    assert "['The quick brown fox jumps over the lazy dog.', 'Something else entirely.']" in out
    result = namespace["result"]
    assert isinstance(result, near_dupes.DuplicateResult)
    assert result.groups == [[0, 1]] and result.keep_indices == [0, 2]


def test_readme_api_dataframe_example():
    df = pd.DataFrame({"name": ["Acme Corp", "ACME Corp.", "Globex"], "city": ["New York", "new york", "Springfield"]})
    assert near_dupes.find_duplicates(df, key=["name", "city"]).groups == [[0, 1]]
    clean = near_dupes.dedupe(df, key=["name", "city"])
    assert list(clean.index) == [0, 2] and list(clean["name"]) == ["Acme Corp", "Globex"]
