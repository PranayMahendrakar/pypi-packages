"""The README quickstart block, run exactly as written."""
import re
from pathlib import Path

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
    assert "ok: True" in out
    assert "(42, 3) (6, 3) (12, 3)" in out
    s = namespace["s"]
    assert len(s.train) + len(s.val) + len(s.test) == 60
    assert s.report().group_overlap["overlapping_groups"] == 0
