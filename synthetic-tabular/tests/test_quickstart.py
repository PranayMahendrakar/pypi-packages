import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def quickstart_block() -> str:
    section = README.read_text(encoding="utf-8").split("## Quickstart", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    assert match, "README has no python block under ## Quickstart"
    return match.group(1)


def test_readme_quickstart_runs_verbatim(capsys):
    code = quickstart_block()
    exec(compile(code, "README-quickstart", "exec"), {"__name__": "__quickstart__"})
    out = capsys.readouterr().out
    assert "Fidelity score" in out
    for column in ("age", "city", "spend"):
        assert column in out
