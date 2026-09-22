"""Runs the README Quickstart block verbatim, so the docs and the code cannot drift apart."""

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def quickstart_code() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1]
    return section.split("```python", 1)[1].split("```", 1)[0]


def test_readme_quickstart_runs_and_detects_drift(capsys):
    code = quickstart_code()
    assert 3 <= len([line for line in code.strip().splitlines() if line.strip()]) <= 6
    namespace = {}
    exec(compile(code, str(README), "exec"), namespace)
    out = capsys.readouterr().out
    assert out.startswith("data-drift-lite: DRIFT DETECTED: 2 of 2 columns drifted (100%)")
    report = namespace["report"]
    assert report.drifted and report.drifted_columns == ["age", "plan"]
    assert "fewer than" not in out  # 20 rows a side: no tiny-batch note in the quickstart
