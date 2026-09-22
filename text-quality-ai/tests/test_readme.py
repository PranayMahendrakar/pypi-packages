"""The README quickstart block, run exactly as written."""
import re
from pathlib import Path

import text_quality_ai

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
    assert "Text quality:" in out
    assert "readability" in out and "clarity" in out
    assert "Fix first:" in out
    report = namespace["report"]
    assert isinstance(report, text_quality_ai.QualityReport)
    assert report.grade == "D"
    assert report.suggestions and "passive" in report.suggestions[0]


def test_readme_api_examples():
    grade = text_quality_ai.readability("Short and clear. That is the whole trick.")["flesch_kincaid_grade"]
    assert isinstance(grade, float)
    repeated = text_quality_ai.repetition("dog dog dog cat cat cat dog")["repeated_words"]
    assert repeated[0]["word"] == "dog"
    delta = text_quality_ai.compare("The report was written by us.", "We wrote the report.")["score"]
    assert delta > 0
    assert text_quality_ai.score(["First draft here.", "Second draft here."]).documents[0].grade in "ABCDF"


def test_readme_documents_every_public_name():
    text = README.read_text(encoding="utf-8")
    for name in text_quality_ai.__all__:
        if name == "__version__":
            continue
        assert name in text, f"{name} is not documented in the README"
