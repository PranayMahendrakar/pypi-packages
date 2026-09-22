"""Runs the README Quickstart block verbatim and checks the documented output, so docs and code cannot drift."""

from pathlib import Path

import pandas as pd

README = Path(__file__).resolve().parents[1] / "README.md"


def quickstart_blocks():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("## What it checks", 1)[0]
    code = section.split("```python", 1)[1].split("```", 1)[0]
    rest = section.split("```python", 1)[1].split("```", 1)[1]
    expected_output = rest.split("```\n", 1)[1].split("```", 1)[0]
    return code, expected_output


def test_readme_quickstart_runs_and_prints_documented_output(capsys):
    code, expected_output = quickstart_blocks()
    assert 3 <= len([line for line in code.strip().splitlines() if line.strip()]) <= 6
    namespace = {}
    exec(compile(code, str(README), "exec"), namespace)
    out = capsys.readouterr().out
    assert out.strip() == expected_output.strip()
    assert out.startswith("Schema validation: FAILED - 2 rows x 3 columns (4 errors, 2 warnings)")

    fixed, new, schema = namespace["fixed"], namespace["new"], namespace["schema"]
    assert list(fixed.columns) == ["id", "city", "score"] == schema.column_names
    assert str(fixed["id"].dtype) == "Int64" and fixed["id"].tolist()[0] == 4 and pd.isna(fixed["id"].iloc[1])
    assert list(new.columns) == ["city", "id", "score"]  # the input was not touched
    assert str(new["id"].dtype) == "float64"


def test_readme_api_section_matches_public_surface():
    import schema_guard

    text = README.read_text(encoding="utf-8")
    for name in ("infer", "load", "validate", "enforce", "guard", "Schema", "ColumnSpec", "ValidationResult", "Problem", "SchemaError"):
        assert name in text and hasattr(schema_guard, name)
    order = [text.index(h) for h in ("## Install", "## Quickstart", "## What it checks", "## API", "## CLI", "## License")]
    assert order == sorted(order)
