"""The README quickstart runs exactly as written and does what the README promises."""
import contextlib
import io
import json
import re
from pathlib import Path

import pandas as pd

from smartclean_df import Action, CleanResult, clean

README = Path(__file__).resolve().parents[1] / "README.md"


def quickstart_block() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    assert match, "README has no python block under ## Quickstart"
    return match.group(1)


def test_readme_quickstart_runs_verbatim():
    namespace = {}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(quickstart_block(), "README quickstart", "exec"), namespace)
    result = namespace["result"]
    assert isinstance(result, CleanResult)
    printed = buffer.getvalue()
    assert printed.startswith("smartclean-df: 4 rows x 3 columns -> 3 rows x 3 columns")
    assert list(result.df.columns) == ["name", "age", "joined"]
    assert result.df["name"].tolist() == ["Ann", "Bob", "Ann"]
    assert result.df["age"].tolist() == [34.0, 37.5, 41.0]
    assert str(result.df["joined"].dtype).startswith("datetime64")
    assert result.df["joined"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-02-10"),
        pd.Timestamp("2024-02-10"),
    ]
    assert [a.kind for a in result.actions] == [
        "rename_column",
        "rename_column",
        "rename_column",
        "strip_whitespace",
        "missing_tokens",
        "missing_tokens",
        "parse_numeric",
        "parse_datetime",
        "drop_duplicates",
        "impute",
        "impute",
        "impute",
    ]
    assert all(isinstance(a, Action) for a in result.actions)
    json.dumps(result.to_dict())


def test_result_unpacks_to_df_and_actions():
    df, actions = clean(pd.DataFrame({"A": [1, 1, 2]}))
    assert isinstance(df, pd.DataFrame)
    assert [a.kind for a in actions] == ["rename_column", "drop_duplicates"]
    assert df["a"].tolist() == [1, 2]
