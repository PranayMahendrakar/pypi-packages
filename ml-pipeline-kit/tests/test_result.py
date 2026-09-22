"""The result object: summary(), to_dict() and the little accessors."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ml_pipeline_kit import Pipeline


def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [31, 45, 29], "city": ["Zurich", "Kyoto", "Lima"]})


def built() -> Pipeline:
    return (
        Pipeline("report")
        .expect_schema({"age": "int"})
        .preprocess(lambda df: df.assign(band=df["age"] // 10))
        .validate(lambda df: True, name="always", severity="warn")
        .predict(lambda df: df.assign(score=df["band"] * 2))
    )


def test_summary_reads_as_a_report():
    text = built().run(frame()).summary()
    lines = text.splitlines()
    assert lines[0].startswith("ml-pipeline-kit: pipeline 'report' ok, 4 of 4 steps ran in ")
    assert "  rows      : 3 in, 3 out" in lines
    assert "  steps     :" in lines
    assert any("1. schema" in line and "schema" in line for line in lines)
    assert any("3 rows in, 3 rows out" in line for line in lines)


def test_summary_is_plain_ascii_and_encodable_anywhere():
    text = built().run(frame()).summary()
    text.encode("ascii")
    for character in ("→", "•", "─", "✓"):
        assert character not in text


def test_summary_shows_failures_and_the_stopping_point():
    pipe = Pipeline("bad").expect_schema({"missing": "int"}).predict(lambda df: df)
    text = pipe.run(frame()).summary()
    assert "FAILED, 1 of 2 steps ran" in text
    assert "  stopped at: 'schema'" in text
    assert "  failures  :" in text
    assert "column 'missing' is missing" in text


def test_to_dict_is_json_safe_and_complete():
    result = built().run(frame())
    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text) == payload
    assert payload["name"] == "report"
    assert payload["ok"] is True
    assert payload["n_planned"] == 4 and payload["n_steps"] == 4
    assert payload["rows_in"] == 3 and payload["rows_out"] == 3
    assert [step["name"] for step in payload["steps"]] == ["schema", "preprocess", "always", "predict"]
    assert set(payload["timings"]) == {"schema", "preprocess", "always", "predict"}
    assert "output" not in payload


def test_step_lookup_by_name_and_its_error_message():
    result = built().run(frame())
    assert result.step("predict").kind == "predict"
    with pytest.raises(KeyError, match="no step named 'nope'"):
        result.step("nope")


def test_failed_steps_lists_only_what_did_not_pass():
    pipe = Pipeline().add(lambda df: df, name="fine").validate(lambda df: False, name="nope")
    result = pipe.run(frame())
    assert [run.name for run in result.failed_steps] == ["nope"]
    assert result.step("nope").to_dict()["status"] == "FAIL"
