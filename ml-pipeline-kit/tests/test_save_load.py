"""save() records configuration, never callables; load() says so."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ml_pipeline_kit import Pipeline, PipelineError


def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [31, 45, 29], "city": ["Zurich", "Kyoto", "Lima"]})


def clean(df):
    return df.assign(age=df["age"].astype("int64"))


def score(df):
    return df.assign(score=df["age"] * 2)


def built() -> Pipeline:
    return (
        Pipeline("scoring")
        .expect_schema({"age": "int", "city": "str"})
        .preprocess(clean)
        .expect_range("age", 18, 100)
        .predict(score, on_error="skip")
        .validate(lambda df: "score" in df.columns, name="scored", severity="warn")
    )


def test_saved_file_holds_names_and_configuration_but_no_callables(tmp_path):
    path = built().save(tmp_path / "scoring.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "ml-pipeline-kit/pipeline"
    assert payload["name"] == "scoring"
    assert "re-register" in payload["note"]
    names = [step["name"] for step in payload["steps"]]
    assert names == ["schema", "clean", "range[age]", "score", "scored"]
    assert payload["steps"][0]["config"]["schema"] == {"age": "int", "city": "str"}
    assert payload["steps"][2]["config"] == {"column": "age", "low": 18.0, "high": 100.0}
    assert payload["steps"][3]["on_error"] == "skip"
    assert payload["steps"][4]["severity"] == "warn"
    text = path.read_text(encoding="utf-8")
    assert "def " not in text and "lambda" not in text


def test_load_restores_checks_and_names_the_steps_that_need_binding(tmp_path):
    path = built().save(tmp_path / "scoring.json")
    loaded = Pipeline.load(path)
    assert loaded.name == "scoring"
    assert loaded.step_names == ["schema", "clean", "range[age]", "score", "scored"]
    assert loaded.unbound == ["clean", "score", "scored"]
    assert "needs bind()" in loaded.describe()

    with pytest.raises(PipelineError, match="clean, score, scored"):
        loaded.run(frame())

    loaded.bind("clean", clean).bind("score", score).bind("scored", lambda df: "score" in df.columns)
    assert loaded.unbound == []
    result = loaded.run(frame())
    assert result.ok
    assert list(result.output["score"]) == [62, 90, 58]


def test_round_trip_keeps_the_saved_description_identical(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    built().save(first)
    Pipeline.load(first).save(second)
    assert json.loads(first.read_text(encoding="utf-8")) == json.loads(
        second.read_text(encoding="utf-8")
    )


def test_bind_rejects_unknown_names_non_callables_and_config_checks(tmp_path):
    loaded = Pipeline.load(built().save(tmp_path / "s.json"))
    with pytest.raises(KeyError, match="no step named 'ghost'"):
        loaded.bind("ghost", clean)
    with pytest.raises(TypeError, match="needs a callable"):
        loaded.bind("clean", 3)
    with pytest.raises(ValueError, match="needs no callable"):
        loaded.bind("schema", clean)


def test_load_rejects_a_file_that_is_not_a_pipeline(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an ml-pipeline-kit pipeline file"):
        Pipeline.load(path)

    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps({"format": "ml-pipeline-kit/pipeline", "steps": [{"kind": "wizardry"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown step kind"):
        Pipeline.load(broken)


def test_save_keeps_unicode_readable(tmp_path):
    path = Pipeline("données").expect_schema({"température": "float"}).save(tmp_path / "u.json")
    text = path.read_text(encoding="utf-8")
    assert "température" in text
    assert Pipeline.load(path).name == "données"


def test_describe_lists_every_step():
    text = built().describe()
    assert text.startswith("pipeline 'scoring' with 5 step(s)")
    assert "age:int, city:str" in text
    assert "age in 18.0 to 100.0" in text
    assert "on_error=skip" in text
    assert "severity=warn" in text


def test_describe_of_an_empty_pipeline():
    assert "(no steps yet)" in Pipeline().describe()
