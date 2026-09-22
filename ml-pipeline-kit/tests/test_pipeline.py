"""The ordinary path: building a pipeline, running it, and reading the result."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml_pipeline_kit
from ml_pipeline_kit import Pipeline, Result, StepRun


def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [31, 45, 29], "income": [50000, 82000, 41000]})


def test_quickstart_from_the_readme():
    customers = frame()
    pipe = (
        Pipeline("scoring")
        .expect_range("age", 18, 100)
        .predict(lambda df: df.assign(score=df["income"] / 1000))
    )
    result = pipe.run(customers)
    assert result.ok
    assert isinstance(result, Result)
    assert list(result.output["score"]) == [50.0, 82.0, 41.0]
    assert "pipeline 'scoring' ok" in result.summary()
    assert result.rows_in == 3 and result.rows_out == 3


def test_steps_run_in_order_and_are_timed():
    pipe = (
        Pipeline("chain")
        .preprocess(lambda df: df.assign(a=1))
        .predict(lambda df: df.assign(b=df["a"] + 1))
        .postprocess(lambda df: df.assign(c=df["b"] + 1))
    )
    result = pipe.run(frame())
    assert [run.kind for run in result.steps] == ["preprocess", "predict", "postprocess"]
    assert list(result.output["c"]) == [3, 3, 3]
    assert list(result.timings) == ["preprocess", "predict", "postprocess"]
    assert all(value >= 0.0 for value in result.timings.values())
    assert result.duration_ms is not None


def test_step_names_come_from_the_callable_then_the_role():
    def clean(df):
        return df

    pipe = Pipeline().add(clean).predict(lambda df: df).preprocess(lambda df: df)
    assert pipe.step_names == ["clean", "predict", "preprocess"]


def test_duplicate_names_are_made_unique():
    pipe = Pipeline().add(lambda d: d).add(lambda d: d).add(lambda d: d)
    assert pipe.step_names == ["step", "step-2", "step-3"]


def test_collect_timings_false_leaves_durations_empty():
    result = Pipeline().add(lambda d: d).run(frame(), collect_timings=False)
    assert result.timings == {}
    assert result.steps[0].duration_ms is None
    assert result.duration_ms is None
    assert "ms" not in result.summary().splitlines()[0]


def test_call_returns_the_output_alone():
    pipe = Pipeline("service").predict(lambda df: df.assign(score=1))
    out = pipe(frame())
    assert isinstance(out, pd.DataFrame)
    assert list(out["score"]) == [1, 1, 1]


def test_row_counts_per_step():
    pipe = Pipeline().add(lambda df: df.head(2), name="head")
    result = pipe.run(frame())
    run = result.step("head")
    assert (run.rows_in, run.rows_out) == (3, 2)
    assert result.rows_out == 2


def test_module_level_run_is_the_one_line_form():
    result = ml_pipeline_kit.run(frame(), lambda df: df.assign(x=1))
    assert result.ok and "x" in result.output.columns
    assert result.name == "pipeline"


def test_works_on_a_plain_list_and_on_a_numpy_array():
    result = Pipeline("lists").add(lambda rows: [r * 2 for r in rows]).run([1, 2, 3])
    assert result.output == [2, 4, 6]
    assert result.rows_in == 3 and result.rows_out == 3

    array_result = Pipeline("array").add(lambda a: a + 1).run(np.array([1, 2, 3]))
    assert list(array_result.output) == [2, 3, 4]


def test_step_run_fields_and_pipeline_len():
    pipe = Pipeline().add(lambda d: d, name="only")
    assert len(pipe) == 1
    run = pipe.run(frame()).steps[0]
    assert isinstance(run, StepRun)
    assert (run.name, run.ok, run.error, run.status) == ("only", True, None, "ok")


def test_add_rejects_a_non_callable_and_a_bad_on_error():
    with pytest.raises(TypeError, match="callable"):
        Pipeline().add(42)
    with pytest.raises(ValueError, match="on_error"):
        Pipeline().add(lambda d: d, on_error="explode")


def test_duplicate_columns_are_named_at_the_entry_point():
    bad = pd.DataFrame([[1, 2]], columns=["age", "age"])
    with pytest.raises(ValueError, match="duplicate column names: age"):
        Pipeline().add(lambda d: d).run(bad)
