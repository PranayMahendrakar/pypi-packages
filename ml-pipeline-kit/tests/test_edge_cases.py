"""The awkward cases: failures, empty data, one row, all-NaN, unicode, mutation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline_kit import Pipeline, PipelineError, StepError


def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [31, 45, 29], "city": ["Zurich", "Kyoto", "Lima"]})


def explode(df):
    raise ValueError("the model file is missing")


# ------------------------------------------------------------- failing steps
def test_a_failing_step_names_itself_the_rows_and_chains_the_original():
    pipe = Pipeline("scoring").predict(explode, name="score")
    with pytest.raises(StepError) as caught:
        pipe.run(frame())
    error = caught.value
    assert error.step == "score"
    assert error.rows_in == 3
    text = str(error)
    assert "pipeline 'scoring' step 'score' failed with 3 rows in" in text
    assert "ValueError: the model file is missing" in text
    assert isinstance(error.__cause__, ValueError)


def test_row_count_reported_is_the_count_that_reached_the_step():
    pipe = Pipeline().add(lambda df: df.head(1), name="head").add(explode, name="boom")
    with pytest.raises(StepError) as caught:
        pipe.run(frame())
    assert caught.value.rows_in == 1
    assert "1 rows in" in str(caught.value)


def test_on_error_skip_carries_on_with_the_previous_data_and_records_it():
    pipe = (
        Pipeline("tolerant")
        .preprocess(lambda df: df.assign(clean=True))
        .add(explode, name="enrich", on_error="skip")
        .postprocess(lambda df: df.assign(done=True))
    )
    result = pipe.run(frame())
    assert not result.ok
    assert result.stopped_at is None
    assert list(result.output.columns) == ["age", "city", "clean", "done"]
    skipped = result.step("enrich")
    assert skipped.status == "skip" and skipped.skipped is True
    assert skipped.rows_in == 3 and skipped.rows_out is None
    assert "was skipped" in result.failures[0]
    assert "3 rows in" in result.failures[0]
    assert "skip" in result.summary()


def test_on_error_stop_ends_the_run_and_reports_it_in_the_result():
    pipe = Pipeline("halting").add(explode, name="boom", on_error="stop").add(
        lambda df: df.assign(never=True), name="after"
    )
    result = pipe.run(frame())
    assert not result.ok
    assert result.stopped_at == "boom"
    assert "never" not in result.output.columns
    assert [run.name for run in result.steps] == ["boom"]


def test_call_raises_a_step_error_when_the_run_stopped():
    pipe = Pipeline("service").add(explode, name="boom", on_error="stop")
    with pytest.raises(StepError, match="stopped at step 'boom'"):
        pipe(frame())


def test_call_still_returns_output_for_a_tolerated_skip():
    pipe = Pipeline("service").add(explode, name="boom", on_error="skip")
    out = pipe(frame())
    pd.testing.assert_frame_equal(out, frame())


# ------------------------------------------------------------------ odd data
def test_an_empty_pipeline_returns_the_input_unchanged_and_is_ok():
    data = frame()
    result = Pipeline("nothing").run(data)
    assert result.ok
    assert result.steps == []
    assert result.stopped_at is None
    pd.testing.assert_frame_equal(result.output, frame())
    assert "has no steps" in result.warnings[0]
    assert "0 of 0 steps ran" in result.summary()


def test_an_empty_dataframe_runs_through():
    empty = pd.DataFrame({"age": pd.Series(dtype="int64"), "city": pd.Series(dtype="object")})
    pipe = Pipeline("empty").expect_schema({"age": "int"}).predict(lambda df: df.assign(score=0))
    result = pipe.run(empty)
    assert result.ok
    assert result.rows_in == 0 and result.rows_out == 0
    assert len(result.output) == 0


def test_a_single_row_runs_through():
    one = frame().head(1)
    result = Pipeline("one").expect_range("age", 18, 100).predict(lambda df: df.assign(score=1)).run(one)
    assert result.ok
    assert result.rows_in == 1 and result.rows_out == 1


def test_an_all_nan_column_records_a_note_instead_of_passing_silently():
    data = pd.DataFrame({"age": [np.nan, np.nan, np.nan], "city": ["a", "b", "c"]})
    result = Pipeline("nan").expect_range("age", 18, 100).run(data)
    assert result.ok
    assert "no usable numbers in column 'age'" in result.warnings[0]
    assert "nan" in result.summary()


def test_missing_values_are_not_counted_as_out_of_range():
    data = pd.DataFrame({"age": [31.0, np.nan, 200.0]})
    result = Pipeline().expect_range("age", 18, 100).run(data)
    assert not result.ok
    assert "1 of 3 values above 100" in result.failures[0]
    assert "below" not in result.failures[0]


def test_mixed_dtypes_are_carried_through_untouched():
    data = pd.DataFrame(
        {
            "n": [1, 2],
            "x": [1.5, 2.5],
            "flag": [True, False],
            "when": pd.to_datetime(["2026-01-01", "2026-06-30"]),
            "text": ["a", "b"],
        }
    )
    schema = {"n": "int", "x": "float", "flag": "bool", "when": "datetime", "text": "str"}
    result = Pipeline("mixed").expect_schema(schema).add(lambda df: df).run(data)
    assert result.ok
    pd.testing.assert_frame_equal(result.output, data)


def test_unicode_text_survives_the_summary_and_the_failure_messages():
    data = pd.DataFrame({"ville": ["Zürich", "東京", "Sao Paulo"], "température": [21.5, 33.0, 99.0]})
    pipe = (
        Pipeline("données")
        .expect_schema({"ville": "str", "température": "float"})
        .expect_range("température", 0, 40)
    )
    result = pipe.run(data)
    assert not result.ok
    assert "température" in result.failures[0]
    assert "données" in result.summary()
    assert result.to_dict()["failures"][0].startswith("check 'range[température]'")


def test_rows_are_none_when_the_data_has_no_length():
    result = Pipeline("scalar").add(lambda value: value * 2, name="double").run(21)
    assert result.output == 42
    assert result.rows_in is None and result.rows_out is None
    assert result.step("double").rows_in is None
    assert "? in, ? out" in result.summary()


def test_text_is_one_value_not_one_row_per_character():
    result = Pipeline("text").add(lambda s: s.upper()).run("naïve")
    assert result.output == "NAÏVE"
    assert result.rows_in is None and result.rows_out is None


# ------------------------------------------------------------- no mutation
def test_the_caller_dataframe_is_never_mutated():
    data = frame()
    before = data.copy(deep=True)

    def edit_in_place(df):
        df["age"] = 0
        df.drop(columns=["city"], inplace=True)
        return df

    result = Pipeline("rude").add(edit_in_place).run(data)
    pd.testing.assert_frame_equal(data, before)
    assert list(result.output.columns) == ["age"]


def test_the_caller_list_and_dict_are_never_mutated():
    rows = [{"a": 1}, {"a": 2}]

    def rude(values):
        values.append({"a": 3})
        values[0]["a"] = 99
        return values

    result = Pipeline().add(rude).run(rows)
    assert rows == [{"a": 1}, {"a": 2}]
    assert len(result.output) == 3


def test_a_numpy_array_is_copied_before_the_first_step():
    array = np.array([1, 2, 3])

    def rude(values):
        values[0] = 99
        return values

    Pipeline().add(rude).run(array)
    assert array[0] == 1


# ---------------------------------------------------------------- guard rails
def test_running_an_unbound_pipeline_names_the_steps_to_re_register(tmp_path):
    path = tmp_path / "pipe.json"
    Pipeline("saved").add(lambda df: df, name="clean").save(path)
    loaded = Pipeline.load(path)
    with pytest.raises(PipelineError) as caught:
        loaded.run(frame())
    assert "clean" in str(caught.value)
    assert "pipeline.bind(name, func)" in str(caught.value)


def test_a_tuple_input_is_copied_before_the_first_step():
    """A tuple is a builtin container, so its mutable members are copied too."""
    inner = [1, 2, 3]

    def rude(values):
        values[0].append(99)
        return values

    Pipeline().add(rude).run((inner,))
    assert inner == [1, 2, 3]
