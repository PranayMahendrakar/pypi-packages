"""validate(), expect_schema() and expect_range()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline_kit import Pipeline, ValidationError


def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [31, 45, 29], "city": ["Zurich", "Kyoto", "Lima"]})


# ------------------------------------------------------------------- validate
def test_a_check_returning_false_stops_the_run_and_is_named():
    pipe = (
        Pipeline("checked")
        .validate(lambda df: len(df) > 10, name="enough_rows")
        .predict(lambda df: df.assign(score=1))
    )
    result = pipe.run(frame())
    assert not result.ok
    assert result.stopped_at == "enough_rows"
    assert "enough_rows" in result.failures[0]
    assert "score" not in result.output.columns
    assert [run.name for run in result.steps] == ["enough_rows"]


def test_a_check_can_return_a_message():
    def has_city(df):
        return "city" in df.columns, "the city column is required for routing"

    result = Pipeline().validate(has_city).run(pd.DataFrame({"age": [1]}))
    assert not result.ok
    assert "the city column is required for routing" in result.failures[0]
    assert result.step("has_city").error.endswith("routing")


def test_a_warn_check_records_but_does_not_stop():
    pipe = (
        Pipeline("soft")
        .validate(lambda df: False, name="tired", severity="warn")
        .predict(lambda df: df.assign(score=1))
    )
    result = pipe.run(frame())
    assert result.ok
    assert result.stopped_at is None
    assert "tired" in result.warnings[0]
    assert "score" in result.output.columns
    assert result.step("tired").status == "warn"


def test_a_check_that_raises_is_a_failed_check_not_a_crash():
    def broken(df):
        raise KeyError("missing_column")

    result = Pipeline().validate(broken).run(frame())
    assert not result.ok
    assert "the check raised KeyError" in result.step("broken").error


def test_a_check_may_return_a_boolean_mask():
    result = Pipeline().validate(lambda df: df["age"] > 30, name="adults").run(frame())
    assert not result.ok
    assert "1 of 3 rows did not pass" in result.failures[0]

    passing = Pipeline().validate(lambda df: df["age"] > 10, name="adults").run(frame())
    assert passing.ok


def test_a_check_returning_none_is_reported_as_a_mistake():
    result = Pipeline().validate(lambda df: None, name="oops").run(frame())
    assert not result.ok
    assert "returned None" in result.step("oops").error


def test_validate_rejects_a_non_callable_and_a_bad_severity():
    with pytest.raises(TypeError, match="callable"):
        Pipeline().validate("not a function")
    with pytest.raises(ValueError, match="severity"):
        Pipeline().validate(lambda d: True, severity="loud")


def test_call_raises_validation_error_when_a_check_stops_the_run():
    pipe = Pipeline("service").validate(lambda df: False, name="never")
    with pytest.raises(ValidationError) as caught:
        pipe(frame())
    assert caught.value.check == "never"
    assert "stopped at check 'never'" in str(caught.value)


# --------------------------------------------------------------- expect_schema
def test_schema_accepts_matching_columns_and_dtypes():
    result = Pipeline().expect_schema({"age": "int", "city": "str"}).run(frame())
    assert result.ok


def test_schema_names_a_missing_column_and_a_wrong_dtype():
    data = pd.DataFrame({"age": ["31", "45"]})
    result = Pipeline().expect_schema({"age": "int", "city": "str"}).run(data)
    assert not result.ok
    message = result.failures[0]
    # The dtype name is pandas', not ours: a column of strings is "object" on pandas 2
    # and "str" on pandas 3. Asserting the spelling tests pandas, so assert the contract
    # instead - which column, and what was expected.
    assert "column 'age' has dtype" in message and "expected int" in message
    assert "column 'city' is missing" in message


def test_schema_accepts_a_list_of_names_a_string_and_a_frame():
    assert Pipeline().expect_schema(["age", "city"]).run(frame()).ok
    assert Pipeline().expect_schema("age").run(frame()).ok
    assert Pipeline().expect_schema(frame()).run(frame()).ok


def test_schema_matches_dtypes_by_family():
    data = pd.DataFrame({"n": np.array([1, 2], dtype="int32"), "x": [1.5, 2.5]})
    assert Pipeline().expect_schema({"n": "int64", "x": "number"}).run(data).ok
    assert not Pipeline().expect_schema({"n": "float"}).run(data).ok


def test_schema_on_something_that_is_not_a_table():
    result = Pipeline().expect_schema(["age"]).run([1, 2, 3])
    assert not result.ok
    assert "got list" in result.failures[0]


def test_schema_rejects_an_empty_or_unreadable_spec():
    with pytest.raises(ValueError, match="at least one column"):
        Pipeline().expect_schema([])
    with pytest.raises(ValueError, match="needs a schema"):
        Pipeline().expect_schema(None)
    with pytest.raises(TypeError):
        Pipeline().expect_schema(42)


# ---------------------------------------------------------------- expect_range
def test_range_passes_inside_the_bounds_and_names_the_offenders():
    assert Pipeline().expect_range("age", 18, 100).run(frame()).ok

    result = Pipeline().expect_range("age", 40, 44).run(frame())
    assert not result.ok
    message = result.failures[0]
    assert "2 of 3 values below 40" in message
    assert "worst 29 at row 2" in message
    assert "1 of 3 values above 44" in message


def test_range_accepts_one_open_side_and_a_bare_series():
    assert Pipeline().expect_range("age", low=18).run(frame()).ok
    assert Pipeline().expect_range("age", high=100).run(frame()).ok
    assert Pipeline().expect_range(None, 0, 10).run(pd.Series([1, 2, 3])).ok


def test_range_reports_a_missing_column_and_a_text_column():
    result = Pipeline().expect_range("weight", 0, 1).run(frame())
    assert "column 'weight' is missing" in result.failures[0]

    text = Pipeline().expect_range("city", 0, 1).run(frame())
    assert "not numeric" in text.failures[0]


def test_range_needs_a_column_for_a_table_and_at_least_one_bound():
    result = Pipeline().expect_range(None, 0, 10).run(frame())
    assert "needs a column name" in result.failures[0]
    with pytest.raises(ValueError, match="at least one of low"):
        Pipeline().expect_range("age")
    with pytest.raises(ValueError, match="above high"):
        Pipeline().expect_range("age", 10, 1)
    with pytest.raises(ValueError, match="needs a number"):
        Pipeline().expect_range("age", "young")


def test_checks_run_in_place_between_steps():
    pipe = (
        Pipeline("ordered")
        .preprocess(lambda df: df.assign(age=df["age"] + 100))
        .expect_range("age", 18, 100)
        .predict(lambda df: df.assign(score=1))
    )
    result = pipe.run(frame())
    assert not result.ok
    assert result.stopped_at == "range[age]"
    assert [run.name for run in result.steps] == ["preprocess", "range[age]"]


# ----------------------------------------- reviewer regressions: partial reads
def test_range_warns_about_values_it_could_not_read_as_numbers():
    """Half a column being junk must be recorded, not silently dropped."""
    messy = pd.DataFrame({"price": ["10.5", "n/a", "20.0", "free"]})
    result = Pipeline("g").expect_range("price", 0, 1000).run(messy)
    assert result.ok
    assert not result.failures
    assert len(result.warnings) == 1
    note = result.warnings[0]
    assert "range[price]" in note
    assert "2 of 4 values that are not numbers" in note


def test_range_reports_both_the_out_of_range_rows_and_the_unreadable_ones():
    messy = pd.DataFrame({"price": ["10.5", "free", "-999"]})
    result = Pipeline().expect_range("price", 0, 1000).run(messy)
    assert not result.ok
    detail = result.failures[0]
    assert "1 of 3 values below 0" in detail
    assert "1 of 3 values that are not numbers" in detail


def test_range_does_not_warn_when_every_value_reads_as_a_number():
    clean = pd.DataFrame({"price": ["10.5", "20.0"]})
    result = Pipeline().expect_range("price", 0, 1000).run(clean)
    assert result.ok
    assert result.warnings == []


def test_range_ignores_missing_values_without_calling_them_unreadable():
    gappy = pd.DataFrame({"price": [10.5, np.nan, 20.0]})
    result = Pipeline().expect_range("price", 0, 1000).run(gappy)
    assert result.ok
    assert result.warnings == []


@pytest.mark.parametrize(
    "column, label",
    [
        (pd.to_datetime(["2024-01-01", "2024-02-01"]), "datetime64[ns]"),
        (pd.to_timedelta([1, 2], unit="D"), "timedelta64[ns]"),
    ],
)
def test_range_calls_a_date_or_duration_a_type_problem_not_a_big_number(column, label):
    """Epoch nanoseconds are never compared against the caller's bounds."""
    result = Pipeline().expect_range("d", 0, 10).run(pd.DataFrame({"d": column}))
    assert not result.ok
    detail = result.failures[0]
    # Likewise the resolution: pandas 2 says datetime64[ns], pandas 3 says
    # datetime64[us], and a timedelta comes back as [s]. The point of the check is that
    # a date is refused as non-numeric, not how pandas spells its dtype today.
    assert "column 'd' is not numeric (dtype" in detail
    assert label.split("[")[0] in detail
    assert "above 10" not in detail


def test_an_all_missing_date_column_records_the_no_numbers_note():
    empty_dates = pd.DataFrame({"d": pd.to_datetime([None, None])})
    result = Pipeline().expect_range("d", 0, 10).run(empty_dates)
    assert result.ok
    assert "no usable numbers" in result.warnings[0]
