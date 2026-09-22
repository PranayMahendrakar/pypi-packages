import json

import numpy as np
import pandas as pd
import pytest

import schema_guard
from schema_guard import PROBLEM_KINDS, Problem, Schema, SchemaError, ValidationResult


def kinds(result):
    return [problem.kind for problem in result.problems]


def test_clean_frame_is_ok(train, schema):
    result = schema.validate(train)
    assert result.ok and result.problems == [] and result.errors == [] and result.warnings == []
    assert result.summary() == "Schema validation: OK - 4 rows x 6 columns (no problems)"
    assert result.to_dict() == {"ok": True, "n_rows": 4, "n_columns": 6, "n_errors": 0, "n_warnings": 0, "problems": []}
    assert result.raise_if_invalid() is result


def test_missing_column(train, schema):
    result = schema.validate(train.drop(columns=["score"]))
    assert not result.ok and kinds(result) == ["missing_column"]
    problem = result.problems[0]
    assert problem.column == "score" and problem.detail == {"expected_position": 2, "family": "float"}
    assert str(problem) == "[error] missing_column 'score': column is missing (expected at position 2)"


def test_extra_column(train, schema):
    result = schema.validate(train.assign(extra=1))
    assert not result.ok and kinds(result) == ["extra_column"]
    assert result.problems[0].column == "extra" and result.problems[0].detail["position"] == 6


def test_wrong_order(train, schema):
    result = schema.validate(train[["city", "id", "score", "active", "signup", "tier"]])
    assert not result.ok and kinds(result) == ["wrong_order"]
    problem = result.problems[0]
    assert problem.column is None
    assert problem.detail["expected"][:2] == ["id", "city"] and problem.detail["actual"][:2] == ["city", "id"]


def test_dtype_mismatch_int_arriving_as_float_with_nan(train, schema):
    result = schema.validate(train.assign(id=[1.0, np.nan, 3.0, 4.0]))
    assert kinds(result) == ["dtype_mismatch", "unexpected_null"]
    mismatch = result.problems[0]
    assert mismatch.detail["expected"] == "int" and mismatch.detail["actual_family"] == "float"
    assert mismatch.detail["whole_numbers"] is True and "Int64" in mismatch.message
    assert result.problems[1].detail == {"null_count": 1, "null_fraction": 0.25}


def test_dtype_mismatch_string_arriving_as_int():
    # The zip repeats, so a category set is recorded (an all-distinct column would not
    # record one - see test_infer.py::test_all_distinct_text_column_records_no_categories).
    schema = Schema.infer(pd.DataFrame({"zip": ["02134", "10001", "02134"]}))
    assert schema["zip"].categories == ["02134", "10001"]
    result = schema.validate(pd.DataFrame({"zip": [2134, 10001]}))
    assert kinds(result) == ["dtype_mismatch", "unknown_category"]
    assert result.problems[0].message == "expected string, got int64 (int)"


def test_dtype_mismatch_datetime_arriving_as_strings(train, schema):
    result = schema.validate(train.assign(signup=train["signup"].dt.strftime("%Y-%m-%d")))
    assert kinds(result) == ["dtype_mismatch"]
    assert result.problems[0].detail["actual_family"] == "string"


def test_dtype_mismatch_category_arriving_as_object_and_vice_versa(train, schema):
    result = schema.validate(train.assign(tier=train["tier"].astype(object)))
    assert kinds(result) == ["dtype_mismatch"]
    assert result.problems[0].message == "expected category, got object (string)"
    result = schema.validate(train.assign(city=train["city"].astype("category")))
    assert kinds(result) == ["dtype_mismatch"] and result.problems[0].column == "city"


def test_object_column_of_python_ints_is_reported_but_does_not_fail(train, schema):
    # Right values, wrong container. It is worth saying - a real dtype is what the rest of
    # the stack wants - but it cannot be an error, or the frame a schema was inferred from
    # would fail that schema (see test_regressions_round2.py).
    result = schema.validate(train.assign(id=pd.Series([1, 2, 3, 4], dtype=object)))
    assert kinds(result) == ["dtype_mismatch"]
    problem = result.problems[0]
    assert result.ok and problem.severity == "warning" and problem.detail["object_backed"] is True
    assert problem.message == "expected a real int dtype, got object holding int values; enforce() converts it"


def test_unknown_category(train, schema):
    result = schema.validate(train.assign(city=["Oslo", "Lima", "Lima", "Rome"], tier=pd.Categorical(["a", "b", "z", "a"])))
    assert kinds(result) == ["unknown_category", "unknown_category"]
    city = result.problems[0]
    assert city.detail == {"unknown_values": ["Lima"], "n_unknown_values": 1, "unknown_count": 2}
    assert result.problems[1].column == "tier" and result.problems[1].detail["unknown_values"] == ["z"]


def test_unexpected_null_only_for_non_nullable_columns():
    schema = Schema.infer(pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]}))
    result = schema.validate(pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, 2.0]}))
    assert kinds(result) == ["unexpected_null"] and result.problems[0].column == "b"


def test_out_of_range_is_a_warning_unless_strict(train, schema):
    batch = train.assign(score=[0.5, 5.0, 0.7, -1.0], id=[0, 2, 3, 4])
    result = schema.validate(batch)
    assert result.ok
    assert kinds(result) == ["out_of_range", "out_of_range"]
    assert [p.severity for p in result.problems] == ["warning", "warning"]
    assert result.warnings == result.problems and result.errors == []
    assert result.summary().startswith("Schema validation: OK - 4 rows x 6 columns (0 errors, 2 warnings)")
    score = result.problems[1]
    assert score.detail == {"min": 0.1, "max": 0.9, "observed_min": -1.0, "observed_max": 5.0, "count": 2, "fraction": 0.5}

    strict = schema.validate(batch, strict_ranges=True)
    assert not strict.ok and [p.severity for p in strict.problems] == ["error", "error"]
    with pytest.raises(SchemaError):
        strict.raise_if_invalid()


def test_problems_come_in_a_stable_order(schema, messy):
    result = schema.validate(messy)
    assert not result.ok
    assert kinds(result) == [
        "extra_column",
        "wrong_order",
        "dtype_mismatch",  # id: float
        "unexpected_null",  # id: NaN
        "out_of_range",  # id: 5 > 4
        "unknown_category",  # city: Lima
        "out_of_range",  # score: 2.0
        "dtype_mismatch",  # active: strings
        "dtype_mismatch",  # signup: strings
        "dtype_mismatch",  # tier: object
        "unknown_category",  # tier: c
    ]
    assert result.summary().splitlines()[0] == "Schema validation: FAILED - 2 rows x 7 columns (9 errors, 2 warnings)"


def test_to_dict_is_json_safe(schema, messy):
    payload = schema.validate(messy).to_dict()
    text = json.dumps(payload)
    assert json.loads(text) == payload
    assert set(payload) == {"ok", "n_rows", "n_columns", "n_errors", "n_warnings", "problems"}
    assert set(payload["problems"][0]) == {"kind", "column", "message", "detail", "severity"}


def test_raise_if_invalid_lists_every_error(schema, messy):
    with pytest.raises(SchemaError) as excinfo:
        schema.validate(messy).raise_if_invalid()
    error = excinfo.value
    assert isinstance(error, ValueError)
    assert len(error.problems) == 9 and all(p.is_error for p in error.problems)
    assert str(error).startswith("DataFrame does not match the schema (9 problems):")
    assert "unknown_category 'city'" in str(error)


def test_validate_never_modifies_the_frame(schema, messy):
    before = messy.copy(deep=True)
    schema.validate(messy)
    pd.testing.assert_frame_equal(messy, before)


def test_untyped_empty_object_column_matches_any_family(train, schema):
    batch = train.assign(signup=pd.Series([None] * 4, dtype=object), active=pd.Series([None] * 4, dtype=object))
    result = schema.validate(batch)
    assert kinds(result) == ["unexpected_null", "unexpected_null"]


def test_validate_accepts_paths_dicts_and_the_top_level_helper(tmp_path, train, schema):
    path = tmp_path / "batch.csv"
    train.assign(city=["Lima"] * 4).to_csv(path, index=False)
    result = schema.validate(path)
    assert "unknown_category" in kinds(result)
    schema_path = schema.save(tmp_path / "schema.json")
    assert schema_guard.validate(train, schema).ok
    assert schema_guard.validate(train, schema.to_dict()).ok
    assert schema_guard.validate(train, schema_path).ok
    assert not schema_guard.validate(train.assign(score=[9.0] * 4), schema, strict_ranges=True).ok


def test_problem_and_result_basics():
    assert set(PROBLEM_KINDS) == {"missing_column", "extra_column", "dtype_mismatch", "unexpected_null", "unknown_category", "out_of_range", "wrong_order"}
    problem = Problem("out_of_range", "x", "too big", {"count": 1}, severity="warning")
    assert not problem.is_error and problem.to_dict()["severity"] == "warning"
    with pytest.raises(ValueError, match="unknown problem kind"):
        Problem("weird", "x", "?")
    with pytest.raises(ValueError, match="unknown severity"):
        Problem("out_of_range", "x", "?", severity="fatal")
    result = ValidationResult([problem], n_rows=1, n_columns=1)
    assert result.ok and result.warnings == [problem]
    assert ValidationResult().ok and ValidationResult().summary() == "Schema validation: OK - 0 rows x 0 columns (no problems)"
