"""Regression tests for the issues found in QA round 2.

Each test names the behaviour that was wrong, so a future change that reintroduces it
fails here with an explanation rather than somewhere far away.
"""

import json
import logging
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import schema_guard as sg
from schema_guard import ColumnSpec, Schema, SchemaError
from schema_guard.cli import main as cli_main


def kinds(result):
    return [problem.kind for problem in result.problems]


def no_warnings(call, *args, **kwargs):
    """Run ``call`` and return (result, warnings it let escape)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call(*args, **kwargs)
    return value, [w.category.__name__ for w in caught]


# ------------------------------------------- category codes must not depend on the dtype
# A CSV reader decides on its own whether a zip column is text or int64. Two batches of the
# same data that enforce to different category codes feed a model two different features.


@pytest.fixture
def zip_category_schema():
    return sg.infer(pd.DataFrame({"zipc": pd.Categorical(["1001", "2002", "1001", "3003"])}))


@pytest.mark.parametrize(
    "values",
    [
        ["1001", "2002"],
        [1001, 2002],
        [1001.0, 2002.0],
        pd.array([1001, 2002], dtype="Int64"),
        pd.Categorical([1001, 2002]),
    ],
    ids=["text", "int64", "float64", "Int64", "categorical-of-ints"],
)
def test_category_codes_are_the_same_however_the_number_arrives(zip_category_schema, values):
    assert zip_category_schema["zipc"].categories == ["1001", "2002", "3003"]
    out = zip_category_schema.enforce(pd.DataFrame({"zipc": values}))["zipc"]
    assert list(out.cat.categories) == ["1001", "2002", "3003"]
    assert list(out.cat.codes) == [0, 1]


def test_validate_accepts_what_enforce_produced_for_a_numeric_category_batch(zip_category_schema):
    # The package's own validator used to reject its own enforcer's output.
    fixed = zip_category_schema.enforce(pd.DataFrame({"zipc": [1001, 2002]}))
    assert zip_category_schema.validate(fixed).ok, zip_category_schema.validate(fixed).summary()


def test_validate_does_not_call_a_numeric_twin_of_a_stored_category_unknown(zip_category_schema):
    # validate() and enforce() go through one normaliser, so they cannot disagree.
    result = zip_category_schema.validate(pd.DataFrame({"zipc": [1001, 2002]}))
    assert "unknown_category" not in kinds(result)


def test_a_null_in_a_numeric_category_batch_stays_a_null_not_a_new_category(zip_category_schema):
    out = zip_category_schema.enforce(pd.DataFrame({"zipc": [1001.0, float("nan")]}))["zipc"]
    assert list(out.cat.categories) == ["1001", "2002", "3003"]
    assert list(out.cat.codes) == [0, -1]


def test_a_stored_number_matches_the_text_form_too():
    schema = Schema([ColumnSpec("k", "category", categories=[1001, 2002])])
    out = schema.enforce(pd.DataFrame({"k": ["1001", "2002"]}))["k"]
    assert list(out.cat.categories) == [1001, 2002] and list(out.cat.codes) == [0, 1]


def test_a_leading_zero_zip_is_not_absorbed_by_its_bare_number():
    # '02134' and 2134 are different categories: the zero is the reason it is text at all.
    # The alias only holds when the text and the number name each other exactly.
    schema = sg.infer(pd.DataFrame({"zip": ["02134", "10001", "02134"]}))
    result = schema.validate(pd.DataFrame({"zip": [2134, 10001]}))
    assert kinds(result) == ["dtype_mismatch", "unknown_category"]
    assert result.problems[1].detail["unknown_values"] == [2134]


# ------------------------------------------ a blank cell must not rewrite every id value
# int64 upcasts to float64 the moment one cell is empty. astype(str) then wrote '1001.0',
# so two batches of identical ids came out different because one of them had a gap.


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1001, 2002], ["1001", "2002"]),
        ([1001.0, 2002.0], ["1001", "2002"]),
        (pd.array([1001, 2002], dtype="Int64"), ["1001", "2002"]),
        (pd.array([1001.0, 2002.0], dtype="Float64"), ["1001", "2002"]),
    ],
    ids=["int64", "float64", "Int64", "Float64"],
)
def test_whole_numbers_become_whole_strings(values, expected):
    schema = sg.infer(pd.DataFrame({"zipc": ["1001", "2002", "1001", "3003"]}))
    assert schema["zipc"].family == "string"
    assert list(schema.enforce(pd.DataFrame({"zipc": values}))["zipc"]) == expected


def test_a_null_does_not_change_the_other_values_in_a_string_column():
    schema = Schema([ColumnSpec("zipc", "string")])
    whole = schema.enforce(pd.DataFrame({"zipc": [1001.0, 2002.0]}))["zipc"]
    gapped = schema.enforce(pd.DataFrame({"zipc": [1001.0, np.nan]}))["zipc"]
    assert whole.iloc[0] == gapped.iloc[0] == "1001"
    assert pd.isna(gapped.iloc[1])


def test_validate_accepts_what_enforce_produced_for_a_float_id_batch():
    schema = sg.infer(pd.DataFrame({"zipc": ["1001", "2002", "1001", "3003"]}))
    fixed = schema.enforce(pd.DataFrame({"zipc": [1001.0, 2002.0]}))
    assert schema.validate(fixed).ok, schema.validate(fixed).summary()


def test_a_real_fraction_keeps_its_decimals():
    schema = Schema([ColumnSpec("s", "string")])
    out = schema.enforce(pd.DataFrame({"s": [1.5, -0.25, 2.0, np.inf]}))["s"]
    assert out.tolist() == ["1.5", "-0.25", "2", "inf"]


def test_enforce_does_not_mutate_the_frame_it_was_given():
    schema = sg.infer(pd.DataFrame({"zipc": ["1001", "2002", "1001", "3003"]}))
    batch = pd.DataFrame({"zipc": [1001.0, 2002.0]})
    before = batch.copy(deep=True)
    schema.enforce(batch)
    pd.testing.assert_frame_equal(batch, before)


# ------------------------------------------------- a frame must satisfy its own schema
# infer() classifies an object column by its values, so calling object backing an error
# made the training frame violate the schema learned from it, at error severity.


@pytest.mark.parametrize(
    "series, family",
    [
        (pd.Series([Decimal("10.00"), Decimal("20.00")], dtype=object), "float"),
        (pd.Series([1, 2, 3], dtype=object), "int"),
        (pd.Series([1.5, 2.5], dtype=object), "float"),
        (pd.Series([True, False], dtype=object), "bool"),
        (pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")], dtype=object), "datetime"),
    ],
    ids=["decimal", "int", "float", "bool", "datetime"],
)
def test_infer_then_validate_the_same_frame_is_ok(series, family):
    df = pd.DataFrame({"a": series})
    schema = Schema.infer(df)
    assert schema["a"].family == family
    result = schema.validate(df)
    assert result.ok, result.summary()
    assert result.raise_if_invalid() is result


def test_object_backing_is_still_reported_and_says_what_the_difference_is():
    df = pd.DataFrame({"amount": pd.Series([Decimal("10.00"), Decimal("20.00")], dtype=object)})
    problem = Schema.infer(df).validate(df).problems[0]
    # The old message named the same family twice: "expected float, got object (float)".
    assert problem.severity == "warning" and problem.detail["object_backed"] is True
    assert problem.message == "expected a real float dtype, got object holding float values; enforce() converts it"


def test_enforce_gives_the_object_backed_column_a_real_dtype():
    df = pd.DataFrame({"amount": pd.Series([Decimal("10.00"), Decimal("20.00")], dtype=object)})
    fixed = Schema.infer(df).enforce(df)
    assert fixed["amount"].dtype == "float64" and fixed["amount"].tolist() == [10.0, 20.0]


def test_a_wrong_family_in_an_object_column_is_still_an_error(train, schema):
    # Only the right-values-wrong-container case softened; a real family change did not.
    result = schema.validate(train.assign(signup=train["signup"].dt.strftime("%Y-%m-%d")))
    assert kinds(result) == ["dtype_mismatch"] and not result.ok


# ------------------------------------------- pandas deprecation noise must not reach users


def test_a_mixed_offset_batch_enforces_without_warning_the_caller():
    schema = Schema([ColumnSpec("d", "datetime", tz="UTC")])
    batch = pd.DataFrame({"d": ["2024-01-01T00:00:00+01:00", "2024-01-01T00:00:00+02:00"]})
    out, caught = no_warnings(schema.enforce, batch)
    assert caught == []
    assert out["d"].tolist() == [
        pd.Timestamp("2023-12-31T23:00:00", tz="UTC"),
        pd.Timestamp("2023-12-31T22:00:00", tz="UTC"),
    ]


def test_a_mixed_offset_batch_enforces_without_warning_a_tz_naive_schema():
    schema = Schema([ColumnSpec("d", "datetime")])
    batch = pd.DataFrame({"d": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00+02:00"]})
    out, caught = no_warnings(schema.enforce, batch)
    assert caught == [] and str(out["d"].dtype) == "datetime64[ns]"


def test_mixed_naive_and_aware_timestamps_coerce_instead_of_being_called_unparseable():
    # What pd.concat of a zoned and a naive frame leaves behind: object dtype, valid values.
    batch = pd.DataFrame({"d": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01", tz="UTC")]})
    out, caught = no_warnings(Schema([ColumnSpec("d", "datetime", tz="UTC")]).enforce, batch)
    assert caught == []
    assert out["d"].tolist() == [pd.Timestamp("2024-01-01", tz="UTC")] * 2


def test_an_out_of_range_date_says_it_is_out_of_range_not_unparseable():
    # '3000-01-01' is a perfectly good date; calling it unparseable sends the user hunting
    # for a data-entry bug that is not there.
    with pytest.raises(SchemaError) as excinfo:
        Schema([ColumnSpec("d", "datetime")]).enforce(pd.DataFrame({"d": ["3000-01-01", "2024-01-01"]}))
    message = excinfo.value.problems[0].message
    assert "outside the dates pandas can represent" in message and "'3000-01-01'" in message


def test_genuinely_unparseable_datetimes_still_say_so():
    with pytest.raises(SchemaError, match="not parseable datetimes"):
        Schema([ColumnSpec("d", "datetime")]).enforce(pd.DataFrame({"d": ["not a date", "2024-01-01"]}))


# ------------------------------------------------ complex values must not be halved quietly


def test_complex_values_are_rejected_rather_than_stripped_of_their_imaginary_part():
    schema = Schema([ColumnSpec("z", "float")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SchemaError) as excinfo:
            schema.enforce(pd.DataFrame({"z": [1 + 2j, 3 + 4j]}))
    assert [w.category.__name__ for w in caught] == []
    problem = excinfo.value.problems[0]
    assert problem.column == "z" and "discard the imaginary part" in problem.message
    assert problem.detail["bad_count"] == 2


def test_a_complex_column_with_no_imaginary_part_converts_cleanly():
    out, caught = no_warnings(
        Schema([ColumnSpec("z", "float")]).enforce, pd.DataFrame({"z": [1 + 0j, 3 + 0j]})
    )
    assert caught == [] and out["z"].tolist() == [1.0, 3.0] and out["z"].dtype == "float64"


# ------------------------------------------------------------------ smaller papercuts


def test_a_one_row_frame_logs_no_all_distinct_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="schema_guard.schema"):
        sg.infer(pd.DataFrame({"a": ["x"], "b": ["y"], "c": ["z"]}))
    assert caplog.records == []


def test_a_repeating_frame_still_warns_when_every_value_is_distinct(caplog):
    with caplog.at_level(logging.WARNING, logger="schema_guard.schema"):
        sg.infer(pd.DataFrame({"a": ["x", "y", "z"]}))
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "all 3 value(s) are distinct" in caplog.records[0].getMessage()


def test_as_schema_is_exported():
    assert "as_schema" in sg.__all__
    namespace: dict = {}
    exec("from schema_guard import *", namespace)  # noqa: S102 - checking the star export
    assert namespace["as_schema"] is sg.as_schema


def test_loading_a_data_file_as_a_schema_names_the_file(tmp_path):
    data = tmp_path / "sales.csv"
    data.write_text("id,city\n1,Oslo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a schema JSON file"):
        Schema.load(data)


def test_cli_swapped_validate_arguments_say_which_way_round_they_go(tmp_path, capsys):
    data = tmp_path / "sales.csv"
    pd.DataFrame({"id": [1, 2], "city": ["Oslo", "Oslo"]}).to_csv(data, index=False)
    schema = Schema.infer(data).save(tmp_path / "schema.json")
    assert cli_main(["validate", str(data), str(schema)]) == 1
    err = capsys.readouterr().err
    assert "sales.csv" in err and "is not a schema JSON file" in err
    assert "take the schema first, then the data file" in err


def test_cli_validate_output_creates_its_parent_directory(tmp_path, capsys):
    data = tmp_path / "train.csv"
    pd.DataFrame({"id": [1, 2], "city": ["Oslo", "Oslo"]}).to_csv(data, index=False)
    schema = Schema.infer(data).save(tmp_path / "schema.json")
    target = tmp_path / "deeper2" / "r.json"
    assert cli_main(["validate", str(schema), str(data), "--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True
    # The summary the user asked for is printed as well, not swallowed by the write.
    assert "Schema validation: OK" in capsys.readouterr().out
