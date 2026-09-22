"""Regression tests for the issues found in QA round 1.

Each test names the behaviour that was wrong, so a future change that reintroduces it
fails here with an explanation rather than somewhere far away.
"""

import json
import logging

import numpy as np
import pandas as pd
import pytest

import schema_guard as sg
from schema_guard import Schema, SchemaError
from schema_guard._types import json_dumps
from schema_guard.cli import main as cli_main


def strict_loads(text: str):
    """json.loads that rejects the bare NaN/Infinity tokens only Python accepts.

    Python's json reads its own non-standard output back, which is exactly why a
    round-trip test cannot catch this; every other parser (JavaScript, Go, jq) rejects it.
    """
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"bare {c} token")))


# --------------------------------------------------------------- inf must not break JSON


def test_schema_json_with_inf_is_standard_json(tmp_path):
    schema = sg.infer(pd.DataFrame({"ratio": [1.0, 2.0, np.inf]}))
    text = schema.save(tmp_path / "inf.json").read_text(encoding="utf-8")
    assert "Infinity" not in text
    payload = strict_loads(text)  # would raise on a bare Infinity literal
    assert payload["columns"][0]["max"] is None
    assert Schema.load(tmp_path / "inf.json") == schema


def test_negative_inf_and_all_inf_columns_are_standard_json(tmp_path):
    schema = sg.infer(pd.DataFrame({"a": [-np.inf, 0.0, np.inf], "b": [np.inf, np.inf, np.inf]}))
    text = schema.to_json()
    assert "Infinity" not in text and "NaN" not in text
    strict_loads(text)
    assert schema["a"].min is None and schema["a"].max is None


def test_validation_result_json_with_inf_is_standard_json():
    schema = sg.infer(pd.DataFrame({"r": [1.0, 2.0, 3.0]}))
    result = schema.validate(pd.DataFrame({"r": [1.0, np.inf]}))
    assert [p.kind for p in result.problems] == ["out_of_range"]
    text = json_dumps(result.to_dict())
    assert "Infinity" not in text
    payload = strict_loads(text)
    assert payload["problems"][0]["detail"]["observed_max"] is None


def test_cli_json_output_with_inf_is_standard_json(tmp_path, capsys):
    data = tmp_path / "inf.csv"
    data.write_text("ratio\n1.0\n2.0\ninf\n", encoding="utf-8")
    schema_path = tmp_path / "s.json"

    assert cli_main(["infer", str(data), "--json", "--output", str(schema_path)]) == 0
    strict_loads(capsys.readouterr().out)
    strict_loads(schema_path.read_text(encoding="utf-8"))

    result_path = tmp_path / "r.json"
    assert cli_main(["validate", str(schema_path), str(data), "--json", "--output", str(result_path)]) in (0, 1)
    strict_loads(capsys.readouterr().out)
    strict_loads(result_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------- timezone is part of the schema


def test_tz_naive_batch_against_a_utc_schema_is_reported():
    utc = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 12:00"]).tz_localize("UTC")})
    naive = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 12:00"])})
    schema = sg.infer(utc)

    result = schema.validate(naive)
    assert not result.ok
    problem = result.problems[0]
    assert problem.kind == "dtype_mismatch" and problem.column == "t"
    assert problem.detail["expected_tz"] == "UTC" and problem.detail["actual_tz"] is None
    with pytest.raises(SchemaError):
        result.raise_if_invalid()
    # and the reverse: a tz-aware batch against a naive schema
    naive_schema = sg.infer(naive)
    assert not naive_schema.validate(utc).ok


def test_enforce_puts_every_batch_in_the_schema_timezone():
    schema = sg.infer(pd.DataFrame({"t": pd.to_datetime(["2024-01-01 12:00"]).tz_localize("UTC")}))
    naive = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 12:00"])})
    oslo = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 13:00"]).tz_localize("Europe/Oslo")})
    strings = pd.DataFrame({"t": ["2024-01-01 12:00"]})

    columns = [schema.enforce(batch)["t"] for batch in (naive, oslo, strings)]
    dtypes = {str(column.dtype) for column in columns}
    assert dtypes == {"datetime64[ns, UTC]"}, dtypes
    # the whole point: batches enforced by one schema still concat to one dtype
    assert str(pd.concat(columns).dtype) == "datetime64[ns, UTC]"
    # converting a zoned batch preserves the instant (13:00 Oslo in January is 12:00 UTC)
    assert columns[1].iloc[0] == pd.Timestamp("2024-01-01 12:00", tz="UTC")


def test_tz_is_recorded_saved_and_loaded(tmp_path):
    schema = sg.infer(pd.DataFrame({"t": pd.to_datetime(["2024-01-01"]).tz_localize("Europe/Oslo")}))
    assert schema["t"].tz == "Europe/Oslo"
    assert schema["t"].to_dict()["tz"] == "Europe/Oslo"
    assert Schema.load(schema.save(tmp_path / "tz.json")) == schema
    # a naive schema for the same family records tz None, and the two are not equal
    assert sg.infer(pd.DataFrame({"t": pd.to_datetime(["2024-01-01"])}))["t"].tz is None
    assert sg.infer(pd.DataFrame({"t": pd.to_datetime(["2024-01-01"])})) != schema


def test_filled_in_datetime_column_carries_the_schema_timezone():
    schema = sg.infer(
        pd.DataFrame({"t": pd.to_datetime(["2024-01-01"]).tz_localize("UTC"), "x": [1]})
    )
    filled = schema.enforce(pd.DataFrame({"x": [1, 2]}))
    present = schema.enforce(pd.DataFrame({"t": pd.to_datetime(["2024-01-01"]).tz_localize("UTC"), "x": [1]}))
    assert str(filled["t"].dtype) == str(present["t"].dtype) == "datetime64[ns, UTC]"
    assert filled["t"].isna().all()


def test_only_datetime_columns_can_carry_a_tz():
    from schema_guard import ColumnSpec

    with pytest.raises(ValueError, match="tz only applies to the datetime family"):
        ColumnSpec("x", "int", tz="UTC")
    with pytest.raises(ValueError, match="not a known timezone"):
        ColumnSpec("t", "datetime", tz="Mars/Olympus")


# --------------------------------------------------------------- all-distinct text is an identifier


def test_all_distinct_text_column_records_no_categories(caplog):
    train = pd.DataFrame({"order_id": [f"ORD-{i:04d}" for i in range(40)]})
    with caplog.at_level(logging.WARNING, logger="schema_guard.schema"):
        schema = sg.infer(train)
    assert schema["order_id"].categories is None
    assert "all 40 value(s) are distinct" in caplog.text

    # the next batch of perfectly good ids must pass
    result = schema.validate(pd.DataFrame({"order_id": ["ORD-0040", "ORD-0041"]}))
    assert result.ok and result.problems == []
    result.raise_if_invalid()


def test_small_sample_of_unique_text_does_not_pin_the_allowed_set(caplog):
    # The README quickstart infers from 3 rows; that must not freeze a text column.
    for size in (1, 3, 5, 20, 50):
        frame = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(size)]})
        with caplog.at_level(logging.WARNING, logger="schema_guard.schema"):
            schema = sg.infer(frame)
        assert schema["email"].categories is None, size
        assert schema.validate(pd.DataFrame({"email": ["someone.else@example.com"]})).ok, size


def test_a_repeating_text_column_still_records_its_categories():
    schema = sg.infer(pd.DataFrame({"tier": ["gold", "silver", "gold", "silver"]}))
    assert schema["tier"].categories == ["gold", "silver"]
    assert not schema.validate(pd.DataFrame({"tier": ["bronze"]})).ok
    # one repeat is enough evidence of a closed set
    assert sg.infer(pd.DataFrame({"t": ["a", "b", "c", "a"]}))["t"].categories == ["a", "b", "c"]


def test_declared_category_dtype_keeps_its_categories_even_when_all_distinct():
    # A category dtype's categories are declared by the user, not guessed from repetition.
    frame = pd.DataFrame({"c": pd.Categorical(["x", "y", "z"])})
    schema = sg.infer(frame)
    assert schema["c"].categories == ["x", "y", "z"]


# --------------------------------------------------------------- column labels JSON cannot keep


def test_pivoted_timestamp_column_labels_are_rejected_not_stringified():
    long = pd.DataFrame(
        {
            "user": ["u1", "u1", "u2", "u2"],
            "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
            "spend": [10.0, 20.0, 30.0, 40.0],
        }
    )
    pivoted = long.pivot_table(index="user", columns="month", values="spend").reset_index()

    with pytest.raises(ValueError, match="column names must be strings or integers"):
        sg.infer(pivoted)
    # the message names the offending label and how to fix it
    with pytest.raises(ValueError, match=r"df\.columns\.map\(str\)"):
        sg.infer(pivoted)

    # once converted, the frame a schema was inferred from validates and enforces cleanly
    fixed = pivoted.copy()
    fixed.columns = fixed.columns.map(str)
    schema = sg.infer(fixed)
    assert schema.validate(fixed).ok
    out = schema.enforce(fixed)
    assert not out.drop(columns=["user"]).isna().any().any()
    assert list(out.columns) == list(fixed.columns)
    # check_names=False: the columns Index carries a .name ('month') from the pivot, which is
    # frame metadata a schema does not store; the labels and every value must still match.
    pd.testing.assert_frame_equal(out, fixed, check_names=False)


@pytest.mark.parametrize(
    "labels",
    [
        [1.5, 2.5],
        [None, "b"],
        [("a", "x"), ("a", "y")],
        [b"a", b"b"],
    ],
)
def test_unsupported_column_labels_raise_valueerror_everywhere(labels):
    frame = pd.DataFrame([[1, 2]], columns=labels)
    for call in (
        lambda: sg.infer(frame),
        lambda: sg.infer(pd.DataFrame([[1, 2]], columns=["a", "b"])).validate(frame),
        lambda: sg.infer(pd.DataFrame([[1, 2]], columns=["a", "b"])).enforce(frame),
    ):
        with pytest.raises(ValueError, match="column names must be strings or integers"):
            call()


def test_guard_never_hands_a_function_an_all_null_frame():
    frame = pd.DataFrame({1.5: [1, 2]})
    with pytest.raises(ValueError, match="column names must be strings or integers"):
        sg.infer(frame)

    # the supported case still works: int labels survive and keep their values
    ints = pd.DataFrame([[1, "a"], [2, "b"]])
    schema = sg.infer(ints)

    @sg.guard(schema)
    def predict(df):
        return df

    out = predict(ints)
    assert out[0].tolist() == [1, 2] and out[1].tolist() == ["a", "b"]


# --------------------------------------------------------------- categories JSON cannot keep raw


def test_datetime_categories_are_not_turned_into_nulls():
    train = pd.DataFrame(
        {"cohort": pd.Categorical(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]))}
    )
    schema = sg.infer(train)
    assert schema["cohort"].categories == ["2024-01-01T00:00:00", "2024-02-01T00:00:00"]

    out = schema.enforce(pd.DataFrame({"cohort": pd.to_datetime(["2024-01-01", "2024-02-01"])}))
    assert not out["cohort"].isna().any()
    assert out["cohort"].cat.codes.tolist() == [0, 1]


def test_bytes_categories_are_not_turned_into_nulls():
    schema = sg.infer(pd.DataFrame({"k": pd.Categorical([b"a", b"b", b"a"])}))
    out = schema.enforce(pd.DataFrame({"k": [b"a", b"b"]}))
    assert not out["k"].isna().any()
    assert out["k"].tolist() == ["a", "b"]


def test_category_codes_are_stable_however_the_value_arrives():
    train = pd.DataFrame(
        {"cohort": pd.Categorical(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]))}
    )
    schema = sg.infer(train)
    as_timestamps = schema.enforce(pd.DataFrame({"cohort": pd.to_datetime(["2024-01-01", "2024-02-01"])}))
    as_strings = schema.enforce(pd.DataFrame({"cohort": ["2024-01-01", "2024-02-01"]}))

    assert list(as_strings["cohort"].cat.categories) == list(as_timestamps["cohort"].cat.categories)
    assert as_strings["cohort"].cat.codes.tolist() == as_timestamps["cohort"].cat.codes.tolist() == [0, 1]
    # no duplicate category for a date that is already in the schema
    assert len(as_strings["cohort"].cat.categories) == 2


def test_validate_agrees_with_enforce_about_datetime_categories():
    train = pd.DataFrame(
        {"cohort": pd.Categorical(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]))}
    )
    schema = sg.infer(train)
    known = pd.DataFrame({"cohort": pd.Categorical(pd.to_datetime(["2024-01-01"]))})
    # validate must not call a value unknown that enforce maps straight onto a category
    assert [p.kind for p in schema.validate(known).problems if p.kind == "unknown_category"] == []

    unknown = pd.DataFrame({"cohort": pd.Categorical(pd.to_datetime(["2024-03-01"]))})
    assert [p.kind for p in schema.validate(unknown).problems if p.kind == "unknown_category"]


def test_unseen_categories_are_appended_never_nulled():
    schema = sg.infer(pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])}))
    out = schema.enforce(pd.DataFrame({"c": ["a", "b", "z"]}))
    assert not out["c"].isna().any()
    assert list(out["c"].cat.categories) == ["a", "b", "z"]


# --------------------------------------------------------------- clear errors for bad paths


def test_load_on_a_directory_raises_a_clear_valueerror(tmp_path):
    with pytest.raises(ValueError, match="is a directory, not a schema JSON file"):
        sg.load(tmp_path)
    with pytest.raises(ValueError, match="is a directory, not a schema JSON file"):
        Schema.load(str(tmp_path))
    # reading data from a directory is just as clear
    with pytest.raises(ValueError, match="is a directory, not a data file"):
        sg.infer(tmp_path)
    # a missing file is still FileNotFoundError, not swallowed
    with pytest.raises(FileNotFoundError):
        sg.load(tmp_path / "nope.json")
