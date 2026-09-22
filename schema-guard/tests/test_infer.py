import logging

import numpy as np
import pandas as pd
import pytest

import schema_guard
from schema_guard import ColumnSpec, Schema


def test_infer_records_family_nullable_categories_ranges_and_order(train, schema):
    assert schema.column_names == ["id", "city", "score", "active", "signup", "tier"]
    assert [spec.family for spec in schema] == ["int", "string", "float", "bool", "datetime", "category"]
    assert not any(spec.nullable for spec in schema)
    assert schema["id"].min == 1 and schema["id"].max == 4 and isinstance(schema["id"].min, int)
    assert schema["score"].min == 0.1 and schema["score"].max == 0.9
    assert schema["city"].categories == ["Oslo", "Paris", "Rome"]  # sorted, deduplicated
    assert schema["tier"].categories == ["a", "b"]  # declared order of the categorical
    for name in ("active", "signup", "tier", "city"):
        assert schema[name].min is None and schema[name].max is None
    for name in ("id", "score", "active", "signup"):
        assert schema[name].categories is None


def test_top_level_infer_is_the_same_as_schema_infer(train):
    assert schema_guard.infer(train) == Schema.infer(train)
    assert schema_guard.infer(train, categorical_max_unique=1).to_dict() == Schema.infer(train, categorical_max_unique=1).to_dict()


def test_nullable_modes():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})
    observed = Schema.infer(df)
    assert observed["a"].nullable is True and observed["b"].nullable is False
    always = Schema.infer(df, nullable="always")
    assert always["a"].nullable and always["b"].nullable
    never = Schema.infer(df, nullable="never")
    assert not never["a"].nullable and not never["b"].nullable
    assert Schema.infer(df, nullable=True) == always
    assert Schema.infer(df, nullable=False) == never
    with pytest.raises(ValueError, match="nullable"):
        Schema.infer(df, nullable="sometimes")


def test_categorical_max_unique_controls_recorded_categories():
    df = pd.DataFrame({"c": list("abcabc"), "k": pd.Categorical(list("xyzxyz"))})
    assert Schema.infer(df, categorical_max_unique=3)["c"].categories == ["a", "b", "c"]
    assert Schema.infer(df, categorical_max_unique=2)["c"].categories is None
    assert Schema.infer(df, categorical_max_unique=2)["k"].categories is None
    assert Schema.infer(df, categorical_max_unique=0)["c"].categories is None
    with pytest.raises(ValueError, match="categorical_max_unique"):
        Schema.infer(df, categorical_max_unique=-1)


def test_numeric_ranges_can_be_disabled():
    df = pd.DataFrame({"i": [3, 1, 2], "f": [0.5, -1.5, 2.5]})
    with_ranges = Schema.infer(df)
    assert (with_ranges["i"].min, with_ranges["i"].max) == (1, 3)
    assert (with_ranges["f"].min, with_ranges["f"].max) == (-1.5, 2.5)
    without = Schema.infer(df, numeric_ranges=False)
    assert without["i"].min is None and without["f"].max is None


def test_nullable_extension_dtypes_and_string_dtype_map_to_families():
    df = pd.DataFrame(
        {
            "i": pd.array([1, 2, None], dtype="Int64"),
            "b": pd.array([True, False, None], dtype="boolean"),
            "f": pd.array([1.5, 0.5, None], dtype="Float64"),
            "s": pd.array(["x", "x", None], dtype="string"),
            "t": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).tz_localize("UTC"),
        }
    )
    schema = Schema.infer(df)
    assert [spec.family for spec in schema] == ["int", "bool", "float", "string", "datetime"]
    assert schema["i"].nullable and schema["b"].nullable and schema["s"].nullable and not schema["t"].nullable
    assert schema["i"].min == 1 and schema["f"].max == 1.5
    assert schema["s"].categories == ["x"]
    assert schema["t"].tz == "UTC"


def test_object_columns_are_classified_by_their_values():
    df = pd.DataFrame(
        {
            "ints": pd.Series([1, 2], dtype=object),
            "floats": pd.Series([1.5, None], dtype=object),
            "bools": pd.Series([True, False], dtype=object),
            "mixed": pd.Series(["a", 1], dtype=object),
            "empty": pd.Series([None, None], dtype=object),
        }
    )
    schema = Schema.infer(df)
    assert [spec.family for spec in schema] == ["int", "float", "bool", "string", "string"]
    assert schema["ints"].min == 1 and schema["ints"].max == 2
    assert schema["empty"].categories is None  # nothing observed: nothing to restrict
    assert schema["empty"].nullable


def test_unsupported_dtype_falls_back_to_string_with_a_warning(caplog):
    df = pd.DataFrame({"delta": pd.to_timedelta([1, 2], unit="D")})
    with caplog.at_level(logging.WARNING, logger="schema_guard.schema"):
        schema = Schema.infer(df)
    assert schema["delta"].family == "string"
    assert "cannot describe" in caplog.text


def test_infer_from_csv_path(tmp_path, train):
    path = tmp_path / "train.csv"
    train.to_csv(path, index=False)
    schema = Schema.infer(path)
    assert schema.column_names == train.columns.tolist()
    assert schema["id"].family == "int" and schema["score"].family == "float" and schema["active"].family == "bool"
    assert schema["signup"].family == "string"  # CSV has no dtypes; dates come back as text
    assert Schema.infer(str(path)) == schema


def test_infer_rejects_frames_a_schema_cannot_describe():
    with pytest.raises(ValueError, match="duplicate column names"):
        Schema.infer(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(ValueError, match="MultiIndex"):
        Schema.infer(pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])))
    with pytest.raises(TypeError, match="DataFrame or a path"):
        Schema.infer([1, 2, 3])
    with pytest.raises(ValueError, match="unsupported file type"):
        Schema.infer("data.xlsx")


def test_columnspec_validates_its_fields():
    spec = ColumnSpec("x", "int", nullable=False, min=0, max=10)
    assert spec.to_dict() == {"name": "x", "family": "int", "nullable": False, "categories": None, "min": 0, "max": 10}
    assert ColumnSpec("c", "string", categories=["b", "a", "b", None]).categories == ["b", "a"]
    with pytest.raises(ValueError, match="unknown family"):
        ColumnSpec("x", "integer")
    with pytest.raises(ValueError, match="greater than max"):
        ColumnSpec("x", "int", min=5, max=1)
    with pytest.raises(TypeError, match="nullable"):
        ColumnSpec("x", "int", nullable="no")
    with pytest.raises(TypeError, match="categories"):
        ColumnSpec("x", "string", categories="abc")
    with pytest.raises(TypeError, match="min"):
        ColumnSpec("x", "int", min="0")
