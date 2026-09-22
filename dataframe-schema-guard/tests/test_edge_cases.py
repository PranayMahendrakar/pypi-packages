import json

import numpy as np
import pandas as pd

from schema_guard import Schema


def test_empty_frame_with_columns(train):
    empty = train.iloc[0:0]
    schema = Schema.infer(empty)
    assert schema.column_names == train.columns.tolist()
    assert [s.family for s in schema] == ["int", "string", "float", "bool", "datetime", "category"]
    assert schema["id"].min is None and schema["city"].categories is None and schema["tier"].categories == ["a", "b"]
    assert schema.validate(empty).ok
    fixed = schema.enforce(empty)
    assert fixed.shape == (0, 6) and list(fixed.columns) == schema.column_names
    # a schema learned from real data accepts an empty batch and enforces it too
    full = Schema.infer(train)
    assert full.validate(empty).ok and full.enforce(empty).shape == (0, 6)
    assert full.enforce(pd.DataFrame({"id": pd.Series([], dtype="int64")})).shape == (0, 6)


def test_frame_with_no_columns_at_all():
    nothing = pd.DataFrame()
    schema = Schema.infer(nothing)
    assert schema.to_dict() == {"format": 1, "columns": []} and len(schema) == 0
    assert schema.validate(nothing).ok and schema.enforce(nothing).shape == (0, 0)
    assert Schema.from_json(schema.to_json()) == schema
    result = schema.validate(pd.DataFrame({"a": [1]}))
    assert [p.kind for p in result.problems] == ["extra_column"]
    assert schema.enforce(pd.DataFrame({"a": [1, 2]})).shape == (2, 0)


def test_single_row(train):
    one = train.iloc[:1]
    schema = Schema.infer(one)
    assert schema["id"].min == schema["id"].max == 1
    # One row means every city is distinct, so no closed category set is recorded; a
    # category dtype still declares its own categories.
    assert schema["city"].categories is None and schema["tier"].categories == ["a", "b"]
    assert schema.validate(one).ok
    pd.testing.assert_frame_equal(schema.enforce(one), one)
    # The whole point: a schema inferred from one row does not reject the real batch.
    result = schema.validate(train)
    assert {p.kind for p in result.problems} == {"out_of_range"} and result.ok


def test_all_nan_and_all_none_columns():
    df = pd.DataFrame({"f": [np.nan, np.nan], "o": [None, None], "n": pd.array([None, None], dtype="Int64")})
    schema = Schema.infer(df)
    assert [s.family for s in schema] == ["float", "string", "int"]
    assert all(s.nullable for s in schema)
    assert schema["f"].min is None and schema["o"].categories is None and schema["n"].max is None
    assert schema.validate(df).ok
    later = pd.DataFrame({"f": [1.5, np.nan], "o": ["x", None], "n": pd.array([3, None], dtype="Int64")})
    assert schema.validate(later).ok
    fixed = schema.enforce(later)
    assert fixed["o"].tolist()[0] == "x" and str(fixed["n"].dtype) == "Int64"
    # an all-null object column is accepted by every family and is coerced to the right dtype
    typed = Schema.infer(pd.DataFrame({"x": [1, 2]}))
    only_nulls = pd.DataFrame({"x": pd.Series([None, None], dtype=object)})
    assert [p.kind for p in typed.validate(only_nulls).problems] == ["unexpected_null"]
    assert str(typed.enforce(only_nulls)["x"].dtype) == "Int64"


def test_mixed_dtypes_frame(tmp_path, train):
    schema = Schema.infer(train)
    pd.testing.assert_frame_equal(schema.enforce(train), train)
    assert Schema.load(schema.save(tmp_path / "s.json")) == schema
    shuffled = train.sample(frac=1, random_state=0)
    assert schema.validate(shuffled).ok
    pd.testing.assert_frame_equal(schema.enforce(shuffled), shuffled)


def test_unicode_values_and_column_names(tmp_path):
    df = pd.DataFrame({"città": ["Zürich", "São Paulo", "東京", "Zürich"], "größe": [1, 2, 3, 4]})
    schema = Schema.infer(df)
    assert schema["città"].categories == ["São Paulo", "Zürich", "東京"]
    path = schema.save(tmp_path / "unicode.json")
    assert "東京" in path.read_text(encoding="utf-8")
    assert Schema.load(path) == schema
    result = schema.validate(df.assign(città=["Zürich", "Ōsaka", "東京", "Zürich"]))
    assert result.problems[0].detail["unknown_values"] == ["Ōsaka"]
    assert "Ōsaka" in result.summary()
    assert json.loads(json.dumps(result.to_dict(), ensure_ascii=False))["problems"][0]["column"] == "città"
    csv = tmp_path / "u.csv"
    df.to_csv(csv, index=False, encoding="utf-8")
    assert schema.validate(csv).ok


def test_integer_column_names_survive_json(tmp_path):
    df = pd.DataFrame([[1, "a"], [2, "b"]])
    schema = Schema.infer(df)
    assert schema.column_names == [0, 1]
    assert Schema.load(schema.save(tmp_path / "ints.json")) == schema
    assert schema.validate(df).ok and list(schema.enforce(df[[1, 0]]).columns) == [0, 1]


def test_timezone_aware_datetimes_and_string_dtype():
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("Europe/Oslo"),
            "s": pd.array(["a", "b"], dtype="string"),
        }
    )
    schema = Schema.infer(df)
    assert schema["t"].family == "datetime" and schema["s"].family == "string"
    assert schema.validate(df).ok
    fixed = schema.enforce(pd.DataFrame({"t": ["2024-01-03", "2024-01-04"], "s": [1, 2]}))
    assert str(fixed["t"].dtype).startswith("datetime64") and fixed["s"].tolist() == ["1", "2"]


def test_many_distinct_values_record_no_categories():
    df = pd.DataFrame({"s": [f"v{i}" for i in range(60)], "c": pd.Categorical([f"v{i}" for i in range(60)])})
    schema = Schema.infer(df)
    assert schema["s"].categories is None and schema["c"].categories is None
    assert schema.validate(pd.DataFrame({"s": ["new"], "c": pd.Categorical(["new"])})).ok


def test_infinite_values_in_ranges(tmp_path):
    df = pd.DataFrame({"f": [-np.inf, 0.0, np.inf]})
    schema = Schema.infer(df)
    # inf is not a bound and has no JSON literal, so "no bound" is what gets recorded.
    assert schema["f"].min is None and schema["f"].max is None
    path = schema.save(tmp_path / "inf.json")
    assert Schema.load(path) == schema
    assert schema.validate(pd.DataFrame({"f": [1e308]})).ok
