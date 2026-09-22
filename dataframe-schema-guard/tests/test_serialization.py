import json

import numpy as np
import pandas as pd
import pytest

import schema_guard
from schema_guard import ColumnSpec, Schema
from schema_guard.schema import as_schema


@pytest.fixture
def rich() -> pd.DataFrame:
    # Text columns repeat a value on purpose: a column whose values are all distinct is an
    # identifier, and infer() deliberately records no category set for one.
    return pd.DataFrame(
        {
            "i": [1, 2, 3, None],
            "n": pd.array([10, None, 30, 40], dtype="Int64"),
            "f": [0.5, 1.5, 2.5, np.inf],
            "b": [True, False, True, True],
            "d": pd.to_datetime(["2024-01-01", None, "2024-03-01", "2024-04-01"]),
            "s": ["a", "b", "a", None],
            "c": pd.Categorical(["p", "q", "p", "q"]),
            "k": pd.Categorical([1, 2, 1, 2]),
            "u": ["héllo", "wörld", "日本", "héllo"],
            "e": pd.Series([None, None, None, None], dtype=object),
        }
    )


def test_save_load_round_trips_exactly(tmp_path, rich):
    schema = Schema.infer(rich)
    path = schema.save(tmp_path / "nested" / "schema.json")
    assert path.exists()
    loaded = Schema.load(path)
    assert loaded == schema
    assert loaded.to_dict() == schema.to_dict()
    assert Schema.load(str(path)) == schema
    assert schema_guard.load(path) == schema
    # a second round trip changes nothing
    assert Schema.load(loaded.save(tmp_path / "again.json")) == schema
    assert Schema.from_dict(schema.to_dict()) == schema
    assert Schema.from_json(schema.to_json()) == schema


def test_json_is_human_readable_and_keeps_unicode(tmp_path, rich):
    schema = Schema.infer(rich)
    text = schema.to_json()
    assert text.startswith('{\n  "format": 1,\n  "columns": [\n')
    assert "héllo" in text and "日本" in text and "\\u" not in text
    payload = json.loads(text)
    assert payload["format"] == 1 and [c["name"] for c in payload["columns"]] == list(rich.columns)
    assert set(payload["columns"][0]) == {"name", "family", "nullable", "categories", "min", "max"}
    assert payload["columns"][7]["categories"] == [1, 2]  # integer categories stay integers
    assert payload["columns"][2]["max"] is None  # inf has no JSON literal, so "no bound" is recorded
    assert payload["columns"][4]["tz"] is None  # only datetime columns carry a tz key
    assert payload["columns"][9]["categories"] is None
    written = tmp_path / "s.json"
    schema.save(written)
    assert written.read_text(encoding="utf-8") == text


def test_from_dict_accepts_bare_lists_and_column_specs():
    specs = [ColumnSpec("a", "int", nullable=False, min=0, max=9), {"name": "b", "family": "string", "categories": ["x"]}]
    schema = Schema.from_dict(specs)
    assert schema.column_names == ["a", "b"] and schema["b"].categories == ["x"] and schema["b"].nullable is True
    assert Schema(specs) == schema and Schema.from_dict(schema) == schema and Schema.from_dict(schema) is not schema
    assert Schema.from_dict({"columns": [{"name": "a", "family": "int"}]}) == Schema([ColumnSpec("a", "int")])


def test_from_dict_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown keys"):
        Schema.from_dict([{"name": "a", "family": "int", "dtype": "int64"}])
    with pytest.raises(ValueError, match="missing the required key 'family'"):
        Schema.from_dict([{"name": "a"}])
    with pytest.raises(ValueError, match="unknown family"):
        Schema.from_dict([{"name": "a", "family": "text"}])
    with pytest.raises(ValueError, match="newer"):
        Schema.from_dict({"format": 99, "columns": []})
    with pytest.raises(ValueError, match="'format'"):
        Schema.from_dict({"format": "1", "columns": []})
    with pytest.raises(ValueError, match="'columns' list"):
        Schema.from_dict({"format": 1})
    with pytest.raises(TypeError):
        Schema.from_dict("not a schema")
    with pytest.raises(TypeError, match="column spec must be a dict"):
        Schema.from_dict(["a"])
    with pytest.raises(ValueError, match="duplicate column names"):
        Schema([ColumnSpec("a", "int"), ColumnSpec("a", "float")])
    with pytest.raises(TypeError, match="ColumnSpec or dict"):
        Schema(["a"])


def test_schema_container_protocol(schema):
    assert len(schema) == 6 and "id" in schema and "nope" not in schema
    assert [spec.name for spec in schema] == schema.column_names
    assert schema["score"].family == "float"
    with pytest.raises(KeyError, match="not in the schema"):
        schema["nope"]
    assert schema.select("score", "id").column_names == ["score", "id"]
    assert schema.drop("tier", "signup").column_names == ["id", "city", "score", "active"]
    with pytest.raises(KeyError):
        schema.drop("nope")
    assert repr(schema) == "Schema([id:int, city:string, score:float, active:bool, signup:datetime, tier:category])"
    lines = schema.summary().splitlines()
    assert lines[0] == "Schema with 6 columns"
    assert lines[1].split() == ["#", "column", "family", "nullable", "range", "categories"]
    assert "1 .. 4" in lines[2] and "3: 'Oslo', 'Paris', 'Rome'" in lines[3]
    assert Schema().summary() == "Schema with 0 columns" and repr(Schema()) == "Schema([])"


def test_as_schema_accepts_every_form(tmp_path, schema):
    path = schema.save(tmp_path / "s.json")
    assert as_schema(schema) is schema
    assert as_schema(schema.to_dict()) == schema
    assert as_schema(schema.to_dict()["columns"]) == schema
    assert as_schema(path) == schema and as_schema(str(path)) == schema
    with pytest.raises(TypeError, match="expected a Schema"):
        as_schema(42)
