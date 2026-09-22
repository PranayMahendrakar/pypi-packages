import pandas as pd
import pytest

from schema_guard import SchemaError, guard


def test_guard_enforces_on_the_first_positional_dataframe(schema, messy):
    calls = []

    @guard(schema)
    def predict(df, threshold=0.5):
        calls.append(df)
        return list(df.columns)

    assert predict(messy) == schema.column_names
    assert predict(messy, threshold=0.9) == schema.column_names
    assert str(calls[0]["id"].dtype) == "Int64"
    assert list(messy.columns)[0] == "city" and str(messy["id"].dtype) == "float64"  # input untouched
    assert predict.__name__ == "predict" and predict.schema is schema


def test_guard_finds_keyword_dataframes_and_skips_self(schema, messy):
    @guard(schema, mode="coerce", extra="keep")
    def run(name, *, frame):
        return name, list(frame.columns)

    assert run("x", frame=messy) == ("x", schema.column_names + ["extra"])

    class Model:
        @guard(schema)
        def predict(self, df):
            return list(df.columns)

    assert Model().predict(messy) == schema.column_names


def test_guard_strict_mode_never_calls_the_function_on_bad_input(schema, train, messy):
    calls = []

    @guard(schema, mode="strict")
    def predict(df):
        calls.append(1)
        return len(df)

    with pytest.raises(SchemaError) as excinfo:
        predict(messy)
    assert calls == [] and len(excinfo.value.problems) >= 3
    assert predict(train) == 4


def test_guard_accepts_a_schema_path_or_dict(tmp_path, schema, messy):
    path = schema.save(tmp_path / "schema.json")

    @guard(str(path), missing="raise")
    def a(df):
        return list(df.columns)

    @guard(schema.to_dict())
    def b(df):
        return list(df.columns)

    assert a(messy) == schema.column_names == b(messy)
    with pytest.raises(SchemaError, match="missing_column"):
        a(pd.DataFrame({"id": [1]}))


def test_guard_rejects_bad_usage(schema):
    with pytest.raises(ValueError, match="mode"):
        guard(schema, mode="lenient")

    @guard(schema)
    def f(x):
        return x

    with pytest.raises(TypeError, match="without a DataFrame argument"):
        f(42)
