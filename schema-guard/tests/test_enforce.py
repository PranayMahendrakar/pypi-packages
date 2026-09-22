import numpy as np
import pandas as pd
import pytest

import schema_guard
from schema_guard import Schema, SchemaError


def test_enforce_coerces_reorders_drops_and_fills(schema, messy):
    fixed = schema.enforce(messy)
    assert list(fixed.columns) == schema.column_names
    assert str(fixed["id"].dtype) == "Int64" and fixed["id"].tolist()[0] == 5 and pd.isna(fixed["id"].iloc[1])
    assert fixed["active"].dtype == bool and fixed["active"].tolist() == [True, False]
    assert str(fixed["signup"].dtype).startswith("datetime64") and fixed["signup"].iloc[0] == pd.Timestamp("2024-05-01")
    assert str(fixed["tier"].dtype) == "category" and list(fixed["tier"].cat.categories) == ["a", "b", "c"]
    assert fixed["tier"].tolist() == ["a", "c"]
    assert fixed["city"].tolist() == ["Paris", "Lima"] and fixed["score"].tolist() == [0.3, 2.0]
    # shape problems are gone; content problems (nulls, unknown values, ranges) are still reported
    remaining = [p.kind for p in schema.validate(fixed).problems]
    assert not {"dtype_mismatch", "wrong_order", "extra_column", "missing_column"} & set(remaining)
    assert "unknown_category" in remaining and "unexpected_null" in remaining


def test_enforce_never_mutates_the_input(schema, messy):
    before = messy.copy(deep=True)
    fixed = schema.enforce(messy)
    pd.testing.assert_frame_equal(messy, before)
    fixed.loc[0, "score"] = 123.0
    fixed.loc[0, "city"] = "Mars"
    pd.testing.assert_frame_equal(messy, before)


def test_enforce_preserves_the_index_and_is_a_no_op_on_matching_data(schema, train):
    df = train.copy()
    df.index = [10, 20, 30, 40]
    fixed = schema.enforce(df)
    assert fixed.index.tolist() == [10, 20, 30, 40]
    pd.testing.assert_frame_equal(fixed, df)
    assert fixed is not df


def test_int_column_arriving_as_float():
    schema = Schema.infer(pd.DataFrame({"a": [1, 2, 3]}))
    with_nan = schema.enforce(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))["a"]
    assert str(with_nan.dtype) == "Int64" and with_nan.tolist()[0] == 1 and pd.isna(with_nan.iloc[1])
    clean = schema.enforce(pd.DataFrame({"a": [1.0, 2.0]}))["a"]
    assert str(clean.dtype) == "int64" and clean.tolist() == [1, 2]
    with pytest.raises(SchemaError, match="not whole numbers"):
        schema.enforce(pd.DataFrame({"a": [1.5, 2.0]}))


def test_string_column_arriving_as_int():
    schema = Schema.infer(pd.DataFrame({"zip": ["02134", "10001"]}))
    fixed = schema.enforce(pd.DataFrame({"zip": [2134, 10001]}))["zip"]
    assert fixed.dtype == object and fixed.tolist() == ["2134", "10001"]
    nulls = schema.enforce(pd.DataFrame({"zip": pd.Series([2134, None], dtype=object)}))["zip"]
    assert nulls.tolist()[0] == "2134" and pd.isna(nulls.iloc[1])


def test_datetime_column_arriving_as_strings():
    schema = Schema.infer(pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-02-01"])}))
    fixed = schema.enforce(pd.DataFrame({"d": ["2024-03-01", "2024-04-15 10:30", None]}))["d"]
    assert str(fixed.dtype).startswith("datetime64")
    assert fixed.iloc[1] == pd.Timestamp("2024-04-15 10:30") and pd.isna(fixed.iloc[2])
    with pytest.raises(SchemaError, match="not parseable datetimes"):
        schema.enforce(pd.DataFrame({"d": ["2024-03-01", "yesterday"]}))
    with pytest.raises(SchemaError, match="ambiguous"):
        schema.enforce(pd.DataFrame({"d": [1700000000, 1700000001]}))


def test_category_column_arriving_as_object_keeps_codes_stable():
    schema = Schema.infer(pd.DataFrame({"c": pd.Categorical(["x", "y", "x"])}))
    fixed = schema.enforce(pd.DataFrame({"c": ["y", "z", None]}))["c"]
    assert str(fixed.dtype) == "category" and list(fixed.cat.categories) == ["x", "y", "z"]
    assert fixed.tolist()[:2] == ["y", "z"] and pd.isna(fixed.iloc[2])
    assert fixed.cat.codes.tolist() == [1, 2, -1]
    # a categorical whose categories are in another order is re-coded to the schema's order
    other = pd.DataFrame({"c": pd.Categorical(["y", "x"], categories=["y", "x"])})
    assert list(schema.enforce(other)["c"].cat.categories) == ["x", "y"]
    # no recorded categories: a plain conversion to category
    free = Schema([{"name": "c", "family": "category"}])
    assert str(free.enforce(pd.DataFrame({"c": ["b", "a"]}))["c"].dtype) == "category"


def test_bool_coercions():
    schema = Schema([{"name": "b", "family": "bool"}])
    words = schema.enforce(pd.DataFrame({"b": ["yes", "No", "TRUE", "f", "1", "0"]}))["b"]
    assert words.dtype == bool and words.tolist() == [True, False, True, False, True, False]
    numbers = schema.enforce(pd.DataFrame({"b": [1, 0, 1]}))["b"]
    assert numbers.dtype == bool and numbers.tolist() == [True, False, True]
    with_null = schema.enforce(pd.DataFrame({"b": [1.0, np.nan]}))["b"]
    assert str(with_null.dtype) == "boolean" and bool(with_null.iloc[0]) is True and pd.isna(with_null.iloc[1])
    with pytest.raises(SchemaError, match="not 0/1"):
        schema.enforce(pd.DataFrame({"b": [1, 2]}))
    with pytest.raises(SchemaError, match="not recognised booleans"):
        schema.enforce(pd.DataFrame({"b": ["yes", "maybe"]}))


def test_numeric_coercions_and_failures():
    schema = Schema([{"name": "i", "family": "int"}, {"name": "f", "family": "float"}])
    fixed = schema.enforce(pd.DataFrame({"i": ["1", "2"], "f": ["2.5", None]}))
    assert str(fixed["i"].dtype) == "int64" and fixed["i"].tolist() == [1, 2]
    assert str(fixed["f"].dtype) == "float64" and fixed["f"].tolist()[0] == 2.5 and np.isnan(fixed["f"].iloc[1])
    booleans = schema.enforce(pd.DataFrame({"i": [True, False], "f": [True, False]}))
    assert booleans["i"].tolist() == [1, 0] and booleans["f"].tolist() == [1.0, 0.0]
    with pytest.raises(SchemaError) as excinfo:
        schema.enforce(pd.DataFrame({"i": ["1", "x"], "f": ["y", "2.0"]}))
    problems = excinfo.value.problems
    assert [p.column for p in problems] == ["i", "f"] and all(p.kind == "dtype_mismatch" for p in problems)
    assert problems[0].detail["bad_values"] == ["x"] and problems[0].detail["bad_count"] == 1
    assert "e.g. 'x'" in problems[0].message
    with pytest.raises(SchemaError, match="datetime values cannot become"):
        schema.enforce(pd.DataFrame({"i": pd.to_datetime(["2024-01-01"]), "f": [1.0]}))


def test_string_coercion_from_other_families():
    schema = Schema([{"name": "s", "family": "string"}])
    dates = schema.enforce(pd.DataFrame({"s": pd.to_datetime(["2024-01-01", None])}))["s"]
    assert dates.dtype == object and str(dates.iloc[0]).startswith("2024-01-01") and pd.isna(dates.iloc[1])
    cat = schema.enforce(pd.DataFrame({"s": pd.Categorical(["p", "q"])}))["s"]
    assert cat.dtype == object and cat.tolist() == ["p", "q"]
    # A whole number reads as a whole number: '2.0' would match nothing downstream, and the
    # same column arriving as int64 already gave '2' (see test_regressions_round2.py).
    floats = schema.enforce(pd.DataFrame({"s": [1.5, 2.0]}))["s"]
    assert floats.tolist() == ["1.5", "2"]


def test_extra_column_policies(schema, train):
    batch = train.assign(extra=["e"] * 4)
    assert "extra" not in schema.enforce(batch).columns
    kept = schema.enforce(batch, extra="keep")
    assert list(kept.columns) == schema.column_names + ["extra"] and kept["extra"].tolist() == ["e"] * 4
    with pytest.raises(SchemaError, match="extra_column 'extra'"):
        schema.enforce(batch, extra="raise")


def test_missing_column_policies(schema):
    batch = pd.DataFrame({"id": [1, 2]})
    filled = schema.enforce(batch)
    assert list(filled.columns) == schema.column_names and len(filled) == 2
    assert str(filled["city"].dtype) == "object" and filled["city"].isna().all()
    assert str(filled["score"].dtype) == "float64" and filled["score"].isna().all()
    assert str(filled["active"].dtype) == "boolean" and filled["active"].isna().all()
    assert str(filled["signup"].dtype).startswith("datetime64") and filled["signup"].isna().all()
    assert str(filled["tier"].dtype) == "category" and list(filled["tier"].cat.categories) == ["a", "b"]
    int_schema = Schema([{"name": "x", "family": "int"}, {"name": "y", "family": "int"}])
    assert str(int_schema.enforce(pd.DataFrame({"x": [1]}))["y"].dtype) == "Int64"
    with pytest.raises(SchemaError) as excinfo:
        schema.enforce(batch, missing="raise")
    assert [p.column for p in excinfo.value.problems] == ["city", "score", "active", "signup", "tier"]


def test_strict_mode_raises_listing_every_problem(schema, train, messy):
    with pytest.raises(SchemaError) as excinfo:
        schema.enforce(messy, mode="strict")
    kinds = [p.kind for p in excinfo.value.problems]
    assert kinds[:2] == ["extra_column", "wrong_order"] and "dtype_mismatch" in kinds and "out_of_range" not in kinds
    same = schema.enforce(train, mode="strict")
    pd.testing.assert_frame_equal(same, train)
    assert same is not train
    kept = schema.enforce(train.assign(extra=1), mode="strict", extra="keep")
    assert "extra" in kept.columns
    with pytest.raises(SchemaError):
        schema.enforce(train.assign(extra=1), mode="strict")
    # out-of-range values are warnings, so strict mode lets them through
    assert len(schema.enforce(train.assign(score=[9.0] * 4), mode="strict")) == 4


def test_invalid_arguments(schema, train):
    for kwargs in ({"mode": "fix"}, {"extra": "ignore"}, {"missing": "drop"}):
        with pytest.raises(ValueError):
            schema.enforce(train, **kwargs)


def test_top_level_enforce_accepts_paths_and_dicts(tmp_path, schema, messy):
    path = tmp_path / "messy.csv"
    messy.to_csv(path, index=False)
    fixed = schema_guard.enforce(path, schema.to_dict(), extra="keep")
    assert list(fixed.columns) == schema.column_names + ["extra"]
    assert str(fixed["id"].dtype) == "Int64"
    schema_path = schema.save(tmp_path / "schema.json")
    assert list(schema_guard.enforce(messy, schema_path).columns) == schema.column_names
