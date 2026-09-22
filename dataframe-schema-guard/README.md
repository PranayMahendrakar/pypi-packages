# dataframe-schema-guard

> Installs as `dataframe-schema-guard`; imports as `schema_guard`. The plain name
> `schema-guard` is blocked on PyPI for being too close to the existing
> `schemaguard` and `schema-guardian` projects.

Stop ML pipelines from breaking when incoming data changes shape: infer a schema from one good DataFrame, then validate or enforce it on every batch that follows.

## Install

```bash
pip install dataframe-schema-guard
```

Reading or writing `.parquet` files needs `pip install "schema-guard[parquet]"`.

## Quickstart

```python
import pandas as pd
import schema_guard

schema = schema_guard.infer(pd.DataFrame({"id": [1, 2, 3], "city": ["Oslo", "Paris", "Oslo"], "score": [0.5, 0.9, 0.7]}))
new = pd.DataFrame({"city": ["Paris", "Rome"], "id": [4.0, None], "score": [0.1, 0.2]})
print(schema.validate(new).summary())   # id arrived as float with a null, unknown city 'Rome', columns out of order
fixed = schema.enforce(new)             # id -> Int64, columns back in schema order; new itself is never modified
```

The `print` lists every problem in a stable order: frame-level problems first, then each column in schema order. Warnings are reported but do not fail validation:

```
Schema validation: FAILED - 2 rows x 3 columns (4 errors, 2 warnings)
  [error] wrong_order: columns are not in schema order: expected ['id', 'city', 'score'], got ['city', 'id', 'score']
  [error] dtype_mismatch 'id': expected int, got float64 (float); values are whole numbers, enforce() coerces them to Int64
  [error] unexpected_null 'id': 1 null value(s) in a non-nullable column
  [warning] out_of_range 'id': 1 value(s) outside the range 1 .. 3 (observed 4.0 .. 4.0)
  [error] unknown_category 'city': 1 value(s) not in the allowed categories (e.g. 'Rome')
  [warning] out_of_range 'score': 2 value(s) outside the range 0.5 .. 0.9 (observed 0.1 .. 0.2)
```

Keep the schema next to the model with `schema.save("schema.json")` and get it back with `schema_guard.load("schema.json")`. The JSON is human-readable, safe to edit by hand, and `load(save(schema)) == schema` always holds. It is standard [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) JSON, so `jq`, JavaScript, Go and anything else read it too: a value with no JSON literal (`inf`, `-inf`, `NaN`) is written as `null`, never as a bare `Infinity` token that only Python accepts.

Column names must be strings or integers - the two label types JSON keeps faithfully. A frame with other labels (`Timestamp` columns from a `pivot_table`, floats, tuples) raises a `ValueError` telling you to convert them first, rather than silently renaming them into something that matches nothing.

## What it checks

`schema.validate(df)` never changes the frame; it reports:

- `missing_column` - a schema column is absent from the frame.
- `extra_column` - the frame has a column the schema does not know.
- `wrong_order` - the schema columns appear in a different order (positional feature arrays care).
- `dtype_mismatch` - the dtype family differs. Families are `int`, `float`, `bool`, `datetime`, `string`, `category`; nullable `Int64`/`boolean` and `object` columns are classified by their values. Typical catches: an int column arriving as `float64` because of nulls, a zip code arriving as `int64`, dates arriving as strings, a `category` column arriving as `object`. The timezone of a `datetime` column is part of the schema, so a UTC column arriving tz-naive (a drift that shifts every timestamp by hours while the dtypes still look compatible) is reported here too. A column whose *values* are the right family but which is backed by `object` - a `NUMERIC` column out of SQL Server arriving as `Decimal`s, a frame that came back through JSON - is reported as a **warning**: `infer()` classifies an object column by its values, so making that an error would have the training frame fail the schema learned from it. `enforce()` gives it a real dtype.
- `unexpected_null` - nulls in a column that had none when the schema was inferred (or is marked `nullable: false`).
- `unknown_category` - text/category values outside the recorded allowed set (recorded for columns with at most `categorical_max_unique` distinct values, 50 by default; a text column that had no values records no set). A text column whose values are **all distinct** records no set either, and logs a warning saying so: every value being unique is the signature of an order id, an email or free text, not of a closed set, and freezing one would reject every later batch. Repetition is the evidence; a declared `category` dtype keeps the categories you declared regardless. A value that names a stored category in another JSON type matches it - a zip code arriving as `1001` or `1001.0` finds the stored `'1001'` - so `validate()` never calls unknown what `enforce()` maps straight onto a category. The two forms have to name each other exactly: `'02134'` is not `2134`, because the leading zero is the reason it is text at all.
- `out_of_range` - numeric values outside the recorded `min`/`max`. This is a **warning**: it does not fail validation unless you pass `strict_ranges=True`.

`schema.enforce(df)` returns a new frame and never mutates the input:

- coerces every column to its family: `float` with `NaN` -> `Int64`, `"12"` -> `12`, `"2024-01-31"` -> `datetime64`, `"yes"`/`"no"`/`1`/`0` -> `bool`, `int` -> `str`, `object` -> `category` using the schema's categories so category codes stay stable between batches (unseen values are appended as new categories, never silently turned into nulls; a date matches its stored category whether it arrives as a `Timestamp` or as `"2024-01-01"`, and a zip code matches whether the CSV reader made the column text, `int64` or `float64`);
- puts every `datetime` column in the schema's timezone: a tz-naive batch is localised to it, a differently-zoned one is converted to it (the instant is preserved), and a tz-aware batch for a naive schema is converted to UTC before the offset is dropped. Every batch therefore enforces to the *same* dtype, so `pd.concat` of two enforced frames does not degrade to `object`;
- writes whole numbers into a `string` column as whole numbers: `1001.0` becomes `"1001"`, not `"1001.0"`. One blank cell upcasts an integer column to `float64`, and an id rewritten that way matches nothing in a join, a lookup or an encoder;
- puts the columns in schema order;
- drops extra columns (`extra="drop"`), keeps them at the end (`extra="keep"`) or rejects them (`extra="raise"`);
- adds missing columns as all-null columns of the right dtype (`missing="fill"`) or rejects them (`missing="raise"`);
- with `mode="strict"` converts nothing: it raises `SchemaError` listing every error unless the frame already matches (warnings - out-of-range values, an object-backed column of otherwise-right values - do not block it);
- values that cannot be converted (`"abc"` into an int column, `2.5` into an int column, numbers into a datetime column) raise `SchemaError` naming every failing column with example values, so nothing is lost silently.

## API

```python
schema_guard.infer(df, *, categorical_max_unique=50, numeric_ranges=True, nullable="observed") -> Schema
schema_guard.load(path) -> Schema
schema_guard.validate(df, schema, *, strict_ranges=False) -> ValidationResult
schema_guard.enforce(df, schema, *, mode="coerce", extra="drop", missing="fill") -> DataFrame
schema_guard.as_schema(obj) -> Schema
```

`df` is a pandas DataFrame or a path to a `.csv`/`.tsv`/`.parquet` file wherever a frame is accepted; `schema` is a `Schema`, a schema dict or a path to a saved JSON - `as_schema()` is the function that turns any of those three into a `Schema`, and is what the module-level helpers call.

**`Schema`** - an ordered list of `ColumnSpec`.

- `Schema.infer(df, *, categorical_max_unique=50, numeric_ranges=True, nullable="observed")` - `nullable` is `"observed"` (nullable only where a null was seen), `"always"` or `"never"`.
- `schema.validate(df, *, strict_ranges=False) -> ValidationResult`
- `schema.enforce(df, *, mode="coerce", extra="drop", missing="fill") -> DataFrame`
- `schema.save(path)` / `Schema.load(path)` - human-readable JSON; `schema.to_dict()` / `Schema.from_dict(d)`; `schema.to_json()` / `Schema.from_json(text)`.
- `schema.column_names`, `schema["city"]`, `"city" in schema`, `len(schema)`, iteration over specs, `schema.select("id", "score")`, `schema.drop("target")`, `schema.summary()`.

**`ColumnSpec(name, family, nullable=True, categories=None, min=None, max=None, tz=None)`** - one column; build them by hand when you want control. `name` is a string or an integer; `tz` is the timezone a `datetime` column must be in (`None` means tz-naive) and applies to no other family.

**`ValidationResult`** - `.ok` (no error-level problems), `.problems` (all of them), `.errors`, `.warnings`, `.summary()` (text), `.to_dict()` (JSON-safe), `.raise_if_invalid()` (raises `SchemaError`, otherwise returns the result so it chains).

**`Problem(kind, column, message, detail, severity="error")`** - `kind` is one of `missing_column`, `extra_column`, `dtype_mismatch`, `unexpected_null`, `unknown_category`, `out_of_range`, `wrong_order`; `column` is `None` for frame-level problems; `detail` holds JSON-safe specifics such as counts and example values.

**`SchemaError`** - a `ValueError` whose message lists every problem and whose `.problems` attribute holds them.

**`@guard(schema, mode="coerce", *, extra="drop", missing="fill")`** - decorator that enforces the schema on the first DataFrame argument (positional first, then keyword) before the function runs; works on methods, and accepts a schema path:

```python
@schema_guard.guard("schema.json")
def predict(df):
    return model.predict(df)
```

The saved JSON looks like this:

```json
{
  "format": 1,
  "columns": [
    {"name": "id", "family": "int", "nullable": false, "categories": null, "min": 1, "max": 3},
    {"name": "city", "family": "string", "nullable": false, "categories": ["Oslo", "Paris"], "min": null, "max": null},
    {"name": "score", "family": "float", "nullable": false, "categories": null, "min": 0.5, "max": 0.9}
  ]
}
```

A `datetime` column carries one more key, `"tz"` - `"UTC"`, `"Europe/Oslo"` or `null` for tz-naive. No other family has one.

## CLI

```
schema-guard data.csv                                   # infer and print the schema (same as: schema-guard infer data.csv)
schema-guard infer data.csv --json --output schema.json # print JSON and save it
schema-guard validate schema.json new.csv               # print the summary; exit code 1 when validation fails
schema-guard validate schema.json new.csv --json --strict-ranges --output result.json
schema-guard enforce schema.json new.csv --output fixed.csv --extra keep --missing raise
schema-guard enforce schema.json new.csv --output fixed.csv --strict
```

`infer` also takes `--max-categories N`, `--no-ranges` and `--nullable observed|always|never`. Input and output files may be `.csv`, `.tsv` or `.parquet`.

## License

MIT
