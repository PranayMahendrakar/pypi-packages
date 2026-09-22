# smartclean-df

Turns a messy table into a tidy one in one call: it finds and fixes missing
values, duplicate rows, outliers, numbers/dates/booleans stored as text, and
dirty column names, and tells you exactly what it changed.

## Install

```
pip install smartclean-df
```

Parquet input needs the optional extra: `pip install "smartclean-df[parquet]"`.

## Quickstart

```python
import pandas as pd
from smartclean_df import clean
df = pd.DataFrame({"Name ": ["Ann", " Bob", "Ann", None], "Age": ["34", "NA", "34", "41"], "Joined": ["2024-01-05", "2024-02-10", "2024-01-05", "-"]})
result = clean(df)
print(result.summary())
print(result.df)
```

`result.df` is the cleaned copy (the input is never modified), `result.actions`
lists every change in the order it was made, and `result.summary()` prints it
as text. `df, actions = clean(df)` also works.

## What it does

Every step runs in this order and every change is logged as an `Action`:

1. **Column names** are stripped, inner whitespace collapsed to `_`, and
   lowercased (`"Order ID "` becomes `order_id`); clashes get a `_2` suffix.
2. **String cells** are stripped of surrounding whitespace, and missing tokens
   (`""`, `NA`, `N/A`, `null`, `None`, `-`, `?`, `nan`, any case) become `NaN`.
3. **Numbers stored as text** are parsed when at least 90% of the non-missing
   values parse: `"1,234"`, `"$12.50"`, `"(300)"`, `"45%"` (the `%` is
   stripped, the value is kept as written: `45.0`), `"  7 "`. Whole-number
   columns become `int64`. Codes with leading zeros (`"00123"`) are left alone.
   Columns that mix numbers and text stay `object`.
4. **Dates stored as text** are parsed when at least 90% parse. Day-first
   versus month-first is inferred from the data, and no pandas warnings leak.
5. **Booleans stored as text** (`yes/no`, `true/false`, `y/n`, `t/f`, `1/0`
   strings) become `bool`. Numeric columns are never touched by this step.
6. **Exact duplicate rows** are dropped, then rows and columns that are
   entirely empty.
7. **Missing values** are imputed: median for numeric columns, mode for
   categorical/boolean columns, forward-fill for datetimes. Integer columns
   stay integer when the median is a whole number. `missing="drop"` drops any
   row with a missing value instead; `missing="none"` leaves them.
8. **Outliers** in numeric columns are found with the IQR rule
   (`Q1 - k*IQR`, `Q3 + k*IQR`, `k = iqr_factor`, default 3.0). `"clip"`
   winsorizes them to the bounds, `"flag"` only reports them, `"drop"` removes
   the rows, `"none"` skips the step. Columns with zero IQR (constants,
   0/1 indicators) are never clipped.

If the input has a default `RangeIndex` and rows were dropped, the result is
re-indexed from 0; any other index is kept so rows stay traceable.

## API

```python
clean(df_or_path, *, missing="auto", outliers="clip", iqr_factor=3.0, duplicates=True,
      normalize_columns=True, parse_numbers=True, parse_dates=True, parse_booleans=True,
      dry_run=False) -> CleanResult
```

`df_or_path` is a `pandas.DataFrame` or a path to a `.csv`, `.tsv` or
`.parquet` file. `missing` is `"auto"`, `"drop"` or `"none"`; `outliers` is
`"clip"`, `"flag"`, `"drop"` or `"none"`. With `dry_run=True` the actions are
computed and reported but `result.df` is an unchanged copy of the input.

`CleanResult`

- `.df` - the cleaned `DataFrame` (unchanged copy when `dry_run=True`)
- `.actions` - `list[Action]`, every change made, in order
- `.input_shape`, `.output_shape` - `(rows, columns)` before and after
- `.summary()` - human-readable text
- `.to_dict()` - JSON-safe dict (`shapes`, `actions`, output `dtypes`)

`Action(column, kind, detail, rows_affected)` - `column` is `None` for
table-wide actions (dropping duplicate rows, dropping rows with missing
values). `kind` is one of `rename_column`, `strip_whitespace`,
`missing_tokens`, `parse_numeric`, `parse_datetime`, `parse_boolean`,
`drop_duplicates`, `drop_empty_rows`, `drop_empty_column`, `impute`,
`drop_missing_rows`, `clip_outliers`, `flag_outliers`, `drop_outliers`.

```python
Cleaner(**same options as clean, except dry_run)
Cleaner.fit(df_or_path) -> Cleaner
Cleaner.transform(df_or_path, *, dry_run=False) -> CleanResult
Cleaner.fit_transform(df_or_path, *, dry_run=False) -> CleanResult
```

`fit` learns which columns to parse (and how), the imputation value of every
column, the outlier bounds, and which all-empty columns to drop. `transform`
applies exactly that to new data, so production batches get the same
treatment as the data you fitted on: learned medians and modes fill new gaps,
learned bounds clip new outliers, and a column that was parsed as a date is
parsed as a date again even if the batch is too small to pass the 90% rule.
Columns not seen during `fit` only get the stateless whitespace / missing
token cleanup.

The library never prints; it logs through `logging.getLogger("smartclean_df")`.

## CLI

```
smartclean-df data.csv                    # print the summary of what would change
smartclean-df data.csv --json             # print to_dict() as JSON
smartclean-df data.csv --output clean.csv # also write the cleaned table (.csv, .tsv, .parquet)
smartclean-df data.csv --missing drop --outliers flag --iqr-factor 1.5
smartclean-df data.csv --dry-run          # report only, never write changed data
smartclean-df --help
```

Flags mirror the Python options: `--missing`, `--outliers`, `--iqr-factor`,
`--keep-duplicates`, `--keep-column-names`, `--no-parse-numbers`,
`--no-parse-dates`, `--no-parse-booleans`, `--dry-run`.

## License

MIT
