# data-drift-lite

Detect whether production data has drifted from training data, column by column, with a single call.

## Install

```
pip install data-drift-lite
```

Reading `.parquet` files needs `pip install "data-drift-lite[parquet]"`.

## Quickstart

```python
import pandas as pd
import data_drift_lite

reference = pd.DataFrame({"age": list(range(20, 60, 2)), "plan": ["basic", "pro"] * 10})
current = pd.DataFrame({"age": list(range(50, 90, 2)), "plan": ["pro"] * 18 + ["enterprise"] * 2})
report = data_drift_lite.detect(reference, current)
print(report.summary())
```

Prints:

```
data-drift-lite: DRIFT DETECTED: 2 of 2 columns drifted (100%)
reference rows: 20 | current rows: 20 | drifted when p < 0.05 or PSI > 0.2

  column  kind         test  statistic  p-value     PSI  status
  age     numeric      ks       0.7500  9.5e-06  6.4703  DRIFTED
  plan    categorical  chi2    14.2857   0.0008  5.1829  DRIFTED

notes:
  - plan: 1 category unseen in reference: 'enterprise'
```

Then `report.drifted` is `True`, `report.drifted_columns` is `["age", "plan"]`, and
`report.to_dict()` is ready for `json.dumps`.

## What it checks

- **Numeric columns** (ints, floats, nullable ints/floats; datetimes and timedeltas
  are compared as int64 nanoseconds): a two-sample Kolmogorov-Smirnov test
  (`scipy.stats.ks_2samp`) plus the Population Stability Index over 10 quantile
  bins built from the reference.
- **Categorical columns** (strings, objects, `category`, and bool): a chi-square
  test on category frequencies (`scipy.stats.chi2_contingency`) plus PSI over the
  reference categories, with every category the reference never saw pooled into a
  single extra bin.
- A column is **drifted** when `p_value < threshold` (default 0.05) **or**
  `psi > psi_threshold` (default 0.2). Pass `None` for either to switch that rule off.
- **Missing values count.** They form their own bin for PSI (and their own category
  in the chi-square table), so a column whose values start disappearing is flagged
  even when the values that remain look the same. The KS test uses the non-missing
  values. `+inf`/`-inf` are treated as missing.
- **Schema drift** is reported alongside: columns missing from the current data,
  new columns, and columns whose type family changed (for example `int64 -> object`).
  Missing columns and type changes set `report.drifted`; new columns are listed but
  do not raise the flag.
- **Guard rails.** Constant columns get PSI 0, never NaN or inf. Numeric columns with
  ten or fewer distinct values get one bin per value, so a 95/5 to 5/95 flip in a
  0/1 column is caught. A reference or batch with fewer than 20 rows produces a
  warning note in the report (and a `logging` warning) instead of a crash. Each side
  is capped at `sample` random rows (default 100,000, seeded by `random_state`) so a
  check stays fast on big tables.

## API

### `detect(reference, current, *, columns=None, threshold=0.05, psi_threshold=0.2, sample=100_000, random_state=0) -> DriftReport`

The one-call path. `reference` and `current` accept a pandas DataFrame, a Series,
or a path to a `.csv` / `.parquet` file.

- `columns`: compare only these columns (default: every reference column).
- `threshold`: p-value below which a column is drifted; `None` disables the rule.
- `psi_threshold`: PSI above which a column is drifted; `None` disables the rule.
- `sample`: cap each side at this many random rows; `None` uses every row.
- `random_state`: seed for that sampling, so results are reproducible.

### `DriftMonitor(reference, *, columns=None, threshold=0.05, psi_threshold=0.2, sample=100_000, random_state=0)`

Profiles the reference once; `monitor.check(batch)` returns a `DriftReport` for
each batch. Use it when scoring many batches against the same training data.

```python
monitor = data_drift_lite.DriftMonitor(train_df, psi_threshold=0.1)
for batch in batches:
    report = monitor.check(batch)
    if report.drifted:
        alert(report.summary())
```

### `DriftReport`

| attribute / method | meaning |
| --- | --- |
| `columns` | `dict[column -> ColumnDrift]` for every compared column, in reference order |
| `drifted_columns` | `list[str]` of the columns flagged as drifted |
| `drift_share` | fraction of compared columns that drifted |
| `drifted` | `True` if any column drifted, a column is missing, or a dtype changed |
| `missing_columns`, `new_columns`, `dtype_changed` | schema drift; `dtype_changed` maps `column -> (reference dtype, current dtype)` |
| `schema` | the same three as a `SchemaDrift` dataclass with its own `.drifted` and `.to_dict()` |
| `reference_rows`, `current_rows`, `threshold`, `psi_threshold`, `notes` | what the check ran on |
| `summary()` | human-readable text (also what `str(report)` returns) |
| `to_dict()` | JSON-safe dict: plain Python numbers, `None` for anything not computable |

### `ColumnDrift`

Dataclass with `kind` (`"numeric"` or `"categorical"`), `test` (`"ks"` or `"chi2"`),
`statistic`, `p_value`, `psi`, `drifted`, `reference_stats`, `current_stats`,
`name`, `notes`, and `to_dict()`. Stats hold `dtype`, `count`, `missing_share`,
then `mean`/`std`/`min`/`median`/`max` for numeric columns (ISO strings for
datetimes) or `n_categories` and the `top` category shares for categorical ones.
Any statistic that could not be computed is `None`, and `notes` says why.

## CLI

```
data-drift-lite train.csv batch.csv
data-drift-lite train.parquet batch.parquet --columns age plan --psi-threshold 0.1
data-drift-lite train.csv batch.csv --json
data-drift-lite train.csv batch.csv --output report.json --fail-on-drift
```

`data-drift-lite REFERENCE CURRENT` prints the summary. Options:

- `--columns COL [COL ...]`, `--threshold P`, `--psi-threshold PSI`,
  `--sample N` (0 disables sampling), `--random-state SEED` mirror `detect()`.
- `--json` prints `to_dict()` as JSON instead of the summary.
- `--output PATH` also writes that JSON to a file.
- `--fail-on-drift` exits with status 1 when drift is detected, for CI and cron jobs.
- `--version`, `--help`.

## License

MIT
