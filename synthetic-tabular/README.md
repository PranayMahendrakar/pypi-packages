# synthetic-tabular

Generate realistic synthetic tabular data that keeps the distributions and correlations of your real table, so you can share, test and prototype without handing out the original rows, and without a GPU.

## Install

```
pip install synthetic-tabular
```

## Quickstart

```python
import pandas as pd
import synthetic_tabular as st

df = pd.DataFrame({"age": [23, 35, 41, 29, 52, 38, 45, 31],
                   "city": ["Pune", "Delhi", "Pune", "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi"],
                   "spend": [120.5, 340.0, 410.2, 210.0, 620.9, 380.4, 455.0, 290.7]})
synthetic = st.generate(df, n=100, random_state=0)  # same columns and dtypes, 100 new rows
print(st.evaluate(df, synthetic).summary())         # how close are the distributions?
```

## What it does

- Fits a **Gaussian copula**: every column is rank-transformed to standard-normal
  scores, the correlation matrix of those scores is learned (repaired to the nearest
  positive-definite matrix when needed), new correlated normals are sampled and mapped
  back through each column's own distribution.
- **Numeric columns** are inverted through their empirical quantile function, so the
  output has the same shape, stays inside the observed min/max, keeps integer columns
  integer and keeps the number of decimals seen in the input.
- **Categorical, boolean and string columns** are sampled by their observed
  frequencies; booleans stay boolean, `category` dtypes keep their categories.
- **Datetime and timedelta columns** are modelled as nanoseconds, converted back to the
  original dtype (time zone included) and rounded to the finest resolution seen in the
  column (days stay days, seconds stay seconds).
- **Missing values** are reinjected per column at the rate seen in the input.
- **Constant columns** and columns with a single category are copied as-is.
- **High-cardinality text** (more than 50% unique strings, for example names or free
  text) cannot be modelled by frequency: those columns are sampled with replacement
  from the original values and flagged with a `UserWarning`. Raise `text_threshold` to
  model them as categories instead.
- Output columns and dtypes match the input; `n` can be larger than the input.
- Paths are accepted wherever a DataFrame is: `.csv`, `.tsv` and `.parquet`
  (`pip install synthetic-tabular[parquet]`). CSV columns whose every value is an
  ISO-8601 date (`2024-03-31`, `2024-03-31 10:15:00`) are parsed as datetimes so they
  are modelled as dates rather than treated as text.
- Deterministic: the same `random_state` always gives the same rows.
- Ships an `evaluate()` fidelity check: per-column KS statistic (numeric) or total
  variation distance (categorical), correlation-matrix difference and a 0-100 score.

Limits worth knowing: a Gaussian copula captures monotone dependence between columns,
not arbitrary non-linear or multi-modal joint structure, and it is not a privacy
guarantee. Text columns that are resampled contain original values verbatim.

## API

```python
synthetic_tabular.generate(df, n=None, *, random_state=0, preserve=("marginals", "correlations")) -> DataFrame
```
One-liner for the common case. `df` is a DataFrame or a path to a `.csv` / `.parquet`
file; `n` defaults to `len(df)`. `preserve` chooses what to keep: drop
`"correlations"` to sample columns independently, drop `"marginals"` to replace the
empirical numeric distributions with a smooth fitted normal (clipped to the observed range).

```python
synthesizer = synthetic_tabular.Synthesizer(random_state=0, *, preserve=..., text_threshold=0.5)
synthesizer.fit(df)                               # returns self
synthesizer.sample(n=None, *, random_state=None)  # DataFrame; same rows on repeat, new seed for a new batch
synthesizer.summary()                             # text: what was learned per column
synthesizer.to_dict()                             # JSON-safe version of the above
```
After `fit`: `columns_`, `column_kinds_` (`numeric`, `datetime`, `timedelta`,
`categorical`, `bool`, `text`, `constant`), `high_cardinality_columns_`,
`correlation_` (DataFrame of the fitted normal-score correlations) and `n_rows_`.

```python
synthetic_tabular.evaluate(real, synthetic) -> FidelityReport
```
`FidelityReport` fields: `score` (0-100), `marginal_score`, `correlation_score`,
`correlation_mad` (mean absolute difference of the Spearman correlation matrices,
`None` with fewer than two non-constant columns), `columns` (dict of
`ColumnFidelity`: `name`, `kind`, `metric` `"ks"`/`"tvd"`, `value`, `score`,
`missing_rate_real`, `missing_rate_synthetic`), `n_real`, `n_synthetic`,
`missing_columns`. Methods: `summary()` (text) and `to_dict()` (JSON-safe).

## CLI

```
synthetic-tabular data.csv                          # generate a synthetic copy and print the fidelity report
synthetic-tabular data.csv --n 1000 -o synth.csv    # write 1000 synthetic rows (csv, tsv or parquet)
synthetic-tabular data.csv --evaluate synth.csv     # score an existing synthetic file
synthetic-tabular data.csv --json                   # report as JSON
synthetic-tabular --help
```
Options: `--random-state SEED`, `--no-correlations` (marginals only), `--quiet`.
Flagged text columns are reported on stderr as `note:` lines.

## License

MIT
