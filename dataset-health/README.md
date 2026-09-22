# dataset-health

One call tells you what is wrong with a dataset before you model it: missing
values, duplicates, leakage into the target, class imbalance, outliers,
correlated features, identifier columns, dates stored as text - each one
explained, ranked by severity and rolled into a single 0-100 score.

## Install

```
pip install dataset-health
```

Parquet input needs the optional extra: `pip install "dataset-health[parquet]"`.

## Quickstart

```python
import pandas as pd
from dataset_health import diagnose

df = pd.DataFrame({"user_id": range(1, 21),
                   "age": [23, 34, 45, None, 31, 29, 38, 52, 41, 27] * 2,
                   "signup": ["2024-01-05"] * 20,
                   "will_churn": [0, 1] * 10,
                   "churn": [0, 1] * 10})
report = diagnose(df, target="churn")
print(report.summary())
```

```
dataset-health: score 56/100 (needs attention) - 1 critical, 4 warning, 0 info
rows: 20 | columns: 5 | target: churn (classification)

CRITICAL
  target_leakage     will_churn               'will_churn' is nearly identical to target 'churn' (|corr| = 1.000)

WARNING
  missing            age                      'age' is 10.0% missing (2 of 20 rows)
  id_like            user_id                  'user_id' is 100.0% unique (consecutive integers); it identifies rows rather than describing them, so drop it before modeling
  constant           signup                   'signup' has a single value ('2024-01-05'); it carries no information
  dates_as_strings   signup                   'signup' holds dates stored as text (e.g. '2024-01-05'); parse it with pd.to_datetime
```

`report.score` is 0-100, `report.critical` and `report.warnings` are the
filtered issue lists, and `report.to_dict()` / `report.to_markdown()` give you
the same findings as JSON or as a report page.

## What it checks

Every finding is an `Issue` with a `kind`. Passing `target=` switches on the two
checks that need a label; everything else runs either way.

| kind | fires when |
|---|---|
| `missing` | a column has missing values (info below 5%, warning at 5%, critical above 50%), or rows have more than half their values missing |
| `empty_column`, `empty_row` | a column or a row is entirely missing |
| `duplicates` | exact duplicate rows exist (info below 1%, warning at 1%, critical at 20%) |
| `constant`, `near_constant` | a column has one value, or more than 99% one value |
| `id_like` | more than 98% unique: unique text, consecutive or increasing integers, or integers in a column named like an identifier |
| `high_cardinality` | a categorical feature has more than 50 distinct values |
| `mixed_types` | an object column mixes Python types, or mixes numbers-as-text with non-numeric values |
| `numeric_as_strings` | every value of a text column parses as a number |
| `dates_as_strings` | at least 90% of a text column parses as a date |
| `class_imbalance` | the classification target's minority class is under 10% of rows (critical under 1%, or when there is only one class) |
| `target_leakage` | a feature is nearly the target: abs(corr) above 0.95, a single threshold that separates the classes, a value-to-class mapping that holds either way, or a correlation ratio above 0.95 for a numeric target - and, as a warning, a feature named after the target |
| `high_correlation` | two numeric features have abs(corr) above 0.9 |
| `outliers` | more than 5% of a numeric column falls outside the 1.5 x IQR fences |
| `skew` | a numeric column has abs(skew) above 3 |
| `empty_dataset` | the table has no rows or no columns |

**Score.** Starts at 100 and subtracts a weighted penalty per issue - heavier for
critical kinds like leakage and imbalance, lighter for info. Repeats of one kind
cost `base * (1 + log2(n))`, so the tenth slightly-missing column hurts far less
than the first. The score is floored at 0, and `report.grade` puts a word on it:
healthy (90+), mostly healthy (75+), needs attention (50+), unhealthy.

**Task detection.** A non-numeric target, a boolean target, or a numeric target
with at most 20 distinct values is treated as classification; anything else is
regression. `report.task` says which was chosen.

**Sampling.** Tables longer than `sample` rows (200,000 by default) are analyzed
on a seeded random sample, and the report says so through `report.sampled`,
`report.n_rows_analyzed` and a line in `report.notes`. Smaller tables are read
whole, so their counts and shares are exact.

## API

```python
diagnose(df_or_path, target=None, *, sample=200_000, random_state=0) -> HealthReport
```

`df_or_path` is a `pandas.DataFrame`, a `Series`, or a path to a `.csv`, `.tsv`
or `.parquet` file. `target` names the label column; it raises `ValueError` -
listing the columns that do exist - when it is not one of them. `sample=None` or
`0` analyzes every row. The input is never modified.

`HealthReport`

- `.score` - 0-100, higher is healthier; `.grade` - the matching word
- `.issues` - all findings, critical first; `.critical`, `.warnings`, `.info` - filtered
- `.by_kind(kind)`, `.by_column(name)` - filtered the other two ways
- `.counts` - `{"critical": n, "warning": n, "info": n}`; `.kinds` - the kinds present
- `.n_rows`, `.n_columns`, `.n_rows_analyzed`, `.sampled` - shape, and whether it sampled
- `.target`, `.task` - the label column and `"classification"` / `"regression"` / `None`
- `.columns` - per-column profile: role, type, dtype, missing share, distinct count
- `.notes` - what the run decided that you should know (sampling, renames, fallbacks)
- `.summary()` - human text grouped by severity; `.to_dict()` - JSON-safe dict;
  `.to_markdown()` - a report page

`Issue(severity, kind, columns, message, detail)`

`severity` is `"critical"`, `"warning"` or `"info"`; `columns` lists the columns
involved (empty for row-level findings); `detail` is a JSON-safe dict with the
numbers behind the message - shares, counts, correlations, example row labels.
`Issue.to_dict()` and `str(issue)` are both available. Every kind name is in
`dataset_health.KINDS` and every severity in `dataset_health.SEVERITIES`.

```python
HealthChecker(*, sample=200_000, random_state=0, missing_warning=0.05, missing_critical=0.5,
              near_constant=0.99, id_unique_ratio=0.98, high_cardinality=50, imbalance=0.10,
              imbalance_critical=0.01, leakage_corr=0.95, high_corr=0.90, outlier_share=0.05,
              skew=3.0, checks=None)
HealthChecker.check(df_or_path, target=None) -> HealthReport
```

The same run with every threshold exposed. `checks=` limits it to a subset of
`KINDS`, e.g. `HealthChecker(checks=["missing", "duplicates"])`. The instance is
callable, so `checker(df, "churn")` works too.

Duplicate column names raise a `ValueError` that names them, instead of failing
later inside pandas; non-text column names are converted to text and noted. The
library never prints; it logs through `logging.getLogger("dataset_health")`.

## CLI

```
dataset-health data.csv                      # print the summary
dataset-health data.csv --target churn       # add the leakage and imbalance checks
dataset-health data.csv --json               # print to_dict() as JSON
dataset-health data.csv --markdown           # print to_markdown()
dataset-health data.csv --output report.md   # also write it (.md -> Markdown, else JSON)
dataset-health data.csv --sample 0           # analyze every row
dataset-health data.csv --fail-below 70      # exit 1 when the score is under 70
dataset-health data.csv --fail-on-critical   # exit 1 on any critical issue
dataset-health --help
```

The two `--fail-*` flags make it a CI gate; without them the command always
exits 0. `python -m dataset_health data.csv` works the same way.

## License

MIT
