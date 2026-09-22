# ml-feature-check

One call tells you which columns should not go into the model: the ones that
carry no signal, the ones that repeat another column, the ones that already know
the answer, and the ones that are really row identifiers - each explained, ranked
by severity, and removable with a single `report.apply(df)`.

## Install

```
pip install ml-feature-check
```

Parquet input needs the optional extra: `pip install "ml-feature-check[parquet]"`.

## Quickstart

```python
import pandas as pd
from ml_feature_check import check

df = pd.DataFrame({"row_id": range(60),
                   "age": [23, 34, 45, 31, 29] * 12,
                   "age_in_years": [23, 34, 45, 31, 29] * 12,
                   "country": ["IN"] * 60,
                   "signed_up": ["2024-01-05", "2024-02-11", "2024-03-18"] * 20,
                   "churn_flag": [0, 1] * 30,
                   "churned": [0, 1] * 30})
report = check(df, target="churned")
print(report.summary())
clean = report.apply(df)          # the same frame minus report.drop_recommended
```

```
ml-feature-check: 6 feature(s), 60 rows, target 'churned'
Drop recommended (4):
  row_id        id_like [high]                   100.0% of the 60 values are distinct and they run consecutively: looks like a row identifier, not a feature
                suspicious_name [low]            the name contains 'row', which usually marks a row identifier or bookkeeping column rather than a feature
  age_in_years  duplicate_of [high]              identical to column 'age': keep one of the two
  country       constant [high]                  only one distinct value ('IN')
  churn_flag    leakage_suspect [high]           a 1-feature decision tree reaches 1.000 accuracy on held-out rows for target 'churned' (baseline 0.500)
Worth a look (2):
  age           zero_importance [low]            a small random forest gives it no measurable importance (+0.0000 against a 0.0000 noise level)
  signed_up     date_as_string [low]             text that parses as dates (for example '2024-01-05'): convert with pandas.to_datetime and feed the parts (year, month, weekday) to the model
                zero_importance [low]            a small random forest gives it no measurable importance (+0.0000 against a 0.0000 noise level)
Keep (2): age, signed_up
Notes:
  - target 'churned' was treated as classification
```

`report.features` maps every feature column to its findings, `report.drop_recommended`
is the ordered hit list, `report.keep` is the rest, and `report.to_dict()` /
`report.to_markdown()` give you the same findings as JSON or as a report page.

## What it checks

Every finding is a `Finding` with a `kind`. Passing `target=` switches on the two
checks that need a label; everything else runs either way.

| kind | severity | fires when |
|---|---|---|
| `constant` | high | the column has one distinct value, or is entirely missing |
| `near_constant` | medium | more than 99% of the non-missing values are the same one |
| `high_missing` | medium (high above 95%) | the missing share is above `missing_threshold` |
| `id_like` | high (medium for timestamps) | more than `cardinality_threshold` of the values are distinct, and the column is text, integer or datetime rather than a continuous measurement |
| `duplicate_of` | high | the column is value-for-value identical to an earlier column |
| `highly_correlated_with` | medium | numeric: `abs(pearson)` above `corr_threshold` with a column kept earlier; categorical: bias-corrected Cramer's V above the same threshold |
| `leakage_suspect` | high | with a target: `abs(corr)` above 0.95 for a numeric target, a cross-fitted 1-feature decision tree above 0.99 accuracy / R2, or a categorical column whose values map onto the target almost one-to-one |
| `date_as_string` | low | at least 90% of a text column parses as a date |
| `suspicious_name` | low | the name contains `id`, `uuid`, `index`, `key`, `timestamp`, `row` or `unnamed` |
| `zero_importance` | low | with a target: a small random forest gives the column no more holdout permutation importance than a shuffled copy of a real column |

**What "drop" means.** A `high` or `medium` finding puts the column on
`drop_recommended`, ordered worst first; `low` findings leave it on `keep` and in
`report.review`, the "worth a look" list. Nothing is ever removed for you until
you call `apply()`.

**Checks that cannot fire twice.** A constant column is not then tested for
missingness or cardinality, a column already reported as a `duplicate_of` another
is not also reported as correlated with it, and the column a duplicate points at
is kept. So one problem produces one finding, on one column.

**The survivor of a redundant group is a column worth keeping.** When several
columns carry the same signal, the one left on `keep` is the one with no
drop-level finding of its own and the fewest missing values, not whichever the
frame happened to list first - so a 90%-missing copy never displaces the complete
column beside it. Columns that tie on both criteria fall back to frame order, so
two equally good copies keep whichever the frame lists first; reordering the
frame can swap which of that pair is dropped, and the report stays correct
either way.

**Leakage without false alarms.** The single-feature tree is cross-fitted on two
halves and scored only on rows it did not train on, and the categorical
value-to-target mapping is scored leave-one-out - each row predicted from the
*other* rows sharing its value. An identifier column therefore scores at chance
instead of scoring perfectly, which is what a plain `groupby` accuracy would do.

**Zero variance is not a warning.** Correlation on a constant column yields no
finding rather than a NaN one, and no numpy warning is emitted anywhere. Two
columns are compared only when they share at least 20 non-missing rows (5% of the
frame once that is larger), and when the overlap is partial the finding says how
many rows it rests on - `correlation +1.000 with column 'a' over 46 shared
non-missing rows` - with the count in `detail["n_shared"]`.

**Sampling.** Frames longer than `sample` rows (200,000 by default) are checked on
a seeded random sample; the forest behind `zero_importance` fits on at most 20,000
of those rows. `report.n_rows`, `report.n_rows_checked` and `report.notes` say
exactly what happened. Smaller frames are read whole, so their counts are exact.

## API

```python
check(df, target=None, *, corr_threshold=0.95, missing_threshold=0.6,
      cardinality_threshold=0.98, sample=200_000, random_state=0) -> FeatureReport
```

`df` is a `pandas.DataFrame` or a path to a `.csv`, `.tsv` or `.parquet` file.
`target` names the label column: it is never reported as a feature, and it
unlocks `leakage_suspect` and `zero_importance`. A target that is not a column
raises `ValueError` listing the columns that do exist, and so do duplicate column
names. The input frame is never modified.

`FeatureReport`

- `.features` - `dict[column -> list[Finding]]`, every feature column, worst finding first; an empty list means the column is clean
- `.drop_recommended` - columns with a high or medium finding, most severe first
- `.keep` - the remaining columns, in the original order
- `.review` - kept columns that still have a low-severity finding
- `.flagged` - every column with any finding; `.columns_with(kind)` - filtered by kind
- `.apply(df)` - a copy of `df` without the `drop_recommended` columns; a frame
  that is missing those columns is logged as a warning on the `ml_feature_check`
  logger rather than passed over in silence
- `.iter_findings()` - `(column, finding)` pairs, most severe first
- `.target`, `.n_features`, `.n_rows`, `.n_rows_checked`, `.params`
- `.notes` - what was skipped or sampled and why
- `.summary()` - human text; `.to_dict()` - JSON-safe dict; `.to_markdown()` - a report page

`Finding(kind, severity, message, detail)`

`kind` is one of `ml_feature_check.KINDS`, `severity` is `"high"`, `"medium"` or
`"low"`, and `detail` is a JSON-safe dict with the numbers behind the message -
the other column, the correlation, the score, the threshold that was crossed.

```python
FeatureChecker(*, corr_threshold=0.95, missing_threshold=0.6, cardinality_threshold=0.98,
               sample=200_000, random_state=0, near_constant_threshold=0.99,
               leakage_score_threshold=0.99, leakage_corr_threshold=0.95, min_id_rows=20,
               max_model_rows=20_000, max_model_categories=100, max_pair_columns=400)
FeatureChecker.check(df, target=None) -> FeatureReport
```

The same run with every threshold exposed, for reuse across many frames.

Without a target the two label-dependent checks are skipped and a note says so;
everything else still runs. The library never prints - it logs through
`logging.getLogger("ml_feature_check")`.

## CLI

```
ml-feature-check train.csv                       # print the summary
ml-feature-check train.csv --target churned      # add the leakage and importance checks
ml-feature-check train.csv --json                # print to_dict() as JSON
ml-feature-check train.csv --markdown            # print to_markdown()
ml-feature-check train.csv --output report.md    # also write it (.json/.md/.txt by suffix)
ml-feature-check train.csv --apply clean.csv     # write the frame without the dropped columns
ml-feature-check train.csv --sample 50000 --corr-threshold 0.9
ml-feature-check --help
```

`python -m ml_feature_check train.csv` works the same way. The command exits 1
with a one-line message on a bad file, a missing target or a bad threshold.

## License

MIT
