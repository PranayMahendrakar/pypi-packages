# quality-predictor

Predict product quality from manufacturing parameters before final inspection, and see
which settings drive it.

## Install

```bash
pip install quality-predictor
```

## Quickstart

```python
import pandas as pd
from quality_predictor import predict_quality

df = pd.DataFrame({
    "temperature": [212, 188, 219, 195, 208, 191, 221, 186, 205, 199, 215, 184, 210, 193, 217, 190],
    "pressure": [12.2, 9.4, 12.8, 10.1, 11.9, 9.7, 12.9, 9.1, 11.6, 10.4, 12.4, 8.9, 12.0, 9.9, 12.6, 9.3],
    "machine": ["A", "A", "B", "A", "B", "B", "A", "B", "A", "B", "A", "A", "B", "A", "A", "B"],
    "quality": ["pass", "fail", "pass", "fail", "pass", "fail", "pass", "fail",
                "pass", "fail", "pass", "fail", "pass", "fail", "pass", "fail"],
})

result = predict_quality(df, "quality")
print(result.summary())
```

```text
quality-predictor: quality (classification, pass/fail) from 3 parameter(s)
rows: 16 | trained on: 12 | held out: 4
held-out scores: accuracy 1.000, precision 1.000, recall 1.000, f1 1.000, roc_auc 1.000
classes: 'fail', 'pass' | good outcome: 'pass' (precision, recall and f1 describe this class)

WHAT DRIVES QUALITY (share of the model's decisions)
  pressure      50.6%  ###############
  temperature   49.4%  ###############
  machine        0.0%

SETTINGS MOST ASSOCIATED WITH GOOD OUTCOMES
  pressure     11.6 to 12.9
  temperature  205 to 221

PREDICTIONS (16 row(s)): pass, fail, pass, fail, pass, ...

NOTES
  - the target is text with 2 distinct value(s), so this is classification
  - only 16 row(s): the model still fits, but the held-out scores are unreliable, ...
  - scores come from 4 held-out row(s); the model you now hold was refitted on all 16
  - class 'pass' reads as the good outcome
  - no new_data was given, so the predictions below are for the rows the model was fitted on
```

## What it does

- Reads the target column and decides on its own whether this is pass/fail
  (classification) or a measured value (regression).
- Builds the whole pipeline for you: median-fill and scale the numbers,
  most-frequent-fill and one-hot the categories, then gradient boosting.
- Holds out a test split during `fit`, so the scores you see are honest, then
  refits on every row so the model you keep has seen all the evidence.
- Ranks the parameters that drive quality by their **original** column names -
  a `machine` column with six values appears once, not as six dummy columns.
- Explains a single part: which of its settings pushed it toward good, and which
  pushed it away.
- Suggests the operating window for each numeric parameter that goes with good
  outcomes. A parameter that moves the outcome too little to act on is given its
  whole observed range instead of an invented setpoint, and a note says so.
- Says when it cannot be trusted: too few rows, a target with one class, columns
  with no values, categories it has never seen.

## API

```python
from quality_predictor import QualityModel, predict_quality
```

**`predict_quality(df, target, new_data=None, **kw)` -> `QualityResult`**
Fit and predict in one call. `df` is a DataFrame or a path to a `.csv`/`.tsv`/
`.parquet` file. `kw` is passed to the model: `task`, `random_state`, `features`,
`test_size`, `good_class`, `higher_is_better`.

**`QualityResult`**
`.metrics`, `.feature_importance` (Series), `.optimal_ranges`, `.predictions`,
`.probabilities`, `.classes`, `.good_class`, `.notes`, `.top_factors`,
`.model` (the fitted `QualityModel`), `.explain(row)`, `.summary()`,
`.to_dict()`, `.to_frame()`.

**`QualityModel(task="auto", random_state=0)`**

| Method | What it gives you |
| --- | --- |
| `.fit(df, target, *, features=None, test_size=0.2)` | the fitted model (returns `self`) |
| `.predict(df)` | an array of predicted outcomes |
| `.predict_proba(df)` | class probabilities, in `.classes_` order (classification only) |
| `.evaluate(df=None, target=None)` | the held-out scores, or scores on fresh rows |
| `.metrics` | the held-out scores recorded during `fit` |
| `.feature_importance` | a Series over the original column names, biggest first |
| `.explain(row)` | an `Explanation` with `.prediction`, `.contributions`, `.summary()` |
| `.optimal_ranges()` | `{parameter: (low, high)}` windows linked to good outcomes |
| `.uninformative_parameters` | the parameters whose window is the whole observed range, for want of a signal |
| `.save(path)` / `QualityModel.load(path)` | round-trip a fitted model |

Classification metrics are accuracy, precision, recall, f1 and roc_auc;
regression metrics are r2, mae and rmse. A score that cannot be computed on the
held-out rows is reported as `None` rather than as a zero.

```python
model = QualityModel(random_state=0).fit(df, "quality")
model.feature_importance.head(3)
model.optimal_ranges()
print(model.explain({"temperature": 187, "pressure": 9.2, "machine": "B"}).summary())
```

## CLI

```bash
quality-predictor runs.csv --target quality
quality-predictor runs.csv --target quality --predict today.csv --explain 0
quality-predictor runs.csv --target quality --json --output report.json
quality-predictor --help
```

With no `--target` the last column is used, and the summary says so. `--json`
prints the machine-readable result, `--output` writes it, `--predictions` writes
the predicted rows back out as a table.

## License

MIT. Copyright 2026 Pranay Mahendrakar.
