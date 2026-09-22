# dataset-splitter

Train/validation/test splits that cannot leak: stratified, grouped, time-aware, and checked, in one call.

## Install

```
pip install dataset-splitter
```

## Quickstart

```python
import pandas as pd
from dataset_splitter import split

df = pd.DataFrame({"user_id": [i // 3 for i in range(60)], "x": range(60), "label": [i % 2 for i in range(60)]})
s = split(df, target="label", group="user_id", test_size=0.2, val_size=0.1)
print(s.train.shape, s.val.shape, s.test.shape)
print(s.report().summary())
```

The summary shows the sizes, the class balance of every part, and the leakage checks: no
`user_id` on two sides, no duplicate row on two sides, rows in exactly one part, `ok: True`.

## What it does

- **Stratified** on `target`: classes are balanced across train/val/test. Numeric targets are
  stratified on quantile bins. Tiny classes are pooled, and a tiny dataset falls back to a plain
  random split, both with a warning instead of an error.
- **Grouped** on `group`: every value of the column (or combination of columns) lands entirely on
  one side, like scikit-learn's `GroupShuffleSplit` but with row-accurate sizes and stratification.
  `group="auto"` detects id-like columns (`customer_id`, `userId`, `session_uuid`, `patient_no`, ...)
  and, when several are found, links rows that share any of them.
- **Time-aware** on `time`: a chronological split with no shuffling; the oldest rows train, the
  newest test. Combined with `group`, groups are ordered by their first timestamp and kept whole.
- **Duplicate-safe**: exact duplicate rows are grouped before splitting (`dedupe=True`), so the
  same row can never sit in both train and test.
- **Checked**: `report()` recomputes everything from the output frames: sizes, class balance,
  group overlap (must be 0), duplicate leakage (must be 0), time ordering, and an exact partition
  check (no row lost or duplicated). `report().ok` is the single flag to look at.
- **Predictable**: the original index values are preserved and listed in `.indices`; row order
  inside each part is the input order; `random_state` makes every split reproducible.

## API

```python
split(df, *, target=None, group=None, time=None, test_size=0.2, val_size=0.1,
      random_state=0, dedupe=True) -> Split
```

`df` is a DataFrame or a path to a `.csv` / `.tsv` / `.parquet` file. `test_size` and `val_size`
are fractions (floats below 1) or absolute row counts (ints); `val_size=0` disables validation.

`Splitter(target=..., group=..., time=..., test_size=..., val_size=..., random_state=...,
dedupe=..., stratify="auto", n_bins=10, max_categories=20)` is the class underneath, for
control over stratification (`stratify=True/False`), the number of quantile bins for numeric
targets, and the cardinality below which a numeric target counts as categorical. `.split(df)`.

`Split`

- `.train`, `.val`, `.test` - DataFrames (`.val` is `None` when `val_size=0`)
- `.indices` - `{"train": [...], "val": [...], "test": [...]}` original index values
- `.strategy` - what was done (method, columns used, number of groups, ...); `.warnings`
- `.report()` - a `SplitReport`
- `.save(dir, format="csv" | "parquet")` - writes `train/val/test.<format>` and `report.json`,
  returns the paths (parquet needs `pip install dataset-splitter[parquet]`)

`SplitReport`

- `.ok` - True when nothing leaks and the parts partition the input exactly
- `.sizes`, `.fractions`, `.class_balance`, `.class_counts`, `.balance_max_deviation`,
  `.target_stats` (numeric targets)
- `.group_overlap`, `.duplicate_leakage`, `.time_ordering`, `.partition`, `.warnings`
- `.summary()` - human-readable text; `.to_dict()` - JSON-safe dict

`detect_id_columns(df)` returns the columns `group="auto"` would use; `load_table(path)` reads
a `.csv` / `.tsv` / `.parquet` file.

## CLI

```
dataset-splitter data.csv --target label --group customer_id
dataset-splitter data.csv --time timestamp --test-size 0.2 --val-size 0.1 --json
dataset-splitter data.csv --group auto --output splits/ --format parquet
```

Prints the report (`--json` for `to_dict()` as JSON), writes the parts with `--output DIR`, and
exits with status 1 when the report is not ok, 2 on bad input. `dataset-splitter --help` lists
every option.

## License

MIT
