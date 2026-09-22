# auto-label

Turn a few keyword, regex or query rules into labels for a whole text list or DataFrame: rules label what they can, a small model trained on those labels covers the rest, and an optional LLM callable takes the items nobody was sure about.

## Install

```bash
pip install auto-label
```

## Quickstart

```python
import auto_label

texts = ["Free prize, click now to claim it", "Team meeting at 10 in room B", "Win a free voucher today",
         "Agenda for the meeting is attached", "Claim your prize before midnight", "Meeting notes from the team",
         "Click now and claim your voucher", "Team notes from room B attached"]
result = auto_label.label(texts, {"spam": ["free", "prize", "win"], "work": ["meeting", "agenda"]})
print(result.summary())
print(result.to_frame())
```

The last two items match no rule; the model trained on the first six labels them `spam` and `work`.

## What it does

- **Rules first.** `keywords` (case-insensitive whole words or phrases), `regex`, a pandas `query`
  string for DataFrames, or any `func(item) -> bool`. A rule fires when every condition it carries holds.
- **Conflicts resolve by weight.** Each firing rule adds its `weight` to its label; the label with the
  largest total wins, ties go to the rule added first. The rule confidence is the share of the winning
  weight (1.0 when only one label fired). A negative weight vetoes.
- **A small model extends the rules.** TF-IDF + logistic regression for text; for a table, each column
  is read for what it holds — free-text columns go through TF-IDF, genuine categories are one-hot
  encoded, numbers are scaled — and `.notes` says which column was read as what. Columns that cannot
  teach the model anything are left out and named there too: a near-unique column of whole numbers in
  order is a row id rather than a measurement (an export sorted by type would otherwise let the id
  outvote the text), and a column that is empty for every rule-labeled row has nothing to learn from.
  Either way the model
  trains on the rule-labeled items and is applied to the rest, with the predicted probability as
  confidence. Training is skipped when fewer than 2 labels or fewer than 5 rule-labeled examples
  exist, or when nothing is left to predict.
- **An optional LLM hook finishes the job.** Items still under `min_confidence` are passed to
  `llm(text, candidate_labels)`; a returned label is used with source `"llm"`. If the callable raises,
  the error is logged and the item stays as it was.
- **Anything else is left `None`.** `coverage` tells you how much was labeled.
- Text input: `list[str]`, a pandas `Series`, or a 1-D array. Tabular input: a `DataFrame`, a dict of
  columns, or a path to `.csv` / `.parquet` (`pip install auto-label[parquet]`). On tables, keyword and
  regex rules see the string-like columns of each row joined by spaces; `func` gets the row as a
  Series, whose `.name` is that row's label in your index.
- Deterministic (`random_state`), no network, no model downloads, logs through `logging`.

## API

```python
auto_label.label(data, rules, *, labels=None, min_confidence=0.6, random_state=0, model=True, llm=None) -> LabelResult
```

`rules` maps a label to a list of keywords, a regex string, a callable, a dict of `add_rule` keyword
arguments (`{"keywords": [...], "regex": ..., "query": ..., "func": ..., "weight": ...}`), or a list of
such dicts.

```python
labeler = auto_label.Labeler(labels=None, min_confidence=0.6, random_state=0)
labeler.add_rule(label, *, keywords=None, regex=None, query=None, func=None, weight=1.0)  # returns self
result = labeler.label(data, *, model=True, llm=None)
labeler.model_        # the fitted scikit-learn pipeline from the last call, or None
```

`labels` (optional) is a closed set: rules must use one of them and LLM answers outside it are ignored.

`LabelResult`

| attribute / method | meaning |
| --- | --- |
| `.labels` | `list[str \| None]`, one per item |
| `.confidence` | `list[float]`; rule share, model probability, 1.0 for the LLM, 0.0 when unlabeled |
| `.source` | `list["rule" \| "model" \| "llm" \| None]` |
| `.coverage` | fraction of items that got a label |
| `.items`, `.index` | the original items (text, or a row dict) and the input index |
| `.counts`, `.source_counts` | items per label, items per source |
| `.to_frame()` | DataFrame `item, label, confidence, source` on the input index |
| `.to_dict()` | JSON-safe dict with totals, counts, notes and every record |
| `.summary()` | a few lines of human-readable text |
| `.model_trained`, `.notes` | whether the model ran, and what each stage did |

## CLI

```bash
auto-label tickets.csv --column text --rule billing=invoice,refund --rule bug=crash,error
auto-label lines.txt --regex urgent="(?i)asap|urgent" --json
auto-label orders.csv --query big="amount > 100" --rules more_rules.json --output labeled.csv
```

`INPUT` is a `.csv` / `.parquet` file (tabular, or one text column with `--column`), a text file with
one item per line, or `-` for stdin. Repeat `--rule LABEL=kw1,kw2`, `--regex LABEL=PATTERN` and
`--query LABEL=EXPR` as needed, or put the same rules in a JSON file for `--rules`. By default the
summary is printed; `--json` prints `to_dict()`; `--output PATH` writes the labeled table as `.csv`
or `.parquet` (`pip install auto-label[parquet]`), or the full result as `.json`. For tabular input
the written table is the input columns plus `label, confidence, source`, so it joins back to the
source file; if the input already uses one of those names, or `index`, the written column gets a free
one (`label_2`, `index_2`) so every header stays unique. The destination is replaced only once the
whole file has been written, so a write that fails leaves the file already at that path intact.

`--encoding` sets the encoding of the input, whether that is a CSV, a text file or piped stdin.
Without it everything is read as UTF-8 and undecodable bytes are replaced, with a warning both logged
and recorded in `.notes`. `--no-model`, `--min-confidence`, `--labels` and `--random-state` mirror the
Python API.

A text file is read one item per line; blank lines are skipped and each record keeps its 0-based
line number in the file as its index, so results map back to the source. `-` reads stdin the same
way, and decodes it as UTF-8 whatever the console's locale encoding happens to be, so the same bytes
give the same labels piped in as they do saved to a file.

## License

MIT
