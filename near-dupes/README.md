# near-dupes

Find near-duplicate texts, table rows and images with one call, then drop the copies and keep the best one.

## Install

```
pip install near-dupes
```

Image support needs Pillow: `pip install "near-dupes[images]"`.

## Quickstart

```python
import near_dupes

texts = ["The quick brown fox jumps over the lazy dog.", "the quick brown fox jumps over the lazy dog", "Something else entirely."]
result = near_dupes.find_duplicates(texts)   # or a DataFrame, or a list of image paths
print(result.summary())                       # 1 duplicate group: [0, 1]
print(result.dedupe(texts))                   # keeps the longest copy plus the unrelated text
```

Rows of a DataFrame work the same way: `near_dupes.dedupe(df, key=["name", "city"])` compares rows on
those columns and returns the frame without the copies.

## What it does

- **Text** (`list[str]`): character n-gram shingles, MinHash (128 permutations, fixed seed) and LSH banding
  propose candidate pairs; every candidate is then scored with the exact Jaccard similarity of its shingle
  sets, so the scores you get back are exact. With 2000 or fewer distinct texts there is no MinHash or
  banding at all: every pair is scored exactly (only when the texts are very long is each pair first screened
  by a MinHash estimate with a wide safety margin). Case, Unicode form (NFKC) and whitespace are normalized first.
- **Records** (`pandas.DataFrame`, a `.csv`/`.parquet` path, or a list of dicts): the key columns of each row
  (default: all columns) are normalized for case, whitespace and punctuation, joined into one string and run
  through the text pipeline. Exact duplicates are always found; NaN cells count as empty.
- **Images** (paths or `PIL.Image` objects): 64-bit difference hash (dHash); similarity is `1 - hamming / 64`.
- `threshold` is the minimum similarity in `(0, 1]`; `1.0` means exact duplicates only.
- Duplicate groups are the connected components of the pairs found. One representative per group is kept:
  the longest text, the first row, or the first image.
- Blank items (empty strings, rows whose key cells are all missing) are never treated as duplicates of each other.
- Result indices are 0-based positions in the input, so they line up with `list` positions and `df.iloc`.
- Everything is deterministic: the MinHash permutations come from a fixed `random_state`.

## API

```python
near_dupes.find_duplicates(items, *, kind="auto", threshold=0.85, key=None, n_gram=3) -> DuplicateResult
near_dupes.dedupe(items, **kw)   # find_duplicates(items, **kw).dedupe(items) in one line
```

`items` is a list of strings, a `pandas.DataFrame` (or a `.csv`/`.parquet` path, or a list of dicts), or a
list of image paths / `PIL.Image` objects. `kind` is detected from the input unless given (`"text"`,
`"records"`, `"images"`). `key` names the DataFrame column(s) to compare rows on; `n_gram` is the character
shingle length.

```python
import pandas as pd
df = pd.DataFrame({"name": ["Acme Corp", "ACME Corp.", "Globex"], "city": ["New York", "new york", "Springfield"]})
near_dupes.find_duplicates(df, key=["name", "city"]).groups   # [[0, 1]]
near_dupes.dedupe(df, key=["name", "city"])                   # rows 0 and 2, original index kept
near_dupes.dedupe(["a.jpg", "a_copy.jpg", "b.jpg"])           # image paths; needs near-dupes[images]
```

`DuplicateFinder(kind="auto", threshold=0.85, key=None, n_gram=3, num_perm=128, random_state=0, normalize=True, all_pairs_max=2000)`
is the class underneath, with `.find(items)` and `.dedupe(items)`, for the extra knobs.
`text_similarity(a, b, n_gram=3)` is the exact shingle Jaccard between two strings.

`DuplicateResult`

- `.groups` - `list[list[int]]`, each group of near-duplicate indices (size >= 2), ascending
- `.pairs` - `list[(i, j, score)]`, the edges found (`i < j`); exact duplicates score `1.0`
- `.representatives` - the index kept for each group; `.keep_indices` - all singletons plus one
  representative per group, ascending; `.drop_indices` - what `dedupe` removes
- `.n_duplicates`, `.n_groups`, `.n_items`, `.kind`, `.threshold`
- `.summary()` - human-readable text; `.to_dict()` - JSON-safe dict
- `.dedupe(items)` - `items` with duplicates removed, same type as the input
  (a DataFrame keeps its original index labels)

## CLI

```
near-dupes INPUT [INPUT ...] [--kind auto|text|records|images] [--threshold 0.85]
           [--key COL [COL ...]] [--n-gram 3] [--json] [--output PATH]
```

- `near-dupes contacts.csv --key name email` prints the summary for a table.
- `near-dupes tickets.txt` treats each line as one text.
- `near-dupes photos/` or `near-dupes a.jpg b.jpg` compares images.
- `--json` prints `to_dict()` as JSON; `--output PATH` writes the deduplicated items
  (`.csv`/`.parquet` for tables, one item per line otherwise).

## License

MIT
