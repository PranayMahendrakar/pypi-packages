# semantic-dedup

Remove passages that repeat the same meaning, not just the same words - "the meeting was postponed"
and "we moved the meeting to a later date" are one idea, and you only need to keep one of them.

## Install

```
pip install semantic-dedup
```

The only dependency is numpy. Nothing is downloaded at runtime, and no model is ever fetched.

## Quickstart

```python
import semantic_dedup

notes = ["The meeting was postponed.", "the meeting was postponed", "We moved the meeting to a later date.", "Lunch is at noon."]
result = semantic_dedup.dedupe(notes)
print(result.summary())
print(result.texts)
```

```
semantic-dedup: 4 texts, 1 duplicate group, 2 removed, 50.0% smaller
  method tfidf, threshold 0.82, keep longest
  group 1 (3 texts, similarity 1.00 to 1.00)
    removed [0] "The meeting was postponed."
    removed [1] "the meeting was postponed"
    kept    [2] "We moved the meeting to a later date."
['We moved the meeting to a later date.', 'Lunch is at noon.']
```

`semantic_dedup.dedupe("notes.txt")` works the same way on a `.txt`, `.csv` or `.jsonl` file.

## What it does

- **Canonicalizes the wording first.** Text is NFKC-normalized, casefolded, split into words, and then
  run through a small built-in English lexicon: a few dozen paraphrase families collapse onto one token
  each (`postponed`, `delayed`, `pushed back`, `at a later date` all become `postpone`), stopwords go, and
  a light suffix stemmer finishes the job. Negations (`not`, `never`, `no`) are deliberately *kept*, so a
  sentence and its opposite do not look alike.
- **Then scores the canonical text.** Word 1- and 2-grams plus character 3- and 4-grams of the canonical
  string are turned into a sublinear TF-IDF vector; similarity is the cosine between two vectors, in
  `[0, 1]`. Because the characters are taken from the *canonical* string, two differently worded
  sentences that canonicalize alike also share characters, and typos still land close together.
- **Numbers and identifiers are evidence, not noise.** `1000`, `2022` and `SKU12` are never stemmed and
  are weighted above ordinary words, because an amount or a ticket number is the most distinguishing
  thing in a passage. "Refund issued for 1000 rupees." and "Refund issued for 100 rupees." are two
  different passages and stay apart; "Ticket 1199 was escalated." and "Ticket 1199 has been escalated."
  are one, and group.
- **Groups and keeps one.** Pairs at or above `threshold` become edges; a duplicate group is a connected
  component; `keep` decides which member survives.
- **How honest is this about "meaning"?** TF-IDF does not understand language. What it has is a lexicon
  of common paraphrases and a robust surface metric, which handles the everyday cases - reworded tickets,
  restated notes, copies with edits - and *will* miss anything whose paraphrase is not in the lexicon
  ("the sprint slipped" vs "we are behind schedule" scores near zero). When you need real semantics, pass
  `embed=` and this package will use your vectors instead; that hook is the honest answer, and it is why
  nothing here depends on a model.
- **Exact duplicates are free.** Texts that are identical after normalization are collapsed before any
  scoring, so they group at *every* threshold and a file full of copies is fast.
- **Indices are the caller's.** `kept`, `removed`, `groups` and `pairs` always refer to positions in the
  input list, whatever happens internally.
- **Deterministic.** The same input gives the same output, every run; the MinHash permutations come from
  a fixed `random_state`.

## API

```python
semantic_dedup.dedupe(texts, *, threshold=0.82, method="auto", keep="longest", embed=None) -> DedupeResult
semantic_dedup.find_duplicates(texts, **kw) -> DedupeResult     # same analysis, drops nothing
semantic_dedup.similarity(a, b, *, method="tfidf") -> float     # 0.0 to 1.0
```

- `texts` - a list of strings, or a path to a `.txt` (one text per line), `.csv`/`.tsv` (a text column is
  picked automatically, or name it with `column=`) or `.jsonl` file.
- `threshold` - minimum similarity, in `(0, 1]`. `1.0` means exact matches only: the similarity search is
  skipped and only texts identical after normalization group.
- `method` -
  `"tfidf"` compares every pair exactly;
  `"minhash"` uses MinHash + LSH banding to propose candidate pairs and then scores each candidate with
  the exact same cosine, so the numbers you get back are never estimates;
  `"embed"` uses the vectors from your `embed` callable;
  `"auto"` (default) is `embed` when you pass one, else `minhash` above 5000 distinct texts, else `tfidf`.
- `keep` - `"longest"` (default), `"first"`, `"last"`, or `"most_complete"`: the member whose words cover
  the rest of the group best, which is usually the one that says everything the others say.
- `embed` - `callable(list[str]) -> ndarray` with one row per text. Rows are L2-normalized for you.

`similarity(a, b)` scores one pair **on its own**, while `dedupe(texts)` weights every word by how rare
it is **across the corpus you passed in** (that is what the IDF in TF-IDF means). The two numbers are
close but not identical - `similarity("The cat sat.", "The cat sat on the mat.")` is `0.65` alone and
`0.63` inside a three-note corpus - so use `similarity()` to get a feel for the scale, and calibrate the
threshold you ship on a sample of the real corpus.

```python
import semantic_dedup
from sentence_transformers import SentenceTransformer        # not a dependency of this package

model = SentenceTransformer("all-MiniLM-L6-v2")
result = semantic_dedup.dedupe(texts, method="embed", embed=lambda batch: model.encode(batch))
```

`Deduper(threshold=0.82, method="auto", keep="longest", embed=None, word_ngram=(1, 2), char_ngram=(3, 4),
num_perm=128, random_state=0, column=None)` is the class underneath, with `.run(texts, drop=True)`.

`DedupeResult`

- `.kept` - `list[int]`, input positions that survived; `.removed` - what was dropped
- `.groups` - `list[list[int]]`, one ascending list per duplicate group (size 2 or more)
- `.pairs` - `list[(i, j, similarity)]` with `i < j`, the edges that formed the groups
- `.texts` - the surviving texts; `.all_texts` - the input as it was read
- `.n_removed`, `.n_kept`, `.n_groups`, `.n_duplicates`, `.n_texts`, `.reduction` (fraction removed)
- `.threshold`, `.method`, `.keep`, `.warnings`
- `.summary()` - the human-readable report shown above; `.to_dict()` - a JSON-safe dict

```python
semantic_dedup.similarity("Our prices went up.", "Costs increased.")        # 1.0
semantic_dedup.similarity("Please fix the login bug.", "Lunch is at noon.") # 0.0
semantic_dedup.find_duplicates(texts).groups                               # report only, nothing dropped
semantic_dedup.dedupe(texts, threshold=1.0)                                # exact duplicates only
```

## CLI

```
semantic-dedup INPUT [--threshold 0.82] [--method auto|tfidf|minhash|embed] [--keep longest|first|last|most_complete]
               [--column NAME] [--report] [--max-groups 5] [--json] [--output PATH]
```

- `semantic-dedup notes.txt` prints the summary.
- `semantic-dedup faq.csv --column question --threshold 0.9` reads one column of a table.
- `--report` finds duplicates without choosing anything to remove.
- `--json` prints `to_dict()` as JSON (UTF-8, never escaped); `--output PATH` writes the surviving texts,
  one per line.

`--json` is an **index report**: `kept`, `removed`, `groups` and `pairs` are positions in the input file,
not the texts, so join it back to that file (line *n* of the `.txt`, row *n* of the `.csv`) to read it.
`pairs` carries one entry per matching pair and can be long on a large, repetitive corpus; `groups` is the
compact view. Use `--output PATH` when what you want is the cleaned text itself.

Output is UTF-8 whatever the console is set to, so piping a summary full of non-ASCII text is safe.

## License

MIT
