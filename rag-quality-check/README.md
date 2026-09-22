# rag-quality-check

Your RAG answers are only as good as what the retriever handed the model, and
"it feels better now" is not a number - this measures whether the right passages
actually came back, and whether the answer stayed inside them.

## Install

```bash
pip install rag-quality-check
```

## Quickstart

```python
import rag_quality_check

cases = [
    {"query": "reset my password", "retrieved": ["p_reset", "p_billing"], "relevant": ["p_reset"]},
    {"query": "cancel my plan", "retrieved": ["p_faq", "p_billing"], "relevant": ["p_cancel"]},
]
print(rag_quality_check.evaluate(cases, k=2).summary())
```

```
rag-quality-check: 2 case(s), top-2, threshold 0.5, similarity by lexical overlap.
retrieval:
  precision_at_k     0.250   relevant items in the top k, divided by k (2 case(s))
  recall_at_k        0.500   relevant items found, divided by all relevant ones (2 case(s))
  mrr                0.500   1 / rank of the first relevant item, averaged over cases (2 case(s))
  ndcg_at_k          0.500   DCG@k / ideal DCG@k, linear gains, log2 discount (2 case(s))
  hit_rate           0.500   share of queries with a relevant item in the top k (2 case(s))
failures (1):
  - case 1 'cancel my plan': nothing relevant in the top 2
weakest cases:
  [1] 0.00  'cancel my plan'
  [0] 0.90  'reset my password'
```

## What it measures

- **precision_at_k**, **recall_at_k**, **mrr**, **ndcg_at_k**, **hit_rate** - the
  five retrieval numbers, each with its exact formula written out in its
  docstring. Users compare these across systems, so an undocumented variant of
  nDCG is worse than no nDCG at all.
- **groundedness** - the share of the answer's words that a retrieved passage
  actually supports. This is the hallucination check. Sentences are weighted by
  how many content words they hold, so a leading `"Yes."` cannot halve the score
  of a paragraph copied straight out of the context.
- **citation_coverage** - the share of retrieved passages the answer drew on. A
  low number beside a high groundedness means the retriever is returning padding.
- **answer_relevance** - how much of the query's own wording the answer repeats,
  and **answer_correctness** when you pass a `ground_truth`.
- Every metric is in `0..1`. Every one of them is defined once, in one place.

It also tells you what it could *not* measure:

- No `relevant` ids on a case? Relevance is **estimated** from similarity to the
  query, and the report says "ESTIMATED" every time it prints, on the report and
  on the case. Estimated numbers are for spotting weak queries, not for
  comparing two systems.
- A case whose `relevant` list is empty is a query you have labelled as having
  no right answer. Every retrieval metric is **excluded** for it and the
  exclusion is noted: retrieving nothing relevant is the wanted result there, so
  scoring it `0.0` would drag the means down and fill `failures` with successes.
- Passed ids in `retrieved` instead of passage text? The retrieval metrics are
  unaffected, but `groundedness` and `citation_coverage` are **excluded** with a
  note rather than reported as `0.0` - an id carries no words for an answer to
  be supported by, and scoring against one would report a hallucination that
  never happened. Pass `{"id": ..., "text": ...}` to measure them.
- `answer_relevance` under the default word overlap is the share of the query's
  content words the answer repeats, and a correct answer routinely repeats none
  of them ("how much does the pro plan cost" answered by "It is 29 dollars per
  seat per month." scores `0.00`). It is reported, noted when it is low, and
  never counted as a failure unless you pass `embed=`, where it is a cosine
  between meanings and does mean what it says.
- If no `relevant` id matches any retrieved id anywhere in the run, the report
  says so, because ids compared in two different forms and a broken retriever
  produce the same zeros.
- An empty `retrieved` list scores zeros, never a `ZeroDivisionError`.
- Asked for top-5 and only 3 came back? `precision_at_k` still divides by 5, and
  the case says so in a note instead of silently shrinking `k`.
- Duplicate passages in `retrieved` are counted once, and the duplicate count is
  noted.
- Unicode queries and passages work; CJK text is compared character by
  character, since it is not written with spaces. One caveat, and only in
  estimated mode: word overlap matches word forms, so an inflected language can
  miss a passage that plainly answers the query (Hindi "बदलें" against "बदलने"
  counts as a miss). Labelled `relevant` ids are unaffected, and `embed=` fixes
  the estimate.
- 1000 cases score in about a second. No model downloads, no network, numpy only.

By default similarity is word overlap. Pass `embed=` any
`callable(list[str]) -> vectors` - a sentence-transformer `.encode`, an API
client, anything - and every similarity becomes cosine instead, so you get real
semantics without this package depending on a model:

```python
report = rag_quality_check.evaluate(cases, embed=model.encode)
```

Texts are de-duplicated and embedded in one batched call for the whole run.

## API

### `evaluate(cases, *, k=5, threshold=0.5, embed=None) -> RagReport`

`cases` is a list of dicts:

| key | required | meaning |
| --- | --- | --- |
| `query` | yes | the question that was asked |
| `retrieved` | yes | what came back, best first: passage strings, ids, or dicts with `id` and/or `text`. Ids alone are enough for the retrieval metrics; `groundedness` and `citation_coverage` need the text |
| `relevant` | no | ids that should have come back - a list, or `{id: gain}` for graded relevance. Without it, relevance is estimated; an empty list means the query has no right answer |
| `answer` | no | the generated answer; turns on the answer metrics |
| `ground_truth` | no | the reference answer; adds `answer_correctness` |

`k` is the cut-off and is always the denominator of `precision_at_k`.
`threshold` is the similarity at which two texts count as a match, and the score
below which a case is reported as a failure - with the single exception of
`answer_relevance` under word overlap, which is reported and never accused.

### `evaluate_case(query, retrieved, **kw) -> CaseResult`

One query, same rules and the same numbers as a one-case `evaluate`. Takes
`relevant`, `answer`, `ground_truth`, `k`, `threshold` and `embed`.

```python
case = rag_quality_check.evaluate_case(
    "how do I reset my password",
    ["To reset your password, open Settings and choose Reset."],
    answer="Open Settings and choose Reset.",
    k=1,
)
print(case.summary())
```

### `RagReport`

| member | what it is |
| --- | --- |
| `.metrics` | `dict[str, float]`, each metric averaged over the cases where it was defined |
| `.support` | how many cases went into each mean |
| `.per_case` | `list[CaseResult]` |
| `.weakest(n=5)` | the `n` worst cases, weakest first |
| `.failures` | `list[str]`, plain language: retrieved nothing, nothing relevant in the top k, answer not grounded, ... |
| `.notes` | everything that changed how a number was produced |
| `.estimated` / `.fully_estimated` | whether any / every case's relevance was guessed |
| `.similarity` | `"lexical overlap"` or `"embeddings"` |
| `.ok` | `True` when no case failed |
| `.summary()` | the human report above |
| `.to_dict()` | JSON-safe, per-case detail included |
| `.to_frame()` | one row per case as a `pandas.DataFrame` (needs `pip install rag-quality-check[frame]`) |

`CaseResult` carries `.query`, `.metrics`, `.score`, `.excluded`, `.notes`,
`.failures`, `.ok`, `.summary()` and `.to_dict()`. A metric that could not be
computed is absent from `.metrics` and named in `.excluded`, never faked as
`0.0`.

## CLI

```bash
rag-quality-check cases.json                      # the summary
rag-quality-check cases.jsonl --k 3 --threshold 0.4
rag-quality-check cases.json --json > report.json # to_dict() as JSON
cat cases.json | rag-quality-check - --weakest 10
```

`cases.json` is a list of the same case objects; `cases.jsonl` is one per line.
`--output PATH` writes the report (`.json` for the dict, otherwise the text).
`rag-quality-check --help` lists every metric and its one-line definition.

## License

MIT
