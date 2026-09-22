# hallucination-check

Check an answer against the sources it claims to use, and get back the exact sentences the
sources do not support.

## Install

```
pip install hallucination-check
```

Only dependency: numpy. Nothing is downloaded at import or at run time.

## Quickstart

```python
import hallucination_check as hc

answer = "The Eiffel Tower is in Paris. It was completed in 1889. It cost 42 million francs."
sources = ["The Eiffel Tower stands in Paris, France, and was completed in 1889."]
report = hc.check(answer, sources)
print(report.summary())                       # 66.7 / 100 grounded, 1 claim flagged
print([c.text for c in report.unsupported])   # ['It cost 42 million francs.']
```

`sources` can be one string, a list of strings, or a list of dicts with `"text"` and an
optional `"id"`, so retrieved passages go straight in:

```python
hc.check(answer, [{"id": "wiki:eiffel#3", "text": "..."}, {"id": "wiki:paris#1", "text": "..."}])
```

## What it checks

- The answer is split into claims (one sentence each by default; `granularity="clause"` or
  `"paragraph"` also work). Every claim is checked on its own.
- For each claim the sources are searched for the best-matching span of one to three
  consecutive sentences. That span, its source id and the score are on the claim, so every
  verdict can be traced back to the text it came from.
- The support score blends three lexical signals: character 4-gram cosine, word
  unigram/bigram cosine, and how much of the claim's content vocabulary the span covers
  (light plural and tense stripping, so "site" matches "sites"). A claim scoring at or above
  `threshold` (default 0.55) counts as supported.
- **A claim carrying a number, date, name or quantity that appears nowhere in the sources is
  flagged whatever its similarity**, because those are what people actually get wrong.
  `12%`, `$4.3 billion`, `1,200`, `twenty-three`, `12 March 2024`, `November 2022` and
  `NASA` are all recognized; numbers written in words and in digits compare equal.
- A fact has to appear in the *same role*, not merely somewhere in the pile of passages.
  A measurement is tied to the unit or head noun after it and a name to the words around
  it, so "reduced symptoms by 1,200 percent" is not backed up by "enrolled 1,200
  participants" in another passage, "11 metres" is not backed up by the 11 in "Apollo 11",
  and "developed at Pfizer" is not backed up by a passage that only mentions Pfizer's
  earnings. A possessive in the source counts as the bare name, so "France's capital"
  grounds a claim about France.
- When the best span states a *different* number or date for otherwise the same sentence,
  the claim also lands in `report.contradictions` with a reason like
  `the sources say 1,200 where the answer says 4,200`.
- A capitalized word at the start of a sentence is not treated as a name unless it is an
  acronym or the start of a multi-word name, so "Cats sleep a lot." is not flagged for
  "Cats".
- Everything is deterministic: the same answer and sources always give the same report.
  Nothing is written, nothing is fetched, and the library never prints.

**Lexical grounding is a signal, not proof.** A claim can score well while quietly reversing
the source's meaning, and a correct paraphrase that shares no vocabulary will score badly.
The `embed` hook is how you strengthen it: pass any function that turns a list of strings
into vectors, and semantic similarity is blended into the score at equal weight (this
package stays free of model dependencies, so the model stays yours).

```python
from sentence_transformers import SentenceTransformer   # your choice, not a dependency
model = SentenceTransformer("all-MiniLM-L6-v2")
report = hc.check(answer, sources, embed=lambda texts: model.encode(texts))
```

With `embed` supplied, every source sentence is embedded (up to `embed_max_sentences`), so
the hook also finds spans that share no words at all with the claim.

## API

```python
hc.check(answer, sources, *, threshold=0.55, granularity="sentence", embed=None) -> GroundingReport
hc.check_batch(pairs, **kw) -> list[GroundingReport]
```

`pairs` is an iterable of `(answer, sources)` tuples, or of dicts with `"answer"` and
`"sources"` keys. `**kw` goes to `GroundingChecker`.

`GroundingReport`

- `.score` - 0-100, the share of the answer's claims the sources support
- `.claims` - `list[Claim]` in answer order
- `.unsupported` - the claims the sources do not back up
- `.contradictions` - unsupported claims where a source states a different number or date
- `.citations` - `dict[claim index -> source id]` for the supported claims
- `.n_claims`, `.n_supported`, `.threshold`, `.granularity`, `.source_ids`, `.warnings`
- `.summary()` - human-readable text; `.to_dict()` - JSON-safe dict

`Claim(text, supported, confidence, best_source, best_span, reason)`

- `.text` - the claim as it appears in the answer
- `.supported` - the verdict
- `.confidence` - 0-1 support score for this claim
- `.best_source` - id of the source the best span came from (`"s1"`, `"s2"`, ... unless you
  gave ids), or `None` when nothing matched
- `.best_span` - the source text that matched best
- `.reason` - one plain sentence saying why, e.g.
  `these appear nowhere in the sources: 42 million`
- `.to_dict()`

```python
report = hc.check(answer, sources)
report.score                     # 66.7
report.claims[1].best_span       # 'The Eiffel Tower stands in Paris, France, and was completed in 1889.'
report.citations                 # {0: 's1', 1: 's1'}
report.to_dict()["unsupported"]  # [2]
```

`GroundingChecker(threshold=0.55, granularity="sentence", embed=None, embed_weight=0.5,
flag_missing_facts=True, char_n=4, window=3, top_k=25, embed_top=5,
embed_max_sentences=2000)` is the class underneath, with `.check(answer, sources)` and
`.check_batch(pairs)`. Set `flag_missing_facts=False` to score on similarity alone.

Edge cases are settled, not accidental: an empty answer scores 100 with no claims; empty
sources score 0 with every claim unsupported; an answer identical to a source scores 100;
the score is always between 0 and 100.

## CLI

```
hallucination-check ANSWER [-s SOURCE ...] [--threshold 0.55]
                    [--granularity sentence|clause|paragraph] [--no-fact-flags]
                    [--json] [--output PATH] [--fail-under SCORE]
```

`ANSWER` and each `SOURCE` may be a file path or the text itself; `-` reads stdin. A value
that can only have meant a file -- one bare word ending in `.txt`, `.md`, `.json`, `.jsonl`
or `.ndjson` -- is an error when no such file exists, rather than being checked as literal
text, so a mistyped path never turns into a confident wrong report.

- `hallucination-check answer.txt -s passage1.txt -s passage2.txt` prints the summary.
- `hallucination-check "Paris is the capital of France." -s "France's capital is Paris."`
  takes text directly.
- `cat answer.txt | hallucination-check - -s retrieved.jsonl --json` reads the answer from
  stdin and the passages from a JSON-lines file (one object per line with `text` and an
  optional `id`), and prints `to_dict()` as JSON.
- `--output PATH` writes that JSON to a file; `--fail-under 80` exits with status 2 when the
  score is below 80, which is the useful thing to put in CI.

## License

MIT
