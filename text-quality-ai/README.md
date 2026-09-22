# text-quality-ai

Score any text for readability, repetition, structure, clarity and vocabulary, and get back a plain-language
list of what to fix first.

## Install

```
pip install text-quality-ai
```

Pure Python plus numpy. Nothing is downloaded at runtime, no model, no corpus, no network.

## Quickstart

```python
import text_quality_ai

text = "It should be noted that the implementation of the solution was undertaken by the team. The evaluation of the results was subsequently performed by the team. The documentation of the findings was then completed by the team."
report = text_quality_ai.score(text)
print(report.summary())
print("Fix first:", report.suggestions[0])
```

That prints the grade, the five measures with the numbers behind each one, the issues found with examples
taken from your own text, and the fixes in the order that would help most:

```
Text quality: 69.1 / 100 (grade D) for a general audience
  37 words, 3 sentences, 1 paragraphs, 62 syllables, about 9s to read

  readability  100.0 [##########]
               Flesch reading ease 52.6 and Flesch-Kincaid grade 9.0; a general audience reads best around grade 8. ...
  clarity       22.4 [##--------]
               Clarity: 3 of 3 sentence(s) read as passive, 1 filler word(s), 0 hedge(s), 3 noun(s) built from a verb, ...
  ...
  What to fix first:
    1. Rewrite the 3 passive sentence(s) so the doer comes first, ...
```

## What it checks

- **readability** - Flesch reading ease, Flesch-Kincaid grade, SMOG, average sentence length, average word
  length. Scored against the reading grade your `target` audience expects.
- **repetition** - words used far more often than the rest, sentence openers that keep repeating, and bigrams
  and trigrams that come back. N-grams never cross a sentence boundary.
- **structure** - how much sentence length varies, whether paragraphs are balanced, and how often sentences
  open with a linking word such as "however" or "for example".
- **clarity** - share of passive sentences, filler and hedge words, nouns built out of verbs
  ("the implementation of" for "we implemented"), and sentences past 30 words. A participle that normally
  works as an adjective ("he was tired", "she is qualified") is not counted as passive unless the sentence
  names the doer, and ordinary nouns that happen to end in `-tion` or `-ment` ("configuration", "version",
  "environment") are not counted as nouns built out of verbs.
- **vocabulary** - type-token ratio, both raw and adjusted for length with a 100-word moving window, and the
  share of words with three or more syllables.

Each measure scores 0 to 100 and carries its own one-sentence explanation. The overall score is a weighted
average; the weights shift with the target, so marketing copy is judged more on clarity and an academic paper
more on structure.

| target | reads best around | long words | nouns built from verbs |
|---|---|---|---|
| `general` (default) | grade 8 | 14% | 3.5% |
| `academic` | grade 14 | 28% | 6.0% |
| `marketing` | grade 7 | 10% | 2.8% |
| `technical` | grade 12 | 22% | 6.0% |
| `simple` | grade 5 | 6% | 2.5% |

The last column is where an issue is raised about nouns built out of verbs. Technical and academic writing
names concepts for a living, so it gets more room than marketing copy; the clarity score shifts with the
target in the same direction.

Being **simpler** than the target only costs points for `academic` and `technical`, where precision is
expected. Everywhere else, easier to read is never penalised.

### Honest limits

- **Syllable counting is a heuristic.** Vowel runs with the usual silent-`e`, `-ed`, `-es` and `-ing`
  corrections for anything that folds to Latin letters; independent vowels, vowel signs and inherent vowels
  for Devanagari; one syllable per character for Han, kana and Hangul; Greek and Cyrillic vowels otherwise.
  Counting runs merges vowels that belong to separate syllables, so a short list of common hiatus words
  (`idea`, `science`, `area`, `create`) is read off a table instead; words outside that table with the same
  shape are still undercounted by one. It is wrong on some words, most often names and loanwords, and
  everything built on it (Flesch, Flesch-Kincaid, SMOG) inherits that error.
- **Han, kana and Hangul have no spaces between words.** Rather than report one enormous word, the text is
  counted by character, the report says so in `notes`, and the readability score switches to characters per
  sentence. The English formulas are still reported, for reference only.
- **Passive voice, fillers and nominalisations are English word lists**, not grammar. They will miss cases and
  occasionally flag an innocent sentence. Every issue shows its examples so you can judge.
- **The passive check errs towards silence.** "To be" plus a past participle also covers every "he was tired"
  in English, so a participle from the built-in adjective list only counts as passive when the sentence names
  the doer ("the door was locked **by the guard**"). The cost is the other way round: a real passive whose
  participle is on that list, such as "the shop is closed at six", is missed.
- **Nouns built out of verbs are judged by suffix**, with a list of ordinary nouns
  ("configuration", "version", "documentation", "environment") left out of the count. Uncommon everyday nouns
  with the same endings may still be counted.
- Empty or whitespace-only text returns a zero-word report, not an error. `None` is a `TypeError`, like any
  other non-string.
- **SMOG assumes 30 sentences or more.** Below that it is scaled and the explanation says to treat it as
  indicative.
- **Below about 25 words** the numbers are noisy; the report adds a `very_short_text` issue saying so.

## API

```python
text_quality_ai.score(text, *, target="general") -> QualityReport
text_quality_ai.readability(text, *, target="general") -> dict
text_quality_ai.repetition(text) -> dict
text_quality_ai.compare(a, b, *, target="general", long_sentence_words=30) -> dict
text_quality_ai.TextScorer(target="general", *, long_sentence_words=30)
text_quality_ai.available_targets() -> list[str]
```

`text` is a string, or a list of strings. With a list, the report covers all of them and `report.documents`
holds one report per input.

`QualityReport`:

| attribute | what it is |
|---|---|
| `.score` | overall quality, 0 to 100 |
| `.grade` | `"A"` to `"F"` |
| `.components` | `{"readability": 100.0, "repetition": 54.3, ...}`, each 0 to 100 |
| `.issues` | `Issue(kind, severity, message, examples, count)`, most worth fixing first |
| `.suggestions` | plain-language fixes, most helpful first |
| `.stats` | `words`, `sentences`, `paragraphs`, `syllables`, `reading_time_seconds` (plus `characters`, `unique_words`, `word_counting`) |
| `.measures` | the full numbers behind each component, plus its `explanation`. For a list of strings the aggregate carries only `score`, `explanation` and `documents`: there is no single Flesch grade for five texts, so the numbers stay on `.documents[i].measures` |
| `.notes` | caveats about how this text was counted |
| `.documents` | one report per input, only when a list was scored |
| `.summary()` | the human-readable report above |
| `.to_dict()` | JSON-safe dict; `.to_json()` gives the string |
| `.explain(name)` | the one-sentence explanation for one measure |

```python
import text_quality_ai

text_quality_ai.readability("Short and clear. That is the whole trick.")["flesch_kincaid_grade"]
text_quality_ai.repetition("dog dog dog cat cat cat dog")["repeated_words"]
text_quality_ai.compare("The report was written by us.", "We wrote the report.")["score"]   # positive: b is better
text_quality_ai.score(["First draft ...", "Second draft ..."]).documents[0].grade
```

`compare(a, b)` returns `b` minus `a` for the overall score, every component and the main stats, plus
`"better"` and a one-line `"summary"`. Both sides are scored with the same settings, so `target` and
`long_sentence_words` apply to `a` and `b` alike and the result records the `long_sentence_words` it used.

## CLI

```
text-quality-ai article.md
text-quality-ai *.md --target marketing
text-quality-ai --text "Your sentence here."
cat draft.txt | text-quality-ai --json > report.json
text-quality-ai draft.txt --compare final.txt
text-quality-ai article.md --fail-under 70
```

`--json` prints `to_dict()`, `--output PATH` writes to a UTF-8 file, `--fail-under SCORE` exits 2 when the
score is below `SCORE` so it can gate a build, and `--target` picks the audience. With no file argument the
text is read from standard input. Output is always written as UTF-8, so piping text in any script is safe.

`--long-sentence-words N` applies to the whole run, `--compare` included, so the report block and the compare
block always agree about the same file.

## License

MIT
