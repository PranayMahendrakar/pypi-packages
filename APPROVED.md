# approved.json - the release gate

`publish-daily.ps1` publishes ONLY the names in `approved.json`. A wheel sitting in a
package folder is not enough; nothing ships until its name is added here by hand.

A package earns its place in this file only after ALL of the following:

1. its own test suite passes against the INSTALLED wheel, not the source tree
2. the README quickstart runs verbatim in a clean virtual environment
3. `<name> --help` works, and piping its output with non-ASCII data does not crash
4. `python -m twine check dist/*` passes
5. any issue an independent reviewer raised is fixed or consciously accepted

`verify.py` in this folder automates steps 1-4. Run it before adding anything:

    python verify.py <package-name>

## Deliberately NOT approved


## Resolved and since approved

- **dataset-splitter** - FIXED. The allocator walked unit midpoints against a row-count boundary, so a
  narrow band could contain no unit at all: three groups of 200 rows left the validation split empty.
  There is now a repair pass, and when units are too few to fill everything the priority is train, then
  test, then val, because a model with no test set cannot be evaluated while a missing val set only
  costs tuning. 600 rows: 1 group gives 600/0/0, 2 gives 300/0/300, 3 gives 200/200/200. 327 tests pass.

- **sensor-anomaly** - FIXED. Two faults, both real.
  (1) Scale was measured from centred rolling-median residuals, which are narrower than the readings
  really are because the median partly tracks the noise it removes. sensitivity=3.0 behaved like about
  2.4. Scoring now uses the larger of that and a lag-one difference estimate, which recovers the true
  noise level and ignores drift. The clean flag rate is now 0.31 percent against the 0.27 percent a
  two-sided 3-sigma tail predicts, and slow drift no longer collects a spurious step-change fault.
  (2) The "is anything wrong" judgement compared the WORST channel against a flat multiple of the mean.
  The maximum of several channels runs high by nature, and flags cluster because one excursion spans
  several readings, so healthy tables were condemned. The allowance is now the expected count plus five
  deviations. Measured over 40 clean six-channel tables: three deviations called 20 percent anomalous,
  four called 8 percent, five calls none, while every planted spike, flatline, stuck-at-zero and step
  change is still caught. 95 tests pass, four of them regressions for exactly these cases.

- **model-watchdog** - FIXED. Predictions and actuals were filtered for missing values
  independently before being paired by position, so one gap slid every later prediction onto
  the wrong actual: a perfect baseline read as 0.03 accuracy, and a model predicting wrong 200
  times running then looked unchanged against it. A monitoring tool reporting healthy through a
  total collapse is the worst possible failure, so pairing is now positional and filtering is
  pairwise. 154 tests pass.

- **ml-pipeline-kit** - FIXED. A validation check returning a list mask fell through every
  branch to bool(raw), and a non-empty list is always truthy, so a check where every row failed
  reported a pass. Same for (False, 404), whose non-string message matched no branch. Lists and
  tuples are now read as masks, and a two-item pair is only a message pair when its second
  element is not itself a boolean, so [False, False] stays a mask. All eleven return shapes are
  covered by tests. 101 tests pass.

- **ml-inference-profiler** - FIXED. Three findings. "decode" was a preprocessing keyword, so a
  pipeline whose cost was output decoding was told its data path was the problem, the exact
  opposite diagnosis; decode is now resolved from the rest of the label (image decode is input,
  token decode is output) and stays unclassified when the label says nothing either way. The
  preprocessing complaint fired at 35 percent, so a healthy 40/35/25 split was called out; it
  now needs half the run. Advice was also checked for run-to-run stability: fifteen identical
  runs now produce identical advice. 104 tests pass.

- **text-quality-ai** - FIXED, two defects. The nominalisation check matched bare -ance, -ence,
  -ity, -ness and -ism suffixes, so it reported "distance", "kindness", "quality", "patience",
  "balance" and "tourism" as nouns built out of verbs. None of them is: -ness and -ity build nouns
  from ADJECTIVES and -ism from nouns, so those suffixes could only ever fire on clean writing.
  Those three are dropped entirely, and -ance/-ence now require the word to be on a list where an
  English verb really is the stem. A check that fires on good prose is worse than no check, so it
  now prefers to miss a real nominalisation over inventing one; bureaucratic prose is still caught.
  Separately, the silent-"es" rule fired on "-les", "-ches" and "-shes", which are pronounced, so
  bottles, candles, tables, watches, dishes and churches each lost a syllable and Flesch reported
  text as easier than it reads. 28 words now count correctly in both directions. 184 tests pass.

- **semantic-dedup** - FIXED. "auto" stayed exhaustive until 5,000 texts, so the default took
  15.2s on 2,700 passages that MinHash finished in 1.6s having removed exactly the same 200
  duplicates. MinHash only picks candidate pairs and every candidate is scored exactly, so nothing
  was traded for the speed; the threshold is now 1,000 and the default runs in 1.3s. A CSV field
  over Python's 131,072-character cap also raised a raw _csv.Error; the cap is now raised and a
  genuinely unreadable file gets a clear ValueError naming it. 116 tests pass.

- **hallucination-check** - FIXED. A faithful paraphrase of a long source sentence scored 0, the
  worst possible failure for a grounding checker, because support was scored by symmetric
  similarity against whole sentences and a shorter paraphrase loses on length alone. It now scores
  100 while a genuinely unsupported claim about a different day, a contractor and a price still
  scores 0. 104 tests pass.

- **rag-chunker** - FIXED. The default method was called "semantic" but split on word overlap
  between neighbouring sentences, which tracks meaning only when the topics happen to use different
  words. On a document with two plainly different subjects it put the boundary one sentence early;
  IDF weighting, four window sizes and TextTiling depth scoring all failed to find it, so the
  signal is genuinely absent for ordinary prose. The default is now "recursive", which is
  predictable and reproduces the input exactly, and an embed hook accepts a real embedding model -
  with one, the same document splits at exactly the right sentence. The CLI defaulted to semantic
  while the library defaulted to recursive, so the same document chunked differently depending on
  how it was called; the test that should have caught that hard-coded the value, and now derives it
  from the library signature. 470 tests pass.

- **llm-router-lite** - FIXED, two defects. A zero-cost model made the cheapest cost zero, so every
  priced model scored 0.0 on cost and tied; the tie fell through to quality and prefer="cheap"
  started listing its fallbacks most-expensive-first, putting a 100x pricier model ahead of a cheap
  one. Cost is now ranked against the cheapest model that actually has a price, so free still wins
  outright and everything else stays in price order. Separately the instruction signal, a fifth of
  the total weight, was dead: instructions were counted per SENTENCE, so a prompt packing four
  requests behind one full stop counted as one and scored zero on exactly the multi-part prompts
  the signal exists to detect. Counting per clause moves a multi-step engineering prompt from
  "moderate" to "hard", which is where it belongs. 154 tests pass.

- **rag-quality-check** - FIXED. Anything longer than 40 characters was assumed to be passage prose
  "which no identifier does", but a source URL and a sha256 chunk id both run well past that. Passing
  either in "retrieved" - which callers legitimately do, and which the retrieval metrics handle fine -
  scored the answer against a bare id, so groundedness came out 0 and a hallucination was reported
  that never happened. Identifiers are now recognised by shape rather than by length. Retrieval
  metrics still work on ids alone, real passages still score 1.0, and an invented answer is still
  caught. 113 tests pass.

- **meeting-intelligence** - FIXED, one of three reported defects real. Actions were deduped on their
  words alone, so when a second person made the same commitment it was silently dropped, and the
  dropped person is exactly the one who never gets chased for it. Actions are now keyed on speaker
  and text; the same speaker repeating themselves is still one action, and a decision restated by
  someone else is still one decision, because that is a confirmation rather than a new decision.
  The other two did not reproduce: owners written in any script were already captured, and a caption
  file with no speaker labels already warned that participation is pooled. 176 tests pass.

- **image-quality-ai** - FIXED. Sharpness was variance of the Laplacian, which is isotropic and so
  cannot see motion blur at all: smearing a frame sideways destroys the horizontal detail while every
  vertical edge survives, and the combined variance barely moves. A heavily smeared photo scored 94
  and graded A, HIGHER than the unblurred original, and passed as usable. The two image axes are now
  measured separately, which makes a smear obvious because one axis collapses while the other does not.
  Recognising it needs two conditions together, since either alone is wrong: a ratio test alone flags
  any striped or text-heavy subject, and an absolute test alone flags any softly-lit photograph.
  A still frame genuinely cannot always tell a smear from a subject that only varies in one direction,
  so that case is reported in the sharpness message rather than failed outright, and the README now
  carries a Limits section saying so. The resolution-invariance claim it used to make is gone, because
  resizing really can move a photo across a threshold. 119 tests pass.

- **predictive-maintenance** - FIXED, and its blocker was the worst defect in the whole family.
  Remaining useful life is measured against the time axis, and a single unreadable timestamp made
  the entire time column count as unreadable, so the axis was discarded for row order and a visibly
  dying machine reported INFINITE remaining life. Two code paths needed it, because a datetime
  column already holding NaT never reached the parsing branch at all. A few missing stamps are now
  interpolated from their neighbours with a note saying how many, and a mostly-unreadable column
  still falls back to row order, which is honest when there is nothing to interpolate between.
  Separately, column names were stringified for lookup and then used against the original labels,
  so a DataFrame built from a numpy array - which has integer labels, an entirely ordinary thing to
  pass - died with a bare KeyError: '0'. 105 tests pass, three of them regressions for these cases.
  The reported index-loss finding did not reproduce: indexing a time series by its time is correct.

- **multilingual-text** - FIXED, two defects. Serbian and Macedonian letters were missing from the
  Cyrillic table, so a Serbian name came back half romanised while the result still reported
  supported=True. That combination is worse than refusing outright: nothing downstream could detect
  the output was incomplete. Those letters are now covered, and the untranslated case still reports
  supported=False. Separately the Greek table implemented a classical scheme - beta to "b", phi to
  "ph", macrons on eta and omega - while the README promised ISO 843, which romanises MODERN Greek
  as "v" and "f". The documentation is what users read, so the table was the thing that was wrong.
  Five existing tests asserted the classical output and were updated, since they encoded exactly the
  disagreement. 1363 tests pass.

- **document-memory** and **smart-crop-ai** - clean on the first review round, scored 9 of 10 each,
  123 and 254 tests. No changes needed.
