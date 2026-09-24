# 50-package programme - status and plan

## Why batches of five, one workflow at a time
Agents stalled ("no progress for 180000ms") in three separate runs. The common factor was
concurrency: a 10-package batch, and later a 5-package batch running alongside a second
workflow. Five packages with ONE reviewer each, one workflow at a time, is what completes.
`verify.py` is the hard gate afterwards and never stalls, so reviewer count can be halved
without weakening the release standard.

## Release gate
Nothing reaches `approved.json` until `python verify.py <name>` passes: clean-venv install,
README quickstart verbatim, CLI help, CLI JSON, piped non-ASCII output, and the package's own
test suite run against the INSTALLED wheel. Domain-specific accuracy is checked by hand on top
(PII detection rates, anomaly recall, split integrity).

## Group 1 - Data and dataset intelligence (10/10 COMPLETE)
dataset-health, near-dupes, synthetic-tabular, data-drift-lite   -> LIVE on PyPI
auto-label, schema-guard, ml-feature-check, smartclean-df,
dataset-splitter, privacy-scan-ml                                 -> approved, queued 17:00

## Group 2 - Industrial and time series (1/7, 1 needs fixing)
predictive-maintenance  BUILT but NOT approved: 6 open findings, incl. a blocker where one NaT
                        in the time column reports "infinite remaining life"; RUL over-predicts
                        4-18x on accelerating wear
timeseries-anomaly, sensor-anomaly, machine-health, quality-predictor   partial source, stalled
energy-analyzer-ai, production-anomaly                                  not started

## Group 3 - Developer and MLOps (0/5)
model-watchdog, model-benchmark, ml-inference-profiler, offline-ml, ml-pipeline-kit

## Group 4 - NLP and LLM (0/10)
text-quality-ai, semantic-dedup, rag-chunker, rag-quality-check, llm-router-lite, prompt-cache,
hallucination-check, document-memory, multilingual-text, meeting-intelligence

## Group 5 - Computer vision (0/10)
vision-anomaly, image-quality-ai, smart-crop-ai, document-quality, ocr-cleaner, image-dedup-ai,
object-counter-ai, camera-health, image-redactor, video-event-detector

## Group 6 - Speech and audio (0/8)
voice-activity-ai, audio-clean-ai, speaker-diarize-lite, voice-commands-ai, audio-anomaly,
speech-quality, call-ai-metrics, offline-stt-router

## Honest notes on three of them
- object-counter-ai and image-redactor need a real detection model. They will be built as
  frameworks that accept a detector the user supplies, plus a classical fallback, rather than
  pretending numpy can detect faces.
- speaker-diarize-lite needs speaker embeddings. Same approach: energy/spectral segmentation
  by default, with a hook for a real embedding model.
- image-dedup-ai overlaps near-dupes, which already handles images. It will be scoped to
  large-scale image-only deduplication with on-disk indexing, or dropped if that is too thin.

## Publishing
`publish-daily.ps1`, Windows task "PyPI Daily Publish", daily 17:00, five per run, 90s apart,
stops on HTTP 429. PyPI enforces a per-day cap on NEW project creation; four in one day
exhausted it, so roughly five per day is the real ceiling regardless of how fast building goes.


## What PyPI actually allows (measured, not guessed)

Two days of evidence, both identical: four new projects published, then every further
upload returns HTTP 429 until roughly the next day.

    21 Sep  dataset-health, near-dupes, synthetic-tabular, data-drift-lite   -> then 429
    22 Sep  ml-feature-check, smartclean-df, dataset-splitter, privacy-scan-ml -> then 429

So the ceiling is about four or five NEW projects per day, and it is a burst quota that
refills rather than a hard daily reset. Publishing an update to a project that already
exists is not affected; only creating a new one is. At four a day the remaining queue
takes about a week, which no amount of building faster will change.

## A second failure that is not a rate limit

`schema-guard` returned 400, not 429: "the name is too similar to an existing project".
`schemaguard` and `schema-guardian` already exist. PyPI blocks confusingly close names,
and this is invisible to an availability check, because a blocked name still answers
"not found" on both the JSON and simple APIs. It was renamed to
`dataframe-schema-guard`, and `publish.sh` now reports a collision differently from a
rate limit, because one needs a rename and the other needs patience.

Any remaining name could hit the same rule. It surfaces immediately and harmlessly on
the first upload attempt, so the plan is to handle each as it appears rather than guess.

## The publishing window, measured

    21 Sep ~19:00 IST   4 published, then 429
    22 Sep ~20:00 IST   4 published, then 429
    23 Sep  11:40 UTC   scheduled run: 429 on all four, ~21 hours after the last success
    24 Sep              retry cadence raised to every 4 hours to find the real edge

So the window is longer than 21 hours and is not a simple daily reset. Guessing at it
one attempt a day learns almost nothing, so the workflow now tries six times a day. Each
run skips anything already on PyPI, so a wasted attempt costs a build and nothing else.

A rate limit no longer fails the run. It is an expected condition while the window is
shut, and marking it red six times a day would train everyone to ignore the signal. The
run asks PyPI afterwards whether the package actually landed and reports that instead.
A genuine problem - a failing test, or a name rejected as too similar to an existing
project - still fails loudly.
