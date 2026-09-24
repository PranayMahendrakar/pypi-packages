# Where things stand

Paused 24 Sep 2026 so the Claude app could be updated. Nothing is mid-flight; the
working tree is clean and pushed.

## Published: 18 on the profile

Your original 8, plus 10 from this work:

    dataset-health        near-dupes            synthetic-tabular     data-drift-lite
    ml-feature-check      smartclean-df         dataset-splitter      privacy-scan-ml
    dataframe-schema-guard  machine-health

## Built, verified, waiting to publish: 25

Listed in `approved.json`. Every one has passed `verify.py`: installed from its own
wheel into a clean environment, README quickstart run verbatim, CLI exercised and piped
with non-ASCII text, and its own test suite run against the installed package rather
than the source beside it.

## Still to build: 13

    Vision   vision-anomaly  ocr-cleaner  image-dedup-ai  object-counter-ai
             video-event-detector
    Audio    speech-quality  voice-activity-ai  audio-anomaly  (partial source on disk,
             not built, not in approved.json)
             speaker-diarize-lite  voice-commands-ai  call-ai-metrics  offline-stt-router
    Other    production-anomaly

## Publishing runs without this machine

GitHub Actions, `.github/workflows/publish-queue.yml`, every four hours. It takes the
next four unpublished names from approved.json, tests them, builds them, and uploads
with the token in the repository secret. The local Windows task is disabled so the two
cannot race.

PyPI allows roughly four new projects before returning 429, and the window is longer
than a day: a run 21 hours after the last success was still refused. Four-hourly retries
exist to find the opening rather than guess at it. A rate limit does not fail the run.

## Two things that will bite the next session

**pandas 3.** This machine has pandas 2; the release pipeline and real users have
pandas 3. Code that writes into a `.to_numpy()` result works here and raises
"assignment destination is read-only" there, because copy-on-write makes those arrays
read-only. timeseries-anomaly failed 52 of its own tests that way while passing all of
them locally. There is a matching environment at
`C:\Users\PRECIS~1\AppData\Local\Temp\p3env` (Python 3.11, pandas 3.0.6) - check every
pandas-touching package against it before approving.

**Name collisions.** PyPI rejects names too close to an existing project, and this is
invisible to an availability check: a blocked name still answers "not found". Two hit
it so far - schema-guard became dataframe-schema-guard, auto-label became
rule-auto-label. It shows up as a 400 on first upload and needs a rename; retrying
never helps.

## Picking it up again

Nothing needs restarting. Publishing continues on its own. To carry on building, take
the next three from "still to build" and follow `CONVENTIONS.md`; three per batch is
what completes reliably, and one workflow at a time.
