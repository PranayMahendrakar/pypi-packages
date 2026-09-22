# model-watchdog

Lightweight production monitoring for any ML model: log predictions, catch drift and silent failure.

A model that has quietly started returning the same value for every request still
returns HTTP 200. `model-watchdog` is the small thing you put in the request path
to notice. It logs each prediction to a JSONL file, and later tells you - in
plain sentences - whether the traffic still looks like the traffic you trained on.

No server, no database, no model downloads. Just pandas and numpy.

## Install

```
pip install model-watchdog
```

Reading a `.parquet` reference needs `pip install "model-watchdog[parquet]"`.

## Quickstart

```python
import model_watchdog

wd = model_watchdog.Watchdog("checkout-model", reference={"prediction": [0.1, 0.2, 0.3, 0.4, 0.5] * 20})
for score in [0.92, 0.93, 0.94, 0.95, 0.96] * 20:
    wd.log(features={"cart_items": 3}, prediction=score, latency_ms=12.0)
print(wd.check().summary())
```

```
model-watchdog: checkout-model - ALERT: 1 of 3 active checks failed (prediction_drift)
records: 100 (window 1000) | 2026-09-22T10:15:03+00:00 to 2026-09-22T10:15:03+00:00
reference: 100 predictions

  check             status    value   threshold  detail
  prediction_drift  FAIL      8.7542  0.2000     PSI 8.7542 against 100 reference predictions: major shift
  feature_drift     inactive  -       -          no reference features; pass reference= with feature columns
  accuracy_drop     inactive  -       -          reference has no accuracy or error to compare against
  latency           inactive  -       -          no reference latency; add latency_ms to the reference
  null_rate         ok        0       0.0500     0.00% of feature values missing across 1 feature (worst: cart_items at 0.00%)
  constant_output   ok        1       50         longest run of one value is 1 of 100 predictions (value: 0.92, flat limit 50)
  volume            inactive  -       -          no reference traffic rate; add a timestamp column to the reference

notes:
  - inactive monitors (not failures, just no data yet): feature_drift, accuracy_drop, latency, volume
```

A monitor that lacks the data it needs reports itself **inactive**. That is not a
failure: it is the report telling you what to log next to switch it on.

In real use the two halves live in different places:

```python
wd.log(features=row, prediction=p, latency_ms=ms)   # in the request handler
print(model_watchdog.check("checkout-model").summary())   # in a cron job
```

## What it checks

- **prediction_drift** - PSI of the logged predictions over reference deciles.
- **feature_drift** - PSI per feature, when features were logged. Names the worst one.
- **accuracy_drop** - rolling accuracy (or mean absolute error) against the
  reference, when actuals were logged.
- **latency** - p50 and p95 against the reference percentiles.
- **null_rate** - share of feature values that arrived missing.
- **constant_output** - the same prediction over and over: the classic silent failure.
- **volume** - traffic far above or below the reference rate.

PSI is read the usual way: under 0.1 is no real shift, under 0.25 a moderate one,
above that a major one. The default threshold is 0.2.

Two of these read the reference before they judge anything, which is what keeps
them quiet on healthy traffic:

- **The drift monitors wait for volume.** A ten-bin PSI needs enough records to
  fill those bins; computed over a handful it reads as drift when nothing has
  moved. Below `thresholds.min_drift_records` (60) they report themselves
  inactive, and above it the bin count is scaled to the records available. A
  comparison of a few labels is binned per value instead, and needs only
  `min_records`.
- **`constant_output` asks what the model normally outputs.** A fraud or churn
  model that answers 0 for 95% of requests repeats itself by design: in a
  healthy 1000-request window its longest run of zeros is around 87. When the
  reference says what the output mix looks like, the limit becomes the longest
  run that mix would produce by chance; `thresholds.constant_run` (50) is the
  floor under it, and all there is to go on with no reference. If the reference
  is so lopsided that the window could not tell stuck from normal either way,
  the monitor says that rather than guessing.

## Built for the request path

- `log()` **never raises into the caller** - not on a full disk, not on a bad
  value, not on a read-only directory. It warns through `logging` and returns.
- Appends are safe from several processes at once: each record is written whole,
  under an exclusive lock, so workers do not overwrite each other.
- That safety has a price: roughly **0.1 to 0.2 ms per call**, 5,000 to 10,000
  records a second depending on the disk, because every record opens, locks,
  writes and closes the file rather than holding a handle open across requests.
  Next to a 10-100 ms inference it does not show; on a sub-millisecond model it
  is real, so sample your logging (`if random.random() < 0.1:`) if you are in
  that range. Reading is unaffected: `check()`, `report()` and `metrics()` each
  chew through 200k records in a few seconds.
- A partial trailing line left by a process that died mid-write is skipped with a
  warning when the log is read, not a crash.
- Timestamps are timezone-aware UTC, always.
- No reference yet? Logging still works, and `check()` tells you which monitors
  are waiting for data.

## API

### `Watchdog(name, *, reference=None, storage=None, alert=None)`

- `name` - what the model is called; also the default folder name.
- `reference` - what normal looks like. A DataFrame of past traffic, a path to a
  `.csv`/`.parquet`, a dict (`{"prediction": [...], "actual": [...]}`), a dict of
  already-known numbers (`{"accuracy": 0.9, "latency_p95": 40}`), or just a list
  of past predictions. Optional. A reference that cannot be read becomes a note
  on every report rather than an exception.
- `storage` - directory for the JSONL log. Defaults to `./.model_watchdog/<name>`.
- `alert` - optional callable taking one `Alert`, called by `check()` when a
  monitor fails. Its own errors are swallowed.

| Method | Returns | Notes |
| --- | --- | --- |
| `.log(features=None, prediction=None, actual=None, latency_ms=None, **meta)` | `None` | Appends one record. Never raises. `ts=` in `**meta` backdates it. |
| `.check(window=1000)` | `WatchReport` | Runs every monitor over the last `window` records, then fires `alert`. |
| `.report(since=None)` | `WatchReport` | Same monitors over everything since a datetime or ISO string. Never alerts. |
| `.metrics()` | `DataFrame` | The whole log: `ts`, `prediction`, `actual`, `latency_ms`, one column per feature, one per meta key. |

`log()` and `metrics()` swallow their own errors. `check()` and `report()` are
the diagnostic path and do raise, so a broken set-up cannot hide behind a green
report: a `window` that is not a positive integer, a `thresholds=` that is not a
`Thresholds`, and a `since=` that cannot be read as a timestamp are all errors
rather than a quietly different question. `Watchdog(name)` wants a real `str`,
since the name is also the folder name.

`.thresholds` is a `Thresholds` dataclass you can assign to; `check()` and
`report()` also take `thresholds=` for a one-off run. `len(wd)` counts the
records, and `wd.reference` can be set after the fact - including from
`wd.metrics()`, to use a good week as the baseline for the next one.

### `WatchReport`

| Member | Type | Meaning |
| --- | --- | --- |
| `.ok` | `bool` | True when no **active** monitor failed *and* something could run. `bool(report)` is the same. |
| `.blind` | `bool` | True when every monitor crashed, so nothing was checked. Makes `.ok` False. |
| `.checks` | `list[Check]` | Every monitor, in report order. |
| `.failed` | `list[Check]` | Only the active monitors that tripped. |
| `.inactive` | `list[Check]` | Monitors that lacked data, each with a reason. |
| `.summary()` | `str` | The block shown above. Plain ASCII, safe to pipe. |
| `.to_dict()` | `dict` | JSON-safe, nothing lost. |
| `.get(name)` | `Check \| None` | One monitor by name; `report["latency"]` also works. |

Each `Check` carries `name`, `ok`, `value`, `threshold`, `message`, `active` and a
`details` dict (per-feature PSI, the repeated value, the latency percentiles).

### `model_watchdog.check(name, *, reference=None, storage=None, window=1000)`

One line, for a cron job or a notebook where the process that wrote the log is
long gone:

```python
report = model_watchdog.check("checkout-model")
```

## CLI

```
model-watchdog checkout-model                  # the summary above
model-watchdog ./logs/checkout --json          # the report as JSON
model-watchdog checkout-model --metrics        # the raw records as CSV
model-watchdog checkout-model --reference last_month.csv
model-watchdog checkout-model --since 2026-09-01 --output report.json
model-watchdog checkout-model --fail-on-alert  # exit 1 when a monitor failed
```

`NAME` is either a watchdog name (resolved under `./.model_watchdog/`) or a path
to a storage directory. `--fail-on-alert` is the one to put in cron. Output is
UTF-8 and safe to pipe, whatever is in your feature names.

## License

MIT
