# model-benchmark

Put several models on the same task and find out which one is actually worth
shipping: latency, memory and accuracy for all of them, side by side, in one call.

## Install

```bash
pip install model-benchmark
```

Writing `.parquet` output needs `pip install model-benchmark[parquet]`.

## Quickstart

```python
import pandas as pd
from model_benchmark import benchmark

X = pd.DataFrame({"x": [0.1, 0.4, 0.6, 0.9] * 25})
y = [0, 0, 1, 1] * 25
report = benchmark({"always_0": lambda d: [0] * len(d),
                    "threshold": lambda d: (d["x"] > 0.5).astype(int)},
                   (X, y), metric="accuracy")
print(report.summary())
```

```text
model-benchmark: 2 models in 0.01 s, 2 measured
  data      : 100 rows
  calls     : 2 warmup + 10 timed, per batch size
  batch     : 1 of 100 rows per timed call
  metric    : accuracy (higher is better)
  fastest   : always_0 (0.000330 ms)
  leanest   : always_0 (0.0156 KB peak, one call at batch size 1)
  best score: threshold (1.0000 accuracy)

  model       latency_ms      p50_ms      p95_ms          per_s     peak_KB      score
  always_0      0.000330    0.000300    0.000455    3,030,306.7      0.0156     0.5000
  threshold       0.0574      0.0458      0.1014       17,406.4       3.300     1.0000
```

`report.best("latency")`, `report.best("score")` and `report.to_frame()` take it
from there.

## What it does

For every model, on exactly the same input, it measures:

- **Latency** - mean, p50, p95, min and max in milliseconds, from
  `time.perf_counter` and nothing else.
- **Throughput** - items per second at each batch size you ask for.
- **Peak memory** - the peak Python allocation growth of **one call at the first
  batch size**, which is one row by default, from `tracemalloc`, started and
  stopped around each model so nothing leaks between them. On a large dataset
  that call is tiny, so the `peak_KB` column reads in fractions of a KB; pass
  `batch_sizes=("all",)` to measure the memory of a whole-input call instead.
  Latency is measured in a separate pass with tracing switched off, because
  tracing makes every call several times slower.
- **RSS delta** - the process resident-size change around that same call, when
  the platform can report it. Noisy by nature; `None` when it cannot be read.
- **Score** - `accuracy`, `f1`, `rmse`, `mae`, `r2` or your own
  `callable(y_true, y_pred)`, computed once on the whole of `X`. `f1` is binary
  when the data holds two labels and macro averaged beyond that; the positive
  class is the **last label in sorted order** (so `1` of `{0, 1}`, `"yes"` of
  `{"no", "yes"}`). With a single label present throughout and predicted
  correctly it scores `1.0`, where scikit-learn's `f1_score` would report `0.0`
  for a `pos_label` that never appears.

And it stays out of your way:

- **One broken model never costs you the run.** Anything that raises is recorded
  in `report.failures` as a readable string and every other model is still
  measured.
- **A model returning the wrong shape is explained, not crashed on.** Three
  predictions for five inputs, or an `(n, k)` probability matrix where labels
  were expected, is reported in plain words and the timings still stand.
- **`timeout=` gives up on a model that hangs** and records it as timed out,
  and the time you pay is bounded by the timeout rather than by the model.
  Python cannot force a running call to stop, so the one call already in flight
  is abandoned rather than killed: it finishes on its own *daemon* thread, which
  means the rest of that model's schedule is skipped and the process still exits
  on time even if the call never returns at all. The report says so in
  `warnings`, and later measurements can be slightly noisier while it drains.
- **`repeats=1` works**, and so does a single-row input. Nothing divides by zero.
- **Deterministic.** Results, tables and rankings always come back in the order
  you listed the models, and ties go to whichever was listed first.
- **No GPU is assumed** and neither torch nor tensorflow is imported. A torch
  module is just a callable, so it works without this package knowing about it.

## API

```python
benchmark(models, data=None, *, metric=None, warmup=2, repeats=10,
          batch_sizes=(1,), timeout=None) -> BenchmarkReport
```

- **models** - `{name: model}`, where a model is a callable or an object with a
  `.predict` method (`.predict` wins when both are available). A list of
  functions works too; they are named after themselves.
- **data** - the input handed to every model. With `metric=` it is the tuple
  `(X, y_true)`. `None` means the models take no argument at all.
- **metric** - `"accuracy"`, `"f1"`, `"rmse"`, `"mae"`, `"r2"`, or any
  `callable(y_true, y_pred) -> float`. A custom callable is assumed to be better
  when bigger unless it carries `higher_is_better = False`.
- **warmup** - untimed calls before each batch size is timed, so import and
  first-call costs stay out of the numbers at every size, not just the first.
- **repeats** - timed calls per batch size.
- **batch_sizes** - how many rows of `X` go into each timed call. `(1,)` measures
  single-item latency, `("all",)` sends the whole input in one call, and
  `(1, 32, 128)` shows how a model scales. Throughput counts items, so the
  numbers stay comparable across batch sizes.
- **timeout** - seconds allowed for one model's whole measurement, or `None`.

```python
compare(models, data, **kw) -> DataFrame
```

The same run, returning just the table.

`BenchmarkReport`:

| Member | What it gives you |
|--------|-------------------|
| `.results` | `dict[name -> ModelResult]`, one entry per model, in your order |
| `.failures` | `dict[name -> str]`, only the models that broke |
| `.best(by)` | winning model name; `by` is `"latency"`, `"score"` or `"memory"` |
| `.rank(by)` | every measured model, best first |
| `.compare(a, b)` | ratios between two models, plus a one-line `summary` |
| `.to_frame()` | one row per model, ready for pandas; `batch_size=` picks the size, and a size that was not benchmarked is a `ValueError` |
| `.batch_frame()` | one row per model and batch size |
| `.summary()` | the plain-text report above |
| `.to_dict()` | JSON-safe dict of everything |
| `.warnings` | anything that had to be adjusted or explained |

`ModelResult` carries `.latency_mean_ms`, `.latency_p50_ms`, `.latency_p95_ms`,
`.latency_min_ms`, `.latency_max_ms`, `.latency_std_ms`, `.throughput_per_s`,
`.peak_memory_kb`, `.rss_delta_kb`, `.score`, `.timings` (per batch size),
`.warnings`, and `.ok` / `.error` / `.timed_out` when things went wrong.

`Benchmark` holds the same options for repeated runs:

```python
from model_benchmark import Benchmark

bench = Benchmark(metric="rmse", repeats=50, batch_sizes=(1, 32), timeout=10)
report = bench.run(models, (X, y))
print(report.compare("linear", "forest")["summary"])
```

## CLI

```bash
model-benchmark --demo
model-benchmark mymodels:fast mymodels:slow --data mymodels:SAMPLE
model-benchmark linear=m:lin forest=m:rf --data rows.csv --target label --metric accuracy
model-benchmark --demo --batch-sizes 1,16,64 --repeats 25
model-benchmark --demo --json > benchmark.json
model-benchmark --demo --output table.csv
```

Each `MODEL` is `name=module:attribute`, imported from the current directory.
`--demo` benchmarks a built-in example so you can see the output without writing
a file first; it brings its own models and data, so combining it with `MODEL`,
`--data` or `--target` is an error rather than a silent no-op. `--json` prints `to_dict()`, `--output` writes `to_frame()` as
`.csv` or `.parquet`, `--best-by latency` prints just the winner's name, and
`--help` lists every option.

## License

MIT
