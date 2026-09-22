# ml-inference-profiler

Find the slow step in an ML inference pipeline: wrap the stages you already have, run it, and get a tree of where the milliseconds went - preprocessing, the model, postprocessing - plus plain-language advice on what to do about it.

## Install

```bash
pip install ml-inference-profiler
```

## Quickstart

```python
import ml_inference_profiler as mip

steps = [("preprocess", lambda r: [x.strip().lower() for x in r]), ("model", lambda r: [len(x) ** 0.5 for x in r]), ("postprocess", lambda s: sum(s) / len(s))]
report = mip.profile_pipeline(steps, ["  Ünicode text  "] * 20000, repeats=5)
print(report.summary())                     # the tree, the bottleneck, and what to do about it
print(report.bottleneck.label, round(report.bottleneck.self_ms, 2))
```

`summary()` prints something like:

```
ml-inference-profiler: pipeline
pipeline: 3 stage(s), 13.92 ms total, 5 repeat(s), 1 warmup
  preprocess        8.41 ms   60.4%  x5
  model             5.30 ms   38.1%  x5
  postprocess       0.21 ms    1.5%  x5
Percent is the share of its own level; the stages under one parent add up to 100%.
Bottleneck: preprocess - 8.41 ms of self time, 60.4% of the run, over 5 call(s).
Timing floor: about 0.0015 ms per stage call (measured); 15 call(s) cost roughly 0.023 ms of the 13.92 ms measured.
Suggestions:
  - Slowest step: preprocess spends 8.41 ms in its own code, 60% of the 13.92 ms run, over 5 call(s). Start there.
  - Preprocessing is 60% of the run (8.41 ms): the data path, not the model, is where the time goes. The model itself accounts for 38%. Cache decoded inputs, move resizing and tokenizing into the loader, or prepare the next batch on a worker thread while the current one runs.
preprocess 8.41
```

`ml-inference-profiler --demo` prints a bigger one, with nested stages, a cold cache and the advice that follows from them.

For code you cannot express as a list of steps, drive the `Profiler` yourself. Stages nest to any depth, and a decorator turns any function into a stage:

```python
import ml_inference_profiler as mip

profiler = mip.Profiler("resnet50")
images = [list(range(60000)) for _ in range(3)]

@profiler.profile                           # every call is recorded under "run_model"
def run_model(batch):
    return sum(x * x for x in batch)

for image in images:
    with profiler.stage("preprocess"):      # stages nest to any depth
        with profiler.stage("resize"):
            image = image[:30000]
        image = [x / 255.0 for x in image]
    run_model(image)                        # a sibling of preprocess, not a child

print(profiler.report().tree())
```

which prints the two halves of the pipeline side by side, each with its share of the run:

```
resnet50: 3 stage(s), 3.90 ms total, 1 repeat(s), 0 warmup
  preprocess       1.98 ms   50.7%  x3  (self 1.85 ms)
    resize         0.13 ms  100.0%  x3
  run_model        1.92 ms   49.3%  x3
```

`preprocess` and `run_model` are siblings, so their shares are the split you came for;
`resize` is a child of `preprocess`, and its 100% is its share of that parent, not of the
run. `(self 1.85 ms)` is `preprocess` minus its children - the time in its own code.

## What it measures

- **Cumulative time** (`total_ms`) - from entering a stage to leaving it, nested stages included.
- **Self time** (`self_ms`) - cumulative time minus the cumulative time of the direct children: the time actually spent in that stage's own code. `report.bottleneck` is the stage with the largest **self** time, because that is the code you would have to make faster; sorting by cumulative time would always crown the outermost stage.
- **Per call**: `calls`, `mean_ms`, `median_ms`, `p95_ms`, `min_ms`, `max_ms` and `first_ms`. A p95 far above the median with the maximum on the first call is the signature of a cold cache.
- **Share of the level** (`share`) - percent of the stages that share the same parent. Stages under one parent add up to 100 percent, at every depth. `pct_of_total` is the percent of the whole run if that is what you want instead.
- **The timing floor** (`overhead_ms_per_stage`) - what one `with profiler.stage(...)` costs on this machine, measured once per process. A stage whose mean is near that floor cannot be timed accurately on its own, and the report says so.
- **Failures** - a stage that raises still records its elapsed time and counts in `Stage.errors`; the exception is re-raised unchanged, so you see the real error, not a profiler error.

Repeated labels aggregate: using `"preprocess"` twice gives one stage with two calls and the summed time, never a silent overwrite. A profiler with nothing recorded returns an empty report (`total_ms == 0.0`, `bottleneck is None`) instead of dividing by zero.

## What it suggests

`report.suggestions` is a list of plain sentences, each naming the numbers it fired on:

- the slowest step by self time, always, so the list is never empty;
- preprocessing dominating the run, with the model's share for comparison;
- a p95 far above the median, pointing at a cold cache or a lazily built model when the slowest call is the first one;
- per-item work that should be batched: many short calls per repeat adding up to a large share;
- timing overhead that has grown into a visible fraction of the run, meaning the stages are too small to time one at a time;
- stages that raised.

## API

```python
mip.profile_pipeline(steps, data, *, name="pipeline", repeats=5, warmup=1) -> ProfileReport
mip.Profiler(name="pipeline")
mip.load_report(path) -> ProfileReport
```

**`Profiler`**

- `profiler.stage(label)` - context manager timing one stage; nestable, and safe to use inside a decorated function.
- `profiler.profile(func)` - decorator recording every call as a stage. Use it bare for the function's own name, or `@profiler.profile("label")` to choose one.
- `profiler.run(pipeline, data, *, repeats=5, warmup=1)` - times a list of `(label, callable)` steps (bare callables work too, using `__name__`). Each step gets the previous step's output; a step returning `None` passes its input along. Every repeat restarts from `data`. Warmup passes run untimed, so cold caches do not skew the numbers - pass `warmup=0` when the cold cost is exactly what you want to see.
- `profiler.report()` - the `ProfileReport` for everything recorded so far; callable at any time.
- `profiler.reset()` - throw the measurements away and start again. `run()` accumulates into what is already there, so reset between unrelated runs.

**`ProfileReport`**

- `.stages` - `list[Stage]` in tree order.
- `.bottleneck` - the `Stage` with the largest self time, or `None` when nothing was recorded.
- `.total_ms` - the run total (the sum of the top-level stages).
- `.tree()` - the indented stage tree with each stage's share of its level.
- `.summary()` - the tree plus the bottleneck, the timing floor and the suggestions. `print(report)` prints the same thing.
- `.suggestions` - `list[str]`.
- `.to_frame()` - one row per stage as a pandas DataFrame.
- `.to_dict()` / `.to_json()` / `.save(path)` - JSON-safe output, UTF-8, `ensure_ascii=False`.
- `.find(label)` - the stage with that label or path, or `None`.
- `.overhead_ms_per_stage`, `.overhead_total_ms`, `.stage_calls`, `.repeats`, `.warmup`, `.threads`.

**`Stage`** - a frozen dataclass: `label`, `calls`, `total_ms`, `mean_ms`, `p95_ms`, `share`, `depth`, `parent`, plus `self_ms`, `pct_of_total`, `self_pct_of_total`, `median_ms`, `min_ms`, `max_ms`, `first_ms`, `errors`, `path`, and `.to_dict()`.

**Threads, honestly.** A profiler can be used from several threads without corrupting itself: the nesting stack is thread-local, so no thread sees another's parent stage, and every write to the shared totals happens under a lock. What it cannot give you is a breakdown of elapsed time when stages run in parallel - each stage measures its own wall clock, so two concurrent stages both count in full and the totals exceed the time the run actually took. Read a multi-threaded profile as "how long each stage took", not as "where the elapsed time went"; `report.threads` says how many threads contributed and the summary points it out. A stage entered on one thread must be left on the same thread. A warmup pass in `run()` pauses recording only on the thread that called it, never process-wide, so a worker thread that opens a stage during a warmup pass keeps its measurement instead of having it dropped without a word - the other side of that promise being that work a step fans out to worker threads is counted even during warmup, so pass `warmup=0` and discard the first report yourself when that matters.

**Time source.** `time.perf_counter` throughout: monotonic, and the highest resolution the standard library offers.

## CLI

```bash
ml-inference-profiler --demo                 # profile a small built-in pipeline and print the report
ml-inference-profiler --demo --tree          # only the stage tree
ml-inference-profiler --demo --repeats 10    # more timed passes for the demo (default 3)
ml-inference-profiler report.json            # print a report saved with ProfileReport.save()
ml-inference-profiler report.json --json     # the same report as JSON, on stdout
ml-inference-profiler report.json -o out.json
ml-inference-profiler --version
```

`--repeats N` only applies to `--demo`; it is ignored when a report file is given, since
that report was already timed. `--json`, `--tree` and `-o/--output` work with either
source.

Output is UTF-8 whatever the console encoding is, so non-ASCII stage labels survive being piped, redirected or captured by CI.

## License

MIT
