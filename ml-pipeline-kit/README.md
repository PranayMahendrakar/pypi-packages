# ml-pipeline-kit

Build a preprocess, predict, validate and log pipeline in a few lines, and get every
step timed, checked and reported without writing any of that yourself.

## Install

```bash
pip install ml-pipeline-kit
```

Reading `.parquet` files on the command line needs `pip install ml-pipeline-kit[parquet]`.

## Quickstart

```python
import pandas as pd
from ml_pipeline_kit import Pipeline

customers = pd.DataFrame({"age": [31, 45, 29], "income": [50000, 82000, 41000]})
pipe = Pipeline("scoring").expect_range("age", 18, 100).predict(lambda df: df.assign(score=df["income"] / 1000))
result = pipe.run(customers)
print(result.summary())
print(result.output)
```

```text
ml-pipeline-kit: pipeline 'scoring' ok, 2 of 2 steps ran in 0.756 ms
  rows      : 3 in, 3 out
  steps     :
    1. range[age]  range    ok    0.406 ms
    2. predict     predict  ok    0.350 ms  3 rows in, 3 rows out
   age  income  score
0   31   50000   50.0
1   45   82000   82.0
2   29   41000   41.0
```

(timings differ from run to run.)

Inside a service, call the pipeline instead of running it and you get the output
value alone - and a clear error, never a half-finished value, if a check stopped it:

```python
scored = pipe(customers)
```

## What it does

- **Times every step** and records the rows that went in and came out, so
  `result.timings` and `result.summary()` are there without any logging code.
- **Names the step that failed.** A step that raises is reported with its name,
  the rows that reached it, and the original exception chained underneath -
  never a bare traceback out of somebody else's function.
- **Checks the data between steps.** `expect_schema`, `expect_range` and
  `validate` run in order with the steps. A failed check stops the run before
  the next step touches bad data, and the result names the check.
- **Decides what a failure means, per step.** `on_error="raise"` lets it out,
  `"skip"` carries on with the data the previous step produced and records that,
  `"stop"` ends the run and reports it in the result.
- **Never mutates your input.** pandas, numpy and the builtin containers
  (including tuples) are copied before the first step, so a step that edits in
  place cannot reach your value. The one thing not copied is an arbitrary object
  you stored inside a DataFrame cell, because `DataFrame.copy(deep=True)` does
  not copy those either.
- **Says nothing it does not know.** Row counts are recorded only when the data
  has a length, and left as `None` otherwise rather than guessed.
- **Does nothing surprising when empty.** A pipeline with no steps returns the
  input unchanged, `ok` is True, and a warning says the pipeline was empty.
- **Saves as configuration.** `Pipeline.save()` writes the step names and their
  configuration, never the callables - the file says so, and `load()` hands back
  a pipeline that names the steps you still have to re-register.

## API

```python
Pipeline(name="pipeline")
```

| Method | What it adds |
|--------|--------------|
| `.add(step, *, name=None, on_error="raise")` | any callable taking and returning the data |
| `.preprocess(func)` / `.predict(func)` / `.postprocess(func)` | the same, labelled with that role |
| `.validate(check, *, name=None, severity="error")` | a check: `callable(data) -> bool` or `(bool, message)` |
| `.expect_schema(schema)` | required columns and dtypes at that point |
| `.expect_range(column, low=None, high=None)` | numeric bounds for a column |
| `.run(data, *, collect_timings=True)` | runs everything, returns a `Result` |
| `.__call__(data)` | runs everything, returns the output value alone |
| `.save(path)` / `Pipeline.load(path)` | step names and configuration as JSON |
| `.bind(name, func)` | give a loaded step its callable back |
| `.describe()` / `.step_names` / `.unbound` | what is registered, and what is still missing |

Every method returns the pipeline, so they chain. `severity="error"` stops the run;
`severity="warn"` records the failure in `result.warnings` and carries on.
`schema` is `{"age": "int"}`, a list of column names when only presence matters, or a
DataFrame to copy the dtypes from; dtypes match by family, so `"int"` accepts any
integer width. `expect_range` ignores missing values, warns with a count when values
are present but cannot be read as numbers, and records a note when a column has no
usable numbers at all instead of passing silently. A date or a duration column is
reported as a type problem, never compared as nanoseconds.

`Result`:

| Member | What it gives you |
|--------|-------------------|
| `.output` | the value the last step produced |
| `.ok` | True when nothing failed (a `warn` check does not change it) |
| `.steps` | `list[StepRun(name, ok, duration_ms, rows_in, rows_out, error, kind, severity, skipped)]` |
| `.failures` / `.warnings` | plain sentences, each naming the step or check |
| `.timings` | `{step name: milliseconds}` |
| `.stopped_at` | the step or check that ended the run early, else `None` |
| `.rows_in` / `.rows_out` / `.duration_ms` | totals for the whole run |
| `.step(name)` | one `StepRun` by name |
| `.summary()` | the plain-text report above |
| `.to_dict()` | JSON-safe dict of the whole run (without `output`, which is your data) |

Errors: `StepError` when a step raises, `ValidationError` when a check stops a
pipeline that was called as a function, both subclasses of `PipelineError`.

For the one-line case there is a module-level `run`:

```python
import ml_pipeline_kit

result = ml_pipeline_kit.run(frame, clean, score)
```

### Saving and loading

`save()` records step names, kinds, `on_error`, `severity` and check configuration.
It never records the callables, because they are code, not data:

```python
pipe.save("scoring.json")

restored = Pipeline.load("scoring.json")
print(restored.unbound)            # ['clean', 'score'] - callables must be re-registered
restored.bind("clean", clean).bind("score", score)
result = restored.run(frame)
```

Schema and range checks come back ready to run. Running before every step is bound
raises a `PipelineError` that names the ones still waiting.

## CLI

The checks are configuration, so they run against a file without writing any Python:

```bash
ml-pipeline-kit sales.csv --expect-schema "id:int,city:str"
ml-pipeline-kit sales.csv --expect-range "price:0:1000" --not-null id
ml-pipeline-kit sales.csv --expect-range "age:18:" --json > report.json
ml-pipeline-kit --describe scoring.json
```

`--help` lists every option. `--json` prints `to_dict()`, `--output PATH` writes it to
a file (creating the folder if it is missing, as `Pipeline.save()` does), and `--warn`
records failures as warnings instead of stopping. The exit code is 0 when every check
passed, 1 when one failed, and 2 when the file could not be read - with `--warn` no
check can fail, so that run always exits 0 and the failures are listed as warnings.

## License

MIT
