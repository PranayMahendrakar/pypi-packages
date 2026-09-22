# machine-health

One continuously updated 0-100 health score per machine, built from every sensor you have
and the limits you already know, with the reasoning attached so you can act on it.

## Install

```
pip install machine-health
```

## Quickstart

```python
import numpy as np, pandas as pd
from machine_health import score

rng = np.random.default_rng(0)
temp = np.r_[rng.normal(60, 1, 800), rng.normal(63, 1.5, 200)]        # the last fifth runs hot
vib = np.r_[rng.normal(0.2, 0.02, 800), rng.normal(0.24, 0.025, 200)]
result = score(pd.DataFrame({"temp": temp, "vibration": vib}), rules={"temp": {"max": 70, "warn_max": 65}})
print(result.summary())
```

```
machine health: 78.2 / 100  (grade C, trend degrading)
window: 800 row(s), baseline: 200 row(s)
channels (2): temp, vibration
components (score, weight, points lost):
  stability      68.3  weight 0.30     -9.50
  compliance     90.0  weight 0.30     -3.00
  anomaly        53.8  weight 0.20     -9.25
  availability  100.0  weight 0.20      0.00
points lost by channel:
  temp                  -16.19
  vibration              -5.57
violations (1):
  [warning] temp above warn_max=65 on 10 of 800 rows (1.2%), worst 66.73
```

`result.value` is the number to put on a dashboard; everything else on the object says where
it came from. The points lost are split two ways - `result.component_penalties` (stability,
compliance, anomaly, availability) and `result.contributors` (temp, vibration) - and each
split adds up to exactly `result.penalty`, which is `100 - result.value`. The printed figures
above are those two splits rounded to two decimals.

## What it scores

Four components, each reported separately on its own 0-100 scale, each measured against a
**baseline period** of rows you call healthy (by default the first 20% of the data):

- **stability** - how far each channel has moved from the way it behaved in the baseline, in
  either of the two ways it can move, whichever costs more. It **swings more**: a channel
  swinging `4x` more than it used to scores 0, and varying less than baseline is never
  penalised. Or it has **settled at another level**: a move of up to one baseline spread is
  free, 1000 spreads scores 0, logarithmic in between. The second half matters because a
  channel that steps to a steady wrong value swings exactly as much as it always did.
- **compliance** - whether your own limits hold. Hard limits (`min` / `max`) are critical, soft
  limits (`warn_min` / `warn_max`) are warnings, and a single breach always costs something,
  so one bad reading in a million rows does not vanish into the average.
- **anomaly** - the share of readings that are outliers against the baseline (robust z above
  3.5, median and MAD based). A channel with 10% or more outlying readings scores 0.
- **availability** - missing readings, plus a penalty for a channel that used to move and is
  now flatlined, which is what a dead sensor looks like.

The components are combined with weights (`stability 0.30, compliance 0.30, anomaly 0.20,
availability 0.20` by default) into `value = 100 - sum(weight * points lost)`, so the score is
always between 0 and 100.

**Grades are fixed bands**, not a curve: `A >= 90`, `B >= 80`, `C >= 70`, `D >= 60`, `F` below 60.
`result.ok` is `True` when the grade is C or better *and* no hard limit was broken.

**A component that cannot be measured is not guessed.** With no rules there is no compliance
score; with a one-row baseline there is no stability score. The weight of a component that
could not be measured is shared out over the ones that could, `result.unmeasured` names it,
`result.to_dict()["components"]` reports it as `null` rather than a free 100, and a note in
`result.notes` says so. That is why an all-NaN table scores 0 on availability alone instead
of inheriting three free 100s.

**Without rules the score is statistical, not absolute.** It knows only how the machine
behaves now compared with its own healthy period - nothing about what its readings *mean*.
That is enough to catch extra noise, outliers, a level shift and dead sensors, but a channel
whose baseline was never healthy in the first place will look fine. Pass your own limits
(`rules={"temp": {"max": 80}}`) whenever you have them; they are the only part of the score
that knows what the numbers are supposed to be.

## API

```python
score(df, *, time=None, channels=None, rules=None, weights=None, baseline=None) -> MachineScore
```

- `df` - a DataFrame or a path to a `.csv` / `.tsv` / `.parquet` file; one row per reading,
  one column per channel.
- `time` - name of the timestamp column. Rows are sorted by it, violations report when they
  first happened, and `result.when` carries the newest timestamp.
- `channels` - which columns to score; defaults to every numeric column. A channel a rule
  names is always included.
- `rules` - `{"temp": {"max": 80}}`, a `Rule`, or a list of either. A rule naming a channel
  that is not in the data raises `ValueError` listing the channels that are.
- `weights` - `{"compliance": 0.5}`; missing names keep their default. Weights that do not sum
  to 1 are normalized and a note is recorded.
- `baseline` - the healthy period: a DataFrame or file path, a fraction (`0.3`), a row count
  (`500`), or a boolean mask over the rows. Default: the first 20% of the rows.

`HealthScorer(time=..., channels=..., rules=..., weights=..., baseline=...)` is the same thing
as an object, when you want to keep the settings and call `.score(df)` repeatedly.

### `MachineScore`

- `.value` - 0-100; `.grade` - `"A"` to `"F"`; `.penalty` - `100 - value`
- `.components` - `{"stability": 68.3, "compliance": 90.0, ...}`, each 0-100
- `.weights` - the weight each component actually got; `.component_penalties` - points each cost
- `.contributors` - `{channel: penalty points}`, summing to `.penalty`; `.top_contributors(n)`
- `.violations` - list of `Violation(channel, limit, value, severity, count, n_rows, fraction,
  worst, first_time)`, critical first; `.critical_violations`
- `.trend` - `"improving"`, `"stable"` or `"degrading"`: the two halves of the window compared
  by how far each sits from the baseline (level shift, extra spread, missing readings and
  breached rules, in baseline spreads). It compares the halves with each other, not with the
  score, so a half that straddles a step never outranks the steady bad level that follows it
- `.unmeasured` - components that could not be scored at all; `.notes` - everything the scorer
  decided for you; `.channels`, `.n_rows`, `.n_baseline_rows`, `.when`, `.ok`
- `.summary()` - the text above; `.to_dict()` - JSON-safe, including the grade bands

### `Rule`

```python
from machine_health import Rule, score
score(df, rules=[Rule("temp", max=80, warn_max=75), Rule("vibration", max=2.5, weight=2.0)])
```

`Rule(channel, min=None, max=None, warn_min=None, warn_max=None, weight=1.0)`. `weight` sets
how much the rule counts against the other rules inside the compliance component.

### `HealthMonitor`

```python
from machine_health import HealthMonitor

monitor = HealthMonitor(healthy_df, rules={"temp": {"max": 70}}, time="ts")
for batch in incoming_batches:
    result = monitor.update(batch)      # same baseline every time, so scores are comparable
print(monitor.history)                  # one row per update
print([a.summary() for a in monitor.alerts])
```

`HealthMonitor(baseline_df, rules=None, weights=None, *, time=None, channels=None)` keeps the
baseline fixed and scores each batch exactly the way `score()` does.

- `.update(batch)` - a `MachineScore`. An empty batch returns the previous score unchanged
  with a note instead of raising. From the second update on, `.trend` compares the batch with
  the mean of the previous three.
- `.history` - DataFrame: `update, when, value, grade, trend, stability, compliance, anomaly,
  availability, rows, violations`. A component that was not measured is `NaN` in its column
  (with no rules, the whole `compliance` column is), never a free 100
- `.alerts` - `Alert(when, severity, message)`, raised when the grade drops a level (critical
  for two levels or a drop to F) or a rule is violated
- `.baseline_score` - the baseline scored against itself, the starting point for comparisons
- `.last_score`, `.summary()`, `.to_dict()`

## CLI

```
machine-health telemetry.csv
machine-health telemetry.csv --time ts --rule "temp:max=80,warn_max=75" --rule "vibration:max=2.5"
machine-health telemetry.csv --json --output health.json
machine-health telemetry.csv --baseline 0.3 --weight compliance=0.5 --fail-under 70
```

`--rule` is repeatable and takes `channel:key=number` pairs (`min`, `max`, `warn_min`,
`warn_max`, `weight`); `--rules FILE` reads the same thing from JSON. `--baseline` takes a
fraction, a row count or a path to a table of healthy rows. The exit status is 0 unless
`--fail-under SCORE` is given and the score is below it (2 on a bad argument or unreadable
file), so it drops straight into a shell check.

## License

MIT
