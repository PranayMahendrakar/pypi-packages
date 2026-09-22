# predictive-maintenance

Estimate how close equipment is to failure from its sensor history, and train a failure model when you have labels.

## Install

```bash
pip install predictive-maintenance
```

Reading `.parquet` files needs `pip install "predictive-maintenance[parquet]"`.

## Quickstart

```python
import numpy as np, pandas as pd, predictive_maintenance as pm

wear = np.concatenate([np.zeros(168), np.linspace(0, 0.7, 120)])   # a bearing starts to go
noise = np.random.default_rng(0).normal(0, 1.0, (2, 288))
df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=288, freq="h"),
                   "vibration": 1 + wear + 0.15 * noise[0],
                   "temp_c": 60 + 2 * wear + 0.8 * noise[1], "rpm": 1500.0})
print(pm.health_score(df).summary())    # 0-100 degradation, and which sensor is driving it
print(pm.estimate_rul(df).summary())    # how long before it crosses the threshold
```

```
predictive-maintenance: health score 18.0/100 [####................] (degrading)
baseline score 0.6 over 58 rows | window 28 rows | 3 channel(s): vibration, temp_c, rpm

  channel                    share of degradation
  vibration                   69.7%
  temp_c                      30.3%
  rpm                          0.0%
predictive-maintenance: remaining useful life 5 days 00:15:56 (confidence: high)
health score 18.0/100 now, threshold 30.0, trend degrading
expected to cross the threshold at: 2026-01-17T23:15:56
trend fit: +3.299e-05 points per second, r-squared 0.77
```

No failure labels are needed for either call. `health_score` takes the first 20% of the
history as the healthy reference and measures how far every later window has drifted from
it; `estimate_rul` extrapolates that drift to the point it crosses a threshold.

## What it does

**`health_score(df)` - unsupervised, no labels.** Each sensor channel is compared against
its own behaviour during the baseline period on three independent axes:

- **level** - how far the rolling mean has moved, counted in baseline standard
  deviations of that channel;
- **variance** - how many doublings (or halvings) the channel's noise is away from the
  spread it had while healthy;
- **spectral energy** - how far the high-frequency share of the channel's power has
  shifted, measured against how much that share already wandered during the baseline.
  This catches a bearing that starts to chatter long before its average value moves.

The three axes are combined per channel as a root-mean-square, then blended across
channels: half the worst channel, half the average of them all, because equipment is as
sick as its sickest sensor but broad drift is worse than one noisy probe. The result is
squashed onto 0 (healthy) to 100 (failed), with four deviation units scoring 50 and eight
scoring 80. A healthy machine sits in the low single digits. `.contributors` splits the
recent degradation across the channels, so the score comes with a culprit, and a channel
that never moves contributes exactly zero rather than `NaN`.

A channel held perfectly constant through the baseline - a controller setpoint, a
quantised reading - has no measured noise to compare against. Its scale is assumed from
the level it was held at, its contribution is capped at the middle of the range, and a
note says so, so last-digit float wobble on a setpoint cannot read as a destroyed bearing.

**`estimate_rul(df)` - how much life is left.** A straight line through the recent half of
the score is extrapolated to `threshold` (30 by default). A flat or improving trend never
gets there, so the answer is `float('inf')` with `confidence="low"` - never a negative
number, and never a crossing date invented from a line that is not rising. That holds even
when the score is already past the threshold: the note then says the machine needs
attention now instead of offering a deadline.

Confidence comes from the fit quality, how many points it was fitted on, and how far past
the data the projection reaches. A trend is only called `"degrading"` when it both moves
the score visibly and fits a line well enough to be more than the ordinary wander of a
healthy machine - but noise is still noise: on pure white noise about one machine in ten
is called `"degrading"` and gets a finite crossing date. None of those come back as
`confidence="high"`, so that field is how you filter them: treat `"low"` as "watch it",
not "book the outage".

On a timestamped frame `.remaining` is a `pandas.Timedelta`. On a numeric axis it is a
count of **rows**, even when the axis itself advances by more than one per row (a cycle
counter stepping by 10 is still answered in rows, with a note); `.eta` stays in the axis's
own units, and `.slope_per_step` is points per row.

**`MaintenanceModel` - supervised, when you have labelled failures.** Builds windowed
features per channel (rolling mean, std, min, max, slope, spectral energy) and trains a
gradient boosting classifier to answer "does a failure land within the next `horizon`
rows?". `.feature_importance` says which sensor and which statistic carried the signal.

The supervised path is the slow one: boosting costs a pass per row, per feature, per tree,
so 100k rows with four sensors takes about 90 seconds at the defaults, against under a
second for `health_score` and `estimate_rul` together on the same frame. It is working,
not hung - the model records a note saying so - and `n_estimators`, `max_depth` and fewer
`channels` are the knobs: `MaintenanceModel(random_state=0, n_estimators=40, max_depth=2)`
fits that same frame in about 15 seconds.

Input is a pandas DataFrame or a path to a `.csv` / `.tsv` / `.parquet` file anywhere a
frame is accepted. Timestamps are found automatically, rows out of time order are sorted,
and gaps in a channel are carried forward. Labels and a positional `baseline` travel
through that sort with their rows, so the same readings in a different order give the same
answer. Nothing prints; nothing is mutated.

## API

```python
predictive_maintenance.health_score(df, *, time=None, channels=None, baseline=None, window=None) -> HealthResult
predictive_maintenance.estimate_rul(df, *, time=None, channels=None, threshold=30.0) -> RULResult
predictive_maintenance.MaintenanceModel(random_state=0)
```

`time` names the timestamp column (auto-detected when omitted; pass `False` to force row
positions). `channels` picks the sensor columns, defaulting to every numeric column.
`window` is the rolling window in rows, about a tenth of the history by default.
`baseline` selects the healthy period and accepts a row count (`40`), a fraction
(`0.25`), a `slice`, a boolean mask, a list of row positions, or a `(start, end)` range on
the time axis. A baseline shorter than the window shortens the window to fit and records
the change in `.notes`.

**`HealthResult`** - `.score` (float, the current 0-100 degradation), `.series` (the score
for every row as a pandas Series on the original time axis), `.trend`
(`"improving"` / `"stable"` / `"degrading"`), `.contributors` (dict of channel to its share
of the degradation, summing to 1), `.top_contributors(3)`, `.degrading`, `.channels`,
`.window`, `.baseline_rows`, `.baseline_score`, `.threshold_hint` (the score `estimate_rul`
calls end of life by default), `.notes`, `.summary()`, `.to_dict()`.

**`RULResult`** - `.remaining` (a `pandas.Timedelta` when the data is timestamped, a float
count of rows otherwise, `float('inf')` when the trend never reaches the threshold),
`.confidence` (`"low"` / `"medium"` / `"high"`), `.eta` (the timestamp of the projected
crossing, or `None`), `.infinite`, `.remaining_text()`, `.score`, `.threshold`, `.trend`,
`.slope_per_step`, `.r_squared`, `.notes`, `.summary()`, `.to_dict()`.

**`MaintenanceModel(random_state=0, *, window=None, n_estimators=150, max_depth=3, learning_rate=0.1)`**

- `.fit(df, labels, *, horizon=10, time=None, channels=None) -> self` - `labels` is a
  boolean Series or array, `True` on the rows where the equipment failed, lined up row for
  row with `df` in whatever order `df` arrived. Labels with no failure in them raise a
  `ValueError` saying so and pointing at `health_score()`, rather than training a model
  that can only answer "no".
- `.predict(df, *, time=None) -> FailureRisk`
- `.evaluate(df, labels, *, time=None) -> dict` - `roc_auc`, `average_precision`,
  `accuracy`, `precision`, `recall`, `f1`, `n_positive`, `positive_rate`. A metric that
  cannot be computed (ROC AUC on single-class labels) comes back as `None` with the reason
  in `notes`.
- `.feature_importance` (dict of `"channel__stat"` to importance, strongest first),
  `.top_features(5)`, `.is_fitted`, `.summary()`, `.to_dict()`.

Every fit is deterministic: the same `random_state` and the same data give the same model.

**`FailureRisk`** - `.probability` (numpy array, one per row), `.risk`
(`"low"` / `"medium"` / `"high"` for the most recent row), `.probability_now`,
`.peak_probability`, `.peak_at`, `.as_series()`, `.horizon`, `.summary()`, `.to_dict()`.

```python
model = pm.MaintenanceModel(random_state=0).fit(df, failures, horizon=12)
risk = model.predict(df)
risk.risk, risk.probability_now, model.top_features(3)
```

## CLI

```
predictive-maintenance sensors.csv                     # health summary (same as: predictive-maintenance health sensors.csv)
predictive-maintenance health sensors.csv --json       # to_dict() as JSON
predictive-maintenance health sensors.csv --channels vibration,temp_c --baseline 0.25
predictive-maintenance rul sensors.csv --threshold 40  # remaining useful life
predictive-maintenance rul sensors.csv --json --output rul.json
```

Both commands take `--time COLUMN`, `--channels A,B,C`, `--baseline SPEC` (a row count or
a fraction), `--window N`, `--json` and `--output PATH`. Input may be `.csv`, `.tsv` or
`.parquet`. Output is UTF-8 everywhere, so sensor names in any script survive a pipe, a
redirect or a CI log.

## License

MIT
