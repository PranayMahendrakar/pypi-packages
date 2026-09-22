# energy-analyzer-ai

Find unusual energy consumption in meter readings, explain what changed, and estimate what it is costing.

A Monday 9am is compared with other Monday 9ams, not with 3am. A cumulative meter
(the kind that only counts up) is detected and differenced instead of being read as
one huge value. Gaps are reported, never quietly filled. And with no tariff, every
cost stays `None` rather than zero.

## Install

```bash
pip install energy-analyzer-ai
```

## Quickstart

```python
import numpy as np, pandas as pd, energy_analyzer_ai as ea

hours = pd.date_range("2026-03-01", periods=24 * 21, freq="h")
kwh = 0.4 + 1.8 * ((hours.hour >= 9) & (hours.hour < 18))
readings = pd.Series(kwh + np.random.default_rng(0).normal(0, 0.05, hours.size), index=hours, name="kwh")
readings.iloc[300] = 9.0                      # one afternoon that went wrong

print(ea.analyze(readings, tariff=0.28).summary())
```

```text
energy-analyzer-ai: 1 unusual period in 504 1h periods (0.2%)
  meter     : kwh over time, 2026-03-01 to 2026-03-21 23:00
  total     : 547.8 kWh, costing 153.39
  always-on : 0.336 kWh per 1h, about 31% of everything used
  baseline  : day of week and hour of day, robust sigma 0.0501, threshold 4 sigmas
  trend     : flat
  unusual   : 6.78 kWh above normal, costing about 1.90
    spike 2026-03-13 12:00: 9 against an expected 2.22 (+6.78, 135.3 sigmas), 1.90 extra
  findings:
    - Used 547.8 kWh across 504 1h periods from 2026-03-01 to 2026-03-21 23:00, costing 153.39.
    - The always-on load is 0.336 kWh per 1h, about 31% of everything used (overnight minimum (00:00-05:00), median of 21 night(s)).
    - 1 period did not look like the same hour on other days: 1 spike.
    - The spikes used 6.78 kWh more than normal, costing about 1.90.
    - The biggest one was 2026-03-13 12:00: 9 kWh against an expected 2.22 kWh.
    - The baseline is steady; there is no clear trend across the window.
```

## What it finds

- **Spikes and drops** against a same-phase baseline: each period is compared with
  the same hour on the same weekday, so the evening peak is not flagged for being
  an evening peak.
- **Standby / always-on load** - the persistent overnight minimum, and what share
  of the bill never switches off.
- **A rising or falling baseline** - a slow drift in the level, with how fast and
  how confident.
- **Step changes** - the level jumped and stayed there, separated from slow drift
  so one change is reported once.
- **Cost** - per period, per anomaly, and in total, from a flat rate or a
  time-of-use `{hour: rate}` map.
- **What it could not do** - gaps, meter resets, negative readings, and every
  fallback it had to take are on the report as `notes` and `warnings`.

## API

```python
analyze(df, *, value=None, time=None, tariff=None, baseline=None, granularity="auto") -> EnergyReport
baseline_load(df, **kw) -> float          # the always-on floor on its own
forecast(df, periods=24, **kw) -> pd.Series   # same-phase seasonal-naive projection
EnergyAnalyzer(...)                       # the class underneath, for the knobs
```

`df` is a DataFrame with a reading column and a timestamp column (or a
`DatetimeIndex`), a Series, a list of numbers, or a path to a `.csv` / `.parquet`
file. `value=` and `time=` name the columns when the guess needs help.

`tariff=` is a flat cost per unit (`0.28`) or a `{hour: rate}` mapping for
time-of-use pricing (`{0: 0.12, 7: 0.31}`). Left out, every cost field is `None`.

`baseline=` is `None` to learn normal behaviour from the data, a number to fix the
expected value per period, or another frame / path to use as a reference period.

`granularity=` is `"auto"`, or `"hourly"`, `"daily"`, `"weekly"`, `"15min"`, or any
fixed pandas offset.

For the knobs - `sensitivity` (robust sigmas, default 4.0), `min_effect`, the
overnight `night=(start, end)` window, and `cumulative=True/False` to override the
cumulative-meter detection - use `EnergyAnalyzer(...)` and call `.analyze(df)` on it.

### EnergyReport

| attribute | what it is |
| --- | --- |
| `.total`, `.total_cost` | units used, and what they cost (`None` without a tariff) |
| `.baseline_load` | always-on load per period; `.baseline_share` of the total |
| `.anomalies` | `list[Anomaly(when, observed, expected, excess, cost)]`, also `.spikes` / `.drops` / `.top(n)` |
| `.excess_units`, `.excess_cost` | how much the spikes used above normal, and its cost |
| `.by_period` | DataFrame: `observed`, `expected`, `excess`, `score`, `is_anomaly`, `is_gap`, and cost columns |
| `.trend` | `Trend` - direction, `pct_per_week`, `r2`, `confident` |
| `.steps` | `list[StepChange]` - sudden level changes that stayed |
| `.findings` | plain-language sentences, in the order a person wants them |
| `.notes`, `.warnings` | what had to be guessed, and what could not be done |
| `.n_gaps`, `.longest_gap`, `.cumulative_meter`, `.negative_readings` | what the readings themselves were like |
| `.summary()` | the short text report above (ASCII only) |
| `.to_dict()` | the whole thing, JSON-safe |

## CLI

```bash
energy-analyzer-ai meter.csv
energy-analyzer-ai meter.csv --value kwh --time recorded_at --tariff 0.28
energy-analyzer-ai meter.csv --tariff '{"0": 0.12, "7": 0.31}' --granularity hourly
energy-analyzer-ai meter.csv --json > report.json
energy-analyzer-ai meter.csv --output by_period.csv --forecast 24
```

`--help` lists every option. Output is UTF-8 whatever the console codepage is, so
piping a report with non-ASCII meter names to a file is safe.

## License

MIT
