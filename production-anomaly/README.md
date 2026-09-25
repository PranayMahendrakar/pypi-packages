# production-anomaly

Find out where a manufacturing line lost output - stoppages, slow running, a speed that sags
shift after shift, micro-stops that add up, and counter glitches - from nothing more than a
table of units produced over time.

## Install

```
pip install production-anomaly
```

## Quickstart

```python
import numpy as np, pandas as pd, production_anomaly as pa

t = pd.date_range("2026-03-02", periods=3 * 288, freq="5min")          # three days of 5-minute counts
units = np.where((t.hour >= 6) & (t.hour < 22), 300.0, 0.0)             # the line runs 06:00-22:00
units[130:142] = 0; units[400:440] *= 0.7; units[700] = 600             # a stop, a slow spell, a double count
report = pa.analyze(pd.DataFrame({"time": t, "units": units}), shift_hours=(6, 22))
print(report.summary())
```

```
production report for 'units' (per-interval counts, 5 min intervals)
period: 2026-03-02 00:00 to 2026-03-05 00:00
schedule: 1 shift: 06:00-22:00
scheduled 48.0 h; 24.0 h outside the shifts not counted
availability   97.9%  (60 min downtime in 1 stoppage)
performance    97.9%  (vs the line's typical rate: 300 per interval = 3,600/h)
oee_partial    95.8%  (availability x performance; quality not measured - it needs scrap data)
units made 165,600; estimated lost 7,200 (downtime 3,600, slow running 3,600)
findings:
  - Downtime: stopped for 60 min of 48.0 h scheduled (2.1%) in 1 stoppage; the longest ran 60 min from 2026-03-02 10:50. About 3,600 units lost.
  - Slow running: 1 period, 3.3 h in total, below 80% of the line's typical rate (240 per interval); the longest ran from 2026-03-03 09:20 to 2026-03-03 12:40. About 3,600 units lost.
  - Spikes: 1 interval far above normal; the largest, 600 units at 2026-03-04 10:20, is 2.0x the typical rate and looks like a double count (the same units counted twice). Spikes are counted at the typical rate so they do not inflate the totals.
stoppages (1, 60 min in total):
  2026-03-02 10:50 to 2026-03-02 11:50  60 min  downtime
events: 1 downtime, 1 slow_running, 1 spike (see report.events)
notes:
  - Quality is not measured: true OEE is availability x performance x quality, and quality needs scrap or reject counts that this data does not have. oee_partial is availability x performance, so it is an upper bound on the line's real OEE.
  - no target_rate given: performance is measured against the line's own typical rate (300 per 5 min interval, 3,600/h: the average of its normal running intervals), so it shows speed lost against its usual pace, not against rated speed.
  - shift_hours given (1 shift: 06:00-22:00); time outside the shifts is unscheduled and is not counted as downtime.
  - output not given; used 'units', the only numeric column.
```

The line is idle every night in that data. Because `shift_hours=(6, 22)` says it is only
scheduled 06:00-22:00, those hours are unscheduled, not downtime. Leave `shift_hours` out and
every hour counts: the nights then show up as downtime, and the report adds a finding that
suggests the `shift_hours` value to pass.

## What it checks

The input is a table with a time column and a units column, in either of two shapes, which
are told apart automatically:

- **units per interval** - each row is what the line made in the interval that starts at its
  timestamp (what `DataFrame.resample(...).sum()` gives you);
- **a running counter** - a total that only goes up. Units per interval are the differences
  between readings.

It then looks for:

- **Downtime** - runs of zero or near-zero output (at most 5% of the line's high rate), merged
  into stoppages with a start, an end and a duration. A stop of 5 minutes or more is
  `downtime` and costs availability; a shorter one is a `micro_stop` and costs performance,
  the way OEE treats minor stops.
- **Slow running** - at least 15 minutes below 80% of the reference rate: the lower of the
  line's own typical rate and `target_rate`. It has to persist (with intervals of 15 minutes
  or longer, at least two in a row) and sit further below the typical rate than the line's
  ordinary interval-to-interval scatter explains, so a noisy line is not called slow.
- **Rate drift** - a sustained decline in running speed across shifts: a robust (Theil-Sen)
  trend over at least 3 shifts that falls by 10% or more, with most shift-to-shift
  comparisons pointing down. One bad hour does not make a drift.
- **Micro-stops** - frequent short drops, each too small to notice, reported together with
  how many units they cost in total and how many minutes of full-speed running that equals.
- **Spikes** - an interval far above normal (over 1.5x the typical rate and well outside its
  usual spread). About 2x usually means a double count; 8x or more usually means a counter
  reset or rollover read as production. Spikes are counted at the typical rate, so they do
  not inflate the totals.
- **Counter resets** - a counter that drops back to zero mid-shift, or a per-interval count
  that goes negative because it was computed from one. It is reported as a reset, never as
  a huge negative rate or as a stoppage, and that interval's output is treated as unknown.

Everything else the data forces on the analysis is written into `report.notes`:

- **Schedule.** With `shift_hours` given, hours outside the shifts are unscheduled and are
  never downtime. Without it every hour counts as scheduled, and `by_shift` uses 8-hour
  blocks from 06:00. `shift_hours` has no days of the week: whole scheduled days on which
  the line made nothing (weekends, say) are downtime, and a finding lists them and says
  whether they are every Saturday and Sunday in the data. If those are days off, leave their
  rows out of `df`: a day without data counts neither as downtime nor as running, and is
  reported as a day without data rather than as a gap.
- **Timestamps.** Irregular timestamps are resampled onto a regular grid at the round
  interval (30 s, 1 min, 5 min...) nearest their typical spacing, each reading's units spread
  over the time until the next reading, and a note says so. Only a gap far beyond the data's
  own spacing is treated as missing data. Timezone-aware timestamps keep their timezone in
  every result, and shifts are read in that local clock. Missing readings are unknown time,
  counted neither as downtime nor as running.
- **Low counts.** When each interval holds only a few whole parts, one part more or less
  would look like a stop or a double count: a 45-second cycle reads 1, 1, 2, 1, 1, 2 per
  minute. Intervals are then merged into the shortest round length that typically holds at
  least 10 parts and fits the shift boundaries (10 minutes for that line), and a note says
  so. Stops are then timed to that length.
- **Quiet choices.** Any automatic choice (which column is time, which is output, counter or
  counts) is recorded. Rows that share a timestamp are always added up (for a counter, the
  highest reading is kept), even when they repeat the same value; a note points out such
  repeats, since a reading logged twice then shows up as a double-count spike. The caller's
  DataFrame is never modified.

**About OEE.** OEE is availability x performance x quality, and quality needs scrap or
reject counts. This package does not have them, so it does not report OEE. It reports
`oee_partial` = availability x performance, which is an upper bound on the real figure: a
line with 90% `oee_partial` and 5% scrap has an OEE near 85.5%. Without `target_rate`,
performance is measured against the line's own typical rate, so it shows speed lost against
its usual pace, not against its rated speed. Pass `target_rate` (units per hour) for a
performance figure you can compare between lines.

## API

```python
report = production_anomaly.analyze(df, *, time=None, output=None, target_rate=None, shift_hours=None)
```

- `df` - a DataFrame, or a path to a `.csv`, `.tsv` or `.parquet` file.
- `time` - the timestamp column. Found automatically when left out (a datetime column, a
  DatetimeIndex, or a column whose text reads as timestamps).
- `output` - the units-produced column. Found automatically when left out, and the choice is
  noted.
- `target_rate` - rated speed in **units per hour**. Without it the line is measured
  against its own typical rate.
- `shift_hours` - when the line is scheduled, in any of these forms: `(6, 22)`;
  `[(6, 14), (14, 22)]`; `{"A": (6, 14), "B": (14, 22), "C": (22, 6)}`;
  `"06:00-14:00,14:00-22:00"`; `"A=6-14,B=14-22"`; or `8` for 8-hour shifts round the clock
  from 06:00. Times are hours or `"HH:MM"`. A shift may run past midnight.

`ProductionReport`:

| attribute | meaning |
| --- | --- |
| `availability` | run time / scheduled time with a usable reading (`None` if nothing was measured) |
| `performance` | units made / what the reference rate would make in the run time, capped at 1 |
| `oee_partial` | `availability * performance`; quality is not included (`quality` is always `None`, `quality_note` says why) |
| `stoppages` | `list[Stoppage(start, end, minutes, kind)]`, `kind` is `"downtime"` or `"micro_stop"` |
| `events` | `list[Event(kind, start, end, minutes, severity, detail, units_lost, value)]` for `downtime`, `micro_stop`, `slow_running`, `rate_drift`, `spike`, `counter_reset` and `data_gap` |
| `by_shift` | DataFrame, one row per shift worked: date, shift, start, end, scheduled/measured/downtime/run minutes, units, rate per hour, availability, performance, oee_partial, stoppage and micro-stop counts |
| `lost_units` | estimated units lost against the reference rate in scheduled time |
| `lost_by_cause` | that estimate split into `downtime`, `micro_stops`, `slow_running` and `speed_loss` |
| `findings` | plain-language sentences, biggest loss first |
| `notes` | how it was measured, and anything about the data worth knowing |
| `intervals` | the regular per-interval table behind it all, with a `state` per interval |
| `summary()` | the text report above |
| `to_dict()` | everything except `intervals`, JSON-safe |

`units` is the total made in scheduled time. `typical_rate` and `reference_rate` are units per
interval, and `interval_minutes` gives the interval length (longer than the data's own
spacing when counts are low, see above). `lost_units` counts only
shortfalls, so running fast at one time
does not cancel a stoppage at another. The losses attached to single events can overlap (a
drift and a slow spell in the same hours), so use `lost_units` for the total.

`ProductionAnalyzer(...)` takes the same arguments plus the thresholds: `interval`
(for example `"1min"`), `counter` (`"auto"`, `True`, `False`), `near_zero=0.05`,
`slow_fraction=0.8`, `min_stop_minutes=5`, `min_slow_minutes=15`, `spike_factor=1.5`,
`drift_threshold=0.10`. Its `.analyze(df)` returns the same report.
`parse_shift_hours(spec)` shows how a `shift_hours` value is read.

## CLI

```
production-anomaly line.csv                                   # print the summary
production-anomaly line.csv --shift-hours 06:00-14:00,14:00-22:00 --target-rate 3600
production-anomaly line.csv --time ts --column units --json   # the full report as JSON
production-anomaly line.csv --output report.json --by-shift shifts.csv
```

`--column` is the units column (`output=` in Python). `--output` writes the JSON report and
`--by-shift` writes the per-shift table. The exit status is 0 on success and 2 when the input
cannot be read.

## License

MIT
