# sensor-anomaly

Spot abnormal behaviour across many industrial sensor channels at once, including
the faults that are only visible *between* channels and the dead sensors that no
outlier test will ever flag.

## Install

```bash
pip install sensor-anomaly
```

## Quickstart

```python
import numpy as np, pandas as pd, sensor_anomaly

rng = np.random.default_rng(0)
df = pd.DataFrame({"temp": rng.normal(70, 1, 500), "flow": rng.normal(12, 0.5, 500)})
df.loc[300:340, "flow"] = 0.0          # this sensor died and now reads zero
report = sensor_anomaly.detect(df)
print(report.summary())
```

That is the whole thing. `report.summary()` prints which channels misbehaved, the
events worth looking at, and a plain-words note for every fault code it used.

## What it checks

Three passes over the same table, all landing in one report:

- **Each channel on its own** - a robust z-score of the rolling residual. Each
  reading is judged against its own recent level, so a slow drift or a daily cycle
  does not bury the result in false alarms, and a handful of genuine spikes cannot
  inflate the threshold the way a standard deviation would.
- **All channels together** - an `IsolationForest` over the standardized channels
  and over each channel's residual against what the others predict for it. This is
  the pass that catches a fault where *no single reading is odd*: flow falling
  while current climbs, both still inside their normal ranges. That broken
  correlation is invisible per channel. The cut is set from how far a reading
  strays from what its neighbours predict, corrected for how many readings the
  table contains, so `sensitivity` means "the chance a whole clean table trips at
  all" rather than a chance per row. Redundant sensors that all track one process
  variable - the normal case in a plant - stay quiet.
- **Sensor faults that are not outliers at all** - a dead sensor is not unusual,
  it is boringly consistent:
  - `flatline` - the reading stopped changing for a long run
  - `stuck-at-zero` - it sat at exactly zero
  - `railed-at-min` / `railed-at-max` - pinned against its measurement limit
  - `missing-burst` - a run of consecutive samples went missing
  - `step-change` - the level jumped and stayed there (a recalibration or a swap)
  - `all-missing` / `no-numeric-data` - a failed sensor, reported rather than crashed on

It also handles the awkward cases without argument: channels full of gaps, an
all-NaN channel, a channel that never moves, a table with a header and no rows,
and a single-channel table (the cross-channel pass is skipped and the report says
so). Your DataFrame is never modified.

A clean table says so. At `sensitivity=3.0` a 3-sigma test flags about 0.27% of
perfectly normal readings by definition; the report counts those as the noise
floor, says so in its notes, and leaves `report.anomalous` `False`.

`IsolationForest` cannot read a missing value, so each channel is imputed with its
own median before fitting rather than silently dropping every row that has a gap.
The report says how many readings were filled and in which channels.

## API

### `detect(df, *, time=None, channels=None, sensitivity=3.0, contamination="auto", random_state=0) -> SensorReport`

| argument | meaning |
| --- | --- |
| `df` | wide DataFrame, one column per channel, or a path to `.csv` / `.tsv` / `.parquet` |
| `time` | timestamp column. Left as `None`, an obvious one is detected, reported, and never scored as a channel |
| `channels` | which columns are sensors. Left as `None`, every numeric column is used and anything skipped is named in the notes |
| `sensitivity` | per-channel threshold in robust sigmas. Higher flags less. Try 4-5 on noisy plant data |
| `contamination` | `"auto"`, or the share of rows you expect to be anomalous |
| `random_state` | seed, so the same table always gives the same answer |

### `SensorReport`

| member | what it gives you |
| --- | --- |
| `.channels` | `dict[name -> ChannelResult]` |
| `.joint` | row positions flagged by the cross-channel model |
| `.events` | merged `Event` records, worst first |
| `.worst_channels` | channel names, worst first: failed, then faulty, then by rate |
| `.anomalous` | one boolean to gate an alarm on: `False` when every flag is inside the false-positive rate `sensitivity` itself predicts |
| `.quiet` | the other side of the same judgement: `True` when the flags are noise |
| `.n_events`, `.n_rows`, `.method`, `.failed_channels` | at a glance |
| `.summary()` | the human report, plain ASCII, safe to pipe anywhere |
| `.to_dict()` | JSON-safe dict of everything |
| `.to_frame()` | one row per channel: status, anomalies, rate, peak score, faults |
| `.events_frame()` | one row per event |
| `.notes` | what was skipped, assumed, imputed or fallen back to |

`ChannelResult` holds `.anomalies`, `.rate`, `.score_max`, `.faults`, `.status`,
`.n_valid`, `.n_missing` and `.notes`. `Event` holds `.start`, `.end`, `.channels`,
`.kind`, `.severity`, `.level` and `.describe()`.

```python
report.channels["flow"].faults          # ['flatline', 'stuck-at-zero', 'step-change']
report.worst_channels[0]                # 'flow'
report.channels_with("flatline")        # ['flow']
[e.describe() for e in report.events]   # one readable line each
report.events_frame()                   # a DataFrame of the same thing
```

Same settings across many tables:

```python
detector = sensor_anomaly.Detector(sensitivity=4.0, time="timestamp")
report = detector.detect(batch)
```

## CLI

```bash
sensor-anomaly plant.csv
sensor-anomaly plant.csv --time timestamp --sensitivity 4
sensor-anomaly plant.csv --channels temp,flow,vibration
sensor-anomaly plant.csv --json > findings.json
sensor-anomaly plant.csv --output channels.csv
```

`--output` writes the per-channel table as `.csv` / `.tsv` / `.parquet`, or the
whole report when the path ends in `.json`. Run `sensor-anomaly --help` for the
rest.

## License

MIT
