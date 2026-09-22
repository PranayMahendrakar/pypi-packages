# timeseries-anomaly

Find the points in a time series or IoT signal that do not belong, with one call and no model training.

## Install

```bash
pip install timeseries-anomaly
```

Reading `.parquet` files needs `pip install timeseries-anomaly[parquet]`.

## Quickstart

```python
import timeseries_anomaly

readings = [10, 11, 10, 12, 11, 10, 11, 60, 10, 11, 12, 10, 11, 10, 12]
result = timeseries_anomaly.detect(readings)
print(result.summary())
print(result.anomalies)
```

```text
timeseries-anomaly: 1 anomaly in 15 points (6.7%)
  series    : value
  method    : zscore (asked for 'auto')
  baseline  : level 11, robust sigma 1.4826 (mad)
  threshold : 3 robust sigmas from expected
  flagged   :
    index 7: value 60, expected 11, 33.05 sigmas
  warnings:
    - method 'auto' chose 'zscore' because the series has only 15 points
[7]
```

It also takes a `pandas.Series` (with or without a `DatetimeIndex`), a `DataFrame`,
a numpy array, or a path to a `.csv` / `.parquet` file:

```python
result = timeseries_anomaly.detect("sensor_log.csv", value="temperature", time="recorded_at")
result.to_frame().to_csv("scored.csv", index=False)
```

## What it does

- Compares every point against what it should have been, and reports the gap in
  **robust sigmas** - median and MAD based, so a handful of extreme points cannot
  hide the rest.
- Picks the method for you (`method="auto"`), or runs every applicable method and
  flags only what a strict majority agree on (`method="all"`).
- Handles seasonal signals: give it `seasonality=24`, or let it infer the cycle
  from a `DatetimeIndex` or from the shape of the series itself.
- Never raises on awkward data. A constant series, an empty series, a single
  point, missing values, unsorted or duplicated timestamps are all handled, and
  whatever had to be adjusted is recorded in `result.warnings`.
- Never invents anomalies out of rounding error. When a series matches its
  baseline exactly - a clean cycle, a steady ramp, a flat line - the spread
  estimate steps down from MAD to standard deviation and then to reporting
  nothing at all, instead of dividing by almost zero.
- Never reports a trend as a fault. A signal that is simply going somewhere - an
  energy meter, a packet counter, an odometer - has its slope taken out before it
  is judged, so every method measures the wobble around the climb rather than the
  climb itself, and a spike on a steep slope is reported without dragging its
  neighbours in with it.
- Covers the first and last reading like any other. Every point is compared
  against its neighbours and never against itself, so an anomaly on the newest
  sample - usually the one that matters - is reported under the default settings.
- Learns a baseline once and reuses it for later batches, for streaming use.
- Deterministic: the same input always gives the same answer, with no sampling
  anywhere and no peeking at future points when scoring a stream.

### Methods

| `method`   | Baseline each point is compared against                                     |
|------------|------------------------------------------------------------------------------|
| `zscore`   | the median of the whole series, spread from the MAD (modified z-score)       |
| `iqr`      | the same flat median, spread from the interquartile range                    |
| `rolling`  | the median of the neighbours either side of it, window `max(7, n // 20)`     |
| `seasonal` | a median trend plus the median of the other points sharing its phase         |
| `ewma`     | the exponential moving average of the points before it                       |
| `auto`     | `rolling` for 30 or more points, otherwise `zscore`                          |
| `all`      | every applicable method above, voting; anomaly when a strict majority agree   |

Every method leaves the point it is explaining out of the baseline explaining it,
and takes the series' own slope out first. So a steadily rising signal does not
report itself as anomalous, an anomaly on the very first or very last reading is
reported like any other, and the spread each point is measured against is the
noise in the data rather than the shape of the fit.

`all` is the conservative choice: it only reports what more than half of the
applicable methods agree on. On a strongly trending series the two flat-baseline
methods (`zscore` and `iqr`) have nothing to say and can outvote the two that do,
in which case nothing is reported and `result.warnings` says that something was
flagged and overruled. Use `rolling` or `ewma` directly on that kind of signal.

`seasonal` infers its cycle from a `DatetimeIndex`, or from the shape of the
series, in `O(n log n)`; passing `seasonality=` explicitly is still a little
faster and removes any doubt about which cycle was used.

## API

```python
detect(data, *, value=None, time=None, method="auto", sensitivity=3.0,
       seasonality=None) -> AnomalyResult
```

- **data** - `pandas.Series`, `DataFrame`, list/array of numbers, or a `.csv` / `.parquet` path.
- **value**, **time** - column names, when `data` is a table. Guessed when omitted.
- **method** - one of the table above.
- **sensitivity** - the threshold in robust sigmas. Higher means fewer anomalies.
- **seasonality** - points per cycle for `seasonal`; inferred when omitted.

`AnomalyResult`:

| Member | What it gives you |
|--------|-------------------|
| `.anomalies` | `list[int]` positional indices of the flagged points |
| `.mask` | numpy bool array aligned to the input |
| `.scores` | numpy float array, robust sigmas from expected |
| `.expected` | numpy float array, the baseline each point was compared against |
| `.n_anomalies` / `.rate` | count, and the share of all points flagged |
| `.method_used` | the method that actually ran, after `auto` and any fallback |
| `.to_frame()` | `DataFrame(time, value, expected, score, is_anomaly)`, keeping your index |
| `.summary()` | the plain-text report above |
| `.to_dict()` | JSON-safe dict of everything, including `warnings` |
| `.plot_data()` | time-ordered lists ready for any plotting library |
| `.top(n)` | the worst points first, flagged or not |
| `.warnings` | what had to be adjusted: fallbacks, sorting, missing values |

`.mask`, `.scores`, `.expected` and `.values` all line up with the input exactly as
you passed it, even when the series had to be sorted by timestamp first.

`Detector` takes the same keyword options and keeps the baseline between batches:

```python
from timeseries_anomaly import Detector

detector = Detector(method="rolling", sensitivity=4.0).fit(history)
result = detector.score(new_batch)      # same baseline, same threshold
print(detector.level_, detector.scale_, detector.method_)
```

`Detector.detect(data)` is the one-shot form, identical to `detect()`.

## CLI

```bash
timeseries-anomaly readings.csv
timeseries-anomaly readings.csv --value temperature --time recorded_at
timeseries-anomaly readings.csv --method seasonal --seasonality 24 --sensitivity 4
timeseries-anomaly readings.csv --json > anomalies.json
timeseries-anomaly readings.csv --output scored.csv
```

`--help` lists every option. `--json` prints `to_dict()`, `--output` writes the
scored table as `.csv` or `.parquet`.

## License

MIT
