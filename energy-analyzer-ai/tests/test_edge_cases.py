"""The awkward inputs real meters produce."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import energy_analyzer_ai as ea
from tests.conftest import meter


# --------------------------------------------------------- cumulative meters
def test_a_cumulative_meter_is_detected_and_differenced():
    """The register only counts up; the analysis must use what was actually used."""
    used = meter(days=21, seed=2)
    register = pd.Series(
        np.cumsum(used.to_numpy()) + 50_000.0, index=used.index, name="meter_kwh"
    )
    report = ea.analyze(register)

    assert report.cumulative_meter is True
    assert report.total == pytest.approx(float(used.iloc[1:].sum()), rel=0.02)
    assert report.total < 1_000  # not the 50,000 on the dial
    assert any("cumulative meter" in note for note in report.notes)
    assert any("cumulative meter" in line for line in report.findings)


def test_a_cumulative_total_is_never_reported_as_a_spike():
    used = meter(days=21, seed=2)
    register = pd.Series(
        np.cumsum(used.to_numpy()) + 50_000.0, index=used.index, name="meter_kwh"
    )
    report = ea.analyze(register)

    assert report.n_anomalies == 0
    assert all(item.observed < 100 for item in report.anomalies)


def test_a_spike_inside_a_cumulative_meter_is_still_found():
    used = meter(days=21, seed=2)
    register = pd.Series(
        np.cumsum(used.to_numpy()) + 50_000.0, index=used.index, name="meter_kwh"
    )
    register.iloc[300:] += 6.0  # one bad hour, then the register carries on
    report = ea.analyze(register)

    assert report.cumulative_meter is True
    assert report.n_anomalies == 1
    assert report.anomalies[0].when == used.index[300]
    assert report.anomalies[0].excess == pytest.approx(6.0, abs=0.3)


def test_cumulative_detection_can_be_forced_off():
    used = meter(days=21, seed=2)
    register = pd.Series(np.cumsum(used.to_numpy()), index=used.index, name="meter_kwh")
    report = ea.EnergyAnalyzer(cumulative=False).analyze(register)

    assert report.cumulative_meter is False
    assert report.total > 10_000


def test_a_meter_rollover_is_left_as_a_gap():
    used = meter(days=14, seed=6)
    register = pd.Series(np.cumsum(used.to_numpy()) + 100.0, index=used.index, name="kwh")
    register.iloc[200:] -= 250.0  # the dial wrapped round to zero
    report = ea.analyze(register)

    assert report.cumulative_meter is True
    assert report.meter_resets == 1
    assert any("reset or rollover" in warning for warning in report.warnings)
    assert report.n_gaps >= 1


def test_rising_consumption_is_not_mistaken_for_a_cumulative_meter():
    """Growth is not a register: only a series that never falls is cumulative."""
    report = ea.analyze(meter(days=21, seed=7))
    assert report.cumulative_meter is False


# ------------------------------------------------------------- gaps and grids
def test_missing_readings_become_reported_gaps_not_filled_values():
    series = meter(days=14, seed=3)
    kept = series.drop(series.index[100:130])
    report = ea.analyze(kept)

    assert report.n_gaps == 30
    assert report.longest_gap == 30
    assert report.n_periods == 24 * 14
    assert int(report.by_period["observed"].isna().sum()) == 30
    assert int(report.by_period["is_gap"].sum()) == 30
    assert any("gaps" in line for line in report.findings)


def test_irregular_readings_are_resampled_onto_a_regular_grid():
    stamps = pd.to_datetime(
        ["2026-03-01 00:03", "2026-03-01 00:58", "2026-03-01 02:10", "2026-03-01 03:01"]
        + ["2026-03-01 {:02d}:{:02d}".format(h, (h * 7) % 60) for h in range(4, 24)]
    )
    series = pd.Series(np.linspace(1.0, 2.0, stamps.size), index=stamps, name="kwh")
    report = ea.analyze(series, granularity="hourly")

    assert report.granularity == "1h"
    assert isinstance(report.by_period.index, pd.DatetimeIndex)
    assert report.n_periods == 24
    assert report.n_gaps == 1  # 01:00 had no reading at all
    assert report.n_readings == 24


def test_duplicate_timestamps_are_added_together():
    stamps = pd.to_datetime(["2026-03-01 00:00"] * 2 + ["2026-03-01 01:00"])
    series = pd.Series([1.0, 2.0, 5.0], index=stamps, name="kwh")
    report = ea.analyze(series, granularity="hourly")

    assert report.total == pytest.approx(8.0)
    assert any("duplicate timestamp" in note for note in report.notes)


def test_out_of_order_timestamps_are_sorted():
    series = meter(days=7, seed=8)
    shuffled = series.iloc[np.random.default_rng(0).permutation(series.size)]
    report = ea.analyze(shuffled)

    assert report.by_period.index.is_monotonic_increasing
    assert any("out of order" in note for note in report.notes)


def test_granularity_can_be_forced_coarser():
    report = ea.analyze(meter(days=28, seed=5), granularity="daily")
    assert report.granularity == "1D"
    assert report.n_periods == 28
    assert report.baseline_method == "day of week"


def test_a_grid_far_finer_than_the_readings_warns():
    daily = meter(days=40, seed=5).resample("1D").sum()
    daily.name = "kwh"
    report = ea.analyze(daily, granularity="hourly")
    assert any("finer than the readings" in warning for warning in report.warnings)


def test_an_impossible_grid_is_refused_clearly():
    index = pd.DatetimeIndex(["1990-01-01", "2000-01-01", "2010-01-01", "2026-01-01"])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="kwh")
    with pytest.raises(ValueError, match="coarser granularity"):
        ea.analyze(series, granularity="1min")


def test_a_nonsense_granularity_is_refused_clearly():
    with pytest.raises(ValueError, match="not a fixed period"):
        ea.analyze(meter(days=3), granularity="banana")


# ------------------------------------------------------------------ negatives
def test_negative_readings_are_flagged_not_crashed():
    series = meter(days=14, seed=4).copy()
    series.iloc[50] = -2.0
    series.iloc[51] = -0.5
    report = ea.analyze(series)

    assert report.negative_readings == 2
    assert any("negative consumption" in warning for warning in report.warnings)
    assert any("negative consumption" in line for line in report.findings)
    assert int(report.by_period["is_negative"].sum()) == 2
    assert report.to_dict()["negative_readings"] == 2


def test_negative_readings_do_not_shape_the_baseline():
    clean = meter(days=14, seed=4)
    dirty = clean.copy()
    dirty.iloc[50] = -500.0
    assert ea.analyze(dirty).baseline_load == pytest.approx(
        ea.analyze(clean).baseline_load, rel=0.05
    )


def test_an_all_negative_series_still_reports():
    index = pd.date_range("2026-03-01", periods=48, freq="h")
    report = ea.analyze(pd.Series(np.full(48, -1.0), index=index, name="export_kwh"))
    assert report.negative_readings == 48
    assert report.baseline_load == 0.0
    assert isinstance(report.summary(), str)


# --------------------------------------------------------- not enough history
def test_a_series_shorter_than_one_cycle_falls_back_with_a_warning():
    report = ea.analyze(meter(days=1, seed=5))

    assert report.baseline_method == "flat median"
    assert any("not enough history" in warning for warning in report.warnings)
    assert report.n_anomalies == 0  # the daily shape itself is not an anomaly


def test_a_big_spike_is_still_found_in_a_short_series():
    series = meter(days=1, seed=5).copy()
    series.iloc[3] = 12.0
    report = ea.analyze(series)

    assert report.baseline_method == "flat median"
    assert report.n_anomalies == 1
    assert report.anomalies[0].observed == pytest.approx(12.0)


def test_a_single_row_is_handled():
    frame = pd.DataFrame({"time": pd.to_datetime(["2026-03-01"]), "kwh": [3.5]})
    report = ea.analyze(frame)

    assert report.n_periods == 1
    assert report.total == pytest.approx(3.5)
    assert report.n_anomalies == 0
    assert isinstance(report.summary(), str)


def test_an_empty_dataframe_is_handled():
    report = ea.analyze(pd.DataFrame({"time": [], "kwh": []}))

    assert report.n_periods == 0
    assert report.total == 0.0
    assert report.anomalies == []
    assert report.span is None
    assert "nothing to analyse" in " ".join(report.findings)
    assert isinstance(report.by_period, pd.DataFrame)
    assert report.to_dict()["periods"] == 0
    assert isinstance(report.summary(), str)


def test_a_frame_with_no_columns_is_handled():
    report = ea.analyze(pd.DataFrame())
    assert report.n_periods == 0
    assert any("no columns" in warning for warning in report.warnings)


def test_an_all_nan_column_is_handled():
    index = pd.date_range("2026-03-01", periods=48, freq="h")
    report = ea.analyze(pd.Series(np.full(48, np.nan), index=index, name="kwh"))

    assert report.total == 0.0
    assert report.n_gaps == 48
    assert report.anomalies == []
    assert isinstance(report.summary(), str)


def test_an_unvarying_series_cannot_flag_anything():
    index = pd.date_range("2026-03-01", periods=24 * 10, freq="h")
    report = ea.analyze(pd.Series(np.full(index.size, 2.0), index=index, name="kwh"))

    assert report.scale is None
    assert report.n_anomalies == 0
    assert "nothing can stand out" in report.summary()


def test_a_plain_list_of_numbers_works():
    report = ea.analyze([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])

    assert report.has_time is False
    assert report.n_periods == 6
    assert report.granularity == "reading"
    assert any("no timestamp column" in warning for warning in report.warnings)
    assert isinstance(report.summary(), str)


# ----------------------------------------------------------------- time zones
@pytest.mark.parametrize("tz", ["Europe/Berlin", "UTC", "America/New_York"])
def test_timezone_is_kept_through_the_whole_report(tz):
    series = meter(days=21, seed=2, tz=tz).copy()
    series.iloc[300] = 9.0
    report = ea.analyze(series, tariff=0.28)

    assert str(report.by_period.index.tz) == tz
    assert str(report.anomalies[0].when.tz) == tz
    assert str(ea.forecast(series, 4).index.tz) == tz
    offset = report.by_period.index[0].isoformat()[-6:]
    assert report.to_dict()["window"]["start"].endswith(offset)


def test_mixed_offsets_are_normalised_to_utc():
    frame = pd.DataFrame(
        {
            "time": ["2026-03-01T00:00:00+01:00", "2026-03-01T02:00:00+05:00"],
            "kwh": [1.0, 2.0],
        }
    )
    report = ea.analyze(frame)
    assert report.n_periods >= 1
    assert isinstance(report.summary(), str)


# --------------------------------------------------------------------- tariffs
def test_no_tariff_leaves_every_cost_as_none(readings):
    report = ea.analyze(readings)

    assert report.total_cost is None
    assert report.excess_cost is None
    assert all(item.cost is None for item in report.anomalies)
    assert [c for c in report.by_period.columns if "cost" in c or c == "rate"] == []
    payload = report.to_dict()
    assert payload["total_cost"] is None and payload["excess_cost"] is None
    assert payload["anomalies"][0]["cost"] is None
    assert "left as None rather than zero" in " ".join(report.findings)


def test_a_flat_tariff_prices_everything(readings):
    report = ea.analyze(readings, tariff=0.25)
    assert report.total_cost == pytest.approx(report.total * 0.25)
    assert report.by_period["rate"].nunique() == 1
    assert report.tariff.kind == "flat"


def test_time_of_use_rates_follow_the_clock(readings):
    rates = {hour: (0.12 if hour < 7 else 0.31) for hour in range(24)}
    report = ea.analyze(readings, tariff=rates)

    assert report.tariff.kind == "time-of-use"
    assert sorted(report.by_period["rate"].unique()) == [0.12, 0.31]
    night = report.by_period[report.by_period.index.hour < 7]
    assert set(night["rate"].unique()) == {0.12}
    assert report.total_cost < report.total * 0.31


def test_a_partial_tariff_warns_and_uses_the_average(readings):
    report = ea.analyze(readings, tariff={0: 0.1, 12: 0.4})
    assert any("does not price" in warning for warning in report.warnings)
    assert report.tariff.mean_rate == pytest.approx(0.25)


def test_time_of_use_on_a_daily_grid_falls_back_to_the_average():
    daily = meter(days=40, seed=5).resample("1D").sum()
    daily.name = "kwh"
    report = ea.analyze(daily, tariff={h: 0.2 + 0.01 * h for h in range(24)})
    assert any("average rate" in warning for warning in report.warnings)


def test_a_bad_tariff_is_refused_clearly(readings):
    with pytest.raises(ValueError, match="outside 0 to 23"):
        ea.analyze(readings, tariff={99: 0.2})
    with pytest.raises(TypeError, match="tariff must be"):
        ea.analyze(readings, tariff="cheap")
    with pytest.raises(ValueError, match="empty mapping"):
        ea.analyze(readings, tariff={})


# ------------------------------------------------------------------ messy tables
def test_duplicate_column_names_are_refused_by_name():
    frame = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["kwh", "kwh", "time"])
    with pytest.raises(ValueError, match="duplicate column names"):
        ea.analyze(frame)


def test_mixed_dtypes_are_handled():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-03-01", periods=8, freq="h"),
            "site": ["north"] * 8,
            "ok": [True] * 8,
            "kwh": ["1.0", "2.0", "oops", "4.0", "1.5", "6.0", "2.0", "8.0"],
        }
    )
    report = ea.analyze(frame, value="kwh")

    assert report.label == "kwh"
    assert report.n_gaps == 1
    assert any("not numbers" in note for note in report.notes)
    assert isinstance(report.summary(), str)


def test_unicode_column_names_and_values_survive():
    frame = pd.DataFrame(
        {
            "Zeitstempel": pd.date_range("2026-03-01", periods=24 * 8, freq="h"),
            "Verbrauch_kWh": np.tile([0.4, 0.9, 2.2], 64),
            "standort": ["München"] * (24 * 8),
        }
    )
    report = ea.analyze(frame, value="Verbrauch_kWh", time="Zeitstempel", tariff=0.3)

    assert report.label == "Verbrauch_kWh"
    assert report.unit == "kWh"
    assert report.to_dict()["meter"] == "Verbrauch_kWh"
    assert isinstance(report.summary(), str)


def test_a_non_ascii_meter_name_does_not_break_the_summary():
    index = pd.date_range("2026-03-01", periods=24 * 8, freq="h")
    series = pd.Series(np.tile([0.4, 0.9, 2.2], 64), index=index, name="замер kWh")
    text = ea.analyze(series).summary()

    assert "замер" in text
    text.encode("utf-8")  # must be encodable for any console we reconfigure


def test_a_missing_named_column_says_what_is_there(frame):
    with pytest.raises(ValueError, match="not in the table"):
        ea.analyze(frame, value="nope")
    with pytest.raises(ValueError, match="not in the table"):
        ea.analyze(frame, time="nope")


def test_a_table_with_nothing_numeric_is_refused_clearly():
    frame = pd.DataFrame({"site": ["a", "b"], "note": ["x", "y"]})
    with pytest.raises(ValueError, match="no numeric column"):
        ea.analyze(frame)


def test_unsupported_input_is_refused_clearly():
    with pytest.raises(TypeError, match="unsupported data type"):
        ea.analyze(object())
    with pytest.raises(ValueError, match="data is None"):
        ea.analyze(None)


# ------------------------------------------------- input errors are ours, not pandas'
def test_a_dict_of_single_values_is_refused_in_our_own_words():
    """pandas says "If using all scalar values..."; the caller never sees that."""
    with pytest.raises(ValueError, match="columns of readings, not single values"):
        ea.analyze({"a": 1})
    with pytest.raises(ValueError, match=r"\{'kwh': \[1\.0, 2\.0, \.\.\.\]\}"):
        ea.analyze({"time": "2026-03-01", "kwh": 1.0})

    try:
        ea.analyze({"a": 1})
    except ValueError as error:
        assert "scalar" not in str(error)
        assert "index" not in str(error)


def test_a_dict_of_uneven_columns_is_refused_in_our_own_words():
    with pytest.raises(ValueError, match="could not be read as columns of readings"):
        ea.analyze({"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]})

    try:
        ea.analyze({"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]})
    except ValueError as error:
        assert "a has 2" in str(error) and "b has 3" in str(error)


def test_a_good_dict_still_loads_including_a_broadcast_label():
    assert ea.analyze({"kwh": [1.0, 2.0, 3.0, 1.0]}).n_periods == 4
    report = ea.analyze(
        {
            "time": pd.date_range("2026-03-01", periods=24 * 8, freq="h"),
            "kwh": np.tile([0.4, 0.9, 2.2], 64),
            "site": "north",  # one label broadcast down the column is fine
        }
    )
    assert report.label == "kwh"
    assert report.n_periods == 24 * 8
