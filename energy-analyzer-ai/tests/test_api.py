"""The rest of the public surface: baseline_load, forecast, the class, files."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import energy_analyzer_ai as ea
from tests.conftest import meter


def test_everything_promised_is_exported():
    for name in (
        "analyze",
        "baseline_load",
        "forecast",
        "EnergyAnalyzer",
        "EnergyReport",
        "Anomaly",
        "Trend",
        "StepChange",
        "Tariff",
        "__version__",
    ):
        assert name in ea.__all__
        assert hasattr(ea, name)
    assert ea.__version__ == "0.1.0"


# ----------------------------------------------------------------- baseline_load
def test_baseline_load_is_the_overnight_floor():
    index = pd.date_range("2026-03-01", periods=24 * 14, freq="h")
    values = np.where((index.hour >= 9) & (index.hour < 18), 3.0, 0.25)
    series = pd.Series(values, index=index, name="kwh")

    assert ea.baseline_load(series) == pytest.approx(0.25, abs=0.02)


def test_baseline_load_accepts_the_night_window():
    series = meter(days=14, seed=1)
    assert ea.baseline_load(series, night=(1, 4)) == pytest.approx(
        ea.baseline_load(series), rel=0.2
    )


def test_baseline_load_on_an_empty_frame_is_zero():
    assert ea.baseline_load(pd.DataFrame({"time": [], "kwh": []})) == 0.0


def test_baseline_load_without_a_clock_uses_the_low_percentile():
    assert ea.baseline_load([1.0, 5.0, 6.0, 5.0, 7.0, 1.2]) == pytest.approx(1.0, abs=0.2)


def test_baseline_load_ignores_a_tariff_keyword():
    series = meter(days=7, seed=1)
    assert ea.baseline_load(series, tariff=0.3) == pytest.approx(ea.baseline_load(series))


# ---------------------------------------------------------------------- forecast
def test_forecast_continues_the_index():
    series = meter(days=21, seed=1)
    projection = ea.forecast(series, 24)

    assert isinstance(projection, pd.Series)
    assert len(projection) == 24
    assert projection.index[0] == series.index[-1] + pd.Timedelta(hours=1)
    assert projection.index.freq is not None or len(projection) == 24
    assert projection.notna().all()


def test_forecast_repeats_the_same_phase():
    """Hour 12 of the projection must look like hour 12 of the history."""
    series = meter(days=28, seed=1)
    projection = ea.forecast(series, 48)
    midday = projection[projection.index.hour == 12]
    night = projection[projection.index.hour == 3]

    assert float(midday.mean()) == pytest.approx(2.2, abs=0.2)
    assert float(night.mean()) == pytest.approx(0.4, abs=0.2)


def test_forecast_carries_the_trend():
    base = meter(days=28, seed=4)
    ramp = 1.0 + 0.06 * np.arange(base.size) / (24 * 7)
    rising = pd.Series(base.to_numpy() * ramp, index=base.index, name="kwh")
    projection = ea.forecast(rising, 24 * 7)

    assert float(projection.mean()) > float(rising.iloc[-24 * 7 :].mean())


def test_forecast_never_goes_negative_for_a_positive_meter():
    series = meter(days=28, seed=2)
    assert (ea.forecast(series, 24 * 14) >= 0).all()


def test_forecast_without_a_clock_uses_positions():
    projection = ea.forecast([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0], 3)
    assert list(projection.index) == [8, 9, 10]


def test_forecast_of_nothing_is_empty():
    assert len(ea.forecast(pd.DataFrame({"time": [], "kwh": []}), 5)) == 0


def test_forecast_refuses_a_silly_period_count():
    series = meter(days=3, seed=1)
    with pytest.raises(ValueError, match="at least 1"):
        ea.forecast(series, 0)
    with pytest.raises(ValueError, match="whole number"):
        ea.forecast(series, "lots")


# ----------------------------------------------------------------- the class
def test_sensitivity_is_the_knob():
    base = meter(days=21, seed=1)
    noise = ea.analyze(base).scale
    series = base.copy()
    series.iloc[100] = base.iloc[100] + 8.0 * noise

    strict = ea.EnergyAnalyzer(sensitivity=20.0).analyze(series)
    default = ea.EnergyAnalyzer().analyze(series)
    loose = ea.EnergyAnalyzer(sensitivity=3.0).analyze(series)

    assert strict.n_anomalies == 0
    assert default.n_anomalies == 1
    assert default.anomalies[0].when == base.index[100]
    assert loose.n_anomalies >= default.n_anomalies
    assert loose.sensitivity == 3.0


def test_the_default_threshold_keeps_ordinary_noise_out():
    """Four sigmas, not three: a meter series is long enough that three flags noise."""
    assert ea.EnergyAnalyzer().sensitivity == 4.0
    for seed in range(6):
        assert ea.analyze(meter(days=21, seed=seed)).n_anomalies == 0


def test_min_effect_keeps_small_wobbles_out():
    series = meter(days=21, seed=1, noise=0.01).copy()
    series.iloc[100] += 0.12  # far outside the noise, but a quarter of the level

    assert ea.EnergyAnalyzer(min_effect=0.5).analyze(series).n_anomalies == 0
    assert ea.EnergyAnalyzer(min_effect=0.0).analyze(series).n_anomalies == 1


def test_bad_knobs_are_refused_clearly():
    with pytest.raises(ValueError, match="sensitivity must be"):
        ea.EnergyAnalyzer(sensitivity=0)
    with pytest.raises(ValueError, match="min_effect must be"):
        ea.EnergyAnalyzer(min_effect=-1)


def test_a_fixed_baseline_number_is_used_as_given():
    index = pd.date_range("2026-03-01", periods=24 * 7, freq="h")
    series = pd.Series(np.full(index.size, 1.0), index=index, name="kwh")
    series.iloc[40] = 4.0
    report = ea.analyze(series, baseline=1.0)

    assert report.baseline_method.startswith("fixed")
    assert report.by_period["expected"].eq(1.0).all()
    assert report.n_anomalies == 1
    assert any("given as a fixed" in note for note in report.notes)


def test_a_reference_period_can_define_normal():
    normal = meter(days=21, seed=1, start="2026-02-01")
    later = meter(days=7, seed=2, start="2026-03-01").copy()
    later.iloc[60] = 9.0
    report = ea.analyze(later, baseline=normal)

    assert "reference period" in report.baseline_method
    assert any("reference period" in note for note in report.notes)
    assert report.n_anomalies >= 1
    assert max(report.spikes, key=lambda a: a.excess).when == later.index[60]


def test_an_empty_reference_period_is_refused_clearly():
    with pytest.raises(ValueError, match="no readings to learn from"):
        ea.analyze(meter(days=3, seed=1), baseline=pd.DataFrame({"time": [], "kwh": []}))


def test_a_nonsense_fixed_baseline_is_refused_clearly():
    with pytest.raises(ValueError, match="finite number"):
        ea.analyze(meter(days=3, seed=1), baseline=float("nan"))


# ------------------------------------------------------------------- csv input
def test_a_csv_path_works(tmp_path, frame):
    path = tmp_path / "meter.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    report = ea.analyze(str(path), tariff=0.28)

    assert report.n_anomalies == 1
    assert report.total_cost is not None
    assert ea.baseline_load(str(path)) > 0


def test_a_missing_file_is_refused_clearly():
    with pytest.raises(FileNotFoundError):
        ea.analyze("no-such-meter.csv")


def test_an_unknown_file_type_is_refused_clearly(tmp_path):
    path = tmp_path / "meter.xlsx"
    path.write_text("not really a spreadsheet", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        ea.analyze(str(path))


# ------------------------------------------------------ the small value objects
def test_anomaly_reads_as_a_sentence(readings):
    item = ea.analyze(readings, tariff=0.28).top(1)[0]
    text = str(item)

    assert "against an expected" in text
    assert "sigmas" in text
    assert "extra" in text
    payload = item.to_dict()
    assert json.dumps(payload)
    assert payload["kind"] == "spike"
    assert payload["cost"] == pytest.approx(item.cost)


def test_trend_and_step_read_as_sentences():
    series = meter(days=28, seed=9).copy()
    series.iloc[24 * 14 :] *= 1.45
    report = ea.analyze(series)

    assert json.dumps(report.trend.to_dict())
    step = report.steps[0]
    assert json.dumps(step.to_dict())
    assert "stayed there" in str(step)
    assert step.to_dict()["when"].startswith("2026-03-15")


def test_tariff_reads_as_a_sentence():
    assert "no tariff given" in str(ea.Tariff())
    flat = ea.analyze(meter(days=7, seed=1), tariff=0.3).tariff
    assert "flat rate" in str(flat)
    tou = ea.analyze(meter(days=7, seed=1), tariff={h: 0.1 * (h + 1) for h in range(24)}).tariff
    assert "time-of-use" in str(tou)
    assert json.dumps(tou.to_dict())


def test_units_are_read_off_the_column_name():
    index = pd.date_range("2026-03-01", periods=48, freq="h")
    for name, unit in (("kwh", "kWh"), ("gas_m3", "m3"), ("demand_kw", "kW"), ("x", "units")):
        series = pd.Series(np.linspace(1, 2, 48), index=index, name=name)
        assert ea.analyze(series).unit == unit


def test_a_baseline_that_flags_everything_says_so():
    """A report where almost every period is an anomaly tells the reader nothing."""
    series = meter(days=14, seed=3)  # quiet nights, busy days
    report = ea.analyze(series, baseline=1.0)  # a flat 1.0 describes neither

    assert report.n_anomalies > 0.9 * report.n_periods
    assert any("does not describe this meter" in item for item in report.warnings)
    assert any("baseline=None" in item for item in report.warnings)
    assert "does not describe this meter" in report.summary()


def test_a_reference_period_that_flags_everything_says_so():
    normal = meter(days=21, seed=1, start="2026-02-01")
    shifted = meter(days=7, seed=2, start="2026-03-01") + 0.9  # a week run higher
    report = ea.analyze(shifted, baseline=normal)

    assert report.n_anomalies > 0.9 * report.n_periods
    assert any("does not describe this meter" in item for item in report.warnings)


def test_a_baseline_that_fits_stays_quiet():
    index = pd.date_range("2026-03-01", periods=24 * 7, freq="h")
    series = pd.Series(np.full(index.size, 1.0), index=index, name="kwh")
    series.iloc[40] = 4.0
    report = ea.analyze(series, baseline=1.0)

    assert report.n_anomalies == 1
    assert not any("does not describe this meter" in item for item in report.warnings)
    # and the default path is untouched
    assert not any(
        "does not describe this meter" in item
        for item in ea.analyze(meter(days=14, seed=3)).warnings
    )
