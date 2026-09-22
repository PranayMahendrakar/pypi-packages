"""One test per defect an independent review found, so none of them can come back.

Each test names the shape of data that broke, not the internal that fixed it, so
the suite still means something if the implementation is rewritten underneath it.
"""
import time
import warnings

import numpy as np
import pytest

from timeseries_anomaly import METHODS, Detector, detect
from timeseries_anomaly._methods import MIN_SEASONAL_CORRELATION, infer_period_from_values

SINGLE = [name for name in METHODS if name not in ("auto", "all")]


def noisy_ramp(seed: int, n: int = 200, slope: float = 0.7, sigma: float = 0.1) -> np.ndarray:
    """An energy meter: a steady climb with a little wobble on it."""
    generator = np.random.default_rng(seed)
    return np.arange(n, dtype=float) * slope + 5.0 + generator.normal(0.0, sigma, n)


def clean_noise(seed: int, n: int) -> np.ndarray:
    """Pure Gaussian noise: every point belongs, none of them is an anomaly."""
    return 100.0 + np.random.default_rng(seed).normal(0.0, 2.0, n)


def seasonal(seed: int, n: int, period: int, sigma: float = 0.3) -> np.ndarray:
    """A clean daily cycle with noise, and no anomaly anywhere in it."""
    shape = 50.0 + 8.0 * np.sin(2.0 * np.pi * np.arange(n) / period)
    return shape + np.random.default_rng(seed).normal(0.0, sigma, n)


# --------------------------------------------------------------------------
# a steadily trending series has no anomalies in it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", list(METHODS))
def test_a_perfect_ramp_is_not_an_anomaly_under_any_method(method):
    """An odometer, a packet counter, an energy meter: climbing is not misbehaving."""
    result = detect([float(i) for i in range(80)], method=method)
    assert result.n_anomalies == 0, result.anomalies
    assert np.all(np.isfinite(result.scores))


@pytest.mark.parametrize("method", list(METHODS))
def test_a_long_clean_ramp_is_not_an_anomaly_under_any_method(method):
    result = detect(np.arange(500.0), method=method)
    assert result.n_anomalies == 0, result.anomalies


@pytest.mark.parametrize("method", SINGLE)
def test_a_ramp_with_real_noise_on_it_stays_quiet(method):
    """The spread measured on a trending series must be its wobble, not its slope."""
    for seed in range(6):
        result = detect(noisy_ramp(seed), method=method)
        assert result.n_anomalies <= 8, (method, seed, result.n_anomalies)
        assert result.rate < 0.05, (method, seed, result.rate)


@pytest.mark.parametrize("method", ["ewma", "rolling", "auto", "seasonal"])
def test_a_spike_on_a_ramp_is_still_found(method):
    """Quieting the slope must not also quiet the thing worth reporting."""
    for seed in range(6):
        values = noisy_ramp(seed)
        values[120] += 25.0
        result = detect(values, method=method)
        assert 120 in result.anomalies, (method, seed, result.anomalies)


def test_a_vote_that_overrules_a_flagged_point_says_so():
    """On a ramp the two flat-baseline methods are blind and outnumber the two that see.

    The vote is entitled to its answer, but it is not entitled to look like
    nothing was ever seen.
    """
    values = noisy_ramp(0)
    values[120] += 25.0
    result = detect(values, method="all")
    assert result.anomalies == []
    assert any("not by a strict majority" in note for note in result.warnings), result.warnings
    assert 120 in detect(values, method="rolling").anomalies


@pytest.mark.parametrize("method", ["rolling", "auto", "seasonal"])
def test_a_spike_on_a_slope_does_not_drag_its_neighbours_in_with_it(method):
    """A neighbourhood median must cost the same on a steep slope as on a flat one."""
    collateral = {}
    for slope in (0.0, 0.02, 0.7):
        extra = 0
        for seed in range(6):
            values = noisy_ramp(seed, slope=slope)
            values[120] += 25.0
            found = detect(values, method=method).anomalies
            assert 120 in found, (method, slope, seed, found)
            extra += len([index for index in found if index != 120])
        collateral[slope] = extra
    assert collateral[0.7] <= collateral[0.0] + 4, collateral


def test_ewma_measures_a_ramp_in_its_own_wobble_not_in_rounding_error():
    """The scale a trending series is judged by must be the size of its noise."""
    result = detect(noisy_ramp(0), method="ewma")
    assert result.scale is not None
    assert 0.03 < result.scale < 0.3, result.scale
    flat = detect(np.arange(500.0), method="ewma")
    assert flat.scale is None
    assert flat.n_anomalies == 0
    assert flat.warnings


# --------------------------------------------------------------------------
# the newest reading, and the oldest, are covered like any other
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("where", [0, -1])
def test_an_anomaly_on_the_very_edge_is_reported(method, where):
    """For monitoring, the newest sample is the one that matters most."""
    for seed in range(5):
        generator = np.random.default_rng(700 + seed)
        values = 20.0 + generator.normal(0.0, 0.4, 150)
        values[where] = 60.0
        result = detect(values, method=method)
        edge = where % 150
        assert edge in result.anomalies, (method, where, seed, result.anomalies)
        assert result.scores[edge] == max(result.scores), (method, where, seed)


@pytest.mark.parametrize("where", [0, -1])
def test_an_edge_point_is_never_compared_against_itself(where):
    """A baseline that copies the point it is explaining can never see it is wrong."""
    values = 20.0 + np.random.default_rng(11).normal(0.0, 0.4, 150)
    values[where] = 60.0
    result = detect(values)
    assert result.expected[where] != pytest.approx(60.0, abs=1.0)
    assert result.scores[where] > 3.0


# --------------------------------------------------------------------------
# clean data does not cry wolf
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("n", [60, 100, 200])
def test_clean_noise_stays_within_a_false_alarm_budget(method, n):
    """A 3-sigma rule should fire on about 1 clean point in 370, not 1 in 30."""
    counts = [detect(clean_noise(1000 + seed, n), method=method).n_anomalies for seed in range(12)]
    assert max(counts) <= 4, (method, n, counts)
    assert sum(counts) <= 12, (method, n, counts)


def test_the_default_method_is_quiet_on_sixty_clean_points():
    """The length a first-time user reaches for first."""
    for seed in (1, 2, 3):
        result = detect(clean_noise(seed, 60))
        assert result.n_anomalies <= 1, (seed, result.anomalies)


# --------------------------------------------------------------------------
# one seasonal anomaly is one anomaly, not a dozen
# --------------------------------------------------------------------------
@pytest.mark.parametrize("period,n", [(12, 120), (24, 240)])
def test_one_seasonal_anomaly_is_not_smeared_over_its_neighbours(period, n):
    values = seasonal(5, n, period)
    values[n // 2] += 12.0
    result = detect(values, method="seasonal", seasonality=period)
    assert result.anomalies == [n // 2], result.anomalies


@pytest.mark.parametrize("period,n", [(12, 120), (24, 240), (24, 480), (6, 72), (10, 200)])
def test_a_clean_seasonal_signal_is_near_silent(period, n):
    """Cyclic data on its own must not read as a few anomalies per hundred points."""
    counts = [detect(seasonal(seed, n, period), method="seasonal", seasonality=period).n_anomalies
              for seed in (5, 6, 7)]
    assert max(counts) <= 4, (period, n, counts)
    assert sum(counts) / (3.0 * n) < 0.02, (period, n, counts)


def test_the_seasonal_spread_is_the_real_noise_not_a_shrunken_one():
    """Every point left out of the median explaining it, so no pile of exact zeros."""
    result = detect(seasonal(5, 240, 24), method="seasonal", seasonality=24)
    assert result.scale == pytest.approx(0.3, rel=0.35), result.scale


# --------------------------------------------------------------------------
# seasonality inference scales
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n,period", [(240, 24), (300, 12), (97, 7), (500, 50)])
def test_inferred_period_matches_a_lag_by_lag_autocorrelation(n, period):
    """The fast transform must find exactly the cycle the slow loop found."""
    generator = np.random.default_rng(4)
    values = 50.0 + 8.0 * np.sin(2.0 * np.pi * np.arange(n) / period) + generator.normal(0, 1.0, n)
    centred = values - values.mean()
    denominator = float(np.dot(centred, centred))
    scores = np.zeros(n // 2 + 2)
    for lag in range(1, n // 2 + 1):
        scores[lag] = float(np.dot(centred[:-lag], centred[lag:])) / denominator
    best, best_score = None, MIN_SEASONAL_CORRELATION
    for lag in range(2, n // 2 + 1):
        if scores[lag] > best_score and scores[lag] > scores[lag - 1] >= 0 and scores[lag] >= scores[lag + 1]:
            best, best_score = lag, scores[lag]
    assert infer_period_from_values(values) == best == period


def test_a_trend_has_no_period_to_infer():
    assert infer_period_from_values(np.arange(200.0)) is None


def test_inferring_a_period_on_a_large_series_is_not_quadratic():
    """300k points is a few days of minute-by-minute telemetry, not an exotic case."""
    n = 300_000
    values = (
        50.0
        + 8.0 * np.sin(2.0 * np.pi * np.arange(n) / 144)
        + np.random.default_rng(3).normal(0.0, 1.0, n)
    )
    start = time.perf_counter()
    result = detect(values, method="seasonal")
    elapsed = time.perf_counter() - start
    assert result.seasonality == 144
    assert elapsed < 8.0, elapsed


# --------------------------------------------------------------------------
# errors are answered in this package's own words
# --------------------------------------------------------------------------
def test_a_dict_of_single_values_is_refused_in_our_own_words():
    with pytest.raises(ValueError, match="sequence of values"):
        detect({"a": 1})


@pytest.mark.parametrize("bad", ["high", b"high", object(), [1, 2]])
def test_a_nonsense_sensitivity_says_what_sensitivity_is(bad):
    with pytest.raises(ValueError, match="sensitivity must be a positive number of sigmas"):
        detect([1.0, 2.0, 3.0], sensitivity=bad)


@pytest.mark.parametrize("bad", ["day", b"day", object()])
def test_a_nonsense_seasonality_says_what_seasonality_is(bad):
    with pytest.raises(ValueError, match="seasonality must be an integer number of points"):
        detect([1.0, 2.0, 3.0], seasonality=bad)


#: the largest finite float, where a median of two middle readings leaves the range
FLOAT_MAX = 1.7976931348623157e308

#: series that once made numpy complain past the caller's own warning handler
EXTREME = {
    "repeated 1e300": [1e300, 1e300, 1e300, -1e300, 1e300] * 8,
    "alternating 1e300": [1e300, -1e300] * 40,
    "ramp of 1e150": list(np.arange(300.0) * 1e150),
    "all missing": [np.nan] * 40,
    "near float max": [1e308, -1e308, 1e308, 1e308, -1e308] * 20,
    "float max either way": [FLOAT_MAX, -FLOAT_MAX] * 50,
    "infinities among huge": [1e308, -1e308, 1e308, np.inf, -np.inf] * 20,
    "infinities among small": [1.0, 2.0, np.inf, 3.0, -np.inf] * 10,
    "infinities and gaps": [1.0, np.inf, np.nan, 2.0, 3.0] * 12,
    "subnormal": list(np.linspace(-1e-320, 1e-320, 100)),
    "huge seasonal": list(1e307 * np.sin(2.0 * np.pi * np.arange(240) / 24.0)),
    "tiny beside huge": [1e-300, 1e300] * 50,
    "constant at the top": [1e308] * 100,
}


@pytest.mark.parametrize("shape", sorted(EXTREME))
@pytest.mark.parametrize("method", list(METHODS))
def test_no_numpy_warning_escapes_to_the_caller(shape, method):
    """A -W error CI run must not turn an overflow deep inside into an exception.

    An even-length median is the mean of its two middle readings, so a series
    near the top of the float range overflows inside ``np.median`` itself - in
    the flat baselines, in the vote's summary of them, and in the shape the
    seasonal fit measures. None of that is news to the caller: an unusable
    baseline is already reported through ``result.warnings``.
    """
    values = EXTREME[shape]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = detect(values, method=method)
        result.summary()
        result.to_dict()
        result.to_frame()
        Detector(method=method).fit(values).score(values[:10])
    assert [f"{item.category.__name__}: {item.message}" for item in caught] == []
    # a reading that is not a finite number is not a measurement, so it is never
    # reported as an anomaly however far from the baseline its score lands
    assert not np.any(result.mask & ~np.isfinite(result.values))


@pytest.mark.parametrize("shape", sorted(EXTREME))
def test_an_unusable_extreme_series_says_so_instead_of_flagging_everything(shape):
    """Overflowing to infinity is not the same as every reading being an anomaly."""
    result = detect(EXTREME[shape])
    assert result.rate <= 0.25, (shape, result.n_anomalies, result.rate)
    if result.scale is None:
        assert result.n_anomalies == 0
        assert result.warnings


# --------------------------------------------------------------------------
# the streaming baseline still never looks ahead
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", SINGLE)
def test_a_frozen_baseline_scores_a_prefix_the_same_as_a_longer_batch(method):
    history = list(np.round(10.0 + np.random.default_rng(2).normal(0, 0.2, 60), 3))
    detector = Detector(method=method, seasonality=6).fit(history)
    prefix = detector.score([10.0, 25.0, 10.0])
    longer = detector.score([10.0, 25.0, 10.0, 10.0, 25.0, 10.0])
    assert np.allclose(prefix.scores, longer.scores[:3], equal_nan=True)
    assert np.allclose(prefix.expected, longer.expected[:3])


# --------------------------------------------------------------------------
# the same, through the command line
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", list(METHODS))
def test_the_cli_reports_a_rising_energy_meter_as_quiet(method, tmp_path, capsys):
    """A meter climbing 0.25 kWh a minute is a working meter, not 300 faults."""
    pd = pytest.importorskip("pandas")
    rows = 300
    generator = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "recorded_at": pd.date_range("2026-05-01", periods=rows, freq="min"),
            "kwh_total": np.arange(rows) * 0.25 + 4200.0 + generator.normal(0, 0.02, rows),
        }
    )
    path = tmp_path / "meter.csv"
    frame.to_csv(path, index=False, encoding="utf-8")

    from timeseries_anomaly.cli import main

    assert main([str(path), "--method", method]) == 0
    report = capsys.readouterr().out
    count = int(report.split("timeseries-anomaly: ")[1].split(" ")[0].replace("no", "0"))
    assert count <= 6, (method, report)
