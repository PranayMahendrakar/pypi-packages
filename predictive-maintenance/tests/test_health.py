"""health_score: the unsupervised degradation score."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import predictive_maintenance as pm
from predictive_maintenance._results import display_width
from conftest import make_frame


def test_quickstart_from_the_readme():
    wear = np.concatenate([np.zeros(168), np.linspace(0, 0.7, 120)])
    noise = np.random.default_rng(0).normal(0, 1.0, (2, 288))
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=288, freq="h"),
                       "vibration": 1 + wear + 0.15 * noise[0],
                       "temp_c": 60 + 2 * wear + 0.8 * noise[1], "rpm": 1500.0})
    health = pm.health_score(df)
    rul = pm.estimate_rul(df)

    assert 0.0 < health.score < 30.0
    assert health.trend == "degrading"
    assert health.top_contributors(1) == ["vibration"]
    assert isinstance(rul.remaining, pd.Timedelta)
    assert rul.remaining > pd.Timedelta(0)
    assert "health score" in health.summary()
    assert "remaining useful life" in rul.summary()


def test_score_is_bounded_and_rises_with_damage():
    scores = [
        pm.health_score(make_frame(amplitude=amp)).score
        for amp in (0.0, 0.5, 1.0, 2.0, 4.0)
    ]
    assert all(0.0 <= score <= 100.0 for score in scores)
    assert scores == sorted(scores)
    assert scores[0] < 5.0
    assert scores[-1] > 60.0


def test_healthy_machine_is_not_flagged(healthy):
    health = pm.health_score(healthy)
    assert health.score < 5.0
    assert health.trend in ("stable", "improving")
    assert health.degrading is False


def test_series_lines_up_with_the_input(degrading):
    health = pm.health_score(degrading)
    assert isinstance(health.series, pd.Series)
    assert len(health.series) == len(degrading)
    assert isinstance(health.series.index, pd.DatetimeIndex)
    assert health.series.iloc[-1] == pytest.approx(health.score)
    assert health.series.notna().all()


def test_contributors_sum_to_one_and_name_the_culprit(degrading):
    health = pm.health_score(degrading)
    assert set(health.contributors) == {"vibration", "temp_c", "rpm"}
    assert sum(health.contributors.values()) == pytest.approx(1.0)
    assert health.contributors["vibration"] > health.contributors["temp_c"]


def test_constant_channel_contributes_zero_not_nan(degrading):
    health = pm.health_score(degrading)
    assert health.contributors["rpm"] == 0.0
    assert not math.isnan(health.contributors["rpm"])
    assert "rpm" not in health.top_contributors(3)


def test_every_channel_constant_gives_zero_score():
    df = pd.DataFrame({"a": [5.0] * 60, "b": [-2.5] * 60})
    health = pm.health_score(df)
    assert health.score == 0.0
    assert health.series.eq(0.0).all()
    assert health.contributors == {"a": 0.0, "b": 0.0}
    assert any("no channel shows any deviation" in note for note in health.notes)


def test_repaired_machine_trends_down(recovering):
    health = pm.health_score(recovering)
    assert health.trend == "improving"
    assert health.degrading is False
    assert health.series.max() > health.score


def test_baseline_accepts_every_documented_spec(degrading):
    n_rows = len(degrading)
    mask = np.zeros(n_rows, dtype=bool)
    mask[:50] = True
    specs = [
        None,
        50,
        0.25,
        slice(0, 50),
        mask,
        list(range(50)),
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03")),
    ]
    for spec in specs:
        health = pm.health_score(degrading, baseline=spec)
        assert health.baseline_rows >= 2
        assert 0.0 <= health.score <= 100.0


def test_baseline_shorter_than_the_window_shortens_the_window(degrading):
    health = pm.health_score(degrading, baseline=6, window=40)
    assert health.window <= health.baseline_rows
    assert health.window >= 2
    assert any("window shortened" in note for note in health.notes)
    assert 0.0 <= health.score <= 100.0


def test_tiny_baseline_falls_back_and_says_so(degrading):
    health = pm.health_score(degrading, baseline=slice(0, 1))
    assert health.baseline_rows > 1
    assert any("fell back" in note for note in health.notes)


def test_bad_baseline_specs_are_rejected(degrading):
    with pytest.raises(ValueError, match="positive"):
        pm.health_score(degrading, baseline=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        pm.health_score(degrading, baseline=1.5)
    with pytest.raises(ValueError, match="baseline must be"):
        pm.health_score(degrading, baseline="last tuesday")


def test_window_selection(degrading):
    assert pm.health_score(degrading, window=12).window == 12
    with pytest.raises(ValueError, match="at least 2 rows"):
        pm.health_score(degrading, window=1)


def test_channels_and_time_can_be_chosen(degrading):
    health = pm.health_score(degrading, time="time", channels=["vibration"])
    assert health.channels == ["vibration"]
    assert list(health.contributors) == ["vibration"]

    positional = pm.health_score(degrading, time=False, channels="vibration")
    assert isinstance(positional.series.index, pd.RangeIndex)


def test_unknown_columns_are_reported_clearly(degrading):
    with pytest.raises(ValueError, match="channels not found"):
        pm.health_score(degrading, channels=["torque"])
    with pytest.raises(ValueError, match="time column"):
        pm.health_score(degrading, time="stamp")


def test_to_dict_is_json_safe(degrading):
    payload = pm.health_score(degrading).to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert "score" in json.loads(text)
    assert payload["trend"] in ("improving", "stable", "degrading")
    assert payload["n_rows"] == len(degrading)


def test_summary_is_plain_ascii(degrading):
    text = pm.health_score(degrading).summary()
    text.encode("ascii")
    assert "share of degradation" in text


def test_deterministic(degrading):
    first = pm.health_score(degrading)
    second = pm.health_score(degrading)
    assert first.score == second.score
    assert first.contributors == second.contributors


def test_input_is_never_mutated(degrading):
    before = degrading.copy(deep=True)
    pm.health_score(degrading)
    pd.testing.assert_frame_equal(degrading, before)


def test_channel_flat_through_the_baseline_does_not_saturate_on_float_dust():
    """A setpoint-held channel that ticks up by 1e-9 is noise, not a failure."""
    n_rows = 300
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "vibration": 1 + rng.normal(0.0, 0.05, n_rows),
            "temp_c": 60 + rng.normal(0.0, 0.5, n_rows),
            "rpm": np.full(n_rows, 1500.0),
        }
    )
    frame.loc[150:, "rpm"] = 1500.0 + 1e-9

    health = pm.health_score(frame)
    assert health.score < 5.0
    assert health.contributors["rpm"] < 0.05

    # and the score must stay monotone in physical significance: float dust on a
    # flat channel scores far below a real 2-sigma shift on a noisy one.
    shift = np.concatenate([rng.normal(0.0, 1.0, 150), rng.normal(2.0, 1.0, 150)])
    real = pm.health_score(
        pd.DataFrame(
            {"time": frame["time"], "v": shift, "steady": np.full(n_rows, 1500.0)}
        )
    )
    assert real.score > 5 * health.score


def test_a_flat_baseline_channel_is_capped_and_says_so():
    """No baseline noise means no calibration: a mid-range score, plus a note."""
    n_rows = 240
    rpm = np.full(n_rows, 1500.0)
    rpm[120:] = 900.0
    frame = pd.DataFrame(
        {"time": pd.date_range("2026-01-01", periods=n_rows, freq="h"), "rpm": rpm}
    )
    health = pm.health_score(frame)
    assert 0.0 < health.score <= 60.0
    assert any("never moved during the baseline" in note for note in health.notes)


def test_a_step_on_a_flat_channel_is_scored_by_its_size():
    """Bigger physical moves score higher; float noise scores nothing."""
    n_rows = 300
    scores = []
    for step in (1e-9, 1e-3, 1.0, 150.0):
        rpm = np.full(n_rows, 1500.0)
        rpm[150:] = 1500.0 + step
        scores.append(
            pm.health_score(
                pd.DataFrame(
                    {"time": pd.date_range("2026-01-01", periods=n_rows, freq="h"), "rpm": rpm}
                )
            ).score
        )
    assert scores == sorted(scores)
    assert scores[0] < 1.0
    assert scores[1] < 1.0
    assert scores[-1] > scores[0]


def test_baseline_range_in_the_wrong_units_is_a_clear_error():
    frame = pd.DataFrame(
        {"cycle": np.arange(0.0, 2000.0, 10.0), "vibration": np.linspace(1.0, 2.0, 200)}
    )
    with pytest.raises(ValueError, match="same units as the time axis"):
        pm.health_score(
            frame, baseline=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))
        )


def test_window_must_be_a_number(degrading):
    with pytest.raises(ValueError, match="window must be a whole number of rows"):
        pm.health_score(degrading, window="big")


def test_summary_columns_line_up_on_wide_channel_names():
    frame = make_frame(n_rows=160, ramp=80).rename(
        columns={"vibration": "振動", "temp_c": "température_°C"}
    )
    lines = [
        line for line in pm.health_score(frame).summary().splitlines() if line.endswith("%")
    ]
    assert len(lines) == 3
    columns = {display_width(line[: line.rindex("%") - 5]) for line in lines}
    assert len(columns) == 1


def test_threshold_hint_is_published(degrading):
    health = pm.health_score(degrading)
    assert health.threshold_hint == 30.0
    assert health.to_dict()["threshold_hint"] == 30.0
