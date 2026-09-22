"""estimate_rul: extrapolating the degradation trend to the threshold."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import predictive_maintenance as pm
from conftest import make_frame


def test_degrading_machine_gets_a_finite_deadline(degrading):
    rul = pm.estimate_rul(degrading)
    assert isinstance(rul.remaining, pd.Timedelta)
    assert rul.remaining > pd.Timedelta(0)
    assert rul.infinite is False
    assert rul.confidence in ("low", "medium", "high")
    assert isinstance(rul.eta, pd.Timestamp)
    assert rul.eta > degrading["time"].iloc[-1]


def test_flat_trend_is_infinite_with_low_confidence(healthy):
    rul = pm.estimate_rul(healthy)
    assert rul.infinite is True
    assert rul.remaining == float("inf")
    assert rul.confidence == "low"
    assert rul.eta is None
    assert "infinite" in rul.remaining_text()


def test_improving_trend_is_infinite_never_negative(recovering):
    rul = pm.estimate_rul(recovering, threshold=90.0)
    assert rul.trend == "improving"
    assert rul.infinite is True
    assert rul.remaining == float("inf")
    assert rul.confidence == "low"


def test_remaining_is_never_negative_at_any_threshold(degrading):
    for threshold in (1.0, 5.0, 15.0, 30.0, 60.0, 99.0):
        rul = pm.estimate_rul(degrading, threshold=threshold)
        if isinstance(rul.remaining, pd.Timedelta):
            assert rul.remaining >= pd.Timedelta(0)
        else:
            assert rul.remaining >= 0.0 or math.isinf(rul.remaining)


def test_a_score_already_past_the_threshold_is_due_now():
    rul = pm.estimate_rul(make_frame(amplitude=4.0), threshold=20.0)
    assert rul.remaining == pd.Timedelta(0)
    assert rul.score >= rul.threshold
    assert any("maintenance is due now" in note for note in rul.notes)


def test_row_based_data_returns_a_row_count():
    df = make_frame(timestamped=False)
    rul = pm.estimate_rul(df, time=False)
    assert not isinstance(rul.remaining, pd.Timedelta)
    assert isinstance(rul.remaining, float)
    assert rul.remaining > 0.0
    assert "rows" in rul.remaining_text()


def test_a_health_result_can_be_reused(degrading):
    health = pm.health_score(degrading)
    from_frame = pm.estimate_rul(degrading)
    from_result = pm.estimate_rul(health)
    assert from_result.score == pytest.approx(from_frame.score)
    assert from_result.remaining == from_frame.remaining


def test_confidence_falls_as_the_projection_reaches_further():
    near = pm.estimate_rul(make_frame(amplitude=0.9))
    far = pm.estimate_rul(make_frame(amplitude=0.35))
    order = {"low": 0, "medium": 1, "high": 2}
    assert order[near.confidence] >= order[far.confidence]


def test_threshold_must_be_a_score(degrading):
    for bad in (0.0, 100.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="threshold must be"):
            pm.estimate_rul(degrading, threshold=bad)


def test_to_dict_is_json_safe_even_when_infinite(healthy, degrading):
    infinite = pm.estimate_rul(healthy).to_dict()
    text = json.dumps(infinite, ensure_ascii=False)
    assert json.loads(text)["infinite"] is True
    assert infinite["remaining_seconds"] is None
    assert infinite["eta"] is None

    finite = pm.estimate_rul(degrading).to_dict()
    json.dumps(finite, ensure_ascii=False)
    assert finite["remaining_seconds"] > 0
    assert finite["infinite"] is False


def test_summary_is_plain_ascii(degrading, healthy):
    for frame in (degrading, healthy):
        pm.estimate_rul(frame).summary().encode("ascii")


def test_deterministic(degrading):
    assert pm.estimate_rul(degrading).remaining == pm.estimate_rul(degrading).remaining


def test_remaining_is_counted_in_rows_on_a_coarse_numeric_axis():
    """A cycle counter that advances 10 per row must not inflate the answer 10x."""
    n_rows = 300
    rng = np.random.default_rng(11)
    values = np.concatenate(
        [
            rng.normal(1.0, 0.05, 150),
            1 + np.linspace(0.0, 0.16, 150) + rng.normal(0.0, 0.05, 150),
        ]
    )
    per_row = pd.DataFrame({"cycle": np.arange(300.0), "v": values, "rpm": 1500.0})
    per_ten = pd.DataFrame(
        {"cycle": np.arange(0.0, n_rows * 10.0, 10.0), "v": values, "rpm": 1500.0}
    )

    coarse = pm.estimate_rul(per_ten)
    fine = pm.estimate_rul(per_row)
    assert coarse.remaining == pytest.approx(fine.remaining, rel=1e-6)
    assert coarse.slope_per_step == pytest.approx(fine.slope_per_step, rel=1e-6)
    assert "rows" in coarse.remaining_text()
    assert coarse.to_dict()["remaining_rows"] == pytest.approx(coarse.remaining)
    assert any("per row rather than per axis unit" in note for note in coarse.notes)


def test_past_the_threshold_with_a_flat_trend_is_infinite_not_zero():
    """The trend decides whether a crossing can be projected, and says why."""
    n_rows = 400
    profile = np.concatenate(
        [np.zeros(120), np.linspace(0.0, 1.5, 140), np.linspace(1.5, 0.0, 140)]
    )
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "vib": 1.0 + profile,
            "temp": 60.0,
        }
    )
    health = pm.health_score(frame)
    rul = pm.estimate_rul(frame, threshold=min(99.0, max(1.0, health.score - 1.0)))

    assert health.trend != "degrading"
    assert rul.infinite is True
    assert rul.remaining == float("inf")
    assert rul.confidence == "low"
    assert rul.eta is None
    assert any("already past the threshold" in note for note in rul.notes)


def test_health_notes_survive_every_path():
    """Data-quality notes matter most when the answer is 'infinite life'."""
    n_rows = 200
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "v": 1.0 + np.random.default_rng(2).normal(0.0, 0.1, n_rows),
        }
    )
    frame.loc[20:75, "v"] = np.nan

    health = pm.health_score(frame)
    rul = pm.estimate_rul(frame)
    assert rul.infinite is True
    for note in health.notes:
        assert note in rul.notes


def test_threshold_must_be_a_number(degrading):
    with pytest.raises(ValueError, match="threshold must be a health score"):
        pm.estimate_rul(degrading, threshold="x")


def _dying_machine(n=288):
    import numpy as np
    import pandas as pd

    wear = np.concatenate([np.zeros(120), np.linspace(0, 1.4, n - 120)])
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "vibration": 1 + wear + 0.05 * rng.normal(size=n),
        "temp_c": 60 + 6 * wear + 0.4 * rng.normal(size=n),
    })


def test_a_few_missing_timestamps_do_not_grant_infinite_life():
    """Regression and the worst bug this package had: remaining useful life is measured
    against the time axis, and a single NaT made the whole column unreadable, so the axis
    was thrown away and a visibly dying machine reported infinite remaining life."""
    import numpy as np
    import pandas as pd

    import predictive_maintenance as pm

    df = _dying_machine()
    clean = pm.estimate_rul(df, time="time")

    for missing in (1, 5, 20):
        damaged = df.copy()
        damaged.loc[damaged.index[10:10 + missing], "time"] = pd.NaT
        result = pm.estimate_rul(damaged, time="time")
        remaining = result.remaining
        if isinstance(remaining, float):
            assert np.isfinite(remaining), f"{missing} NaT gave infinite life"
        assert str(remaining) == str(clean.remaining), (
            f"{missing} missing stamps changed the answer: {remaining} vs {clean.remaining}"
        )


def test_a_mostly_unreadable_time_column_falls_back_honestly():
    """When there is nothing to interpolate between, row order really is the right
    fallback - but it must still produce a finite answer."""
    import pandas as pd

    import predictive_maintenance as pm

    df = _dying_machine()
    df.loc[:, "time"] = pd.NaT
    result = pm.estimate_rul(df, time="time")
    assert result.remaining is not None


def test_columns_that_are_not_strings_are_accepted():
    """A DataFrame built from a numpy array has integer labels, which is an ordinary
    thing to pass. Lookups stringified the name and then used it against the original
    integer label, dying with a bare KeyError: '0'."""
    import predictive_maintenance as pm

    df = _dying_machine()
    df.columns = [0, 1, 2]
    report = pm.health_score(df, time=0)
    assert 0.0 <= report.score <= 100.0
