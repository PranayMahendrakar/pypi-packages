"""Regressions found by review. Every test here is a bug that shipped once.

1. A steady level shift cost nothing once the outlier share saturated, so a
   thermometer reading a million against a 60 baseline scored the same passing
   grade C as one reading 65.
2. score().trend read "improving" for a permanent step to a worse level, and gave
   a directional verdict on nearly half of all stationary runs.
3. numpy overflow warnings escaped to the caller on absurd magnitudes.
4. to_dict() reported 100.0 for a component that was never measured.
5. The CLI surfaced pandas' and the OS's own wording without naming the argument.
"""

from __future__ import annotations

import collections
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from machine_health import HealthMonitor, score
from machine_health.cli import main


def stepped(level, *, seed=0, n_base=400, n_after=600, sd=1.0):
    """400 healthy rows at 60, then a permanent steady step to ``level``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"temp": np.r_[rng.normal(60, sd, n_base), rng.normal(level, sd, n_after)]}
    )


# --------------------------------------------------------------- level shifts


def test_a_steady_level_shift_keeps_costing_points_without_rules():
    """Severity beyond the saturated outlier share must still move the number."""
    results = [score(stepped(level), baseline=400) for level in (65, 70, 100, 1000, 1e6)]
    values = [r.value for r in results]
    assert values == sorted(values, reverse=True), values
    assert values[0] - values[-1] > 20  # not a flat floor any more
    for level, result in zip((65, 70, 100, 1000, 1e6), results):
        assert not result.ok, f"{level} reported ok"
        assert result.grade in {"D", "F"}, f"{level} graded {result.grade}"
        assert 0.0 <= result.value <= 100.0


def test_a_level_shift_shows_up_in_stability_and_is_explained():
    result = score(stepped(100), baseline=400)
    assert result.components["stability"] < 60
    assert result.contributors["temp"] == pytest.approx(result.penalty)
    assert sum(result.component_penalties.values()) == pytest.approx(result.penalty, abs=1e-9)


def test_rules_still_beat_the_unruled_score_on_the_same_data():
    """The documented remedy must stay at least as strict as the default."""
    data = stepped(100)
    assert score(data, baseline=400, rules={"temp": {"max": 65}}).value <= score(
        data, baseline=400
    ).value


def test_a_level_shift_inside_normal_wander_is_not_punished():
    """The other side of the fix: ordinary noise must not cost stability points."""
    grades = collections.Counter()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            {"temp": rng.normal(60, 1, 1000), "vibration": rng.normal(0.2, 0.02, 1000)}
        )
        result = score(frame)
        grades[result.grade] += 1
        assert result.components["stability"] > 90
    assert grades == {"A": 20}, grades


def test_a_flat_baseline_that_moves_to_another_flat_level_is_not_perfect():
    result = score(pd.DataFrame({"x": [1.0] * 50 + [900.0] * 50}), baseline=50)
    assert result.components["stability"] < 100
    assert not result.ok


# --------------------------------------------------------------------- trend


def test_a_permanent_step_to_a_worse_level_never_reads_improving():
    trends = collections.Counter()
    for seed in range(25):
        rng = np.random.default_rng(seed)
        values = np.r_[rng.normal(60, 1, 500), rng.normal(72, 1, 500)]
        trends[score(pd.DataFrame({"temp": values})).trend] += 1
    assert trends == {"degrading": 25}, trends


def test_a_recovering_machine_still_reads_improving():
    trends = collections.Counter()
    for seed in range(25):
        rng = np.random.default_rng(seed)
        values = np.r_[rng.normal(60, 1, 200), rng.normal(72, 1, 400), rng.normal(60, 1, 400)]
        trends[score(pd.DataFrame({"temp": values})).trend] += 1
    assert trends == {"improving": 25}, trends


@pytest.mark.parametrize("rules", [None, {"x": {"max": 56}}])
def test_stationary_data_does_not_get_a_directional_verdict(rules):
    """A stray 3-sigma row in one half used to be enough to call a direction."""
    trends = collections.Counter()
    for seed in range(100):
        rng = np.random.default_rng(seed)
        trends[score(pd.DataFrame({"x": rng.normal(50, 2, 600)}), rules=rules).trend] += 1
    assert trends["stable"] >= 95, trends


def test_a_single_breaching_row_does_not_flip_the_trend():
    values = np.full(400, 10.0)
    values[120] = 99.0  # one bad row, early in the window
    result = score(pd.DataFrame({"x": values}), rules={"x": {"max": 50}})
    assert result.violations  # it is still reported
    assert result.trend == "stable"


def test_the_trend_still_reacts_to_real_change():
    rng = np.random.default_rng(3)
    louder = np.r_[rng.normal(60, 1, 500), rng.normal(60, 4, 500)]
    assert score(pd.DataFrame({"temp": louder})).trend == "degrading"
    dying = pd.DataFrame({"temp": np.r_[rng.normal(60, 1, 600), np.full(400, np.nan)]})
    assert score(dying).trend == "degrading"


# ------------------------------------------------------------------ warnings


@pytest.mark.parametrize(
    "values",
    [
        [1e308] * 60,
        list(np.random.default_rng(0).normal(1e307, 1e305, 60)),
        [1e308] * 30 + [1.0] * 30,
        [1e-308] * 30 + [1e308] * 30,
    ],
)
def test_no_numpy_warning_escapes_on_extreme_magnitudes(values):
    frame = pd.DataFrame({"a": values})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all becomes an exception
        result = score(frame)
        score(frame, rules={"a": {"max": 1e300}})
        monitor = HealthMonitor(frame.iloc[:30])
        monitor.update(frame.iloc[30:])
    assert 0.0 <= result.value <= 100.0


def test_an_unmeasurable_scale_is_recorded_as_a_note():
    result = score(pd.DataFrame({"a": [1e308] * 60}))
    assert any("too large to measure" in note for note in result.notes)
    assert "stability" in result.unmeasured


# --------------------------------------------------------------- to_dict json


def test_an_unmeasured_component_is_null_not_a_free_hundred():
    result = score(pd.DataFrame({"a": np.random.default_rng(0).normal(0, 1, 200)}))
    payload = result.to_dict()
    assert payload["components"]["compliance"] is None
    assert payload["weights"]["compliance"] == 0.0
    assert payload["unmeasured"] == ["compliance"]
    assert payload["components"]["stability"] is not None
    json.dumps(payload)  # null is JSON, NaN is not
    assert "n/a" in result.summary()
    assert "not scored" in result.summary()


def test_a_measured_component_keeps_its_number():
    result = score(
        pd.DataFrame({"a": [1.0] * 40 + [9.0] * 10}), rules={"a": {"max": 5}}, baseline=40
    )
    payload = result.to_dict()
    assert payload["unmeasured"] == []
    for name, value in payload["components"].items():
        assert value is not None, name
    assert payload["components"]["compliance"] < 100


def test_monitor_history_does_not_invent_a_compliance_column():
    rng = np.random.default_rng(1)
    monitor = HealthMonitor(pd.DataFrame({"temp": rng.normal(5, 1, 200)}))
    monitor.update(pd.DataFrame({"temp": rng.normal(5, 1, 100)}))
    assert monitor.history["compliance"].isna().all()
    assert monitor.history["availability"].notna().all()
    assert monitor.to_dict()["history"][0]["compliance"] is None


# -------------------------------------------------------------------- the CLI


def test_an_empty_csv_names_the_file(tmp_path, capsys):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert main([str(path)]) == 2
    error = capsys.readouterr().err
    assert "empty.csv" in error
    assert "no columns to read" in error
    assert "Traceback" not in error


def test_a_missing_rules_file_names_the_flag(tmp_path, capsys):
    data = tmp_path / "t.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_csv(data, index=False)
    assert main([str(data), "--rules", str(tmp_path / "no_such_file.json")]) == 2
    error = capsys.readouterr().err
    assert "--rules" in error
    assert "no such file" in error
    assert "Errno" not in error


def test_a_malformed_rules_file_names_the_flag(tmp_path, capsys):
    data = tmp_path / "t.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_csv(data, index=False)
    broken = tmp_path / "rules.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main([str(data), "--rules", str(broken)]) == 2
    error = capsys.readouterr().err
    assert "--rules" in error
    assert "not valid JSON" in error
