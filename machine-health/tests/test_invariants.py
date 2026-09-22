"""The promises the score makes: bounded, explainable and reproducible."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine_health import COMPONENTS, HealthMonitor, score

rng = np.random.default_rng(42)


def awkward_frames():
    """Every shape of input worth checking the 0-100 bound against."""
    yield "single row", pd.DataFrame({"x": [1.0]})
    yield "two rows", pd.DataFrame({"x": [1.0, 2.0]})
    yield "constant", pd.DataFrame({"x": [7.0] * 100})
    yield "all nan", pd.DataFrame({"x": [np.nan] * 100})
    yield "half nan", pd.DataFrame({"x": [1.0, np.nan] * 50})
    yield "one value one nan", pd.DataFrame({"x": [1.0, np.nan]})
    yield "infinities", pd.DataFrame({"x": [np.inf, -np.inf] * 25 + [1.0] * 50})
    yield "huge", pd.DataFrame({"x": [1e18, -1e18] * 50})
    yield "tiny", pd.DataFrame({"x": [1e-18, -1e-18] * 50})
    yield "explosion", pd.DataFrame({"x": np.r_[np.zeros(50), rng.normal(0, 1e6, 50)]})
    yield "mixed channels", pd.DataFrame(
        {"a": rng.normal(0, 1, 100), "b": [np.nan] * 100, "c": [3.0] * 100}
    )
    yield "negative", pd.DataFrame({"x": -np.abs(rng.normal(0, 1, 100))})
    yield "bools", pd.DataFrame({"x": [True, False] * 50})
    yield "ints", pd.DataFrame({"x": np.arange(100)})


@pytest.mark.parametrize("name, df", list(awkward_frames()), ids=lambda v: v if isinstance(v, str) else "")
def test_score_is_always_between_0_and_100(name, df):
    for rules in (None, {"x": {"min": -1, "max": 1}} if "x" in df.columns else None):
        result = score(df, rules=rules)
        assert 0.0 <= result.value <= 100.0, f"{name}: {result.value}"
        assert result.grade in {"A", "B", "C", "D", "F"}
        for component, value in result.components.items():
            assert 0.0 <= value <= 100.0, f"{name}/{component}: {value}"


@pytest.mark.parametrize("name, df", list(awkward_frames()), ids=lambda v: v if isinstance(v, str) else "")
def test_components_and_contributors_sum_to_the_penalty(name, df):
    result = score(df)
    assert sum(result.component_penalties.values()) == pytest.approx(result.penalty, abs=1e-9)
    assert sum(result.contributors.values()) == pytest.approx(result.penalty, abs=1e-9)


def test_the_two_breakdowns_agree_on_a_real_machine(drifting_df):
    result = score(drifting_df, rules={"temp": {"max": 62}, "vibration": {"max": 0.25}})
    assert result.penalty == pytest.approx(100.0 - result.value)
    assert sum(result.component_penalties.values()) == pytest.approx(result.penalty, abs=1e-9)
    assert sum(result.contributors.values()) == pytest.approx(result.penalty, abs=1e-9)
    for name in COMPONENTS:
        expected = result.weights[name] * (100.0 - result.components[name])
        assert result.component_penalties[name] == pytest.approx(expected, abs=1e-9)


def test_effective_weights_sum_to_one_when_anything_is_measurable(drifting_df):
    result = score(drifting_df, rules={"temp": {"max": 70}})
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_scoring_is_reproducible(drifting_df):
    first = score(drifting_df, rules={"temp": {"max": 62}})
    second = score(drifting_df, rules={"temp": {"max": 62}})
    assert first.to_dict() == second.to_dict()


def test_a_worse_machine_never_scores_higher():
    rng_local = np.random.default_rng(3)
    base = rng_local.normal(10, 1, 400)
    mild = pd.DataFrame({"x": np.r_[base, rng_local.normal(10.5, 1.2, 200)]})
    severe = pd.DataFrame({"x": np.r_[base, rng_local.normal(20.0, 6.0, 200)]})
    assert score(severe).value < score(mild).value


def test_no_rules_still_scores_the_other_three(drifting_df):
    result = score(drifting_df)
    assert 0 < result.value < 100
    assert result.weights["compliance"] == 0.0
    assert sum(result.weights[name] for name in ("stability", "anomaly", "availability")) == (
        pytest.approx(1.0)
    )
    assert any("no rules given" in note for note in result.notes)
    assert result.violations == []


def test_weights_that_do_not_sum_to_one_are_normalized_with_a_note(drifting_df):
    result = score(drifting_df, rules={"temp": {"max": 70}}, weights={"stability": 3, "compliance": 3, "anomaly": 2, "availability": 2})
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert result.weights["stability"] == pytest.approx(0.3)
    assert any("did not sum to 1" in note for note in result.notes)


def test_normalized_weights_match_the_equivalent_fractions(drifting_df):
    scaled = score(drifting_df, rules={"temp": {"max": 70}}, weights={"stability": 60, "compliance": 60, "anomaly": 40, "availability": 40})
    plain = score(drifting_df, rules={"temp": {"max": 70}})
    assert scaled.value == pytest.approx(plain.value)


@pytest.mark.parametrize(
    "bad", [{"stability": -1}, {"nonsense": 0.5}, {"stability": float("nan")}, {"stability": "heavy"},
            {"stability": 0, "compliance": 0, "anomaly": 0, "availability": 0}]
)
def test_bad_weights_are_rejected(bad, drifting_df):
    with pytest.raises(ValueError):
        score(drifting_df, weights=bad)


def test_weights_must_be_a_mapping(drifting_df):
    with pytest.raises(TypeError):
        score(drifting_df, weights=[0.25, 0.25, 0.25, 0.25])


def test_monitor_scores_match_score_on_the_same_window(drifting_df):
    baseline = drifting_df.iloc[:200]
    batch = drifting_df.iloc[200:]
    monitor = HealthMonitor(baseline, rules={"temp": {"max": 70}})
    direct = score(drifting_df, rules={"temp": {"max": 70}}, baseline=200)
    assert monitor.update(batch).value == pytest.approx(direct.value)
