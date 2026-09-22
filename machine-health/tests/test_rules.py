"""Limits, violations and the errors a mistyped rule produces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine_health import Rule, Violation, score
from machine_health.rules import evaluate_rule, normalize_rules


@pytest.fixture
def hot_df():
    return pd.DataFrame({"temp": [60.0] * 50 + [85.0] * 10, "vibration": [0.2] * 60})


def test_a_broken_hard_limit_is_critical(hot_df):
    result = score(hot_df, rules={"temp": {"max": 80}})
    assert result.violations
    violation = result.violations[0]
    assert isinstance(violation, Violation)
    assert violation.channel == "temp"
    assert violation.limit == "max"
    assert violation.severity == "critical"
    assert violation.count == 10
    assert violation.worst == pytest.approx(85.0)
    assert "temp above max=80" in violation.message
    assert not result.ok


def test_a_broken_soft_limit_is_only_a_warning(hot_df):
    result = score(hot_df, rules={"temp": {"warn_max": 80}})
    assert [v.severity for v in result.violations] == ["warning"]
    assert result.components["compliance"] > score(hot_df, rules={"temp": {"max": 80}}).components[
        "compliance"
    ]


def test_hard_and_soft_limits_do_not_double_count(hot_df):
    result = score(hot_df, rules={"temp": {"max": 80, "warn_max": 70}})
    limits = {v.limit: v.count for v in result.violations}
    assert limits["max"] == 10
    assert "warn_max" not in limits  # the same rows are not warned about twice


def test_a_single_bad_reading_still_costs_something():
    values = [10.0] * 999 + [99.0]
    result = score(pd.DataFrame({"x": values}), rules={"x": {"max": 50}})
    assert result.violations
    assert result.components["compliance"] <= 75.0


def test_min_limit_and_worst_value():
    result = score(pd.DataFrame({"x": [5.0] * 40 + [-3.0] * 10}), rules={"x": {"min": 0}})
    violation = result.violations[0]
    assert violation.limit == "min"
    assert violation.worst == pytest.approx(-3.0)
    assert "below min=0" in violation.message


def test_rule_weight_shifts_the_blame():
    df = pd.DataFrame({"a": [0.0] * 40 + [9.0] * 10, "b": [0.0] * 40 + [9.0] * 10})
    light = score(df, rules=[Rule("a", max=1, weight=1), Rule("b", max=1, weight=1)])
    heavy = score(df, rules=[Rule("a", max=1, weight=9), Rule("b", max=1, weight=1)])
    assert light.contributors["a"] == pytest.approx(light.contributors["b"])
    assert heavy.contributors["a"] > heavy.contributors["b"]


def test_unknown_channel_lists_what_is_available():
    df = pd.DataFrame({"temp": [1.0, 2.0, 3.0], "vibration": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError) as excinfo:
        score(df, rules={"presure": {"max": 3}})
    message = str(excinfo.value)
    assert "presure" in message
    assert "temp" in message and "vibration" in message
    assert "available channels" in message


def test_rule_on_a_non_numeric_column_is_reported():
    df = pd.DataFrame({"temp": [1.0, 2.0, 3.0], "site": ["a", "b", "c"]})
    with pytest.raises(ValueError) as excinfo:
        score(df, rules={"site": {"max": 3}})
    assert "site" in str(excinfo.value)


def test_normalize_rules_accepts_every_documented_shape():
    assert normalize_rules(None) == []
    assert len(normalize_rules(Rule("a", max=1))) == 1
    assert len(normalize_rules({"a": {"max": 1}, "b": {"min": 0}})) == 2
    assert len(normalize_rules([{"channel": "a", "max": 1}])) == 1
    assert normalize_rules({"a": (0, 10)})[0].max == 10


@pytest.mark.parametrize(
    "bad, error",
    [
        ({"a": {"maxx": 1}}, ValueError),
        ({"a": {}}, ValueError),
        ([{"max": 1}], ValueError),
        ("temp:max=1", TypeError),
        ([object()], TypeError),
        ({"a": "high"}, TypeError),
    ],
)
def test_bad_rule_shapes_raise_clearly(bad, error):
    with pytest.raises(error):
        normalize_rules(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": 10, "max": 1},
        {"max": float("nan")},
        {"max": 1, "weight": 0},
        {"max": 1, "weight": -2},
        {"max": "hot"},
    ],
)
def test_bad_rule_values_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        Rule("temp", **kwargs)


def test_rule_needs_a_limit_and_a_channel():
    with pytest.raises(ValueError):
        Rule("temp")
    with pytest.raises(ValueError):
        Rule("", max=1)


def test_rule_describe_and_to_dict():
    rule = Rule("temp", max=80.0, warn_max=75.0, weight=2.0)
    assert rule.describe() == "temp: max=80, warn_max=75, weight=2"
    assert rule.to_dict() == {"channel": "temp", "weight": 2.0, "max": 80.0, "warn_max": 75.0}


def test_evaluate_rule_on_an_empty_channel():
    deficit, violations = evaluate_rule(Rule("x", max=1), np.array([np.nan, np.nan]))
    assert deficit is None
    assert violations == []


def test_violation_to_dict_is_json_safe(hot_df):
    import json

    result = score(hot_df, rules={"temp": {"max": 80}})
    payload = result.violations[0].to_dict()
    json.dumps(payload)
    assert payload["severity"] == "critical"
    # the window is the 48 rows after the default 20% baseline, and 10 of them are hot
    assert payload["n_rows"] == 48
    assert payload["fraction"] == pytest.approx(10 / 48)


def test_violation_reports_when_it_first_happened():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-03-01", periods=60, freq="h"),
            "temp": [60.0] * 50 + [85.0] * 10,
        }
    )
    result = score(df, time="ts", rules={"temp": {"max": 80}})
    assert result.violations[0].first_time == pd.Timestamp("2026-03-03 02:00:00")
