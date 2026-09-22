"""The four components, the grade bands and the arguments of score()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine_health import (
    COMPONENTS,
    DEFAULT_WEIGHTS,
    GRADE_BANDS,
    HealthScorer,
    MachineScore,
    Rule,
    grade_for,
    score,
)


def test_healthy_machine_scores_a(healthy_df):
    result = score(healthy_df)
    assert result.value > 90
    assert result.grade == "A"
    assert result.trend == "stable"
    assert not result.violations
    assert result.ok


def test_drifting_machine_scores_worse_than_healthy(healthy_df, drifting_df):
    assert score(drifting_df).value < score(healthy_df).value - 10


def test_components_are_reported_separately(drifting_df):
    result = score(drifting_df, rules={"temp": {"max": 70}})
    assert set(result.components) == set(COMPONENTS)
    for name, value in result.components.items():
        assert 0.0 <= value <= 100.0, name


def test_stability_reacts_to_extra_variation():
    steady = pd.DataFrame({"x": [10.0, 10.5, 10.0, 9.5] * 50})
    swinging = pd.DataFrame({"x": [10.0, 10.5, 10.0, 9.5] * 25 + [5.0, 15.0, 4.0, 16.0] * 25})
    assert score(steady).components["stability"] > 95
    assert score(swinging).components["stability"] < 40


def test_stability_ignores_a_channel_that_calms_down():
    calming = pd.DataFrame({"x": [0.0, 20.0] * 50 + [10.0, 10.1] * 50})
    assert score(calming).components["stability"] == pytest.approx(100.0)


def test_anomaly_counts_outliers_against_the_baseline():
    rng = np.random.default_rng(5)
    values = rng.normal(0, 1, 400)
    values[300:340] = 40.0  # 10% of the window is far away
    result = score(pd.DataFrame({"x": values}))
    assert result.components["anomaly"] == pytest.approx(0.0, abs=1.0)


def test_availability_reacts_to_missing_data():
    values = np.arange(200, dtype="float64")
    values[100:150] = np.nan
    result = score(pd.DataFrame({"x": values}))
    assert result.components["availability"] == pytest.approx(100 * (1 - 50 / 160), abs=0.5)


def test_availability_reacts_to_a_flatlined_channel():
    rng = np.random.default_rng(9)
    values = np.r_[rng.normal(5, 1, 100), np.full(100, 5.0)]
    result = score(pd.DataFrame({"x": values}), baseline=100)
    assert result.components["availability"] == pytest.approx(50.0)
    # the flatline is surfaced as availability points, charged to the channel itself
    assert result.component_penalties["availability"] == pytest.approx(result.penalty)
    assert result.contributors["x"] == pytest.approx(result.penalty)
    assert result.top_contributors()[0][0] == "x"


def test_trend_sees_a_machine_getting_worse():
    rng = np.random.default_rng(11)
    worsening = np.r_[rng.normal(10, 0.5, 200), rng.normal(10, 0.6, 200), rng.normal(14, 3, 200)]
    assert score(pd.DataFrame({"x": worsening})).trend == "degrading"


def test_trend_sees_a_machine_recovering():
    rng = np.random.default_rng(12)
    recovering = np.r_[rng.normal(10, 0.5, 200), rng.normal(15, 3, 300), rng.normal(10, 0.5, 300)]
    assert score(pd.DataFrame({"x": recovering})).trend == "improving"


@pytest.mark.parametrize(
    "value, grade",
    [(100, "A"), (90, "A"), (89.99, "B"), (80, "B"), (79.5, "C"), (70, "C"), (65, "D"), (60, "D"), (59.9, "F"), (0, "F")],
)
def test_grades_are_fixed_bands(value, grade):
    assert grade_for(value) == grade


def test_grade_bands_are_documented():
    readme = (pytest.importorskip("pathlib").Path(__file__).resolve().parents[1] / "README.md")
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        for letter, floor in GRADE_BANDS[:-1]:
            assert f"{letter} >= {int(floor)}" in text
    assert [g for g, _ in GRADE_BANDS] == ["A", "B", "C", "D", "F"]
    assert score(pd.DataFrame({"x": [1.0, 1.0, 1.0]})).to_dict()["grade_bands"]["A"] == 90.0


def test_baseline_can_be_a_fraction_a_count_or_a_frame(drifting_df):
    by_fraction = score(drifting_df, baseline=0.2)
    by_count = score(drifting_df, baseline=200)
    assert by_fraction.value == pytest.approx(by_count.value)
    assert by_fraction.n_baseline_rows == 200

    separate = score(drifting_df, baseline=drifting_df.iloc[:200])
    assert separate.n_rows == len(drifting_df)
    assert separate.n_baseline_rows == 200


def test_baseline_can_be_a_boolean_mask(drifting_df):
    mask = np.zeros(len(drifting_df), dtype=bool)
    mask[:300] = True
    result = score(drifting_df, baseline=mask)
    assert result.n_baseline_rows == 300
    assert result.n_rows == len(drifting_df) - 300


def test_baseline_defaults_to_the_first_fifth(drifting_df):
    assert score(drifting_df).n_baseline_rows == len(drifting_df) // 5


def test_channels_argument_limits_what_is_scored(drifting_df):
    result = score(drifting_df, channels=["temp"])
    assert result.channels == ["temp"]
    assert set(result.contributors) == {"temp"}


def test_a_rule_pulls_its_channel_back_in(drifting_df):
    result = score(drifting_df, channels=["temp"], rules={"vibration": {"max": 0.5}})
    assert set(result.channels) == {"temp", "vibration"}
    assert any("rule names them" in note for note in result.notes)


def test_time_column_orders_rows_and_is_excluded(timed_df):
    shuffled = timed_df.sample(frac=1.0, random_state=0)
    ordered = score(timed_df, time="ts")
    reordered = score(shuffled, time="ts")
    assert reordered.value == pytest.approx(ordered.value)
    assert "ts" not in ordered.channels
    assert ordered.when == timed_df["ts"].iloc[-1]
    assert any("sorted by" in note for note in reordered.notes)


def test_scorer_class_matches_the_function(drifting_df):
    scorer = HealthScorer(rules={"temp": {"max": 70}}, baseline=0.25)
    assert scorer.score(drifting_df).value == pytest.approx(
        score(drifting_df, rules={"temp": {"max": 70}}, baseline=0.25).value
    )


def test_reading_from_a_csv_path(tmp_path, drifting_df):
    path = tmp_path / "telemetry.csv"
    drifting_df.to_csv(path, index=False)
    assert score(path).value == pytest.approx(score(drifting_df).value)


def test_result_is_json_safe_and_explains_itself(timed_df):
    import json

    result = score(timed_df, time="ts", rules={"temp": {"max": 62}})
    data = result.to_dict()
    json.dumps(data, ensure_ascii=False)
    assert data["value"] == pytest.approx(result.value)
    assert set(data["component_meaning"]) == set(COMPONENTS)
    assert isinstance(result, MachineScore)
    assert "machine health:" in result.summary()


def test_weights_change_the_answer(drifting_df):
    compliance_heavy = score(
        drifting_df, rules={"temp": {"max": 61}}, weights={"compliance": 0.9}
    )
    stability_heavy = score(
        drifting_df, rules={"temp": {"max": 61}}, weights={"stability": 0.9}
    )
    assert compliance_heavy.value != pytest.approx(stability_heavy.value)
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_rule_objects_and_dicts_agree(drifting_df):
    from_object = score(drifting_df, rules=[Rule("temp", max=70, warn_max=65)])
    from_dict = score(drifting_df, rules={"temp": {"max": 70, "warn_max": 65}})
    assert from_object.value == pytest.approx(from_dict.value)
