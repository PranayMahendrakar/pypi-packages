"""HealthMonitor: a fixed baseline, one score per batch, history and alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine_health import Alert, HealthMonitor
from machine_health.result import COMPONENTS


def batches(levels, seed=0, n=120, start="2026-02-01"):
    """One 120-row batch per (mean, sd) pair, each an hour after the last."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (mu, sd) in enumerate(levels):
        out.append(
            pd.DataFrame(
                {
                    "ts": pd.date_range(start, periods=n, freq="min")
                    + pd.Timedelta(hours=i + 1),
                    "temp": rng.normal(mu, sd, n),
                    "vibration": rng.normal(0.2, 0.02, n),
                }
            )
        )
    return out


@pytest.fixture
def baseline_df():
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-02-01", periods=300, freq="min"),
            "temp": rng.normal(60, 1, 300),
            "vibration": rng.normal(0.2, 0.02, 300),
        }
    )


def test_baseline_scores_itself_well(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    assert monitor.baseline_score.value > 95
    assert monitor.last_score is monitor.baseline_score
    assert monitor.n_updates == 0
    assert monitor.history.empty
    assert list(monitor.history.columns)[:5] == ["update", "when", "value", "grade", "trend"]


def test_each_batch_is_scored_against_the_same_baseline(baseline_df):
    monitor = HealthMonitor(baseline_df, rules={"temp": {"max": 70}}, time="ts")
    values = [monitor.update(b).value for b in batches([(60, 1), (63, 2), (70, 5)])]
    assert values[0] > values[1] > values[2]
    assert monitor.n_updates == 3
    assert len(monitor.history) == 3
    assert set(COMPONENTS).issubset(monitor.history.columns)
    assert monitor.history["value"].tolist() == [pytest.approx(v) for v in values]


def test_history_carries_timestamps_and_grades(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    for batch in batches([(60, 1), (72, 6)]):
        monitor.update(batch)
    history = monitor.history
    assert history["when"].notna().all()
    assert set(history["grade"]) <= {"A", "B", "C", "D", "F"}
    assert history["rows"].tolist() == [120, 120]


def test_a_grade_drop_raises_an_alert(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    monitor.update(batches([(60, 1)])[0])
    monitor.update(batches([(75, 8)], seed=5)[0])
    drops = [a for a in monitor.alerts if "grade dropped" in a.message]
    assert drops
    assert isinstance(drops[0], Alert)
    assert drops[0].severity in {"warning", "critical"}
    assert drops[0].when is not None


def test_a_broken_rule_raises_an_alert(baseline_df):
    monitor = HealthMonitor(baseline_df, rules={"temp": {"max": 65}}, time="ts")
    monitor.update(batches([(70, 1)], seed=2)[0])
    rule_alerts = [a for a in monitor.alerts if "above max=65" in a.message]
    assert rule_alerts
    assert rule_alerts[0].severity == "critical"
    assert "[critical]" in rule_alerts[0].summary()
    assert rule_alerts[0].to_dict()["severity"] == "critical"


def test_a_steady_machine_raises_nothing(baseline_df):
    monitor = HealthMonitor(baseline_df, rules={"temp": {"max": 90}}, time="ts")
    for batch in batches([(60, 1), (60, 1), (60, 1)], seed=11):
        monitor.update(batch)
    assert monitor.alerts == []


def test_empty_batch_returns_the_previous_score_with_a_note(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    first = monitor.update(batches([(63, 2)])[0])
    same = monitor.update(pd.DataFrame(columns=["ts", "temp", "vibration"]))
    assert same.value == pytest.approx(first.value)
    assert same.grade == first.grade
    assert same.notes[-1] == "empty batch: the previous score is returned unchanged"
    assert monitor.n_updates == 1
    assert len(monitor.history) == 1
    assert first.notes[-1] != same.notes[-1]  # the previous score itself is untouched


def test_empty_first_batch_falls_back_to_the_baseline_score(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    result = monitor.update(pd.DataFrame(columns=["ts", "temp", "vibration"]))
    assert result.value == pytest.approx(monitor.baseline_score.value)
    assert monitor.n_updates == 0


def test_trend_follows_the_history(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    trends = [monitor.update(b).trend for b in batches([(60, 1), (66, 3), (72, 6), (60, 1)])]
    assert trends[1] == "degrading"
    assert trends[-1] == "improving"
    assert any("compares this score" in n for n in monitor.last_score.notes)


def test_a_batch_missing_a_channel_counts_as_missing_data(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    batch = batches([(60, 1)])[0].drop(columns=["vibration"])
    result = monitor.update(batch)
    assert any("no column for channel" in note for note in result.notes)
    assert result.components["availability"] < 100


def test_extra_batch_columns_are_ignored_with_a_note(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    batch = batches([(60, 1)])[0].assign(pressure=1.0)
    result = monitor.update(batch)
    assert "pressure" not in result.channels
    assert any("not baseline channels" in note for note in result.notes)


def test_a_batch_without_the_time_column_still_scores(baseline_df):
    monitor = HealthMonitor(baseline_df, time="ts")
    batch = batches([(60, 1)])[0].drop(columns=["ts"])
    result = monitor.update(batch)
    assert any("no 'ts' column" in note for note in result.notes)
    assert 0 <= result.value <= 100


def test_batches_out_of_order_are_sorted(baseline_df):
    monitor_a = HealthMonitor(baseline_df, time="ts")
    monitor_b = HealthMonitor(baseline_df, time="ts")
    batch = batches([(64, 2)])[0]
    assert monitor_a.update(batch).value == pytest.approx(
        monitor_b.update(batch.sample(frac=1.0, random_state=1)).value
    )


def test_monitor_channels_argument_and_rules_together(baseline_df):
    monitor = HealthMonitor(
        baseline_df, rules={"vibration": {"max": 0.3}}, channels=["temp"], time="ts"
    )
    assert set(monitor.channels) == {"temp", "vibration"}


def test_monitor_rejects_a_rule_for_an_absent_channel(baseline_df):
    with pytest.raises(ValueError) as excinfo:
        HealthMonitor(baseline_df, rules={"pressure": {"max": 3}})
    assert "pressure" in str(excinfo.value)
    assert "available channels" in str(excinfo.value)


def test_monitor_weights_are_normalized(baseline_df):
    monitor = HealthMonitor(baseline_df, weights={"stability": 2, "compliance": 2, "anomaly": 1, "availability": 1})
    assert sum(monitor.weights.values()) == pytest.approx(1.0)
    assert any("did not sum to 1" in note for note in monitor.baseline_score.notes)


def test_monitor_summary_and_to_dict(baseline_df):
    import json

    monitor = HealthMonitor(baseline_df, rules={"temp": {"max": 65}}, time="ts")
    monitor.update(batches([(70, 2)], seed=8)[0])
    text = monitor.summary()
    assert "monitor:" in text
    payload = monitor.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["updates"] == 1
    assert payload["alerts"]
    assert payload["rules"][0]["channel"] == "temp"
    assert repr(monitor).startswith("HealthMonitor(")


def test_monitor_without_a_time_column():
    rng = np.random.default_rng(1)
    baseline = pd.DataFrame({"temp": rng.normal(5, 1, 200)})
    monitor = HealthMonitor(baseline)
    result = monitor.update(pd.DataFrame({"temp": rng.normal(5, 1, 100)}))
    assert result.when is None
    assert 0 <= result.value <= 100


def test_monitor_accepts_a_csv_path(tmp_path, baseline_df):
    path = tmp_path / "baseline.csv"
    baseline_df.to_csv(path, index=False)
    monitor = HealthMonitor(path, time="ts")
    assert monitor.baseline_score.value > 90
