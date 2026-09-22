"""Core detection behaviour: the three stages and the contract of the report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sensor_anomaly
from sensor_anomaly import Detector, SensorReport, detect


def make_plant(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A small, healthy plant: two correlated channels and one independent."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "temp": 70 + 5 * base + rng.normal(0, 0.3, n),
            "flow": 20 + 2 * base + rng.normal(0, 0.15, n),
            "vibration": rng.normal(2.0, 0.1, n),
        }
    )


def test_quickstart_from_the_readme():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"temp": rng.normal(70, 1, 500), "flow": rng.normal(12, 0.5, 500)}
    )
    df.loc[300:340, "flow"] = 0.0
    report = detect(df)

    assert isinstance(report, SensorReport)
    assert report.n_rows == 500
    assert "stuck-at-zero" in report.channels["flow"].faults
    assert report.worst_channels[0] == "flow"
    assert report.summary()
    assert report.anomalous


def test_report_explains_itself():
    report = detect(make_plant())
    text = report.summary()
    assert "sensor-anomaly" in text
    assert "sensitivity" in text
    assert text == str(report)
    assert text.isascii(), "summary must stay printable on any console"
    assert "SensorReport(" in repr(report)


def test_to_dict_is_json_safe():
    import json

    report = detect(make_plant())
    payload = report.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert payload["n_rows"] == 600
    assert set(payload["channels"]) == {"temp", "flow", "vibration"}
    assert payload["n_events"] == report.n_events


def test_to_frame_has_one_row_per_channel():
    report = detect(make_plant())
    frame = report.to_frame()
    assert list(frame["channel"]) == report.channel_names
    assert len(frame) == 3
    assert {"status", "anomalies", "rate", "score_max", "faults"} <= set(frame.columns)


def test_events_frame_matches_events():
    report = detect(make_plant())
    frame = report.events_frame()
    assert len(frame) == report.n_events
    if report.n_events:
        assert frame.iloc[0]["start"] == report.events[0].start
    # No time column was given, so no timestamp columns are invented.
    assert "start_time" not in frame.columns


def test_a_spike_in_one_channel_is_found():
    df = make_plant()
    df.loc[300, "vibration"] = 40.0
    report = detect(df)
    assert 300 in report.channels["vibration"].anomalies
    assert report.channels["vibration"].score_max > 3.0
    assert any(e.kind == "spike" and e.start <= 300 <= e.end for e in report.events)


def test_broken_correlation_is_caught_without_any_channel_being_odd():
    """The headline case: no single reading is out of range, only the relationship."""
    rng = np.random.default_rng(7)
    n = 2000
    base = rng.normal(0, 1, n)
    a = 50 + 5 * base + rng.normal(0, 0.3, n)
    b = 20 + 2 * base + rng.normal(0, 0.15, n)
    c = rng.normal(100, 3, n)
    b[900:950] = 20 - 2 * base[900:950] + rng.normal(0, 0.15, 50)
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    # Every channel stays inside the range it shows elsewhere, so a per-channel
    # test has nothing to find.
    assert b[900:950].min() > b.min() and b[900:950].max() < b.max()

    report = detect(df)
    flagged = np.array(report.joint, dtype=int)
    assert flagged.size > 0, "the cross-channel stage found nothing"
    inside = int(((flagged >= 900) & (flagged < 950)).sum())
    assert inside / flagged.size >= 0.7, "most cross-channel flags should be in the fault"
    assert any(
        e.kind == "joint" and e.start < 950 and e.end >= 900 for e in report.events
    )


def test_clean_data_stays_quiet():
    """A healthy plant must not be buried in flags, and must say why."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({name: rng.normal(0, 1, 3000) for name in "wxyz"})
    report = detect(df)

    assert report.joint == [], "clean data must not trip the cross-channel stage"
    assert not any(res.faults for res in report.channels.values())
    for res in report.channels.values():
        assert res.rate < 0.01
    assert any("Nothing here stands out" in note for note in report.notes)


def test_higher_sensitivity_flags_less():
    df = make_plant()
    df.loc[300, "vibration"] = 8.0
    loose = detect(df, sensitivity=2.5)
    tight = detect(df, sensitivity=6.0)
    assert sum(c.n_anomalies for c in tight.channels.values()) <= sum(
        c.n_anomalies for c in loose.channels.values()
    )


def test_worst_channels_puts_broken_sensors_first():
    df = make_plant()
    df["dead"] = np.nan
    df["stuck"] = 3.0
    order = detect(df).worst_channels
    assert order[0] == "dead", "a failed sensor outranks everything"
    assert order.index("stuck") < order.index("vibration")


def test_detector_class_matches_the_function():
    df = make_plant()
    assert Detector(sensitivity=4.0).detect(df).to_dict() == detect(
        df, sensitivity=4.0
    ).to_dict()
    assert Detector()(df).n_rows == 600


def test_package_surface():
    assert sensor_anomaly.__version__ == "0.1.0"
    for name in sensor_anomaly.__all__:
        assert hasattr(sensor_anomaly, name), name


# --- regression: the report used to call clean multi-channel data anomalous -------------

def _clean(seed, n=500, k=6):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"ch{i}": rng.normal(50.0, 1.0, n) for i in range(k)})


def test_clean_multichannel_data_is_not_called_anomalous():
    """Six healthy channels used to come back with five of them flagged."""
    import sensor_anomaly as sa

    noisy = [s for s in range(20) if sa.detect(_clean(s)).anomalous]
    assert not noisy, f"clean tables reported as anomalous for seeds {noisy}"


def test_real_faults_are_still_caught():
    import sensor_anomaly as sa

    base = _clean(0)

    spike = base.copy()
    spike.loc[250:253, "ch2"] = 500.0
    assert sa.detect(spike).anomalous
    assert sa.detect(spike).channels["ch2"].anomalies

    flat = base.copy()
    flat.loc[200:, "ch3"] = 50.0
    assert "flatline" in sa.detect(flat).channels["ch3"].faults

    step = base.copy()
    step.loc[250:, "ch5"] = step.loc[250:, "ch5"] + 20.0
    assert "step-change" in sa.detect(step).channels["ch5"].faults

    zero = base.copy()
    zero.loc[100:, "ch1"] = 0.0
    assert sa.detect(zero).anomalous


def test_slow_drift_is_not_reported_as_a_step_change():
    """A channel drifting smoothly used to collect a spurious step-change fault,
    and it got worse the longer the table was."""
    import numpy as np

    import sensor_anomaly as sa

    for n in (300, 1000, 3000):
        df = _clean(1, n=n)
        df["ch4"] = np.linspace(50.0, 80.0, n) + np.random.default_rng(2).normal(0, 1.0, n)
        assert "step-change" not in sa.detect(df).channels["ch4"].faults, f"n={n}"


def test_flag_rate_on_clean_data_matches_the_documented_sensitivity():
    """sensitivity=3.0 must behave like 3 sigma, not like 2.4."""
    import numpy as np

    import sensor_anomaly as sa

    flagged = 0
    total = 0
    for seed in range(12):
        df = _clean(seed)
        report = sa.detect(df)
        for result in report.channels.values():
            flagged += len(result.anomalies)
        total += df.size
    rate = flagged / total
    # a two-sided 3-sigma normal tail is 0.0027; allow generous room for clustering
    assert rate < 0.008, f"clean flag rate {rate:.4f} is far above the 3-sigma tail"
