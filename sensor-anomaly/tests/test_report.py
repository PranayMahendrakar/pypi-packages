"""The result objects themselves: every public member reachable from the report."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sensor_anomaly
from sensor_anomaly import ChannelResult, Event, FAULT_HELP, FAULT_ORDER, detect


def test_every_fault_code_has_a_plain_words_explanation():
    assert set(FAULT_HELP) == set(FAULT_ORDER)
    assert all(isinstance(text, str) and text for text in FAULT_HELP.values())


def test_faults_reported_are_always_explained_in_the_notes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "good": rng.normal(0, 1, 400),
            "zeroed": rng.normal(10, 1, 400),
            "dead": [np.nan] * 400,
        }
    )
    df.loc[100:200, "zeroed"] = 0.0
    report = detect(df)

    seen = {code for res in report.channels.values() for code in res.faults}
    assert seen, "this table is meant to be faulty"
    for code in seen:
        assert code in FAULT_HELP, code
        assert any(note.startswith(code + ":") for note in report.notes), code


def test_channels_with_selects_by_fault():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 300), "frozen": [7.0] * 300})
    report = detect(df)
    assert report.channels_with("flatline") == ["frozen"]
    assert report.channels_with("railed-at-max") == []


def test_event_level_bands_and_json():
    low = Event(start=0, end=0, channels=["a"], kind="spike", severity=1.0)
    medium = Event(start=1, end=3, channels=[], kind="joint", severity=2.0)
    high = Event(start=4, end=4, channels=["b"], kind="step", severity=9.0)
    assert (low.level, medium.level, high.level) == ("low", "medium", "high")
    assert medium.n_rows == 3
    assert "no single channel" in medium.describe()
    payload = high.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["level"] == "high" and payload["start_time"] is None


def test_channel_result_defaults_and_dict():
    empty = ChannelResult()
    assert empty.ok and empty.status == "ok" and not empty.failed
    assert empty.n_anomalies == 0
    payload = ChannelResult(anomalies=[1, 2], rate=0.5, score_max=4.0, faults=["flatline"]).to_dict()
    assert payload["status"] == "faulty"
    assert payload["n_anomalies"] == 2
    assert json.loads(json.dumps(payload)) == payload


def test_a_table_with_no_numeric_columns_says_so_instead_of_failing():
    df = pd.DataFrame({"note": ["ok", "ok", "bad"], "tag": ["x", "y", "z"]})
    report = detect(df)
    assert report.channels == {}
    assert not report.anomalous
    assert "no usable sensor channels" in report.summary()
    assert report.to_dict()["n_channels"] == 0


def test_a_non_default_index_does_not_confuse_row_positions():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame(
        {"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)},
        index=pd.date_range("2026-03-01", periods=n, freq="min"),
    )
    df.iloc[150, 0] = 40.0
    report = detect(df)
    assert 150 in report.channels["a"].anomalies
    assert report.n_rows == n


def test_summary_stays_ascii_for_ascii_channel_names():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": [0.0] * 300})
    text = detect(df).summary()
    assert text.isascii(), "summary must survive any console encoding"
    assert "notes:" in text


def test_limiting_how_many_events_the_summary_prints():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(0, 1, 800), "b": rng.normal(0, 1, 800)})
    df.loc[100:130, "a"] = 0.0
    report = detect(df)
    assert report.n_events > 1
    short = report.summary(max_events=1)
    assert "more" in short
    assert len(short) < len(report.summary(max_events=100))


def test_report_repr_and_names():
    rng = np.random.default_rng(5)
    report = detect(pd.DataFrame({"z": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200)}))
    assert report.channel_names == ["z", "y"]
    assert sorted(report.worst_channels) == ["y", "z"]
    assert "SensorReport(channels=2" in repr(report)
    assert sensor_anomaly.__doc__
