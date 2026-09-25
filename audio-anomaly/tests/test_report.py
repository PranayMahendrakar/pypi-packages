"""The result objects explain themselves and serialise cleanly."""

import json

import numpy as np
import pytest

import audio_anomaly
from audio_anomaly import Event
from _signals import SR, add_knock, add_tone, hum, scale_span


@pytest.fixture(scope="module")
def busy_report():
    x = add_tone(add_knock(hum(seconds=6.0, seed=1), at=1.0), start=3.0)
    x = add_knock(x, at=4.5, amp=0.2, seed=7)
    x = scale_span(x, at=2.0, seconds=0.3, factor=0.0)
    return audio_anomaly.detect(x, sample_rate=SR)


def test_event_fields_are_exactly_the_documented_five():
    e = Event(1.0, 2.0, "burst", 5.0, "a knock")
    assert (e.start_s, e.end_s, e.kind, e.score, e.message) == (1.0, 2.0, "burst", 5.0, "a knock")
    assert e.duration_s == 1.0
    assert set(e.to_dict()) == {"start_s", "end_s", "duration_s", "kind", "score", "message"}
    assert "burst at 1.000-2.000 s" in str(e)
    with pytest.raises(Exception):
        e.score = 1.0  # frozen


def test_event_coerces_numpy_scalars():
    e = Event(np.float64(1), np.float32(2), "burst", np.float64(3), "m")
    assert type(e.start_s) is float and type(e.score) is float


def test_busy_report_finds_every_planted_fault(busy_report):
    kinds = sorted({e.kind for e in busy_report.anomalies})
    assert kinds == ["burst", "dropout", "tonal"]
    starts = [e.start_s for e in busy_report.anomalies]
    assert starts == sorted(starts)


def test_loudest_orders_by_score(busy_report):
    top = busy_report.loudest(2)
    assert len(top) == 2
    assert top[0].score >= top[1].score
    assert all(top[-1].score >= e.score for e in busy_report.anomalies if e not in top)
    assert busy_report.loudest(0) == []
    assert len(busy_report.loudest(100)) == len(busy_report.anomalies)
    assert busy_report.loudest()[0] == top[0]


def test_counts_and_of_kind(busy_report):
    counts = busy_report.counts
    assert sum(counts.values()) == len(busy_report.anomalies)
    assert list(counts) == [k for k in audio_anomaly.KINDS if k in counts]
    assert len(busy_report.of_kind("burst")) == counts["burst"]
    assert busy_report.has_anomalies


def test_to_dict_is_json_safe_and_complete(busy_report):
    data = busy_report.to_dict()
    text = json.dumps(data, ensure_ascii=False)
    back = json.loads(text)
    assert back["mode"] == "self"
    assert len(back["scores"]) == len(back["frame_times"]) == busy_report.n_frames
    assert len(back["anomalies"]) == len(busy_report.anomalies)
    assert 0.0 < back["anomaly_ratio"] < 1.0
    for key in ("sample_rate", "duration_s", "frame_ms", "sensitivity", "counts", "notes", "warnings"):
        assert key in back


def test_summary_is_plain_ascii_and_explains_itself(busy_report):
    text = busy_report.summary()
    assert text.isascii()
    assert "What the kinds mean:" in text
    assert "threshold 3.0" in text
    for kind in busy_report.counts:
        assert kind in text
    assert "Normal means:" in text


def test_clean_summary_says_how_much_headroom_there_was():
    report = audio_anomaly.detect(hum(seed=2), sample_rate=SR)
    text = report.summary()
    assert "no anomalies" in text
    assert "highest frame score was" in text
    assert repr(report).startswith("AudioAnomalyReport(0 anomalies")


def test_long_lists_are_trimmed_in_the_summary():
    x = hum(seconds=12.0, seed=3)
    for k in range(20):
        x = add_knock(x, at=0.5 + 0.55 * k, amp=0.3, seed=k)
    report = audio_anomaly.detect(x, sample_rate=SR)
    assert len(report.anomalies) == 20
    assert "The 15 strongest of 20 anomalies" in report.summary()
    assert "5 more in report.anomalies" in report.summary()


def test_anomaly_ratio_matches_event_time(busy_report):
    total = 0.0
    spans = sorted((e.start_s, e.end_s) for e in busy_report.anomalies)
    current = None
    for a, b in spans:
        if current is None or a > current[1]:
            if current:
                total += current[1] - current[0]
            current = [a, b]
        else:
            current[1] = max(current[1], b)
    total += current[1] - current[0]
    assert busy_report.anomaly_ratio == pytest.approx(total / busy_report.duration_s)
