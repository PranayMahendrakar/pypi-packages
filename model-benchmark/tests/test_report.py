"""The report object: best, rank, compare, to_frame, to_dict, summary."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from model_benchmark import benchmark


def _three_models():
    def quick(rows):
        return [0] * len(rows)

    def slow(rows):
        # Same answers as `quick`, just a lot more work to get there.
        for _ in range(len(rows)):
            sum(range(3000))
        return [0] * len(rows)

    def wrong(rows):
        return [1] * len(rows)

    X = np.zeros((20, 2))
    y = np.zeros(20, dtype=int)
    return benchmark(
        {"quick": quick, "slow": slow, "wrong": wrong},
        (X, y),
        metric="accuracy",
        repeats=3,
    )


def test_best_by_latency_score_and_memory():
    report = _three_models()
    assert report.best("latency") in {"quick", "wrong"}
    assert report.best("score") in {"quick", "wrong"}
    assert report.best("memory") in report.names
    assert report.results[report.best("score")].score == pytest.approx(1.0)


def test_best_rejects_an_unknown_key():
    report = _three_models()
    with pytest.raises(ValueError, match="by must be one of"):
        report.best("vibes")
    with pytest.raises(ValueError, match="by must be one of"):
        report.rank("vibes")


def test_best_by_score_needs_a_metric():
    report = benchmark({"identity": lambda a: a}, [1, 2, 3], repeats=2)
    with pytest.raises(ValueError, match="no model has a score"):
        report.best("score")
    assert report.rank("score") == []


def test_rank_is_ordered_and_complete():
    report = _three_models()
    ranked = report.rank("latency")
    assert set(ranked) == {"quick", "slow", "wrong"}
    latencies = [report.results[name].latency_mean_ms for name in ranked]
    assert latencies == sorted(latencies)


def test_lower_is_better_metrics_flip_the_winner():
    X = np.arange(10.0).reshape(5, 2)
    y = np.zeros(5)
    report = benchmark(
        {"near": lambda a: np.full(len(a), 0.1), "far": lambda a: np.full(len(a), 9.0)},
        (X, y),
        metric="rmse",
        repeats=2,
    )
    assert report.higher_is_better is False
    assert report.best("score") == "near"
    assert report.rank("score") == ["near", "far"]


def test_compare_gives_ratios_and_a_sentence():
    report = _three_models()
    out = report.compare("slow", "quick")
    assert out["a"] == "slow" and out["b"] == "quick"
    assert out["latency_ratio"] > 1
    assert out["speedup"] == pytest.approx(1.0 / out["latency_ratio"])
    assert out["faster"] == "quick"
    assert out["score_delta"] == pytest.approx(0.0)
    assert "slower than quick" in out["summary"]
    assert out["metric_name"] == "accuracy"


def test_compare_complains_about_unknown_or_failed_models():
    report = benchmark({"ok": lambda a: a, "bad": _raiser}, [1, 2, 3], repeats=2)
    with pytest.raises(ValueError, match="no model named"):
        report.compare("ok", "ghost")
    with pytest.raises(ValueError, match="failed and has nothing to compare"):
        report.compare("ok", "bad")


def _raiser(rows):
    raise ValueError("nope")


def test_to_frame_has_one_row_per_model_in_order():
    report = _three_models()
    frame = report.to_frame()
    assert list(frame["model"]) == ["quick", "slow", "wrong"]
    assert len(frame) == len(report.results)
    assert frame["ok"].all()
    assert set(
        ["model", "ok", "latency_ms", "p50_ms", "p95_ms", "throughput_per_s", "score", "error"]
    ).issubset(frame.columns)


def test_to_frame_can_select_a_batch_size():
    report = benchmark({"identity": lambda a: a}, np.arange(50.0), batch_sizes=(1, 25), repeats=2)
    first = report.to_frame(batch_size=1)
    second = report.to_frame(batch_size=25)
    assert len(first) == len(second) == 1
    assert first.loc[0, "throughput_per_s"] != second.loc[0, "throughput_per_s"]


def test_to_dict_is_json_safe():
    report = benchmark({"ok": lambda a: a, "bad": _raiser}, [1, 2, 3], repeats=1)
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert "NaN" not in text and "Infinity" not in text
    assert payload["n_models"] == 2 and payload["n_failed"] == 1
    assert payload["results"]["bad"]["latency_mean_ms"] is None
    assert payload["best"]["latency"] == "ok"
    assert payload["best"]["score"] is None
    assert json.loads(text)["failures"]["bad"].startswith("ValueError")


def test_summary_is_plain_ascii_and_mentions_everything():
    report = _three_models()
    text = report.summary()
    assert text.isascii()
    for token in ("model-benchmark:", "fastest", "leanest", "best score", "quick", "slow"):
        assert token in text
    assert str(report) == text
    assert "BenchmarkReport(" in repr(report)


def test_summary_limit_truncates_the_table():
    report = _three_models()
    text = report.summary(limit=1)
    assert "... and 2 more" in text


def test_report_behaves_like_a_mapping():
    report = _three_models()
    assert len(report) == 3
    assert "quick" in report
    assert report["quick"].name == "quick"
    assert [result.name for result in report] == report.names
    assert set(report.ok_results) == {"quick", "slow", "wrong"}


def test_failed_models_appear_in_the_table_without_numbers():
    report = benchmark({"ok": lambda a: a, "bad": _raiser}, [1, 2, 3], repeats=1)
    frame = report.to_frame()
    assert list(frame["model"]) == ["ok", "bad"]
    assert bool(frame.loc[1, "ok"]) is False
    assert math.isnan(float(frame.loc[1, "latency_ms"]))
    assert "-" in report.summary()
