"""The awkward cases: breakage, wrong shapes, one repeat, timeouts, odd data."""
from __future__ import annotations

import json
import math
import threading

import numpy as np
import pandas as pd
import pytest

from model_benchmark import benchmark, compare


# --------------------------------------------------------------- broken models
def test_a_model_that_raises_is_recorded_and_the_run_continues():
    def broken(rows):
        raise ValueError("boom")

    report = benchmark(
        {"first": lambda d: d, "broken": broken, "last": lambda d: list(d)},
        [1, 2, 3],
        repeats=2,
    )
    assert set(report.failures) == {"broken"}
    assert report.failures["broken"] == "ValueError: boom"
    assert report.results["broken"].ok is False
    assert report.results["broken"].timed_out is False
    assert report.results["first"].ok and report.results["last"].ok
    assert report.names == ["first", "broken", "last"]
    assert "boom" in report.summary()


def test_a_model_that_only_fails_during_warmup_is_still_a_failure():
    calls = {"n": 0}

    def flaky(rows):
        calls["n"] += 1
        raise KeyError("missing feature")

    report = benchmark({"flaky": flaky}, [1], warmup=1, repeats=1)
    assert calls["n"] == 1  # it stopped at the first warmup call
    assert "KeyError" in report.failures["flaky"]
    assert report.ok_results == {}


def test_every_model_failing_leaves_a_readable_report():
    def broken(rows):
        raise RuntimeError("nope")

    report = benchmark({"a": broken, "b": broken}, [1], repeats=1)
    assert len(report.failures) == 2
    with pytest.raises(ValueError, match="no model produced measurements"):
        report.best("latency")
    assert report.summary().isascii()


# ------------------------------------------------------------------ bad shapes
def test_wrong_number_of_predictions_is_explained_not_crashed_on():
    report = benchmark(
        {"short": lambda a: [0, 0, 0]},
        (np.zeros((5, 2)), np.zeros(5)),
        metric="accuracy",
        repeats=1,
    )
    result = report.results["short"]
    assert result.ok  # the timings are still valid
    assert result.score is None
    assert result.score_error == "returned 3 predictions for 5 inputs"
    assert result.n_predictions == 3
    assert "3 predictions for 5 inputs" in report.summary()
    assert report.failures == {}


def test_a_two_dimensional_prediction_is_explained():
    report = benchmark(
        {"probabilities": lambda a: np.zeros((len(a), 3))},
        (np.zeros((4, 2)), np.zeros(4)),
        metric="accuracy",
        repeats=1,
    )
    result = report.results["probabilities"]
    assert result.score is None
    assert "argmax" in result.score_error
    assert "shape" in result.score_error


def test_a_prediction_with_no_length_is_explained():
    report = benchmark(
        {"scalar": lambda a: 7},
        (np.zeros((4, 2)), np.zeros(4)),
        metric="accuracy",
        repeats=1,
    )
    result = report.results["scalar"]
    assert result.score is None
    assert "no length" in result.score_error


# ------------------------------------------------------------- tiny settings
def test_one_repeat_still_produces_statistics():
    report = benchmark({"identity": lambda a: a}, [1, 2, 3], repeats=1, warmup=0)
    result = report.results["identity"]
    assert result.repeats == 1
    for value in (
        result.latency_mean_ms,
        result.latency_p50_ms,
        result.latency_p95_ms,
        result.latency_min_ms,
        result.latency_max_ms,
    ):
        assert not math.isnan(value)
    assert result.latency_std_ms == pytest.approx(0.0)
    assert result.latency_mean_ms == pytest.approx(result.latency_p50_ms)
    assert result.throughput_per_s > 0
    assert math.isfinite(result.throughput_per_s)


def test_zero_warmup_is_allowed():
    report = benchmark({"identity": lambda a: a}, [1], warmup=0, repeats=1)
    assert report.warmup == 0
    assert report.results["identity"].ok


def test_bad_settings_are_rejected_clearly():
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        benchmark({"m": lambda a: a}, [1], repeats=0)
    with pytest.raises(ValueError, match="warmup cannot be negative"):
        benchmark({"m": lambda a: a}, [1], warmup=-1)
    with pytest.raises(ValueError, match="timeout must be positive"):
        benchmark({"m": lambda a: a}, [1], timeout=0)
    with pytest.raises(ValueError, match="batch size 0 is not positive"):
        benchmark({"m": lambda a: a}, [1], batch_sizes=(0,))
    with pytest.raises(ValueError, match="models is empty"):
        benchmark({}, [1])
    with pytest.raises(TypeError, match="models must be a dict"):
        benchmark("not models", [1])
    with pytest.raises(ValueError, match="duplicate model names"):
        benchmark([("a", lambda d: d), ("a", lambda d: d)], [1])


# ----------------------------------------------------------------- timeouts
def test_a_hanging_model_is_cancelled_and_recorded_as_timed_out():
    release = threading.Event()

    def hangs(rows):
        release.wait(timeout=30)
        return rows

    try:
        report = benchmark(
            {"fine": lambda d: d, "hangs": hangs, "also_fine": lambda d: list(d)},
            [1, 2, 3],
            repeats=2,
            timeout=0.2,
        )
    finally:
        release.set()

    assert report.failures["hangs"] == "timed out after 0.2 s"
    assert report.results["hangs"].timed_out is True
    assert report.results["hangs"].ok is False
    assert report.results["fine"].ok and report.results["also_fine"].ok
    assert any("background thread" in note for note in report.warnings)
    assert "timed out" in report.summary()


def test_a_generous_timeout_does_not_disturb_a_healthy_model():
    report = benchmark({"identity": lambda a: a}, [1, 2, 3], repeats=2, timeout=30)
    assert report.failures == {}
    assert report.results["identity"].ok
    assert report.timeout == 30.0
    assert "timeout" in report.summary()


# --------------------------------------------------------------- awkward data
def test_empty_dataframe():
    frame = pd.DataFrame({"x": pd.Series([], dtype=float)})
    report = benchmark({"count": lambda d: [0] * len(d)}, frame, repeats=2)
    assert report.n_samples == 0
    assert report.results["count"].ok
    assert report.summary().isascii()


def test_empty_dataframe_with_a_metric():
    frame = pd.DataFrame({"x": pd.Series([], dtype=float)})
    report = benchmark(
        {"count": lambda d: np.zeros(len(d), dtype=int)},
        (frame, np.array([], dtype=int)),
        metric="accuracy",
        repeats=1,
    )
    assert report.results["count"].ok
    assert report.to_dict()["results"]["count"]["score"] is None  # NaN never leaks into JSON


def test_single_row():
    frame = pd.DataFrame({"x": [1.0]})
    report = benchmark({"first": lambda d: d["x"].tolist()}, frame, repeats=2)
    assert report.n_samples == 1
    assert report.results["first"].ok


def test_all_nan_column():
    frame = pd.DataFrame({"x": [float("nan")] * 8})
    report = benchmark({"mean": lambda d: [float(d["x"].mean())] * len(d)}, frame, repeats=2)
    assert report.results["mean"].ok
    assert report.failures == {}


def test_mixed_dtypes():
    frame = pd.DataFrame(
        {
            "number": [1, 2, 3, 4],
            "amount": [0.5, 1.5, 2.5, 3.5],
            "label": ["a", "b", "c", "d"],
            "flag": [True, False, True, False],
        }
    )
    report = benchmark({"widths": lambda d: [len(str(v)) for v in d["label"]]}, frame, repeats=2)
    assert report.results["widths"].ok


def test_unicode_text_in_data_and_model_names():
    frame = pd.DataFrame({"texte": ["café", "naïve", "日本語", "Здравствуй"]})
    report = benchmark(
        {
            "modèle-rapide": lambda d: [len(v) for v in d["texte"]],
            "モデル": lambda d: [v.upper() for v in d["texte"]],
        },
        frame,
        repeats=2,
    )
    assert report.failures == {}
    text = report.summary()
    assert "modèle-rapide" in text and "モデル" in text
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "モデル" in payload
    assert report.best("latency") in {"modèle-rapide", "モデル"}


def test_a_single_opaque_input_is_passed_whole():
    report = benchmark(
        {"length": lambda text: len(text)}, "a prompt that is not a table", repeats=2
    )
    assert report.n_samples is None
    assert report.batch_sizes == (1,)
    assert report.results["length"].ok


def test_batch_sizes_are_ignored_for_an_opaque_input_and_a_warning_says_so():
    report = benchmark({"length": lambda text: len(text)}, "prompt", batch_sizes=(1, 8), repeats=1)
    assert report.batch_sizes == (1,)
    assert any("passed whole" in note for note in report.warnings)


def test_a_dict_input_is_one_input_not_one_row_per_key():
    report = benchmark({"keys": lambda payload: list(payload)}, {"a": 1, "b": 2}, repeats=2)
    assert report.n_samples is None
    assert report.results["keys"].ok


def test_compare_helper_on_awkward_data():
    frame = pd.DataFrame({"x": []})
    table = compare({"m": lambda d: []}, frame, repeats=1)
    assert len(table) == 1


def test_duplicate_column_names_are_refused_by_name():
    frame = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "a"])
    with pytest.raises(ValueError, match="duplicate column names: a"):
        benchmark({"identity": lambda d: d}, frame, repeats=1, warmup=0)
