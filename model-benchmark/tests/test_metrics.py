"""The built-in metrics, and custom callables."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from model_benchmark import METRICS, benchmark
from model_benchmark._metrics import accuracy, f1, mae, r2, resolve_metric, rmse


def test_every_metric_name_is_registered():
    assert sorted(METRICS) == ["accuracy", "f1", "mae", "r2", "rmse"]


def test_accuracy():
    assert accuracy([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 1]) == pytest.approx(0.5)
    assert math.isnan(accuracy([], []))


def test_accuracy_handles_text_labels():
    assert accuracy(["oui", "non"], ["oui", "non"]) == pytest.approx(1.0)
    assert accuracy(pd.Series(["a", "b"]), np.array(["a", "c"])) == pytest.approx(0.5)


def test_f1_binary_and_multiclass():
    assert f1([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert f1([1, 1, 0, 0], [0, 0, 0, 0]) == pytest.approx(0.0)
    macro = f1([0, 1, 2, 0], [0, 1, 2, 0])
    assert macro == pytest.approx(1.0)
    assert 0.0 <= f1([0, 1, 2, 2], [0, 1, 1, 2]) <= 1.0


def test_regression_metrics():
    assert rmse([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert mae([0.0, 0.0], [3.0, 5.0]) == pytest.approx(4.0)
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r2_survives_a_constant_truth():
    assert r2([2.0, 2.0, 2.0], [2.0, 2.0, 2.0]) == pytest.approx(1.0)
    assert r2([2.0, 2.0, 2.0], [1.0, 3.0, 2.0]) == pytest.approx(0.0)


def test_resolve_metric_accepts_names_callables_and_none():
    name, function, higher = resolve_metric("ACCURACY")
    assert name == "accuracy" and higher is True and function is accuracy

    name, function, higher = resolve_metric("rmse")
    assert name == "rmse" and higher is False

    def custom(y_true, y_pred):
        return 1.0

    name, function, higher = resolve_metric(custom)
    assert name == "custom" and higher is True and function is custom

    assert resolve_metric(None) == (None, None, True)


def test_resolve_metric_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown metric"):
        resolve_metric("perplexity")
    with pytest.raises(TypeError, match="metric must be"):
        resolve_metric(7)


def test_custom_metric_can_declare_that_lower_is_better():
    def loss(y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    loss.higher_is_better = False
    X = np.zeros((4, 2))
    y = np.zeros(4)
    report = benchmark(
        {"near": lambda a: np.full(len(a), 0.1), "far": lambda a: np.full(len(a), 5.0)},
        (X, y),
        metric=loss,
        repeats=2,
    )
    assert report.metric_name == "loss"
    assert report.higher_is_better is False
    assert report.best("score") == "near"


def test_a_metric_that_raises_is_reported_not_propagated():
    def explodes(y_true, y_pred):
        raise RuntimeError("metric is broken")

    report = benchmark({"model": lambda a: a}, ([1, 2, 3], [1, 2, 3]), metric=explodes, repeats=1)
    result = report.results["model"]
    assert result.ok
    assert result.score is None
    assert "RuntimeError" in result.score_error
    assert any("could not score" in note for note in result.warnings)


def test_metric_requires_the_data_tuple():
    with pytest.raises(ValueError, match=r"data must be the tuple"):
        benchmark({"m": lambda a: a}, [1, 2, 3], metric="accuracy")
