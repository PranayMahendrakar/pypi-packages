"""The main path: fit, predict, score, explain, and the parameter windows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quality_predictor import Explanation, QualityModel
from quality_predictor._metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS


def test_fit_returns_self_and_detects_classification(fitted, runs):
    assert fitted.task == "classification"
    assert fitted.target == "quality"
    assert fitted.n_rows == len(runs)
    assert fitted.n_train + fitted.n_test == len(runs)
    assert fitted.n_test > 0
    assert set(fitted.classes_) == {"pass", "fail"}
    assert fitted.good_class == "pass"


def test_metrics_are_recorded_during_fit(fitted):
    assert set(fitted.metrics) == set(CLASSIFICATION_METRICS)
    assert fitted.metrics["accuracy"] > 0.6
    assert fitted.evaluate() == fitted.metrics


def test_predict_and_predict_proba(fitted, runs):
    predictions = fitted.predict(runs)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(runs)
    assert set(predictions.tolist()) <= {"pass", "fail"}

    proba = fitted.predict_proba(runs)
    assert proba.shape == (len(runs), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # the column order follows classes_, and the winning column is the prediction
    winners = [fitted.classes_[i] for i in proba.argmax(axis=1)]
    assert winners == predictions.tolist()


def test_evaluate_on_fresh_rows(fitted, runs):
    scores = fitted.evaluate(runs.head(40))
    assert set(scores) == set(CLASSIFICATION_METRICS)
    assert 0.0 <= scores["accuracy"] <= 1.0


def test_feature_importance_uses_original_column_names(fitted, runs):
    importance = fitted.feature_importance
    assert isinstance(importance, pd.Series)
    # one row per ORIGINAL column: 'machine' has three levels but appears once
    assert list(importance.index) != []
    assert set(importance.index) == {"temperature", "pressure", "speed", "machine", "operator"}
    assert "machine_line-1" not in importance.index
    assert not any("_line-" in str(name) for name in importance.index)
    assert importance.is_monotonic_decreasing
    assert importance.sum() == pytest.approx(1.0)
    # the two parameters that actually decide the outcome lead the ranking
    assert set(importance.head(2).index) == {"temperature", "pressure"}


def test_feature_importance_sums_one_hot_levels_back_together():
    rows = 60
    machine = ["m{0}".format(i % 6) for i in range(rows)]
    df = pd.DataFrame(
        {
            "machine": machine,
            "temperature": [180 + (i % 6) * 8 for i in range(rows)],
            "quality": ["pass" if (i % 6) < 3 else "fail" for i in range(rows)],
        }
    )
    model = QualityModel().fit(df, "quality")
    importance = model.feature_importance
    assert list(importance.index) == sorted(importance.index, key=lambda n: -importance[n])
    assert set(importance.index) == {"machine", "temperature"}
    assert importance["machine"] > 0


def test_explain_one_row(fitted, runs):
    explanation = fitted.explain({
        "temperature": 225.0,
        "pressure": 13.0,
        "speed": 60.0,
        "machine": "line-1",
        "operator": "ana",
    })
    assert isinstance(explanation, Explanation)
    assert explanation.prediction in ("pass", "fail")
    assert set(explanation.contributions) == set(fitted.features)
    assert all(isinstance(v, float) for v in explanation.contributions.values())
    assert 0.0 <= explanation.score <= 1.0
    text = explanation.summary()
    assert "WHAT DROVE IT" in text
    assert "temperature" in text
    json.dumps(explanation.to_dict())
    # good settings push the score toward the good class
    assert explanation.contributions["temperature"] > 0


def test_explain_accepts_several_row_shapes(fitted, runs):
    from_series = fitted.explain(runs.iloc[0])
    from_frame = fitted.explain(runs.iloc[[0]])
    from_index = fitted.explain(0)
    assert from_series.prediction == from_frame.prediction
    assert isinstance(from_index, Explanation)
    assert fitted.explain().prediction is not None


def test_optimal_ranges(fitted):
    ranges = fitted.optimal_ranges()
    assert set(ranges) == {"temperature", "pressure", "speed"}
    for name, bounds in ranges.items():
        low, high = bounds
        assert low <= high
    low, high = ranges["temperature"]
    # the data passes above 205 degrees, so the window must sit in the top half
    assert high > 205.0
    assert fitted.optimal_ranges() == ranges


def test_regression_path(fitted_regression, measurements):
    assert fitted_regression.task == "regression"
    assert set(fitted_regression.metrics) == set(REGRESSION_METRICS)
    assert fitted_regression.metrics["r2"] > 0.5
    predictions = fitted_regression.predict(measurements.head(5))
    assert predictions.dtype.kind == "f"
    assert len(predictions) == 5
    ranges = fitted_regression.optimal_ranges()
    assert "temperature" in ranges
    explanation = fitted_regression.explain(0)
    assert explanation.good_class is None
    assert "predicted as" in explanation.summary()


def test_regression_lower_is_better(measurements):
    model = QualityModel().fit(measurements, "strength", higher_is_better=False)
    ranges = model.optimal_ranges()
    low, high = ranges["temperature"]
    assert low < 205.0
    assert any("lower" in note for note in model.notes)


def test_deterministic_under_random_state(runs):
    first = QualityModel(random_state=3).fit(runs, "quality")
    second = QualityModel(random_state=3).fit(runs, "quality")
    assert np.array_equal(first.predict(runs), second.predict(runs))
    assert first.metrics == second.metrics
    assert first.feature_importance.equals(second.feature_importance)


def test_save_and_load_reproduce_identical_predictions(fitted, runs, tmp_path):
    path = fitted.save(tmp_path / "models" / "quality.pkl")
    assert path.exists()
    reloaded = QualityModel.load(path)
    assert np.array_equal(reloaded.predict(runs), fitted.predict(runs))
    assert np.allclose(reloaded.predict_proba(runs), fitted.predict_proba(runs))
    assert reloaded.metrics == fitted.metrics
    assert reloaded.feature_importance.equals(fitted.feature_importance)
    assert reloaded.optimal_ranges() == fitted.optimal_ranges()


def test_features_argument_limits_the_columns(runs):
    model = QualityModel().fit(runs, "quality", features=["temperature", "pressure"])
    assert model.features == ["temperature", "pressure"]
    assert set(model.feature_importance.index) == {"temperature", "pressure"}


def test_forced_task_and_good_class(runs):
    model = QualityModel(task="classification").fit(runs, "quality", good_class="fail")
    assert model.good_class == "fail"
    assert any("task was set" in note for note in model.notes)


def test_model_to_dict_is_json_safe(fitted):
    payload = fitted.to_dict()
    json.dumps(payload)
    assert payload["target"] == "quality"
    assert payload["task"] == "classification"
    assert payload["feature_importance"]
