"""The one-line entry point, the result object, and the README quickstart itself."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quality_predictor import QualityResult, predict_quality

README = Path(__file__).resolve().parents[1] / "README.md"


def _quickstart_code() -> str:
    """The python block under '## Quickstart' in README.md, exactly as published."""
    text = README.read_text(encoding="utf-8")
    after = text.split("## Quickstart", 1)[1]
    return after.split("```python", 1)[1].split("```", 1)[0]


def test_readme_quickstart_runs_verbatim(capsys):
    exec(compile(_quickstart_code(), str(README), "exec"), {"__name__": "__main__"})
    printed = capsys.readouterr().out
    assert "quality-predictor: quality" in printed
    assert "WHAT DRIVES QUALITY" in printed


def test_predict_quality_returns_a_useful_result(quickstart):
    result = predict_quality(quickstart, "quality")
    assert isinstance(result, QualityResult)
    assert result.task == "classification"
    assert result.target == "quality"
    assert len(result.predictions) == len(quickstart)
    assert result.probabilities.shape == (len(quickstart), 2)
    assert set(result.classes) == {"pass", "fail"}
    assert result.good_class == "pass"
    assert result.feature_importance.sum() == pytest.approx(1.0)
    assert set(result.optimal_ranges) == {"temperature", "pressure"}
    assert any("fitted on" in note for note in result.notes)
    assert len(result.top_factors) <= 5


def test_predict_quality_with_new_data(runs):
    new_rows = runs.drop(columns=["quality"]).head(4)
    result = predict_quality(runs, "quality", new_rows)
    assert len(result.predictions) == 4
    assert not any("no new_data" in note for note in result.notes)
    explanation = result.explain()
    assert explanation.prediction == result.predictions[0]


def test_result_summary_and_serialisation(quickstart):
    result = predict_quality(quickstart, "quality")
    text = result.summary()
    for heading in ("WHAT DRIVES QUALITY", "SETTINGS MOST ASSOCIATED", "PREDICTIONS", "NOTES"):
        assert heading in text
    assert text == text.rstrip()
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["metrics"]
    assert payload["optimal_ranges"]
    frame = result.to_frame()
    assert list(frame.columns) == ["feature", "importance"]
    assert len(frame) == len(result.feature_importance)


def test_predict_quality_passes_options_through(runs):
    result = predict_quality(
        runs, "quality", features=["temperature", "pressure"], random_state=5, test_size=0.3
    )
    assert result.features == ["temperature", "pressure"]
    assert result.n_test == pytest.approx(len(runs) * 0.3, abs=1)
    assert result.model.random_state == 5


def test_predict_quality_rejects_unknown_options(quickstart):
    with pytest.raises(TypeError, match="unexpected keyword"):
        predict_quality(quickstart, "quality", n_estimators=500)


def test_accepts_a_csv_path(quickstart, tmp_path):
    path = tmp_path / "runs.csv"
    quickstart.to_csv(path, index=False, encoding="utf-8")
    result = predict_quality(str(path), "quality")
    assert result.n_rows == len(quickstart)
    assert isinstance(result.predictions, np.ndarray)


def test_regression_convenience(measurements):
    result = predict_quality(measurements, "strength")
    assert result.task == "regression"
    assert result.classes is None
    assert "r2" in result.metrics
    assert "measured value" in result.summary()
    assert isinstance(result.feature_importance, pd.Series)
