"""The streaming path: learn a baseline once, score later batches unchanged."""
import numpy as np
import pandas as pd
import pytest

from timeseries_anomaly import Detector

HISTORY = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 9.9] * 5


def test_fit_then_score_flags_the_new_spike():
    detector = Detector(method="rolling", sensitivity=3.0).fit(HISTORY)
    result = detector.score([10.0, 10.1, 25.0, 9.9])
    assert result.anomalies == [2]
    assert result.n_points == 4
    assert detector.is_fitted
    assert detector.method_ == "rolling"
    assert detector.n_reference_ == len(HISTORY)
    assert detector.scale_ is not None
    assert detector.scale_kind_ in ("mad", "iqr", "std")
    assert detector.level_ == pytest.approx(10.0, abs=0.5)
    assert isinstance(detector.warnings_, list)
    assert any("frozen on" in note for note in result.warnings)


def test_the_baseline_does_not_move_between_batches():
    detector = Detector(method="zscore").fit(HISTORY)
    level, scale = detector.level_, detector.scale_
    quiet = detector.score([10.0, 10.1, 9.9, 10.0])
    loud = detector.score([40.0, 41.0, 39.0, 40.0])
    assert (detector.level_, detector.scale_) == (level, scale)
    assert quiet.n_anomalies == 0
    assert loud.n_anomalies == 4
    assert quiet.expected.tolist() == pytest.approx([level] * 4)


def test_scoring_never_peeks_at_later_points():
    detector = Detector(method="rolling").fit(HISTORY)
    prefix = detector.score([10.0, 25.0, 10.0])
    longer = detector.score([10.0, 25.0, 10.0, 10.0, 25.0, 10.0])
    assert np.allclose(prefix.scores, longer.scores[:3])
    assert np.allclose(prefix.expected, longer.expected[:3])
    assert prefix.anomalies == [1]
    assert longer.anomalies == [1, 4]


def test_scoring_is_deterministic():
    detector = Detector().fit(HISTORY)
    batch = [10.0, 30.0, 10.0, 9.8]
    assert detector.score(batch).to_dict() == detector.score(batch).to_dict()


def test_a_seasonal_baseline_keeps_its_phase():
    index = pd.date_range("2026-01-01", periods=96, freq="h")
    wobble = np.random.default_rng(0).normal(0.0, 0.2, 96)
    values = 20 + 5 * np.sin(2 * np.pi * np.arange(96) / 24.0) + wobble
    reference = pd.Series(values, index=index)
    detector = Detector(method="seasonal", seasonality=24).fit(reference)
    assert detector.method_ == "seasonal"
    following = 20 + 5 * np.sin(2 * np.pi * np.arange(96, 120) / 24.0)
    following[10] = 60.0
    result = detector.score(list(following))
    assert 10 in result.anomalies


def test_detector_one_shot_detect_matches_the_function():
    from timeseries_anomaly import detect

    values = [1.0, 1.1, 0.9, 1.0, 9.0, 1.0, 1.1, 0.9, 1.0, 1.2]
    assert Detector(method="zscore").detect(values).anomalies == detect(
        values, method="zscore"
    ).anomalies


def test_detector_accepts_a_dataframe_reference():
    frame = pd.DataFrame(
        {"t": pd.date_range("2026-01-01", periods=50, freq="min"), "v": HISTORY}
    )
    detector = Detector(value="v", time="t", method="zscore").fit(frame)
    assert detector.n_reference_ == 50
    assert detector.score([10.0, 50.0]).anomalies == [1]


def test_all_cannot_be_frozen_so_it_steps_down_and_says_so():
    detector = Detector(method="all").fit(HISTORY)
    assert detector.method_ in ("rolling", "zscore")
    assert any("only works on a whole series" in note for note in detector.warnings_)


def test_scoring_an_empty_batch_is_empty_not_an_error():
    detector = Detector().fit(HISTORY)
    result = detector.score([])
    assert result.n_points == 0
    assert result.anomalies == []


def test_unfitted_and_empty_reference_raise_clear_errors():
    with pytest.raises(ValueError, match="not fitted yet"):
        Detector().score([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not fitted yet"):
        _ = Detector().method_
    with pytest.raises(ValueError, match="empty reference"):
        Detector().fit([])
    with pytest.raises(ValueError, match="unknown method"):
        Detector(method="magic")


def test_a_flat_reference_scores_nothing():
    detector = Detector(method="zscore").fit([7.0] * 40)
    result = detector.score([7.0, 7.0, 100.0])
    assert detector.scale_ is None
    assert result.n_anomalies == 0
    assert any("no point can be an anomaly" in note for note in result.warnings)
