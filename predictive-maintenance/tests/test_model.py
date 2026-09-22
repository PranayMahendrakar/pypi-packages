"""MaintenanceModel: the supervised failure classifier."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import predictive_maintenance as pm
from conftest import make_labelled


def test_fit_predict_round_trip(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    risk = model.predict(df)

    assert model.is_fitted is True
    assert model.horizon == 12
    assert isinstance(risk.probability, np.ndarray)
    assert risk.probability.shape == (len(df),)
    assert ((risk.probability >= 0.0) & (risk.probability <= 1.0)).all()
    assert risk.risk in ("low", "medium", "high")


def test_risk_is_high_right_before_a_failure(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    probability = model.predict(df).probability
    about_to_fail = np.asarray(labels, dtype=bool)
    assert probability[about_to_fail].mean() > probability[~about_to_fail].mean()
    assert model.predict(df).risk == "high"


def test_no_failures_in_the_labels_is_a_clear_error(labelled):
    df, labels = labelled
    with pytest.raises(ValueError) as caught:
        pm.MaintenanceModel().fit(df, pd.Series([False] * len(df)))
    message = str(caught.value)
    assert "no failures" in message
    assert "health_score()" in message


def test_all_rows_positive_is_a_clear_error(labelled):
    df, labels = labelled
    with pytest.raises(ValueError, match="every row counts as a failure"):
        pm.MaintenanceModel().fit(df, pd.Series([True] * len(df)), horizon=5)


def test_labels_must_line_up(labelled):
    df, labels = labelled
    with pytest.raises(ValueError, match="line up"):
        pm.MaintenanceModel().fit(df, labels.iloc[:-5])


def test_horizon_must_be_positive(labelled):
    df, labels = labelled
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        pm.MaintenanceModel().fit(df, labels, horizon=0)


def test_predict_before_fit_is_a_clear_error(labelled):
    df, _ = labelled
    with pytest.raises(ValueError, match="not fitted yet"):
        pm.MaintenanceModel().predict(df)


def test_feature_importance_is_ranked_and_named(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    values = list(model.feature_importance.values())
    assert values == sorted(values, reverse=True)
    assert all("__" in name for name in model.feature_importance)
    top = model.top_features(3)
    assert 0 < len(top) <= 3
    assert top[0] in model.feature_importance


def test_evaluate_returns_metrics(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    metrics = model.evaluate(df, labels)
    for key in ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"):
        assert key in metrics
    assert metrics["roc_auc"] > 0.8
    assert metrics["n_positive"] > 0
    assert 0.0 < metrics["positive_rate"] < 1.0
    json.dumps(metrics, ensure_ascii=False)


def test_evaluate_on_single_class_labels_explains_itself(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    metrics = model.evaluate(df, pd.Series([False] * len(df)))
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert any("single class" in note for note in metrics["notes"])
    json.dumps(metrics, ensure_ascii=False)


def test_deterministic_under_random_state(labelled):
    df, labels = labelled
    first = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    second = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    np.testing.assert_array_equal(first.predict(df).probability, second.predict(df).probability)
    assert first.feature_importance == second.feature_importance


def test_labels_accept_arrays_and_zero_one(labelled):
    df, labels = labelled
    as_ints = labels.astype(int)
    from_series = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=10).predict(df)
    from_array = (
        pm.MaintenanceModel(random_state=0)
        .fit(df, as_ints.to_numpy(), horizon=10)
        .predict(df)
    )
    np.testing.assert_allclose(from_series.probability, from_array.probability)


def test_labels_as_one_column_frame(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels.to_frame(), horizon=10)
    assert model.is_fitted
    with pytest.raises(ValueError, match="one column"):
        pm.MaintenanceModel().fit(df, df[["vibration", "temp_c"]])


def test_failure_risk_result_surface(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    risk = model.predict(df)

    assert 0.0 <= risk.probability_now <= 1.0
    assert risk.peak_probability >= risk.probability_now
    assert isinstance(risk.peak_at, pd.Timestamp)
    series = risk.as_series()
    assert isinstance(series, pd.Series)
    assert len(series) == len(df)
    risk.summary().encode("ascii")
    json.dumps(risk.to_dict(), ensure_ascii=False)


def test_model_summary_and_to_dict():
    df, labels = make_labelled(cycles=4, life=60)
    unfitted = pm.MaintenanceModel()
    assert "not fitted" in unfitted.summary()
    assert unfitted.to_dict()["fitted"] is False

    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=8)
    model.summary().encode("ascii")
    payload = model.to_dict()
    assert payload["fitted"] is True
    assert payload["horizon"] == 8
    assert payload["random_state"] == 0
    json.dumps(payload, ensure_ascii=False)


def test_predict_on_a_shorter_frame(labelled):
    df, labels = labelled
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    tail = df.tail(30).reset_index(drop=True)
    risk = model.predict(tail)

    assert risk.probability.shape == (30,)
    assert ((risk.probability >= 0.0) & (risk.probability <= 1.0)).all()
    # the tail ends on a failure, so the model has to say so
    assert risk.risk == "high"
    assert risk.probability_now > 0.6
    assert any("same single window" in note for note in risk.notes)


def test_labels_follow_their_rows_through_the_sort():
    """The same physical rows in a different order must give the same model."""
    df, labels = make_labelled(cycles=4, life=60)
    order = np.random.default_rng(1).permutation(len(df))
    shuffled = df.iloc[order].reset_index(drop=True)
    shuffled_labels = labels.iloc[order].reset_index(drop=True)

    ordered = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    reordered = pm.MaintenanceModel(random_state=0).fit(shuffled, shuffled_labels, horizon=12)

    np.testing.assert_allclose(
        ordered.predict(df).probability, reordered.predict(df).probability, atol=1e-12
    )
    assert ordered.feature_importance == reordered.feature_importance
    assert any("carried through the sort" in note for note in reordered.notes)


def test_a_newest_first_frame_still_finds_the_failure():
    """A descending CSV must not invert the answer."""
    n_rows = 300
    wear = np.concatenate([np.zeros(270), np.linspace(0.0, 8.0, 30)])
    ascending = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "vib": 1.0 + wear,
            "temp": 60.0 + 4 * wear,
        }
    )
    labels = pd.Series(False, index=ascending.index)
    labels.iloc[-1] = True
    descending = ascending.iloc[::-1].reset_index(drop=True)
    descending_labels = labels.iloc[::-1].reset_index(drop=True)

    from_ascending = pm.MaintenanceModel(random_state=0).fit(
        ascending, labels, horizon=15
    ).predict(ascending)
    from_descending = pm.MaintenanceModel(random_state=0).fit(
        descending, descending_labels, horizon=15
    ).predict(descending)

    assert from_ascending.risk == "high"
    assert from_descending.risk == "high"
    assert from_descending.probability_now == pytest.approx(
        from_ascending.probability_now, abs=1e-12
    )


def test_evaluate_is_not_fooled_by_unsorted_rows():
    df, labels = make_labelled(cycles=4, life=60)
    order = np.random.default_rng(2).permutation(len(df))
    model = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)

    ordered = model.evaluate(df, labels)
    reordered = model.evaluate(
        df.iloc[order].reset_index(drop=True), labels.iloc[order].reset_index(drop=True)
    )
    for key in ("roc_auc", "average_precision", "accuracy", "recall", "n_positive"):
        assert reordered[key] == pytest.approx(ordered[key])
    assert any("carried through the sort" in note for note in reordered["notes"])


def test_a_long_fit_warns_instead_of_looking_hung():
    """The slow supervised path announces itself; the cost knobs really cut it."""
    df, labels = make_labelled(cycles=5, life=60)

    quiet = pm.MaintenanceModel(random_state=0).fit(df, labels, horizon=12)
    assert not any("takes minutes" in note for note in quiet.notes)

    heavy = pm.MaintenanceModel(random_state=0, n_estimators=1200).fit(
        df, labels, horizon=12
    )
    note = [n for n in heavy.notes if "takes minutes" in n]
    assert note, heavy.notes
    assert "n_estimators" in note[0]

    cheap = pm.MaintenanceModel(random_state=0, n_estimators=40, max_depth=2).fit(
        df, labels, horizon=12
    )
    assert not any("takes minutes" in n for n in cheap.notes)
    assert len(cheap._estimator.estimators_) == 40
    assert len(quiet._estimator.estimators_) == 150
    assert cheap.predict(df).risk in ("low", "medium", "high")
