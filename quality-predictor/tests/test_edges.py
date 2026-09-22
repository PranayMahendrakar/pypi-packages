"""The awkward data: one class, tiny frames, unseen categories, empty columns, unicode."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quality_predictor import QualityModel, predict_quality


def _tiny(rows: int = 12) -> pd.DataFrame:
    """A small pass/fail frame where temperature decides the outcome."""
    temps = [180 + i * 5 for i in range(rows)]
    return pd.DataFrame(
        {
            "temperature": temps,
            "pressure": [8.0 + i * 0.4 for i in range(rows)],
            "machine": ["A" if i % 2 else "B" for i in range(rows)],
            "quality": ["fail" if t < 180 + (rows // 2) * 5 else "pass" for t in temps],
        }
    )


def test_single_class_target_raises_a_clear_error():
    df = _tiny()
    df["quality"] = "pass"
    with pytest.raises(ValueError, match="only one distinct class"):
        QualityModel().fit(df, "quality")


def test_single_value_regression_target_raises():
    df = _tiny()
    df["strength"] = 42.0
    with pytest.raises(ValueError, match="only one distinct"):
        QualityModel(task="regression").fit(df, "strength")


def test_few_rows_still_fit_but_record_a_warning(caplog):
    df = _tiny(rows=10)
    with caplog.at_level("WARNING", logger="quality_predictor._model"):
        model = QualityModel().fit(df, "quality")
    assert model.fitted
    assert len(model.predict(df)) == 10
    assert any("unreliable" in note for note in model.notes)
    assert any("unreliable" in record.message for record in caplog.records)


def test_unseen_categories_do_not_crash_at_predict_time():
    model = QualityModel().fit(_tiny(), "quality")
    fresh = pd.DataFrame(
        {"temperature": [215.0], "pressure": [12.0], "machine": ["Z-brand-new"]}
    )
    prediction = model.predict(fresh)
    assert len(prediction) == 1
    proba = model.predict_proba(fresh)
    assert proba.shape == (1, 2)
    assert model.explain(fresh).prediction == prediction[0]


def test_predict_before_fit_raises_a_clear_error():
    model = QualityModel()
    for call in (
        lambda: model.predict(_tiny()),
        lambda: model.predict_proba(_tiny()),
        lambda: model.evaluate(),
        lambda: model.explain(0),
        lambda: model.optimal_ranges(),
        lambda: model.feature_importance,
        lambda: model.save("never-written.pkl"),
    ):
        with pytest.raises(ValueError, match="not fitted yet"):
            call()


def test_empty_dataframe():
    empty = pd.DataFrame({"temperature": [], "quality": []})
    with pytest.raises(ValueError, match="no rows"):
        QualityModel().fit(empty, "quality")


def test_single_row():
    one = _tiny().head(1)
    with pytest.raises(ValueError, match="only one distinct"):
        QualityModel().fit(one, "quality")


def test_two_rows_fit_without_a_holdout():
    df = _tiny().iloc[[0, -1]]
    model = QualityModel().fit(df, "quality")
    assert model.fitted
    assert model.n_test in (0, 1)
    assert len(model.predict(df)) == 2


def test_all_nan_column_is_skipped():
    df = _tiny()
    df["dead_sensor"] = np.nan
    model = QualityModel().fit(df, "quality")
    assert "dead_sensor" not in model.features
    assert any("no values at all" in note for note in model.notes)
    assert "dead_sensor" not in model.feature_importance.index


def test_missing_values_are_imputed_not_dropped():
    df = _tiny(rows=16)
    df.loc[0:3, "pressure"] = np.nan
    df.loc[4:5, "machine"] = None
    model = QualityModel().fit(df, "quality")
    assert model.n_rows == 16
    assert len(model.predict(df)) == 16


def test_rows_with_no_target_are_left_out():
    df = _tiny(rows=14)
    df.loc[0, "quality"] = np.nan
    model = QualityModel().fit(df, "quality")
    assert model.n_rows == 13
    assert any("had no quality" in note for note in model.notes)


def test_mixed_dtypes_including_bool_and_datetime():
    rows = 16
    df = pd.DataFrame(
        {
            "temperature": [180.0 + i * 4 for i in range(rows)],
            "recalibrated": [bool(i % 2) for i in range(rows)],
            "shift_start": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "machine": pd.Categorical(["A" if i % 3 else "B" for i in range(rows)]),
            "batch": ["b{0}".format(i) for i in range(rows)],
            "quality": [i >= rows // 2 for i in range(rows)],
        }
    )
    model = QualityModel().fit(df, "quality")
    assert model.task == "classification"
    assert set(model.classes_) == {False, True}
    assert "recalibrated" in model.numeric_features
    assert "shift_start" in model.numeric_features
    assert "machine" in model.categorical_features
    assert len(model.predict(df)) == rows


def test_unicode_columns_and_values(unicode_runs):
    result = predict_quality(unicode_runs, "qualité")
    text = result.summary()
    assert "température" in text
    assert "qualité" in text
    assert result.explain().summary()
    assert "北京" in result.explain(unicode_runs.iloc[[1]]).summary()
    assert "qualité" in json.dumps(result.to_dict(), ensure_ascii=False)


def test_duplicate_column_names_are_named_plainly():
    df = pd.DataFrame([[1, 2, "pass"], [3, 4, "fail"]], columns=["a", "a", "quality"])
    with pytest.raises(ValueError, match="duplicate column names"):
        QualityModel().fit(df, "quality")


def test_target_not_in_data():
    with pytest.raises(ValueError, match="is not in the data"):
        QualityModel().fit(_tiny(), "nope")


def test_unknown_feature_column():
    with pytest.raises(ValueError, match="not in the data"):
        QualityModel().fit(_tiny(), "quality", features=["temperature", "ghost"])


def test_no_features_left():
    df = pd.DataFrame({"quality": ["pass", "fail", "pass", "fail"]})
    with pytest.raises(ValueError, match="at least one parameter column"):
        QualityModel().fit(df, "quality")


def test_bad_task_name():
    with pytest.raises(ValueError, match="task must be one of"):
        QualityModel(task="clustering")


def test_regression_task_on_text_target():
    with pytest.raises(ValueError, match="needs a numeric target"):
        QualityModel(task="regression").fit(_tiny(), "quality")


def test_predict_proba_on_a_regression_model(fitted_regression, measurements):
    with pytest.raises(ValueError, match="only available for classification"):
        fitted_regression.predict_proba(measurements)


def test_predict_with_a_missing_column(fitted, runs):
    with pytest.raises(ValueError, match="not in the data"):
        fitted.predict(runs.drop(columns=["speed"]))


def test_predict_on_an_empty_frame(fitted, runs):
    with pytest.raises(ValueError, match="nothing to predict"):
        fitted.predict(runs.head(0))


def test_bad_test_size():
    with pytest.raises(ValueError, match="fraction between 0 and 1"):
        QualityModel().fit(_tiny(), "quality", test_size=1.5)


def test_unknown_good_class():
    with pytest.raises(ValueError, match="not one of the classes"):
        QualityModel().fit(_tiny(), "quality", good_class="perfect")


def test_target_listed_as_a_feature_is_dropped():
    model = QualityModel().fit(
        _tiny(), "quality", features=["temperature", "quality"]
    )
    assert model.features == ["temperature"]
    assert any("was listed as a feature" in note for note in model.notes)


def test_whole_number_target_reads_as_classes():
    rows = 20
    df = pd.DataFrame(
        {
            "temperature": [180 + i * 3 for i in range(rows)],
            "grade": [1 if i < rows // 2 else 3 for i in range(rows)],
        }
    )
    model = QualityModel().fit(df, "grade")
    assert model.task == "classification"
    assert any("whole numbers" in note or "distinct value" in note for note in model.notes)


def test_load_rejects_a_file_that_is_not_a_model(tmp_path):
    import pickle

    path = tmp_path / "not-a-model.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"model": 17}, handle)
    with pytest.raises(ValueError, match="does not hold a QualityModel"):
        QualityModel.load(path)
    with pytest.raises(FileNotFoundError):
        QualityModel.load(tmp_path / "missing.pkl")


def test_unsupported_input_type():
    with pytest.raises(TypeError, match="expected a pandas DataFrame"):
        QualityModel().fit(12345, "quality")
    with pytest.raises(ValueError, match="unsupported file type"):
        QualityModel().fit("runs.xlsx", "quality")
