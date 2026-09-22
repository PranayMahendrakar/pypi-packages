"""Regressions for the problems the independent review round raised.

Each test names the symptom a reader would have seen, so a future change that
brings one back fails with the complaint rather than with a diff.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quality_predictor import QualityModel, predict_quality


def _noise_frame(seed: int = 2, n: int = 200) -> pd.DataFrame:
    """Two real drivers and one sensor that drives nothing at all."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "temperature": rng.uniform(180, 230, n),
            "pressure": rng.uniform(8, 14, n),
            "ambient_rh": rng.uniform(20, 80, n),
        }
    )
    df["strength"] = 0.45 * df.temperature + 1.8 * df.pressure + rng.normal(0, 0.5, n)
    return df


def _span(values: pd.Series) -> float:
    """How wide the observed range of a parameter is."""
    return float(values.max() - values.min())


# ---------------------------------------------------- major: bogus setpoints


def test_a_parameter_that_drives_nothing_gets_no_setpoint():
    """A pure-noise column must never be printed as a window to hold.

    The old code scaled the keep band to the curve's own wobble, so a sensor with
    0.08% importance collapsed onto a single grid point and read as "hold ambient
    humidity at 21.14".
    """
    df = _noise_frame()
    result = predict_quality(df, "strength")

    low, high = result.optimal_ranges["ambient_rh"]
    span = _span(df["ambient_rh"])
    assert high > low, "a noise column was given a zero-width window"
    assert (high - low) / span > 0.5, "a noise column was given a narrow setpoint"

    assert "ambient_rh" in result.model.uninformative_parameters
    assert any(
        "ambient_rh" in note and "too little" in note for note in result.notes
    ), "the degenerate window was not reported in NOTES"
    text = result.summary()
    assert "ambient_rh" in text
    assert "too little to call" in text

    # the parameters that really drive the outcome keep their honest windows
    assert not {"temperature", "pressure"} & set(result.model.uninformative_parameters)
    t_low, t_high = result.optimal_ranges["temperature"]
    assert (t_high - t_low) / _span(df["temperature"]) < 0.5
    assert t_high > 205.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("higher_is_better", [True, False])
def test_noise_column_never_degenerates_across_seeds(seed, higher_is_better):
    """The review swept seeds 0-5 both directions; 5 of 12 runs were zero-width."""
    df = _noise_frame(seed=seed)
    model = QualityModel(random_state=0).fit(
        df, "strength", higher_is_better=higher_is_better
    )
    low, high = model.optimal_ranges()["ambient_rh"]
    assert (high - low) / _span(df["ambient_rh"]) > 0.5
    assert "ambient_rh" in model.uninformative_parameters
    assert any("too little" in note for note in model.notes)


def test_classification_noise_column_is_flagged_not_silently_listed():
    """The mirror-image symptom: a full-range window printed with no warning."""
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame(
        {
            "temperature": rng.uniform(180, 230, n),
            "line_speed": rng.uniform(1, 5, n),
        }
    )
    df["quality"] = np.where(df.temperature > 205, "pass", "fail")
    model = QualityModel(random_state=0).fit(df, "quality")
    ranges = model.optimal_ranges()

    assert "line_speed" in ranges, "the parameter should still be listed, just explained"
    assert "line_speed" in model.uninformative_parameters
    assert any("line_speed" in note and "too little" in note for note in model.notes)
    assert "temperature" not in model.uninformative_parameters


def test_the_warning_travels_with_the_result_not_just_the_model():
    """predict_quality copies notes; the window warning has to be in that copy."""
    result = predict_quality(_noise_frame(), "strength")
    assert any("too little" in note for note in result.notes)
    assert "too little" in result.summary()
    json.dumps(result.to_dict())


def test_a_real_driver_keeps_a_tight_window():
    """The floor must not flatten everything: a genuine driver still gets a window."""
    df = _noise_frame()
    model = QualityModel(random_state=0).fit(df, "strength")
    low, high = model.optimal_ranges()["pressure"]
    assert (high - low) / _span(df["pressure"]) < 0.6
    assert high > 12.0
    assert model.uninformative_parameters == ["ambient_rh"]


# -------------------------------------------- minor: non-English pass/fail


@pytest.mark.parametrize(
    "good, bad",
    [
        ("pass", "fail"),
        ("gut", "schlecht"),
        ("bueno", "malo"),
        ("パス", "不合格"),
        ("réussi", "échec"),
        ("合格", "不良"),
    ],
)
def test_non_english_labels_do_not_invert_the_windows(good, bad):
    """gut/schlecht used to pick the BAD label and report the inverted window."""
    rng = np.random.default_rng(0)
    temperature = rng.uniform(180, 230, 150)
    df = pd.DataFrame({"temperature": temperature})
    df["q"] = np.where(temperature > 205, good, bad)

    model = QualityModel(random_state=0).fit(df, "q")
    assert model.good_class == good
    assert model.good_class_guessed is False
    low, high = model.optimal_ranges()["temperature"]
    assert low > 200.0, "the good-outcome window came out upside down"


def test_an_unreadable_pair_says_so_loudly_in_the_windows_heading():
    """With nothing to read, the fallback must name itself where the windows are."""
    rng = np.random.default_rng(0)
    temperature = rng.uniform(180, 230, 150)
    df = pd.DataFrame({"temperature": temperature})
    df["q"] = np.where(temperature > 205, "alpha", "omega")

    result = predict_quality(df, "q")
    assert result.model.good_class_guessed is True
    text = result.summary()
    heading = [ln for ln in text.splitlines() if ln.startswith("SETTINGS MOST")][0]
    assert "good outcome" in heading
    assert "good_class=" in heading
    assert str(result.good_class) in heading


# ---------------------------------------- minor: numpy reprs in human text


def test_numeric_class_labels_print_without_a_numpy_wrapper():
    """0/1 is the commonest pass/fail encoding; it must not read np.int64(1)."""
    rng = np.random.default_rng(0)
    temperature = rng.uniform(180, 230, 120)
    df = pd.DataFrame(
        {"temperature": temperature, "pass_fail": (temperature > 205).astype(int)}
    )

    result = predict_quality(df, "pass_fail")
    text = result.summary()
    assert "np.int64" not in text
    assert "np.float64" not in text
    classes_line = [ln for ln in text.splitlines() if ln.startswith("classes:")][0]
    assert classes_line.startswith("classes: 0, 1 | good outcome: 1")
    assert not any("np.int64" in note for note in result.notes)
    assert result.to_dict()["classes"] == [0, 1]


def test_boolean_labels_also_print_plainly():
    """A True/False target is the other common encoding."""
    rng = np.random.default_rng(1)
    temperature = rng.uniform(180, 230, 120)
    df = pd.DataFrame({"temperature": temperature, "ok": temperature > 205})
    text = predict_quality(df, "ok").summary()
    assert "np.bool_" not in text
    assert "classes: False, True" in text


def test_constant_numeric_target_is_called_a_value_and_prints_plainly():
    """The message used to read 'one distinct class (np.float64(5.0))'."""
    df = pd.DataFrame({"temperature": [180.0, 185.0, 190.0, 195.0], "pass_fail": 5.0})
    with pytest.raises(ValueError) as caught:
        QualityModel().fit(df, "pass_fail")
    message = str(caught.value)
    assert "only one distinct value (5)" in message
    assert "np.float64" not in message


def test_unknown_good_class_error_prints_plain_labels():
    """The list of real classes in that error went through repr() too."""
    rng = np.random.default_rng(0)
    temperature = rng.uniform(180, 230, 60)
    df = pd.DataFrame(
        {"temperature": temperature, "pass_fail": (temperature > 205).astype(int)}
    )
    with pytest.raises(ValueError) as caught:
        QualityModel().fit(df, "pass_fail", good_class=7)
    message = str(caught.value)
    assert "not one of the classes" in message
    assert "np.int64" not in message


# -------------------------------- minor: row count versus prediction count


def test_prediction_count_reconciles_with_the_rows_that_had_no_outcome():
    """'rows: 90' beside 'PREDICTIONS (100 row(s))' read like an arithmetic error."""
    rng = np.random.default_rng(4)
    n = 100
    df = pd.DataFrame({"t": rng.uniform(180, 230, n), "p": rng.uniform(8, 14, n)})
    df["q"] = np.where(df.t > 205, "pass", "fail")
    df.loc[df.index[:10], "q"] = np.nan

    result = predict_quality(df, "q")
    assert result.n_rows == 90
    assert len(result.predictions) == 100
    assert result.model.n_dropped_no_outcome == 10

    line = [ln for ln in result.summary().splitlines() if ln.startswith("PREDICTIONS")][0]
    assert "100 row(s)" in line
    assert "the 90 above plus the 10 with no recorded q" in line


def test_prediction_line_stays_quiet_when_nothing_was_dropped():
    """The reconciliation clause must not appear when there is nothing to explain."""
    rng = np.random.default_rng(4)
    n = 60
    df = pd.DataFrame({"t": rng.uniform(180, 230, n)})
    df["q"] = np.where(df.t > 205, "pass", "fail")
    result = predict_quality(df, "q")
    line = [ln for ln in result.summary().splitlines() if ln.startswith("PREDICTIONS")][0]
    assert "plus the" not in line
    assert result.model.n_dropped_no_outcome == 0
