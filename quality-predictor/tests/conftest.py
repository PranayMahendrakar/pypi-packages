"""Shared fixtures: small, deterministic factory data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quality_predictor import QualityModel

N_ROWS = 90


def _base(seed: int = 7) -> pd.DataFrame:
    """Process parameters for a run of parts, with a believable amount of noise."""
    rng = np.random.RandomState(seed)
    temperature = rng.uniform(180.0, 230.0, N_ROWS)
    pressure = rng.uniform(8.0, 14.0, N_ROWS)
    speed = rng.uniform(40.0, 90.0, N_ROWS)
    machine = rng.choice(["line-1", "line-2", "line-3"], N_ROWS)
    operator = rng.choice(["ana", "bo", "chen"], N_ROWS)
    return pd.DataFrame(
        {
            "temperature": temperature,
            "pressure": pressure,
            "speed": speed,
            "machine": machine,
            "operator": operator,
        }
    )


@pytest.fixture(scope="session")
def runs() -> pd.DataFrame:
    """A pass/fail data set where temperature and pressure decide the outcome."""
    df = _base()
    good = (df["temperature"] > 205.0) & (df["pressure"] > 10.5)
    df["quality"] = np.where(good, "pass", "fail")
    return df


@pytest.fixture(scope="session")
def measurements() -> pd.DataFrame:
    """A numeric-target data set: a strength reading driven by temperature."""
    df = _base(seed=11)
    df["strength"] = (
        0.45 * df["temperature"] + 1.8 * df["pressure"] - 0.2 * df["speed"] + 12.0
    )
    return df


@pytest.fixture(scope="session")
def fitted(runs: pd.DataFrame) -> QualityModel:
    """One fitted classification model, shared so the suite stays quick."""
    return QualityModel(random_state=0).fit(runs, "quality")


@pytest.fixture(scope="session")
def fitted_regression(measurements: pd.DataFrame) -> QualityModel:
    """One fitted regression model, shared so the suite stays quick."""
    return QualityModel(random_state=0).fit(measurements, "strength")


@pytest.fixture(scope="session")
def quickstart() -> pd.DataFrame:
    """The exact frame printed in the README quickstart."""
    return pd.DataFrame(
        {
            "temperature": [
                212, 188, 219, 195, 208, 191, 221, 186,
                205, 199, 215, 184, 210, 193, 217, 190,
            ],
            "pressure": [
                12.2, 9.4, 12.8, 10.1, 11.9, 9.7, 12.9, 9.1,
                11.6, 10.4, 12.4, 8.9, 12.0, 9.9, 12.6, 9.3,
            ],
            "machine": [
                "A", "A", "B", "A", "B", "B", "A", "B",
                "A", "B", "A", "A", "B", "A", "A", "B",
            ],
            "quality": [
                "pass", "fail", "pass", "fail", "pass", "fail", "pass", "fail",
                "pass", "fail", "pass", "fail", "pass", "fail", "pass", "fail",
            ],
        }
    )


@pytest.fixture()
def unicode_runs() -> pd.DataFrame:
    """Non-ASCII column names and category values, for encoding safety."""
    return pd.DataFrame(
        {
            "température": [212, 188, 219, 195, 208, 191, 221, 186, 205, 199, 215, 184],
            "opérateur": ["Müller", "北京", "Müller", "北京", "Müller", "北京",
                          "Müller", "北京", "Müller", "北京", "Müller", "北京"],
            "qualité": ["réussi", "échec", "réussi", "échec", "réussi", "échec",
                        "réussi", "échec", "réussi", "échec", "réussi", "échec"],
        }
    )
